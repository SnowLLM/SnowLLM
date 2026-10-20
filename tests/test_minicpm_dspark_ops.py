# SPDX-License-Identifier: Apache-2.0

import gc
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

import _harness
from snowllm import ops
from snowllm.engine import (_draft_geometry, _draft_source, draft_device_weight_bytes,
                            draft_load_peak_bytes)
from snowllm.models.qwen3_5.dflash_draft import load


def norm(x, weight, eps):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()


def reference(src, geo, taps, noise, lengths, block):
    def weight(name):
        return src.dense(name).cuda().bfloat16()
    def linear(x, name):
        return F.linear(x, weight(name))
    ctx = norm(linear(taps, "fc.weight"), weight("hidden_norm.weight"), geo.eps).bfloat16()
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    pos = torch.tensor([p + 37 for n in lengths for p in range(n + block)], device="cuda")
    freq = torch.arange(0, geo.head_size, 2, device="cuda").float() / geo.head_size
    angle = pos[:, None].float() * (geo.rope_theta ** -freq)[None]
    cos, sin = angle.cos().repeat(1, 2), angle.sin().repeat(1, 2)
    def rotate(x, gamma, at):
        x = norm(x, gamma, geo.eps)
        a, b = x.chunk(2, dim=-1)
        rotated = torch.cat((-b, a), -1)
        return (x * cos[at, None] + rotated * sin[at, None]).bfloat16()
    ctx_rows, draft_rows, at = [], [], 0
    for n in lengths:
        ctx_rows.extend(range(at, at + n))
        draft_rows.extend(range(at + n, at + n + block))
        at += n + block
    h = noise
    for i in range(geo.num_layers):
        p = f"layers.{i}."
        x = norm(h, weight(p + "input_layernorm.weight"), geo.eps).bfloat16()
        q = linear(x, p + "self_attn.q_proj.weight").view(-1, geo.num_heads, geo.head_size)
        k = linear(x, p + "self_attn.k_proj.weight").view(-1, geo.num_kv_heads, geo.head_size)
        v = linear(x, p + "self_attn.v_proj.weight").view(-1, geo.num_kv_heads, geo.head_size)
        ck = linear(ctx, p + "self_attn.k_proj.weight").view(-1, geo.num_kv_heads, geo.head_size)
        cv = linear(ctx, p + "self_attn.v_proj.weight").view(-1, geo.num_kv_heads, geo.head_size)
        q = rotate(q, weight(p + "self_attn.q_norm.weight"), draft_rows)
        k = rotate(k, weight(p + "self_attn.k_norm.weight"), draft_rows)
        ck = rotate(ck, weight(p + "self_attn.k_norm.weight"), ctx_rows)
        outputs = []
        for b, n in enumerate(lengths):
            dr = slice(b*block, (b+1)*block)
            cr = slice(offsets[b], offsets[b+1])
            key = torch.cat((ck[cr], k[dr])).repeat_interleave(geo.num_heads // geo.num_kv_heads, 1)
            val = torch.cat((cv[cr], v[dr])).repeat_interleave(geo.num_heads // geo.num_kv_heads, 1)
            scores = torch.einsum("thd,shd->hts", q[dr].float(), key.float()) * geo.head_size ** -.5
            out = torch.einsum("hts,shd->thd", scores.softmax(-1), val.float())
            outputs.append(out.reshape(block, -1).bfloat16())
        h = (h + linear(torch.cat(outputs), p + "self_attn.o_proj.weight")).bfloat16()
        x = norm(h, weight(p + "post_attention_layernorm.weight"), geo.eps).bfloat16()
        gate = linear(x, p + "mlp.gate_proj.weight").float()
        up = linear(x, p + "mlp.up_proj.weight").float()
        h = (h + linear((F.silu(gate)*up).bfloat16(), p + "mlp.down_proj.weight")).bfloat16()
    return norm(h, weight("norm.weight"), geo.eps)


def main():
    root = Path(os.environ.get("SNOWLLM_MINICPM_DSPARK",
                Path.home() / "models/MiniCPM-V-4.7-35B-A3B-DSpark"))
    if not root.exists():
        _harness.skip(f"no DSpark checkpoint at {root}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(4728)
    ck = _harness.Checks(62)
    geo = _draft_geometry(str(root))
    ck("official taps are layer outputs, shared by BF16 and GGUF",
       geo.tap_layers == (3, 11, 19, 27, 35))
    ck("anchor-first DSpark block and Markov rank", geo.block_size == 15 and geo.markov_rank == 256)
    planned = draft_device_weight_bytes(str(root), geo)
    planned_peak = draft_load_peak_bytes(str(root), geo)
    src = _draft_source(str(root), dry=False)
    before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    torch.cuda.reset_peak_memory_stats()
    draft = load(src, geo, [], ops.Arena(), 16)
    live = torch.cuda.memory_stats()["requested_bytes.all.current"] - before
    peak = torch.cuda.memory_stats()["requested_bytes.all.peak"] - before
    ck("planned weight storage equals actual requested storage", planned == live,
       f"planned={planned}, actual={live}")
    ck("loading and conversion peak equals shared dry-load plan", peak == planned_peak,
       f"planned={planned_peak}, actual={peak}")
    print(f"DSpark resident weights: {live / (1<<20):.2f} MiB")
    for block, lengths in ((1, [3]), (4, [3, 9]), (8, [7]), (15, [3, 9])):
        b, page = len(lengths), 16
        tables = torch.arange(b*4, dtype=torch.int32, device="cuda").view(b, 4)
        draft.pools = [(torch.zeros(b*4*page*geo.kv_dim, dtype=torch.bfloat16, device="cuda"),
                        torch.zeros(b*4*page*geo.kv_dim, dtype=torch.bfloat16, device="cuda"))
                       for _ in range(geo.num_layers)]
        taps = torch.randn(sum(lengths), geo.fc_k, dtype=torch.bfloat16, device="cuda")
        noise = torch.randn(b*block, geo.hidden, dtype=torch.bfloat16, device="cuda")
        pos = torch.tensor([p+37 for n in lengths for p in range(n)], dtype=torch.int64, device="cuda")
        slots = torch.tensor([i*4*page+p for i,n in enumerate(lengths) for p in range(n)],
                             dtype=torch.int32, device="cuda")
        with draft.arena.frame():
            feat = draft.context_feature(taps)
            expected = norm(F.linear(taps, src.dense("fc.weight").cuda().bfloat16()),
                            src.dense("hidden_norm.weight").cuda(), geo.eps)
            ck.close(f"context projection and norm block={block}", feat, expected, .008)
            draft.write_context(feat, pos, slots)
        pos = torch.tensor([p+37 for n in lengths for p in range(n,n+block)], dtype=torch.int64, device="cuda")
        slots = torch.tensor([i*4*page+p for i,n in enumerate(lengths) for p in range(n,n+block)],
                             dtype=torch.int32, device="cuda")
        seq = torch.tensor([n+block for n in lengths], dtype=torch.int32, device="cuda")
        actual = draft.forward(noise, pos, slots, tables, seq)
        expected = reference(src, geo, taps, noise, lengths, block)
        ck.close(f"five-layer noncausal backbone B={b} block={block}", actual, expected, .04)
    b, block, vocab = 2, 4, geo.vocab_size
    hidden = torch.randn(b*block, geo.hidden, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn(b*block, draft.w["markov_w1"].shape[0], device="cuda") * 2
    logits[:, vocab:] = -float("inf")
    anchor = torch.tensor([123, 456], dtype=torch.int64, device="cuda")
    actual = torch.empty(b, block, dtype=torch.int64, device="cuda")
    conf = torch.empty(b, block, device="cuda")
    draft.propose(hidden, logits, anchor, actual, conf)
    w1 = src.dense("markov_head.markov_w1.weight").cuda().bfloat16()
    w2 = src.dense("markov_head.markov_w2.weight").cuda().bfloat16()
    cw = src.dense("confidence_head.proj.weight").cuda().float().view(-1)
    cb = src.dense("confidence_head.proj.bias").cuda().float()
    previous, expected, confidence = anchor, [], []
    for t in range(block):
        latent = w1[previous]
        bias = F.linear(latent, w2).float()
        ids = (logits.view(b, block, -1)[:, t, :vocab] + bias).argmax(-1)
        expected.append(ids)
        features = torch.cat((hidden.view(b, block, -1)[:, t].float(), latent.float()), -1)
        confidence.append((features @ cw + cb).sigmoid())
        previous = ids
    ck.exact("sequential Markov tokens versus independent PyTorch", actual, torch.stack(expected, 1))
    ck.close("confidence uses normalized hidden and previous-token latent", conf,
             torch.stack(confidence, 1), .0001)
    ck("padded vocabulary never enters a proposal", bool((actual < vocab).all()))
    src.close()
    del draft
    gc.collect()
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())

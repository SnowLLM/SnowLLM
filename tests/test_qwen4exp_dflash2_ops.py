# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

import _harness
from snowllm import ops
from snowllm.engine import _draft_geometry, _draft_source, draft_device_weight_bytes
from snowllm.models.qwen3_5.dflash_draft import context_workspace_bytes, load


def check_window(ck, geo, block, lengths, window):
    d = ops.dflash
    page = 16
    batch = len(lengths)
    pages = (max(lengths) + page - 1) // page
    tables = torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(batch, pages)
    kc = torch.zeros(batch * pages * page * geo.kv_dim, dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    expected = []
    for seq, length in enumerate(lengths):
        key = torch.zeros(length, geo.kv_dim, dtype=torch.bfloat16, device="cuda")
        value = torch.full_like(key, seq * 4)
        first = length - block
        marked = min(length - 1, max(0, first - window + 1) + 2)
        value[marked] += 1024
        positions = torch.arange(length, device="cuda", dtype=torch.int64)
        slots = (tables[seq, positions // page] * page + positions % page).int()
        d.reshape_and_cache(key, value, kc, vc, slots, page)
        for query in range(first, length):
            lo, hi = max(0, query - window + 1), min(length, query + window)
            expected.append(seq * 4 + (1024 / (hi - lo) if lo <= marked < hi else 0))
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    view, view_lens = ops.paged_window_view(
        tables, lens, 0, window + block + page - 2, page, end_offset=-1)
    count = d.paged_attn_num_slots(batch)
    plan = d.paged_attn_plan(view_lens, count, page)
    workspace = torch.empty(d.paged_attn_workspace_size(count, block),
                            dtype=torch.uint8, device="cuda")
    query = torch.zeros(batch * block, geo.num_heads, geo.head_size,
                        dtype=torch.bfloat16, device="cuda")
    actual = torch.empty_like(query)
    d.paged_attn(query, view_lens, kc, vc, actual, view, plan, count, workspace,
                 geo.head_size ** -.5, block, page, window=window)
    expected = torch.tensor(expected, device="cuda").view(-1, 1, 1).expand_as(actual)
    ck(f"window={window} block={block}: exact per-query boundaries and paged batches",
       torch.allclose(actual.float(), expected, rtol=.005, atol=.002),
       f"max delta={(actual.float() - expected).abs().max().item():.5f}")


def check_selector(ck, geo, draft, source, block):
    batch, length = 2, block - 1
    rank, top_k = geo.selector_rank, geo.selector_top_k
    hidden = torch.randn(batch * length, rank).bfloat16()
    logits = torch.randn(batch * length, geo.vocab_size)
    anchor = torch.randint(geo.vocab_size, (batch,))
    values, candidates = logits.topk(top_k, dim=-1)
    predecessor = source.dense("candidate_selector.predecessor_codebook").float()
    successor = source.dense("candidate_selector.successor_codebook").float()
    expected = []
    for seq in range(batch):
        previous, path = anchor[seq], []
        for position in range(length):
            row = seq * length + position
            scores = values[row] + (successor[candidates[row]]
                                    * predecessor[previous] * hidden[row].float()).sum(-1)
            previous = candidates[row, scores.argmax()]
            path.append(previous.item())
        expected.append(path)
    actual = torch.empty(batch, length, dtype=torch.int64, device="cuda")
    scratch = torch.empty(ops.dflash.select_scratch_bytes(batch * length),
                          dtype=torch.uint8, device="cuda")
    ops.dflash.select_path(hidden.cuda(), logits.cuda(), draft.w["sel_pred"], draft.w["sel_succ"],
                           anchor.cuda(), actual, scratch)
    ck.exact(f"selector block={block}: independent predecessor-conditioned walk",
             actual.cpu(), torch.tensor(expected))


def main():
    root = Path(os.environ.get("SNOWLLM_QWEN4EXP_DFLASH2",
                Path.home() / "models/Qwen3.8-Flash-Next-UD-Q3_K_XL/dflash2"))
    if not (root / "model.safetensors").is_file():
        _harness.skip(f"no BF16 Flash-Next drafter at {root}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(73)
    ck = _harness.Checks(60)
    geo = _draft_geometry(str(root))
    planned = draft_device_weight_bytes(str(root), geo)
    src = _draft_source(str(root), dry=False)
    ck("published checkpoint contains only BF16 weights",
       all(t.dtype == torch.bfloat16 for t in src.sd.values()))
    torch.cuda.empty_cache()
    def requested_bytes():
        return sum(b["requested_size"] for s in torch.cuda.memory_snapshot() for b in s["blocks"]
                   if b["state"] == "active_allocated")

    before = requested_bytes()
    draft = load(src, geo, [], ops.Arena(), 16)
    live = requested_bytes() - before
    ck("planned BF16 storage matches actual allocation requests",
       planned == live, f"planned={planned}, requested={live}")
    d = ops.dflash
    for block, lengths in ((1, (2048, 2051)), (4, (2049, 2054)), (8, (2054, 8196)),
                           (15, (2060, 8192)), (16, (2070, 4096))):
        check_window(ck, geo, block, lengths, geo.sliding_window)
    for block in (4, 8):
        check_window(ck, geo, block, (23, 38), 5)
        check_selector(ck, geo, draft, src, block)
    p = d.Proj
    cases = [
        (p.FC, draft.w["fc"], ["fc.weight"]),
        (p.CTX_KV, draft.w["ctx_kv"], [f"layers.{i}.self_attn.{kv}_proj.weight"
                                     for i in range(geo.num_layers) for kv in ("k", "v")]),
        (p.QKV, draft.w["layers"][0]["qkv"],
         [f"layers.0.self_attn.{qkv}_proj.weight" for qkv in ("q", "k", "v")]),
        (p.O, draft.w["layers"][0]["o"], ["layers.0.self_attn.o_proj.weight"]),
        (p.GATE_UP, draft.w["layers"][0]["gate_up"],
         ["layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"]),
        (p.DOWN, draft.w["layers"][0]["down"], ["layers.0.mlp.down_proj.weight"]),
        (p.CONV, draft.w["layers"][0]["attn_conv"]["proj"],
         ["layers.0.attention_conv.kernel_projection.weight"]),
        (p.SELECTOR_HIDDEN, draft.w["sel_hidden"],
         ["candidate_selector.hidden_projection.weight"]),
    ]
    for which, packed, names in cases:
        weight = torch.cat([src.dense(name) for name in names]).float()
        n, k = weight.shape
        ck(which.name + ": native dimensions match checkpoint", d.proj_shape(which) == (n, k))
        for m in (1, 8, 129):
            x = torch.randn(m, k).bfloat16()
            actual = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            scratch = torch.empty(d.proj_workspace_bytes(which, packed, m), dtype=torch.uint8,
                                  device="cuda")
            d.proj(which, x.cuda(), packed, scratch, actual)
            ck.close(f"{which.name} M={m}: versus CPU FP32 linear", actual.cpu(),
                     F.linear(x.float(), weight), .005)

    for block in (4, 8):
        m = 2 * block
        x = torch.randn(m, geo.hidden, device="cuda", dtype=torch.bfloat16)
        dyn = torch.randn(m, geo.conv_proj_n, device="cuda", dtype=torch.bfloat16)
        base = draft.w["layers"][0]["attn_conv"]["base"]
        actual = torch.empty_like(x)
        for site in (0, 1):
            d.dyn_conv(x, dyn, base, actual, block, site)
            coeff = (dyn.float().view(m, 2, 2, -1)[:, site]
                     .repeat_interleave(geo.conv_group, dim=-1) + base[site].float())
            previous = torch.zeros_like(x)
            previous.view(2, block, -1)[:, 1:] = x.view(2, block, -1)[:, :-1]
            expected = coeff[:, 0] * x.float() + coeff[:, 1] * previous.float()
            ck.close(f"conv block={block} site={site}: independent request boundaries",
                     actual, expected, .005)
    for rows in (1, 8, 129):
        arena = draft.arena = ops.Arena()
        needed = context_workspace_bytes(geo, rows)
        arena.freeze(needed)
        x = torch.randn(rows, geo.fc_k, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(rows, dtype=torch.int64, device="cuda")
        slots = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
        for _ in range(2):
            with arena.frame():
                features = draft.context_feature(x)
                draft.write_context(features, positions, slots)
        ck(f"context M={rows}: measured arena fits exact plan and reuses storage",
           arena.high == needed and arena.at == 0, f"{arena.high} vs {needed}")
    src.close()
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

import _harness
from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.checkpoint.gguf.dflash import GGUFDraftSource
from snowllm.engine import Engine, SamplingParams


def norm(x, weight, eps):
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
            * weight.float()).to(torch.bfloat16)


def rotary(x, weight, angle, eps):
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    left, right = x.chunk(2, -1)
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    return torch.cat((left*cos-right*sin, right*cos+left*sin), -1).to(torch.bfloat16)


def cache_rows(pool, pages, heads, dim, page, *, key):
    x = pool.view(torch.bfloat16)
    if key:
        x = x.view(-1, heads, dim//16, page, 16)[pages].permute(0, 3, 1, 2, 4)
    else:
        x = x.view(-1, heads, page//4, dim, 4)[pages].permute(0, 2, 4, 1, 3)
    return x.reshape(-1, heads, dim)


@torch.no_grad()
def main():
    root = pathlib.Path(os.environ.get('SNOWLLM_LFM_DIR',
        pathlib.Path.home()/'models/LFM2.5-8B-A1B-Q4_K_M'))
    draft_root = pathlib.Path(os.environ.get('SNOWLLM_LFM_DSPARK_DIR',
        pathlib.Path.home()/'models/LFM2.5-8B-A1B-DSpark-Q8_0'))
    if not any(root.glob('*.gguf')) or not any(draft_root.glob('*.gguf')):
        _harness.skip('no complete LFM target/draft checkpoint')
    tok, _ = loader.load_tokenizer(root)
    model = loader.load(root, mtp=False, vision=False,
                        kv=dict(ctx=4096, slots=2, util=.9, chunk=512))
    engine = Engine(model, max_num_seqs=2, max_model_len=4096, prefill_chunk=512,
        dflash_path=str(draft_root), dflash_block=9, enforce_eager=True)
    draft, ck = engine.dflash.draft, _harness.Checks()
    g, page = draft.geo, draft.block_size
    names = ['norm.weight', 'hidden_norm.weight', 'fc.weight',
             'confidence_head.proj.weight',
             'markov_head.markov_w1.weight', 'markov_head.markov_w2.weight']
    names += [f'layers.{i}.{key}' for i in range(g.num_layers) for key in (
        'input_layernorm.weight', 'post_attention_layernorm.weight',
        'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
        'self_attn.o_proj.weight', 'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
        'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')]
    src = GGUFDraftSource(next(draft_root.glob('*.gguf')))
    if src.has('confidence_head.proj.bias'):
        names.append('confidence_head.proj.bias')
    weights = {name: src.dense(name) for name in names}
    src.close()
    context = draft.context_feature
    def feature(taps):
        actual = context(taps)
        expected = norm(F.linear(taps, weights['fc.weight']),
                        weights['hidden_norm.weight'], g.eps)
        ck.close('draft context projection and norm', actual, expected, .02)
        return actual
    draft.context_feature = feature
    write_context = draft.write_context
    def write(feat, positions, slots):
        feat, positions, slots = feat.clone(), positions.clone(), slots.clone()
        write_context(feat, positions, slots)
        live = slots >= 0
        feat, positions, slots = feat[live], positions[live], slots[live].long()
        angle = positions[:, None].float()*draft.inv_freq[None]
        blocks, cells = slots//page, slots % page
        for i, (kc, vc) in enumerate(draft.pools):
            w = lambda key: weights[f'layers.{i}.{key}']
            k = rotary(F.linear(feat, w('self_attn.k_proj.weight')).view(
                -1, g.num_kv_heads, g.head_size), w('self_attn.k_norm.weight'), angle, g.eps)
            v = F.linear(feat, w('self_attn.v_proj.weight')).view_as(k)
            actual_k = kc.view(torch.bfloat16).view(
                -1, g.num_kv_heads, g.head_size//16, page, 16)[blocks, :, :, cells, :].view_as(k)
            actual_v = vc.view(torch.bfloat16).view(
                -1, g.num_kv_heads, page//4, g.head_size, 4)[blocks, :, cells//4, :, cells % 4]
            ck.close(f'draft context layer {i} K', actual_k, k, .005)
            ck.close(f'draft context layer {i} V', actual_v, v, .005)
    draft.write_context = write
    original = draft.forward
    def forward(noise, positions, slots, tables, lens):
        # Arena metadata may be reused after the native forward consumes it.
        positions, lens = positions.clone(), lens.clone()
        captured = []
        native = ops.dflash.paged_attn
        def attention(*args, **kwargs):
            native(*args, **kwargs)
            captured.append((args[0].clone(), args[4].clone()))
        ops.dflash.paged_attn = attention
        try:
            actual = original(noise, positions, slots, tables, lens).clone()
        finally:
            ops.dflash.paged_attn = native
        rows, length = noise.shape[0], int(lens[0])
        past = length-rows
        angle = positions[:, None].float()*draft.inv_freq[None]
        pages = tables[0, :ops.kv_blocks_for(length, page)].long()
        groups = g.num_heads//g.num_kv_heads
        def attend(q, k, v):
            k = k.repeat_interleave(groups, 1).transpose(0, 1)
            v = v.repeat_interleave(groups, 1).transpose(0, 1)
            scores = q.transpose(0, 1).float() @ k.float().transpose(-1, -2)
            probabilities = (scores*g.head_size**-.5).softmax(-1).to(torch.bfloat16)
            return (probabilities @ v).transpose(0, 1).contiguous().view(rows, -1)
        h = noise.clone()
        for i, (kc, vc) in enumerate(draft.pools):
            w = lambda key: weights[f'layers.{i}.{key}']
            x = norm(h, w('input_layernorm.weight'), g.eps)
            q = rotary(F.linear(x, w('self_attn.q_proj.weight')).view(
                rows, g.num_heads, g.head_size), w('self_attn.q_norm.weight'), angle, g.eps)
            k = rotary(F.linear(x, w('self_attn.k_proj.weight')).view(
                rows, g.num_kv_heads, g.head_size), w('self_attn.k_norm.weight'), angle, g.eps)
            v = F.linear(x, w('self_attn.v_proj.weight')).view(rows, g.num_kv_heads, g.head_size)
            keys = cache_rows(kc, pages, g.num_kv_heads, g.head_size, page, key=True)[:length]
            values = cache_rows(vc, pages, g.num_kv_heads, g.head_size, page, key=False)[:length]
            ck.close(f'draft layer {i} attention at identical QKV', captured[i][1],
                     attend(captured[i][0], keys, values), .005)
            attn = attend(q, torch.cat((keys[:past], k)), torch.cat((values[:past], v)))
            h = h+F.linear(attn, w('self_attn.o_proj.weight'))
            x = norm(h, w('post_attention_layernorm.weight'), g.eps)
            gate, up = (F.linear(x, w(f'mlp.{name}_proj.weight')) for name in ('gate', 'up'))
            act = (F.silu(gate.float())*up.float()).to(torch.bfloat16)
            h = h+F.linear(act, w('mlp.down_proj.weight'))
        expected = norm(h, weights['norm.weight'], g.eps)
        ck.close('five-layer draft backbone', actual, expected, .03)
        scratch = torch.empty(ops.lm_head_scratch_bytes(rows), device='cuda', dtype=torch.uint8)
        logits = model.lm_head(actual, engine.runner.draft_logits, scratch=scratch).clone()
        anchor = engine.dflash.d_i64[engine.dflash.width:engine.dflash.width+1].clone()
        previous, reference, confidence_ref = anchor, [], []
        conf_bias = weights.get('confidence_head.proj.bias')
        for row, hidden in zip(logits, actual):
            latent = weights['markov_head.markov_w1.weight'][previous]
            bias = F.linear(latent, weights['markov_head.markov_w2.weight'])
            confidence_ref.append(F.linear(torch.cat((hidden, latent[0])).float(),
                weights['confidence_head.proj.weight'].float().view(1, -1),
                conf_bias.float().view(1) if conf_bias is not None else None).sigmoid())
            previous = (row.float()+bias[0].float()).argmax().view(1)
            reference.append(int(previous))
        path = torch.empty(1, rows, device='cuda', dtype=torch.int64)
        confidence = torch.empty(1, rows, device='cuda', dtype=torch.float32)
        workspace = torch.empty(ops.dspark_markov_workspace_bytes(1, g.vocab_size,
            g.markov_rank, rows), device='cuda', dtype=torch.uint8)
        ops.dspark_markov(logits, draft.w['markov_w1'], draft.w['markov_w2'], anchor,
            rows, pre_head=actual, conf_weight=draft.w['conf_weight'],
            conf_bias=draft.w['conf_bias'], confidence=confidence, workspace=workspace, out=path)
        ck('Markov proposals match PyTorch', path[0].tolist() == reference)
        ck.close('confidence head matches PyTorch', confidence,
                 torch.cat(confidence_ref).view_as(confidence), 1e-5)
        return actual
    draft.forward = forward
    corpus = (pathlib.Path(__file__).parents[1]/'benchmarks/data/pg19_context_bench.txt').read_text()
    ids = tok.encode(corpus.split('\n<<<SNOWLLM-BENCH-SPLIT>>>\n')[0], add_special_tokens=False)
    request = engine.add(ids[:2048], SamplingParams(temperature=0, max_new_tokens=32))
    while not request.prefilled:
        engine.step()
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())

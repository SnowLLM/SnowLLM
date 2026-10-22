# SPDX-License-Identifier: Apache-2.0
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.engine import Engine, Request, SamplingParams


def cache_checks(ck):
    e = Engine.__new__(Engine)
    e._active_factor, e._loaded_rope_factor = None, 1.
    e.runner = SimpleNamespace(tunable_rope=True, set_rope=Mock(), block_size=16)
    e._rope_table = lambda factor: (torch.ones(32), factor)
    e.cache = Mock()
    e.cache.evict.side_effect = [True, False]
    e._alloc_blocks = lambda n, keep: None
    def lookup(tokens, limit):
        ck('factor switch evicts prefixes before lookup', e.cache.evict.call_count == 2)
        return None
    e.cache.lookup.side_effect = lookup
    r = Request(prompt=[1, 2, 3], params=SamplingParams(max_new_tokens=1), rope_factor=2.)
    e._admit(r)
    e._activate_rope(2.)
    ck('unchanged YaRN factor retains its prefix cache', e.cache.evict.call_count == 2)
    e.default_rope_factor, e._rope_orig_max, e.max_model_len = 2., 262144, 524288
    params = SamplingParams(max_new_tokens=262144)
    ck('Engine.add defaults to the configured extended context',
       e.add([1], params, queue=False).rope_factor == 2.)
    for factor, count in ((1., 262144), (None, 524288)):
        refused = False
        try:
            e.add([1], SamplingParams(max_new_tokens=count), rope_factor=factor, queue=False)
        except ops.SnowLLMError:
            refused = True
        ck(f'factor {factor}: request length limit remains enforced', refused)


def table_checks(ck):
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    cfg = dict(head_dim=256, max_position_embeddings=262144,
               rope_parameters=dict(rope_theta=10000000., partial_rotary_factor=.25))
    for factor in (1., 1.25, 2., 4.):
        inv, scale = loader.yarn_rope_table(cfg, factor)
        rp = dict(cfg['rope_parameters'], rope_type='yarn', factor=factor,
                  original_max_position_embeddings=262144)
        ref, ref_scale = _compute_yarn_parameters(Qwen4ExpTextConfig(**dict(cfg, rope_parameters=rp)),
                                                  device=torch.device('cpu'))
        ck.exact(f'factor {factor}: Transformers YaRN frequencies', inv, ref)
        ck(f'factor {factor}: Transformers attention scale', abs(scale - ref_scale) < 1e-7)


def index_checks(ck):
    from snowllm._capi import select_geometry, GEO_QWEN38_FLASH_NEXT
    select_geometry(GEO_QWEN38_FLASH_NEXT)
    M, D, ratio = 16, 128, 4
    raw = torch.randn(M, D, dtype=torch.bfloat16, device='cuda')
    gamma = torch.randn(D, dtype=torch.bfloat16, device='cuda')
    cfg = dict(head_dim=256, max_position_embeddings=262144,
               rope_parameters=dict(rope_theta=10000000., partial_rotary_factor=.25))
    inv, scale = loader.yarn_rope_table(cfg, 2.)
    inv = inv.cuda()
    mscale = torch.tensor([scale], dtype=torch.float32, device='cuda')
    positions = torch.arange(300000, 300000 + M, device='cuda').expand(3, M).contiguous()
    positions[1] += 100
    positions[2] += 200
    pool = torch.zeros(M // ratio + 1, D, dtype=torch.bfloat16, device='cuda')
    carry = torch.zeros(2, ratio - 1, D, dtype=torch.bfloat16, device='cuda')
    carry_pos = torch.zeros(2, ratio - 1, 3, dtype=torch.int64, device='cuda')
    slots = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    for lo, hi in ((0, 3), (3, 8)):
        rows = torch.tensor([*range(lo, hi), *range(8 + lo, 8 + hi)], device='cuda')
        chunk = raw[rows].contiguous()
        cu = torch.tensor([0, hi-lo, 2*(hi-lo)], dtype=torch.int32, device='cuda')
        lens = torch.tensor([hi, hi], dtype=torch.int32, device='cuda')
        ops.qwen4exp_qsa_produce(chunk, carry, carry_pos, slots, slots, cu, lens,
                                 positions[:, rows].contiguous(), rows.int(), gamma, inv,
                                 pool, ratio, 1e-6, mscale=mscale)
    pooled = raw.float().view(-1, ratio, D).mean(1)
    normed = (pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-6)
              * gamma.float()).bfloat16().float()
    starts = positions[:, ::ratio]
    cos = torch.empty(M // ratio, 64, dtype=torch.float32, device='cuda')
    sin = torch.empty_like(cos)
    ops.rope_cos_sin(starts.contiguous(), inv, cos, sin, mscale=mscale)
    x1, x2 = normed[:, :32], normed[:, 32:64]
    expected = normed.clone()
    expected[:, :32] = x1*cos[:, :32] - x2*sin[:, :32]
    expected[:, 32:64] = x2*cos[:, :32] + x1*sin[:, :32]
    ck.close('scaled pooled keys across chunks and shared snapshots', pool[:-1], expected, .006)
    ck.exact('pooled key nonrotary dimensions retain their scale', pool[:-1, 64:],
             expected[:, 64:].bfloat16())


def axis_checks(ck):
    for n_blocks in (65536, 65537, 131072, 262144):
        for tile_rows in (4, 16):
            rows, stride, ratio = tile_rows + 3, 16, 4
            picks = []
            cells = []
            counts = []
            for row in range(rows):
                chosen = sorted({0, 17, 65534, 65535, 65536, n_blocks//2, n_blocks-2})
                chosen = [x for x in chosen if x < n_blocks-1 and (x + row) % 3]
                if row % 7 == 0:
                    chosen = []
                counts.append(len(chosen))
                picks.append(chosen + [0]*(stride-len(chosen)))
                cells.append((n_blocks-1)*ratio + row % ratio)
            sel = torch.tensor(picks, dtype=torch.int32, device='cuda')
            cnt = torch.tensor(counts, dtype=torch.int32, device='cuda')
            pos = torch.tensor(cells, dtype=torch.int64, device='cuda')
            tiles = (rows + tile_rows - 1) // tile_rows
            width = ((tile_rows*(stride+1)*ratio + 63)//64)*64
            axis = torch.full((tiles, width), -1, dtype=torch.int32, device='cuda')
            lens = torch.zeros(tiles, dtype=torch.int32, device='cuda')
            mask = torch.full((rows, width), 23, dtype=torch.int8, device='cuda')
            ops.qwen4exp_qsa_tile_axis(sel, cnt, pos, axis, lens, mask, n_blocks, tile_rows, ratio)
            axis, lens, mask = axis.cpu(), lens.cpu(), mask.cpu()
            good = True
            for tile in range(tiles):
                first, end = tile*tile_rows, min(rows, (tile+1)*tile_rows)
                expected = set()
                visible = []
                for row in range(first, end):
                    blocks = set(picks[row][:counts[row]])
                    if (cells[row]+1) % ratio:
                        blocks.add(cells[row]//ratio)
                    expected.update(b*ratio+t for b in blocks for t in range(ratio))
                    visible.append({b*ratio+t for b in blocks for t in range(ratio)
                                    if b*ratio+t <= cells[row]})
                actual = axis[tile, :int(lens[tile])].tolist()
                good &= len(actual) == len(set(actual)) and set(actual) == expected
                for row in range(first, end):
                    good &= all((int(mask[row, j]) == 0) == (cell in visible[row-first])
                                for j, cell in enumerate(actual))
                rounded = ((len(actual)+63)//64)*64
                good &= bool((axis[tile, len(actual):rounded] == 0).all())
                good &= bool((mask[first:end, len(actual):rounded] != 0).all())
            ck(f'{n_blocks*ratio} tokens, tile {tile_rows}: exact QSA axis and masks', good)


def model_checks(ck):
    from snowllm import cli

    root = _harness.checkpoint('Qwen3.8-Flash-Next-UD-Q3_K_XL')
    state = cli.build(str(root), max_num_seqs=1, max_model_len=524288, num_spec=2,
                      num_kv_blocks=None, default_max_tokens=None, seed=71, profile_dir=None,
                      served_names=['auto', 'native=1', 'extended=4'],
                      prefill_chunk=8192, prefix_memory_ratio=0., gpu_memory_utilization=.95)
    e = state.engine.engine
    ck('512K automatically selects factor 2', e.default_rope_factor == 2.)
    ck('API aliases retain automatic and explicit context limits',
       [(a.factor, a.ctx) for a in state.aliases.values()] ==
       [(2., 524288), (1., 262144), (4., 524288)])
    ids = state.tokenizer.encode('The capital of France is', add_special_tokens=False)
    params = SamplingParams(temperature=0, max_new_tokens=8)
    for factor in (None, 1., None):
        r = e.add(ids, params, rope_factor=factor)
        e.run()
        ck(f'factor {factor}: graph and MTP decode finish', len(r.out) == 8)
        ck(f'factor {factor}: expected frequency table is active',
           torch.equal(state.model.inv_freq.cpu(), loader.yarn_rope_table(state.model.config,
                       2. if factor is None else factor)[0]))
        print(f'  output: {state.tokenizer.decode(r.out)!r}', flush=True)
    if os.environ.get('SNOWLLM_YARN_LONG'):
        text = ('Background filler. ' * 150000)
        filler = state.tokenizer.encode(text, add_special_tokens=False)[:270000]
        suffix = state.tokenizer.encode('\nWhat is 1+1? Answer with only the number:',
                                         add_special_tokens=False)
        r = e.add(filler + suffix, SamplingParams(temperature=0, max_new_tokens=16))
        ck('request beyond 256K is admitted with factor 2',
           len(r.prompt) > 262144 and r.rope_factor == 2.)
        e.run()
        out = state.tokenizer.decode(r.out)
        ck('request beyond 256K finishes and answers', r.done and '2' in out)
        print(f'  long output ({len(r.prompt)} input tokens): {out!r}', flush=True)
    state.engine._worker.shutdown(wait=True)


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(71)
    ck = _harness.Checks(70)
    table_checks(ck)
    cache_checks(ck)
    index_checks(ck)
    axis_checks(ck)
    if os.environ.get('SNOWLLM_YARN_MODEL'):
        model_checks(ck)
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())

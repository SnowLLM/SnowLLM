# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import numpy as np
import torch

from snowllm import _capi, ops
from snowllm.checkpoint.loader import load_gguf
from snowllm.engine.forward_context import PLAN_MAIN, Batch, Qwen4ExpContext
from snowllm.models.qwen4exp import ngram_rows

import _harness
from _reference import Reference

REF_DIR = pathlib.Path.home() / "SnowLLM-Kernels/fixtures/qwen4exp"
CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")
BLOCK = 16
PROBE = (0, 4, 20, 39, 46)
TOL_MIX, TOL_INJECT = 0.03, 0.005


def _err(got: torch.Tensor, want: torch.Tensor) -> float:
    g, w = got.float().cpu().flatten(), want.float().flatten()
    return ((g - w).norm() / w.norm()).item()


def _pools(m: object, T: int) -> None:
    geo = m.geo
    nb = ops.kv_blocks_for(T + BLOCK, BLOCK)
    for lay in m.model.layers:
        if lay.is_full:
            kb, vb = ops.kv_pool_bytes(nb, False, BLOCK)
            lay.attn.kv = (ops.empty_bytes(kb).zero_(), ops.empty_bytes(vb).zero_())
            ix, rows = lay.attn.indexer, nb * (BLOCK // geo.index_ratio) + 1
            ix.pool = torch.zeros(rows, geo.index_head_dim, dtype=torch.bfloat16, device="cuda")
            ix.carry = torch.zeros(2, geo.index_ratio - 1, geo.index_head_dim,
                                   dtype=torch.bfloat16, device="cuda")
            ix.carry_pos = torch.zeros(2, geo.index_ratio - 1, 3, dtype=torch.int64,
                                       device="cuda")
        else:
            lay.attn.state = (
                torch.zeros(2, geo.lin_conv_state, geo.lin_conv_dim, dtype=torch.bfloat16,
                            device="cuda"),
                torch.zeros(2, geo.lin_num_v_heads, geo.lin_head_k, geo.lin_head_v,
                            dtype=torch.float32, device="cuda"))


def _batch(toks: list[int]) -> Batch:
    T = len(toks)
    nb = ops.kv_blocks_for(T + BLOCK, BLOCK)
    total, qmap = ops.prefill_q_plan([T])
    return Batch(
        input_ids=torch.tensor(toks, dtype=torch.int64, device="cuda"),
        positions=torch.arange(T, dtype=torch.int64, device="cuda").expand(3, T).contiguous(),
        slot_mapping=torch.arange(T, dtype=torch.int32, device="cuda"),
        block_tables=torch.arange(nb, dtype=torch.int32, device="cuda").reshape(1, nb),
        seq_lens=torch.tensor([T], dtype=torch.int32, device="cuda"),
        state_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        has_state=torch.zeros(1, dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device="cuda"),
        total_q_blocks=total, q_block_map=qmap,
        last_row=torch.tensor([T - 1], dtype=torch.int64, device="cuda"),
        is_prefill=True, num_tokens=T, need_logits=True)


def main() -> int:
    ref = Reference(REF_DIR)
    if not ref:
        print(f"== skipped: no reference dump in {REF_DIR}")
        return 0
    if not _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT):
        _harness.skip("this build carries no Qwen3.8-Flash-Next geometry")
    ck = _harness.Checks(54)

    toks = ref.tokens
    T = len(toks)
    m = load_gguf(CKPT, mtp=False, vision=False)
    geo = m.geo
    _pools(m, T)

    seen: dict[int, torch.Tensor] = {}
    for i, lay in enumerate(m.model.layers):
        lay.register_forward_hook(
            lambda mod, args, out, i=i: seen.__setitem__(i, args[1].detach().clone()))

    ctx = Qwen4ExpContext(
        batch=_batch(toks), M=T, eps=geo.eps, path=ops.Path.PREFILL, arena=ops.Arena(),
        plan_key=PLAN_MAIN, num_slots=ops.paged_decode_num_slots(1), block_size=BLOCK, geo=geo,
        ple_emb=m.ple_table.gather(ngram_rows(np.asarray(toks), None, geo)),
        inv_freq=m.model.inv_freq,
        x=torch.zeros(T, geo.hidden, dtype=torch.bfloat16, device="cuda"),
        mscale=torch.ones(1, dtype=torch.float32, device="cuda"))
    out = m(ctx)
    logits = m.lm_head(out, torch.empty(1, geo.vocab_size, dtype=torch.float32, device="cuda"))
    torch.cuda.synchronize()

    errs = [_err(seen[i].reshape(T, -1), ref.get(f"l_last-{i}").reshape(T, -1))
            for i in range(len(m.model.layers) - 1)]
    ck("layer 0 lands inside one bf16 rounding of llama.cpp", errs[0] < 0.02, f"{errs[0]:.4f}")
    ck("and the drift down the stack stays under 20%", max(errs) < 0.20,
       f"worst {max(errs):.4f} at layer {errs.index(max(errs))}")

    e = _err(out.reshape(-1, geo.hidden), ref.get("result_norm").reshape(-1, geo.hidden)[-1:])
    ck("the head fold reproduces result_norm", e < 0.12, f"rel L2 {e:.4f}")

    want = ref.get("result_output").reshape(-1, geo.vocab_size)[-1]
    top = logits[0].topk(2).indices.tolist()
    ck("and the two most likely tokens are llama.cpp's",
       top == want.topk(2).indices.tolist(), f"{top} vs {want.topk(2).indices.tolist()}")

    x = torch.empty(T, geo.hidden, dtype=torch.bfloat16, device="cuda")
    o = torch.empty(T, geo.hidden, dtype=torch.bfloat16, device="cuda")
    bare = Qwen4ExpContext(batch=None, M=T, eps=geo.eps, path=ops.Path.PREFILL, arena=ops.Arena(),
                           plan_key=(), geo=geo, x=x)
    for i in PROBE:
        with bare.arena.frame():
            streams = ref.get(f"hc_combine-{i}").reshape(T, geo.hc_count, geo.hidden)
            mixed, inject = m.model.layers[i].mlp_hc(
                bare, streams.cuda().to(torch.bfloat16).contiguous())
            torch.cuda.synchronize()
            e = _err(mixed, ref.get(f"hc_mixed-{i}", 1).reshape(T, geo.hidden))
            ck(f"the hyper-connection mix on the dump's own streams at layer {i}", e < TOL_MIX,
               f"rel L2 {e:.4f}")
            e = _err(inject, ref.get(f"hc_inject-{i}", 1).reshape(T, -1))
            ck(f"and the inject it hands the combine at layer {i}", e < TOL_INJECT,
               f"rel L2 {e:.4f}")

    for i in PROBE:
        x.copy_(ref.get(f"hc_mixed-{i}", 1).reshape(T, geo.hidden))
        m.model.layers[i].mlp(bare, x, o)
        torch.cuda.synchronize()
        e = _err(o, ref.get(f"ffn_out-{i}").reshape(T, geo.hidden))
        ck(f"the MoE on the dump's own input at layer {i}", e < 0.04, f"rel L2 {e:.4f}")

    return 0 if ck.ok else 1


sys.exit(main())

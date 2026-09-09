# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
import os
import pathlib
import sys
from collections.abc import Callable

import torch

from snowllm import ops
from snowllm.checkpoint.gguf import GGUF, dspark as dspark_gguf
from snowllm.checkpoint.gguf.source import GGUFReader, find_dspark_gguf, find_gguf
from snowllm.engine.forward_context import Dsv4Walker
from snowllm.models.deepseek_v4 import dspark
from snowllm.models.deepseek_v4.layers import VocabEmbedding
from snowllm.models.geometry import DSparkGeometry, ModelGeometry

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))

CTX = 300
TAPS = (41, 42, 43)


class Borrowed:
    def __init__(self, rd: GGUFReader, geo: DSparkGeometry) -> None:
        from snowllm import _capi
        from snowllm.checkpoint.gguf import deepseek4 as deepseek4_gguf
        from snowllm.models.deepseek_v4.deepseek4_weights import KQUANT
        from snowllm.models.geometry import DeepSeekV4Geometry
        _capi.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
        self.geo = DeepSeekV4Geometry.from_config(deepseek4_gguf.config(rd.gguf))
        self.embed = VocabEmbedding(rd.tensor("token_embd.weight").reshape(
            geo.vocab_size, geo.hidden).cuda().contiguous())
        self.head = ops.lm_head_kquant_shuffle_weight(
            rd.raw("output.weight"), KQUANT[rd.gguf["output.weight"].quant.name])


def main() -> int:
    draft_path = find_dspark_gguf(MODEL_DIR)
    if draft_path is None:
        print(f"== skipped: no dspark-*.gguf under {MODEL_DIR}")
        return 0

    ck = _harness.Checks(19)
    geo = DSparkGeometry.from_config(dspark_gguf.config(GGUF(draft_path)))
    st = geo.stack

    with GGUFReader(find_gguf(MODEL_DIR)) as trd:
        target = Borrowed(trd, st)
    with GGUFReader(draft_path) as drd:
        draft = dspark.load(drd, geo, target, Dsv4Walker())

    print("\n=== it borrows the target's two shared tensors ===")
    ck("the embedding is the target's, not a copy",
       draft.stack.embed.weight is target.embed.weight)
    ck("and so is the head", draft.stack.head is target.head)
    ck("it brought three stages of its own", len(draft.stack.layers) == st.num_layers == 3)
    ck("and it took the file's taps as they stand, the last being the stack's output",
       draft.tap_layers == TAPS
       and max(draft.tap_layers) == target.geo.num_layers, str(draft.tap_layers))

    print("\n=== the context encoder ===")
    taps = torch.randn(CTX, geo.fc_k, dtype=torch.bfloat16, device="cuda") * 0.05
    feat = draft.context_feature(taps)
    ck("fc folds the taps to one hidden", feat.shape == (CTX, st.hidden), str(feat.shape))
    ck("and it is finite", bool(torch.isfinite(feat).all()))
    ck("a tap width it was not built for is refused",
       _raises(draft.context_feature, taps[:, :geo.fc_k - st.hidden]))

    print("\n=== the drafter's own KV ring ===")
    blk = geo.block_size
    need = ops.kv_blocks_for(CTX + blk, ops.KV_BLOCK_SIZES[0]) + 1
    draft.pools(need, ops.KV_BLOCK_SIZES[0])
    ck("every stage got a raw pool", all(lp.raw is not None for lp in draft.cache.layers))
    ck("and none got a compressed one, which is what ratio 0 means",
       all(lp.comp is None for lp in draft.cache.layers) and not draft.cache.ratios)

    pos = torch.arange(CTX, dtype=torch.int64, device="cuda")
    draft.write_context(feat, pos, torch.arange(CTX, dtype=torch.int32, device="cuda"))
    k0 = draft.cache.layers[0].raw.k
    ck("the injected latents landed and are finite", bool(torch.isfinite(k0).all()))

    print("\n=== a block of noise through three DSV4 stages ===")
    noise = torch.empty(blk, st.hidden, dtype=torch.bfloat16, device="cuda")
    ids = torch.full((blk,), geo.mask_token_id, dtype=torch.int64, device="cuda")
    draft.stack.embed(ids, noise)
    hid = draft.forward(
        noise, torch.arange(CTX, CTX + blk, dtype=torch.int64, device="cuda"),
        torch.arange(CTX, CTX + blk, dtype=torch.int32, device="cuda"), None,
        torch.tensor([CTX + blk], dtype=torch.int32, device="cuda"))
    ck("one hidden per block position", hid.shape == (blk, st.hidden), str(hid.shape))
    ck("and every one is finite", bool(torch.isfinite(hid).all()))

    block_attention(ck, st, draft.stack.layers[0].attn.sinks)

    print("\n=== the borrowed head, then the Markov chain over it ===")
    base = draft.stack.lm_head(draft.walker.arena, hid)
    ck("the head speaks the target's vocabulary", base.shape == (blk, st.vocab_size),
       str(base.shape))
    anchor = torch.tensor([11], dtype=torch.int64, device="cuda")
    drafts = draft.markov(base, anchor, blk)
    ck("it drafts a token for every row of the block, not one fewer -- DSpark denoises "
       "the anchor row too", drafts.shape == (1, blk), str(drafts.shape))
    ck("all of them inside the vocabulary",
       bool(((drafts >= 0) & (drafts < st.vocab_size)).all()), str(drafts.tolist()))
    ck("the chain is sequential, so a different anchor moves it",
       not torch.equal(drafts, draft.markov(base, anchor + 1, blk)))

    return ck.done()


def cpu_block_mla(q: torch.Tensor, kv: torch.Tensor, sinks: torch.Tensor, window: int,
                  scale: float, causal: bool) -> torch.Tensor:
    S, H, D = q.shape
    N = kv.shape[0]
    ctx = N - S
    s = torch.einsum("ihd,jd->ijh", q.float(), kv.float()) * scale
    j = torch.arange(N, device=q.device)[None, :]
    if causal:
        i = torch.arange(ctx, N, device=q.device)[:, None]
        vis = (j <= i) & (j > i - window)
    else:
        vis = j >= ctx - window
    s = s.masked_fill(~vis[:, :, None], float("-inf"))
    s = torch.cat([s, sinks.float().expand(S, 1, H)], dim=1)
    return torch.einsum("ijh,jd->ihd", s.softmax(dim=1)[:, :N], kv.float())


def block_attention(ck: _harness.Checks, st: ModelGeometry, sinks: torch.Tensor) -> None:
    print("\n=== and the block is attended BOTH WAYS, not causally ===")
    torch.manual_seed(0)
    S, D, W = 5, st.head_size, st.sliding_window
    N = 300
    q = (torch.randn(S, st.num_heads, D, device="cuda") * 0.3).to(torch.bfloat16)
    kv = (torch.randn(N, D, device="cuda") * 0.3).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    nb = ops.kv_blocks_for(N, ops.KV_BLOCK_SIZES[0])
    kb, vb = ops.kv_pool_bytes(nb, False, ops.KV_BLOCK_SIZES[0])
    kc, vc = ops.zero_bytes(kb).view(torch.bfloat16), ops.zero_bytes(vb).view(torch.bfloat16)
    ops.reshape_and_cache(kv, kv, kc, vc, torch.arange(N, dtype=torch.int32, device="cuda"), D, D,
                          ops.KV_BLOCK_SIZES[0])

    total, qmap = ops.prefill_q_plan([S])
    args = (kc, vc, torch.tensor([0, S], dtype=torch.int32, device="cuda"),
            torch.arange(nb, dtype=torch.int32, device="cuda").reshape(1, nb),
            torch.tensor([N], dtype=torch.int32, device="cuda"), total, scale, qmap, sinks, W,
            ops.KV_BLOCK_SIZES[0])
    out = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill_block(q, args[0], args[1], out, *args[2:])
    causal = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill(q, args[0], args[1], causal, *args[2:])
    torch.cuda.synchronize()

    want = cpu_block_mla(q, kv, sinks, W, scale, causal=False)
    tol = 2e-2 * float(want.abs().max())

    def err(got: torch.Tensor, ref: torch.Tensor) -> float:
        return float((got.float() - ref).abs().max())

    ck("every row of the block sees every other row, later ones included",
       err(out, want) < tol, f"max |delta| {err(out, want):.3e} vs tol {tol:.3e}")
    ck(f"its window holds the last {W} CONTEXT keys, so the block does not push them out",
       err(out, cpu_block_mla(q, kv, sinks, W - S, scale, False)) > tol,
       f"a {W - S}-key window reads {err(out, cpu_block_mla(q, kv, sinks, W - S, scale, False)):.3e}")
    ck("and the causal entry beside it matches the causal reference and nothing else, so the "
       "two modes really are different attentions",
       err(causal, cpu_block_mla(q, kv, sinks, W, scale, True)) < tol
       and err(causal, want) > tol,
       f"causal vs block reference {err(causal, want):.3e}")


def _raises(fn: Callable, *a: object) -> bool:
    from snowllm._capi import SnowLLMError
    try:
        fn(*a)
        return False
    except SnowLLMError:
        return True


if __name__ == "__main__":
    sys.exit(main())

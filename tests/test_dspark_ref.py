# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import pathlib
import sys
from collections.abc import Callable

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf import GGUF, dspark as dspark_gguf
from snowllm.checkpoint.gguf.source import GGUFReader, find_dspark_gguf, find_gguf
from snowllm.engine.forward_context import Dsv4Walker
from snowllm.models.deepseek_v4 import dspark
from snowllm.models.geometry import DSparkGeometry

import _harness
from test_dspark_forward import Borrowed

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))

ROWS = 16


def rms(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got.float().cpu() - ref).abs().max() / ref.abs().max().clamp_min(1e-9))


def main() -> int:
    draft_path = find_dspark_gguf(MODEL_DIR)
    if draft_path is None:
        _harness.skip(f"no dspark-*.gguf under {MODEL_DIR}")

    geo = DSparkGeometry.from_config(dspark_gguf.config(GGUF(draft_path)))
    st = geo.stack
    ck = _harness.Checks(14)

    with GGUFReader(find_gguf(MODEL_DIR)) as trd:
        target = Borrowed(trd, st)
    _capi.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    with GGUFReader(draft_path) as drd:
        draft = dspark.load(drd, geo, target, Dsv4Walker())
        fc32 = drd.tensor("fc.weight", torch.float32).reshape(st.hidden, geo.fc_k).cpu()
        enc32 = drd.tensor("enc.output_norm.weight", torch.float32).cpu()
        w1_32 = drd.tensor("markov_w1.weight", torch.float32).reshape(
            -1, geo.markov_rank).cpu()
        w2_32 = drd.tensor("markov_w2.weight", torch.float32).reshape(
            -1, geo.markov_rank).cpu()
        kv32 = [drd.tensor(f"blk.{i}.attn_kv.weight", torch.float32).reshape(
            st.kv_dim, st.hidden).cpu() for i in range(st.num_layers)]
        kvn32 = [drd.tensor(f"blk.{i}.attn_kv_a_norm.weight", torch.float32).cpu()
                 for i in range(st.num_layers)]

    print("\n=== the context encoder, against a CPU recomputation of the same weights ===")
    ck("fc is stored as enc's Linear needs it, [hidden, num_taps * hidden]",
       tuple(fc32.shape) == (st.hidden, geo.fc_k), str(tuple(fc32.shape)))

    torch.manual_seed(0)
    taps = (torch.randn(ROWS, geo.fc_k) * 0.05).to(torch.bfloat16)
    lin = torch.einsum("tk,ok->to", taps.float(), fc32.to(torch.bfloat16).float())
    got = draft.context_feature(taps.cuda())
    e = rel(got, lin)
    ck("context_feature is that Linear alone -- the encoder's norm rides on the projection that "
       "reads it", e < 3e-2, f"rel {e:.2e}")

    ck("a tap row that is not num_taps wide is refused",
       _raises(draft.context_feature, taps[:, :geo.fc_k - st.hidden].cuda()))

    print("\n=== every stage's kv projection is the tensor the file names ===")
    feat = draft.context_feature(taps.cuda())
    ctx = draft.walker.bare_context(draft.stack, feat.shape[0])
    gamma, eps = draft.enc_norm.gamma, st.eps
    normed = rms(feat.float().cpu(), enc32, eps)
    first = None
    for i, layer in enumerate(draft.stack.layers):
        proj = layer.attn.fanout.kv(ctx, feat, gamma, eps)
        r = torch.einsum("th,oh->to", normed, kv32[i].to(torch.bfloat16).float())
        e = rel(proj, r)
        first = e if first is None else first
        ck(f"stage {i} norms with the encoder's gamma and projects through blk.{i}.attn_kv, "
           f"not another stage's", e < 8e-2, f"rel {e:.2e}")

    wrong = rel(draft.stack.layers[0].attn.fanout.kv(ctx, feat, gamma, eps),
                torch.einsum("th,oh->to",
                             rms(feat.float().cpu(), torch.ones_like(enc32), eps),
                             kv32[0].to(torch.bfloat16).float()))
    ck("and that gamma is really applied inside it, not dropped", wrong > 10 * first,
       f"rel against gamma=1 is {wrong:.2e} vs {first:.2e}")

    other = rel(draft.stack.layers[0].attn.fanout.kv(ctx, feat, gamma, eps),
                torch.einsum("th,oh->to", normed, kv32[1].to(torch.bfloat16).float()))
    ck("and the three are distinguishable, so that check can fail", other > 0.5,
       f"stage 0 against stage 1's weight: rel {other:.2e}")

    print("\n=== the Markov chain, token for token ===")
    ck("w1 and w2 are both [vocab, rank], the low-rank pair the head factors into",
       tuple(w1_32.shape) == (st.vocab_size, geo.markov_rank)
       and tuple(w2_32.shape) == (st.vocab_size, geo.markov_rank),
       f"{tuple(w1_32.shape)} {tuple(w2_32.shape)}")

    blk = geo.block_size
    base = (torch.randn(blk, st.vocab_size) * 2.0)
    anchor = 11
    prev, want = anchor, []
    for i in range(blk):
        col = base[i].double() + (w1_32[prev].to(torch.bfloat16).float().double()
                                  @ w2_32.to(torch.bfloat16).float().double().t())
        prev = int(col.argmax())
        want.append(prev)
    got = draft.markov(base.cuda(), torch.tensor([anchor], dtype=torch.int64, device="cuda"),
                       blk)[0].tolist()
    ck("the chain reproduces a CPU recomputation of w1[prev] @ w2.T + logits, step for step",
       got == want, f"{got} vs {want}")

    swapped, prev = [], anchor
    for i in range(blk):
        col = base[i].double() + (w2_32[prev].to(torch.bfloat16).float().double()
                                  @ w1_32.to(torch.bfloat16).float().double().t())
        prev = int(col.argmax())
        swapped.append(prev)
    ck("and w1/w2 are not interchangeable, so that check has teeth", swapped != want,
       f"swapped {swapped}")

    print("\n=== the confidence head ===")
    names = set(GGUF(draft_path).tensors)
    ck("it is in the file", "conf_proj.weight" in names,
       str(sorted(n for n in names if "conf" in n)))
    ck("and it is loaded, one weight per hidden plus one per markov rank",
       draft.conf_proj is not None
       and draft.conf_proj.numel() == geo.stack.hidden + geo.markov_rank,
       str(None if draft.conf_proj is None else draft.conf_proj.numel()))

    return ck.done()


def _raises(fn: Callable, *a: object) -> bool:
    try:
        fn(*a)
    except Exception:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())

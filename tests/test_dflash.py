# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any

import _harness

if TYPE_CHECKING:
    import torch

    from snowllm.models.geometry import DFlashGeometry
    from snowllm.models.qwen3_5.dflash_draft import DFlashDraft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

DRAFT = "Qwen3.6-35B-A3B-DFlash"

check = _harness.Checks(46)
GEO = None
WINDOW = None


def _reference_module(ckpt: pathlib.Path) -> "tuple[Any, dict[str, torch.Tensor], Any]":
    import json

    import torch
    from safetensors.torch import load_file
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    kernels = pathlib.Path.home() / "SnowLLM-Kernels" / "scratchpad"
    if not (kernels / "dflash_ref_model.py").is_file():
        _harness.skip(f"no vendored reference at {kernels}/dflash_ref_model.py")
    sys.path.insert(0, str(kernels))
    from dflash_ref_model import DFlashDraftModel

    cfg_json = json.loads((ckpt / "config.json").read_text())
    cfg = Qwen3Config(**{k: v for k, v in cfg_json.items() if k != "auto_map"})
    cfg.block_size = cfg_json["dflash_config"]["block_size"]
    cfg.dflash_config = cfg_json["dflash_config"]
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = DFlashDraftModel(cfg)
    sd = load_file(str(ckpt / "model.safetensors"))
    model.load_state_dict(sd, assign=True, strict=True)
    model.rotary_emb = Qwen3RotaryEmbedding(cfg)
    return model.to("cuda").to(torch.bfloat16).eval(), sd, cfg


def _geometry(ckpt: pathlib.Path) -> "DFlashGeometry":
    import json

    from snowllm.models.geometry import DFlashGeometry

    return DFlashGeometry.from_config(json.loads((ckpt / "config.json").read_text()))


def _build(sd: "dict[str, torch.Tensor]",
           pages: int) -> "tuple[DFlashDraft, list[tuple[torch.Tensor, torch.Tensor]]]":
    import torch

    from snowllm import ops
    from snowllm.checkpoint.gguf import dflash as gguf_dflash
    from snowllm.models.qwen3_5 import dflash_draft

    pools = [(torch.zeros(pages * ops.KV_BLOCK_SIZES[0] * GEO.kv_dim, dtype=torch.bfloat16,
                          device="cuda"),
              torch.zeros(pages * ops.KV_BLOCK_SIZES[0] * GEO.kv_dim, dtype=torch.bfloat16,
                          device="cuda"))
             for _ in range(GEO.num_layers)]
    src = gguf_dflash.DictDraftSource(sd)
    return dflash_draft.load(src, GEO, pools, ops.Arena(), ops.KV_BLOCK_SIZES[0]), pools


def _step(draft: "DFlashDraft", pools: "list[tuple[torch.Tensor, torch.Tensor]]", pages: int,
          ops: ModuleType, target_hidden: "torch.Tensor", noise_embedding: "torch.Tensor",
          positions: "torch.Tensor") -> "torch.Tensor":
    import torch

    d = ops.dflash
    ctx_len, blk = target_hidden.shape[0], noise_embedding.shape[0]
    for kc, vc in pools:
        kc.zero_()
        vc.zero_()
    bt = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, pages)

    def slots(first: int, n: int) -> "torch.Tensor":
        idx = torch.arange(first, first + n, device="cuda")
        return (bt[0][idx // ops.KV_BLOCK_SIZES[0]] * ops.KV_BLOCK_SIZES[0]
                + idx % ops.KV_BLOCK_SIZES[0]).to(torch.int32)

    ctx_feat = draft.context_feature(target_hidden)
    draft.write_context(ctx_feat, positions[:ctx_len].contiguous(), slots(0, ctx_len))
    seq_lens = torch.full((1,), ctx_len + blk, dtype=torch.int32, device="cuda")
    return draft.forward(noise_embedding, positions[ctx_len:].contiguous(),
                         slots(ctx_len, blk), bt, seq_lens)


SHORT = ((6, 8), (1, 8), (31, 16))


def main() -> None:
    global GEO, WINDOW, LONG
    ckpt = _harness.checkpoint(DRAFT)
    GEO = _geometry(ckpt)
    WINDOW = GEO.sliding_window
    LONG = ((2048, 8), (WINDOW + 200, 8))
    import torch

    from snowllm import _capi, ops

    ops.dflash.select(GEO)

    sys.path.insert(0, str(pathlib.Path.home() / "SnowLLM-Kernels" / "scratchpad"))
    from dflash_probe import build, reference_forward

    m32, cfg = build(torch.float32)
    n_taps = GEO.num_taps
    golds = {}
    for ctx_len, blk in SHORT + LONG:
        th, ne, pos = _inputs(torch, cfg, n_taps, ctx_len, blk, torch.float32)
        with torch.inference_mode():
            golds[(ctx_len, blk)] = reference_forward(
                m32, cfg, th, ne, pos, swa=ctx_len + blk > WINDOW, swa_uniform=True)[0]
    del m32
    torch.cuda.empty_cache()

    model, sd, cfg = _reference_module(ckpt)
    check("tap count", len(model.target_layer_ids) == n_taps, str(n_taps))
    check("tap layers", tuple(model.target_layer_ids) == GEO.tap_layers,
          str(tuple(model.target_layer_ids)))
    check("the kernels carry a draft of this checkpoint's shape",
          ops.dflash.selected() is GEO, _capi.draft_name(0))

    for ctx_len, blk in SHORT + LONG:
        pages = (ctx_len + blk + ops.KV_BLOCK_SIZES[0] - 1) // ops.KV_BLOCK_SIZES[0]
        draft, pools = _build(sd, pages)
        th, ne, pos = _inputs(torch, cfg, n_taps, ctx_len, blk, torch.bfloat16)
        got = _step(draft, pools, pages, ops, th[0].contiguous(), ne[0].contiguous(),
                    pos[0].contiguous())
        gold = golds[(ctx_len, blk)]
        check.close(f"vs fp32  ctx={ctx_len} blk={blk}", got, gold, 4e-2)

        if (ctx_len, blk) in SHORT:
            with torch.inference_mode():
                mod = model(position_ids=pos, noise_embedding=ne, target_hidden=th,
                            past_key_values=None, use_cache=False, is_causal=False)[0]
            check.close(f"vs module ctx={ctx_len} blk={blk}", got, mod, 6e-2)

    _check_window_matters(torch, ops, model, cfg, n_taps, reference_forward)
    _check_batched(torch, ops, sd, cfg, n_taps)
    sys.exit(check.done())


def _check_batched(torch: ModuleType, ops: ModuleType, sd: "dict[str, torch.Tensor]",
                   cfg: Any, n_taps: int) -> None:
    d = ops.dflash
    ctxs, blk = [6, 1, 11], 8
    pages = (max(ctxs) + blk + ops.KV_BLOCK_SIZES[0] - 1) // ops.KV_BLOCK_SIZES[0]
    B = len(ctxs)

    torch.manual_seed(9)
    th = [torch.randn(c, n_taps * cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
          for c in ctxs]
    ne = [torch.randn(blk, cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02
          for _ in ctxs]

    alone = []
    for i, c in enumerate(ctxs):
        draft, pools = _build(sd, pages)
        bt = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, pages)
        idx = torch.arange(c, device="cuda")
        sl = (bt[0][idx // ops.KV_BLOCK_SIZES[0]] * ops.KV_BLOCK_SIZES[0]
              + idx % ops.KV_BLOCK_SIZES[0]).to(torch.int32)
        pos = torch.arange(c + blk, device="cuda")
        cf = draft.context_feature(th[i])
        draft.write_context(cf, pos[:c].contiguous(), sl)
        bidx = torch.arange(c, c + blk, device="cuda")
        bsl = (bt[0][bidx // ops.KV_BLOCK_SIZES[0]] * ops.KV_BLOCK_SIZES[0]
               + bidx % ops.KV_BLOCK_SIZES[0]).to(torch.int32)
        alone.append(draft.forward(ne[i], pos[c:].contiguous(), bsl, bt,
                                   torch.full((1,), c + blk, dtype=torch.int32,
                                              device="cuda")))

    draft, pools = _build(sd, pages * B)
    bt = torch.arange(B * pages, dtype=torch.int32, device="cuda").view(B, pages)
    seqs, cpos = [], []
    for i, c in enumerate(ctxs):
        seqs += [i] * c
        cpos += list(range(c))
    cf = draft.context_feature(torch.cat(th, 0))
    draft.write_context(cf, torch.tensor(cpos, dtype=torch.int64, device="cuda"),
                        ops.resolve_slots(bt, i32_(torch, seqs), i32_(torch, cpos), ops.KV_BLOCK_SIZES[0]))
    bseq = [i for i in range(B) for _ in range(blk)]
    bpos = [c + t for c in ctxs for t in range(blk)]
    got = draft.forward(torch.cat(ne, 0),
                        torch.tensor(bpos, dtype=torch.int64, device="cuda"),
                        ops.resolve_slots(bt, i32_(torch, bseq), i32_(torch, bpos), ops.KV_BLOCK_SIZES[0]), bt,
                        torch.tensor([c + blk for c in ctxs], dtype=torch.int32,
                                     device="cuda"))

    for i, c in enumerate(ctxs):
        check.close(f"batched request {i} (ctx={c})", got[i * blk:(i + 1) * blk], alone[i], 1e-2)


def i32_(torch: ModuleType, xs: Sequence[int]) -> "torch.Tensor":
    return torch.tensor(xs, dtype=torch.int32, device="cuda")


def _inputs(torch: ModuleType, cfg: Any, n_taps: int, ctx_len: int, blk: int,
            dtype: "torch.dtype") -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    torch.manual_seed(ctx_len * 1000 + blk)
    return (torch.randn(1, ctx_len, n_taps * cfg.hidden_size, device="cuda", dtype=dtype) * 0.1,
            torch.randn(1, blk, cfg.hidden_size, device="cuda", dtype=dtype) * 0.02,
            torch.arange(ctx_len + blk, device="cuda").unsqueeze(0))


def _check_window_matters(torch: ModuleType, ops: ModuleType, model: Any, cfg: Any, n_taps: int,
                          reference_forward: Callable) -> None:
    ctx_len, blk = WINDOW + 200, 8
    th, ne, pos = _inputs(torch, cfg, n_taps, ctx_len, blk, torch.bfloat16)
    with torch.inference_mode():
        win = reference_forward(model, cfg, th, ne, pos, swa=True, swa_uniform=True)[0]
        rowwise = reference_forward(model, cfg, th, ne, pos, swa=True)[0]
        none = reference_forward(model, cfg, th, ne, pos)[0]

    def r(a: "torch.Tensor", b: "torch.Tensor") -> float:
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    check("window changes the answer", r(win, none) > 0.05,
          f"windowed vs not: rel L2 {r(win, none):.3f}")
    check("block-shared window edge is small", r(rowwise, win) < 0.05,
          f"per-row vs block-uniform: rel L2 {r(rowwise, win):.4f}")


if __name__ == "__main__":
    main()

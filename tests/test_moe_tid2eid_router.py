# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _harness  # noqa: E402

from snowllm import _capi, ops  # noqa: E402

CASES = (1, 4, 16, 512, 4096)


def reference(hidden, router, tid2eid, token_ids, topk, scale, norm):
    logits = hidden.float() @ router.float().t()
    scores = torch.sqrt(torch.nn.functional.softplus(logits, beta=1, threshold=20))
    ids = tid2eid[token_ids.to(torch.int64)].to(torch.int64)
    w = torch.gather(scores, 1, ids)
    if norm:
        w = w / w.sum(1, keepdim=True).clamp_min(6.103515625e-5)
    return w * scale, ids.to(torch.int32)


def main() -> int:
    if not torch.cuda.is_available():
        print("== skipped: no GPU")
        return 0
    _capi.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    g = _capi.build_geometry()
    E, H, TOPK, ALL = g.moe_num_experts, g.hidden, g.moe_topk, g.moe_topk_all
    V = 4096  # a stand-in vocabulary: the table is keyed by token id and nothing reads it further
    ck = _harness.Checks()
    rng = torch.Generator(device="cuda").manual_seed(0)

    router = torch.randn(E, H, dtype=torch.bfloat16, device="cuda", generator=rng) * 0.05
    bias = torch.randn(E, dtype=torch.float32, device="cuda", generator=rng)
    tid2eid = torch.randint(0, E, (V, TOPK), dtype=torch.int32, device="cuda", generator=rng)

    print("\n=== the table router is the torch chain it replaced ===")
    for router_w, tag in ((ops.moe_shuffle_router(router, bias), "with a selection bias"),
                          (ops.moe_shuffle_router(router, None), "without one")):
        for M in CASES:
            hidden = torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=rng)
            tok = torch.randint(0, V, (M,), dtype=torch.int64, device="cuda", generator=rng)
            w, ids = ops.moe_router_tid2eid(hidden, router_w, tok, tid2eid)
            wref, idref = reference(hidden, router, tid2eid, tok, TOPK, 1.5, True)
            bad = int((ids[:, :TOPK] != idref).sum())
            ck(f"M={M:<5} {tag}: the picked experts are the table's", bad == 0, f"{bad} differ")
            err = (w[:, :TOPK] - wref).abs().max().item()
            ck(f"M={M:<5} {tag}: and their weights are the router's", err < 2e-3,
               f"max |diff| {err:.3g}")
            ck(f"M={M:<5} {tag}: the shared expert's slot is appended", int(ids[0, TOPK]) == E,
               f"slot {TOPK} holds {int(ids[0, TOPK])}, want {E}")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())

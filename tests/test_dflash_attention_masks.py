# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

import torch

import _harness
from snowllm import ops
from snowllm.engine import _draft_geometry
from snowllm.models.geometry import DFlashGeometry


def attention(ck, geo, block, lengths, causal, window):
    d, page = ops.dflash, 16
    batch = len(lengths)
    pages = (max(lengths) + page - 1) // page
    table = torch.randperm(batch*pages, device="cuda", dtype=torch.int32).view(batch, pages)
    kc = torch.zeros(batch*pages*page*geo.kv_dim, dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    query = torch.randn(batch*block, geo.num_heads, geo.head_size, device="cuda", dtype=torch.bfloat16)
    expected = []
    for b, n in enumerate(lengths):
        key = torch.randn(n, geo.num_kv_heads, geo.head_size, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        pos = torch.arange(n, device="cuda")
        slots = (table[b, pos // page]*page + pos % page).int()
        d.reshape_and_cache(key.view(n, -1), value.view(n, -1), kc, vc, slots, page)
        k = key.repeat_interleave(geo.num_heads // geo.num_kv_heads, 1)
        v = value.repeat_interleave(geo.num_heads // geo.num_kv_heads, 1)
        q = query[b*block:(b+1)*block]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * geo.head_size**-.5
        qp = torch.arange(n-block, n, device="cuda")[:, None]
        kp = pos[None]
        visible = kp <= qp if causal else torch.ones(block, n, device="cuda", dtype=torch.bool)
        if window:
            visible &= (kp-qp).abs() < window
        scores.masked_fill_(~visible[None], -torch.inf)
        expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()))
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    if window:
        table, lens = ops.paged_window_view(table, lens, 0, window+block+page-2, page, end_offset=-1)
    count = d.paged_attn_num_slots(batch)
    plan = d.paged_attn_plan(lens, count, page)
    workspace = torch.empty(d.paged_attn_workspace_size(count, block), dtype=torch.uint8, device="cuda")
    out = torch.empty_like(query)
    d.paged_attn(query, lens, kc, vc, out, table, plan, count, workspace,
                 geo.head_size**-.5, block, page, window=window, causal=causal)
    ck.close(f"block={block} causal={causal} window={window}: shuffled pages and per-query mask",
             out, torch.cat(expected), .015)


def main():
    root = Path.home() / "models"
    paths = [root / "Qwen3.6-35B-A3B-UD-Q4_K_XL/dflash",
             root / "Qwen3.8-27B-UD-Q4_K_XL/dflash2"]
    if not all(p.exists() for p in paths):
        _harness.skip("Qwen DFlash1/DFlash2 checkpoint metadata is missing")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(7367)
    ck = _harness.Checks(75)
    for path in paths:
        native = _draft_geometry(str(path))
        cfg = json.loads((path / "config.json").read_text())
        hf = DFlashGeometry.from_config(cfg)
        flags = tuple(native.causal(i) for i in range(native.num_layers))
        ck(f"{path.name}: GGUF and HF causality agree",
           flags == tuple(hf.causal(i) for i in range(hf.num_layers)))
        expected = (True,)*5 + (False,) if not native.dflash2 else (False,)*5
        ck(f"{path.name}: checkpoint layer mask contract", flags == expected)
        for explicit in (False, True):
            configured = DFlashGeometry.from_config(dict(cfg, is_causal=explicit))
            ck(f"{path.name}: explicit is_causal={explicit} applies to every layer",
               all(configured.causal(i) == explicit for i in range(configured.num_layers)))
        d = ops.dflash
        d.select(native)
        for block in (1, 4, 8, 16):
            for causal in (False, True):
                attention(ck, native, block, [native.sliding_window+3, native.sliding_window+19],
                          causal, native.sliding_window)
        attention(ck, native, 8, [19, 35], False, 0)
        attention(ck, native, 8, [19, 35], True, 0)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())

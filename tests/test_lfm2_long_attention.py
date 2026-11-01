# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
import _harness
from snowllm import cli
from snowllm.engine import SamplingParams
from test_lfm2_dspark_reference import cache_rows


@torch.no_grad()
def main():
    root = _harness.checkpoint("LFM2.5-8B-A1B-Q4_K_M", env="SNOWLLM_LFM_DIR")
    length = int(os.environ.get("SNOWLLM_LFM_ATTENTION_TOKENS", "8192"))
    state = cli.build(str(root), max_num_seqs=1, max_model_len=length+32,
        num_kv_blocks=None, default_max_tokens=1, profile_dir=None, seed=0,
        prefill_chunk=512, num_spec=0, prefix_memory_ratio=0)
    engine, tok = state.engine.engine, state.tokenizer
    check = _harness.Checks(55)
    checked = []
    for i, layer in enumerate(state.model.layers):
        if not hasattr(layer.attn, "geo"):
            continue
        original = layer.attn._attend

        def attend(ctx, q, pools, out, proj, original=original, layer=i, geo=layer.attn.geo):
            original(ctx, q, pools, out, proj)
            b = ctx.batch
            if not b.is_prefill or b.host_seq_lens[0] != length:
                return
            page = ctx.block_size
            pages = b.block_tables[0, : (length+page-1)//page].long()
            heads, dim = geo.num_kv_heads, geo.head_size
            k = cache_rows(pools[0], pages, heads, dim, page, key=True)[:length].float()
            v = cache_rows(pools[1], pages, heads, dim, page, key=False)[:length].float()
            rows = min(3, len(q))
            query = q[-rows:].float().view(rows, heads, geo.num_heads//heads, dim)
            scores = torch.einsum("bghd,tgd->bght", query, k) * dim**-.5
            positions = torch.arange(length, device="cuda")
            ends = torch.arange(length-rows+1, length+1, device="cuda")
            scores.masked_fill_(positions[None, None, None] >= ends[:, None, None, None], -float("inf"))
            expected = torch.einsum("bght,tgd->bghd", scores.softmax(-1), v).reshape(rows, geo.num_heads, dim)
            error = _harness.rel(out[-rows:], expected)
            check(f"layer {layer} long attention versus FP32", error < .005, f"relL2={error:.6f}")
            checked.append(layer)

        layer.attn._attend = attend
    unit = tok.encode("这是一条普通的仓库记录，内容为日常库存更新。\n", add_special_tokens=False)
    ids = (unit * (1+length//len(unit)))[:length]
    r = engine.add(ids, SamplingParams(temperature=0, max_new_tokens=1))
    engine.run()
    check("all six attention layers checked", len(checked) == 6, checked)
    check("long prefill finishes", r.done and len(r.out) == 1)
    state.engine._worker.shutdown(wait=True)
    return check.done()


if __name__ == "__main__":
    sys.exit(main())

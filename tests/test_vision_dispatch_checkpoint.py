# SPDX-License-Identifier: Apache-2.0
"""Real GGUF vision towers: Torch reference, HF FP32 and image isolation."""
import argparse
import json
from pathlib import Path
import sys

import torch

import _harness
import _vision_reference as ref
from snowllm import ops
from snowllm.checkpoint.gguf.names import vision_config
from snowllm.checkpoint.gguf.source import GGUFReader, find_mmproj, vision_state
from snowllm.models.vision import VisionModel, _bilinear_pos_inputs


def load_tower(root):
    path = find_mmproj(root)
    if path is None:
        raise ValueError(f"no vision checkpoint in {root}")
    with GGUFReader(path) as rd:
        cfg = vision_config(rd.gguf)
        sd = {k: v.clone() for k, v in vision_state(rd).items()}
    return VisionModel.from_hf_state(sd, cfg), sd, cfg, path


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path,
                   default=_harness.checkpoint_path(_harness.FP8, env="SNOWLLM_QWEN_DIR"))
    p.add_argument("--out", type=Path, default=Path("/tmp/snowllm-vision-checkpoint.json"))
    p.add_argument("--hf", action="store_true")
    args = p.parse_args()
    if not args.model.is_dir() or find_mmproj(args.model) is None:
        _harness.skip("no vision checkpoint")
    torch.manual_seed(1911)
    torch.set_num_threads(4)
    check = _harness.Checks(56)
    model, sd, cfg, path = load_tower(args.model)
    print(path, cfg, flush=True)
    hf = None
    if args.hf:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
        hf_cfg = Qwen3_5MoeVisionConfig(**cfg)
        hf_cfg._attn_implementation = "eager"
        hf = Qwen3_5MoeVisionModel(hf_cfg).eval().cuda().float()
        hf.load_state_dict({k.removeprefix("model.visual."): v.float() for k, v in sd.items()}, strict=True)
    rows = []
    for grid in ([[1, 2, 2]], [[1, 16, 16]], [[1, 16, 24]],
                 [[1, 16, 16], [1, 8, 12]], [[2, 6, 10], [1, 4, 8]],
                 [[1, 32, 32]], [[1, 64, 64]], [[1, 16, 16]]):
        grid = torch.tensor(grid, dtype=torch.int64)
        m = sum(t * h * w for t, h, w in grid.tolist())
        x = torch.randn(m, model.geo.patch_in, device="cuda", dtype=torch.bfloat16)
        ix, wt = _bilinear_pos_inputs(grid, model.geo.grid_side, model.geo.spatial_merge)
        original = torch.randn(m, model.geo.hidden, device="cuda", dtype=torch.bfloat16)
        pos = ref._bilinear_pos_embed(grid, model.pos_embed, model.geo.grid_side, model.geo.spatial_merge)
        expected = (original.float() + pos).bfloat16()
        ops.vision_pos_embed_add(original, model.pos_embed, ix.cuda(), wt.cuda())
        check.exact(f"real position grid={grid.tolist()}", original, expected)
        want = ref.forward(model, x, grid).clone()
        got = model.forward(x, grid).clone()
        check.exact(f"real complete tower grid={grid.tolist()}", got, want)
        check("real complete tower finite", bool(torch.isfinite(got).all()))
        result = dict(grid=grid.tolist(), patches=m, old_native_exact=got.equal(want))
        # Large HF eager attention is quadratic in memory; native old/new still
        # cover 4096 patches, while independent HF reference covers <=384 here.
        if hf is not None and m <= 384:
            want_hf = hf(x.float(), grid.cuda()).pooler_output
            err = _harness.rel(got, want_hf)
            check.close("complete tower vs HF FP32", got, want_hf, 3.5e-2)
            result["hf_relative_l2"] = err
        rows.append(result)
    # Compare a batched video+image call with independent frame/image calls.
    grid = torch.tensor([[2, 6, 10], [1, 4, 8]])
    x = torch.randn(152, model.geo.patch_in, device="cuda", dtype=torch.bfloat16)
    batch = model.forward(x, grid).clone()
    start = 0
    for i, (h, w) in enumerate(((6, 10), (6, 10), (4, 8))):
        m = h * w
        single = model.forward(x[start:start+m], torch.tensor([[1, h, w]])).clone()
        # The underlying GEMM has the same contraction/rounding for any M tile.
        check.exact(f"complete tower frame/image isolation {i}",
                    batch[start // model.geo.merge_unit:(start+m) // model.geo.merge_unit], single)
        start += m

    # CPU grid mistakes must fail before dispatch, without launching bad GPU indices.
    from snowllm._capi import SnowLLMError
    for tag, bad in (("empty", torch.empty(0, 3, dtype=torch.int64)),
                     ("merge", torch.tensor([[1, 3, 4]])),
                     ("zero", torch.tensor([[0, 4, 4]])),
                     ("count", torch.tensor([[1, 4, 4]]))):
        try:
            model.forward(x, bad)
        except SnowLLMError:
            check(f"grid rejects {tag}", True)
        else:
            check(f"grid rejects {tag}", False)
    check.exact("GPU grid input preserves CPU geometry preparation",
                model.forward(x, grid.cuda()), batch)
    args.out.write_text(json.dumps(dict(checkpoint=str(path), config=cfg, results=rows, ok=check.ok), indent=2) + "\n")
    return check.done()


if __name__ == "__main__":
    sys.exit(main())

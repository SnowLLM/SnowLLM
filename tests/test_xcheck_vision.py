# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The vision tower against the checkpoint's own Qwen3_5MoeVisionModel, on its own weights.

The checkpoint is Qwen3_5MoeForConditionalGeneration (a VL model); this checks the from-scratch HIP
vision encoder end to end against HF's module fed the SAME real weights and the SAME patch input, so
nothing in the reference comes from our own code. Needs the real checkpoint; skips (exit 0) if it is
not in the HF cache.
"""

import json
import sys

import torch

import _harness

CKPT = _harness.checkpoint()

import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeVisionConfig,
)

from snowllm import ops, vision  # noqa: E402

check = _harness.Checks(30)


def vis_state():
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    out, handles = {}, {}
    for k, shard in idx.items():
        if not k.startswith("model.visual."):
            continue
        if shard not in handles:
            handles[shard] = safe_open(str(CKPT / shard), framework="pt", device="cuda")
        out[k[len("model.visual."):]] = handles[shard].get_tensor(k)
    return out


def main():
    torch.manual_seed(7)
    cfg = Qwen3_5MoeVisionConfig(**json.loads((CKPT / "config.json").read_text())["vision_config"])
    sd = vis_state()

    hf = mod.Qwen3_5MoeVisionModel(cfg).cuda().eval()
    hf.load_state_dict(sd, strict=True)
    hf_f32 = hf.float()
    hf_bf16 = mod.Qwen3_5MoeVisionModel(cfg).cuda().eval()
    hf_bf16.load_state_dict(sd, strict=True)
    hf_bf16 = hf_bf16.to(torch.bfloat16)

    ours = vision.VisionModel.from_hf_state(
        sd, json.loads((CKPT / "config.json").read_text())["vision_config"], prefix="")

    # The tower is 27 bf16 residual blocks; error against the fp32 reference is accumulation, not
    # a bug -- every op in it is checked against torch on its own elsewhere. The bf16-vs-bf16
    # column shows what is left once both sides round the same way.
    for grid in ([[1, 16, 16]], [[1, 16, 24]], [[1, 16, 16], [1, 8, 8]]):
        grid_thw = torch.tensor(grid, dtype=torch.int64)
        M = sum(int(t) * int(h) * int(w) for t, h, w in grid)
        px = torch.randn(M, ours.geo.patch_in)

        with torch.no_grad():
            want_f32 = hf_f32(px.cuda().float(), grid_thw.cuda()).pooler_output  # [M/4, 2048]
            want_bf16 = hf_bf16(px.cuda().to(torch.bfloat16), grid_thw.cuda()).pooler_output
        got = ours.forward(px, grid_thw)
        ops.synchronize()
        print(f"    (vs HF bf16: rel L2 {_harness.rel(got, want_bf16):.5f})")
        check.close(f"merged grid={grid}", got, want_f32, 3.5e-2)

    return check.done()


if __name__ == "__main__":
    sys.exit(main())

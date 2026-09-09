# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from transformers import PretrainedConfig  # noqa: E402
from transformers.modeling_rope_utils import _compute_yarn_parameters  # noqa: E402

from snowllm.checkpoint import loader  # noqa: E402

HEAD_DIM, PRF, THETA, ORIG_MAX = 256, 0.25, 10000000, 262144
CFG = {"head_dim": HEAD_DIM, "max_position_embeddings": ORIG_MAX,
       "rope_parameters": {"rope_theta": THETA, "partial_rotary_factor": PRF}}


def _ref(factor: float) -> tuple[torch.Tensor, float]:
    cfg = PretrainedConfig(
        head_dim=HEAD_DIM, hidden_size=2048, num_attention_heads=16,
        max_position_embeddings=ORIG_MAX,
        rope_parameters={"rope_type": "yarn", "rope_theta": THETA, "partial_rotary_factor": PRF,
                         "factor": factor, "original_max_position_embeddings": ORIG_MAX},
    )
    inv, mscale = _compute_yarn_parameters(cfg)
    return inv.float().cpu(), float(mscale)


def main() -> None:
    dr = int(HEAD_DIM * PRF)
    i = torch.arange(0, dr, 2, dtype=torch.float32)
    base_inv = 1.0 / (THETA ** (i / dr))
    got_inv, got_ms = loader.yarn_rope_table(CFG, 1.0, orig_max_pos=ORIG_MAX)
    assert torch.equal(got_inv, base_inv), "factor=1.0 diverged from the base inv_freq"
    assert got_ms == 1.0, f"factor=1.0 mscale should be 1.0, got {got_ms}"
    print("factor=1.0  bit-exact base table  OK")

    for factor in (2.0, 4.0):
        got_inv, got_ms = loader.yarn_rope_table(CFG, factor, orig_max_pos=ORIG_MAX)
        ref_inv, ref_ms = _ref(factor)
        max_abs = (got_inv - ref_inv).abs().max().item()
        assert torch.allclose(got_inv, ref_inv, rtol=0, atol=1e-12), \
            f"factor={factor} inv_freq off by {max_abs}"
        assert abs(got_ms - ref_ms) < 1e-9, f"factor={factor} mscale {got_ms} vs ref {ref_ms}"
        print(f"factor={factor}  inv_freq max|d|={max_abs:.2e}  mscale={got_ms:.6f}  OK")

    print("test_yarn_rope PASS")


if __name__ == "__main__":
    main()
    sys.exit(0)

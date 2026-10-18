# SPDX-License-Identifier: Apache-2.0
"""Independent pre-composition pipeline, retained only in tests."""
import torch
from snowllm import ops
from snowllm.models.glm5_next import layers


def weighted_norm(streams, weights, gamma, eps):
    summed = torch.empty(streams.shape[0], streams.shape[2], dtype=streams.dtype, device=streams.device)
    ops.dsv4_hc_weighted_sum(streams, weights[:, :streams.shape[1]], summed)
    return layers.norm(summed, gamma, eps)

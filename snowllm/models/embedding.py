# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch
from torch import nn

from .. import ops


class VocabEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, quant: int = 0) -> None:
        super().__init__()
        self.weight, self.quant = weight, quant

    def forward(self, input_ids: torch.Tensor, out: torch.Tensor) -> None:
        if self.quant == 4:
            ops.gather_embedding_q4_k(input_ids, self.weight, out)
        else:
            ops.gather_embedding(input_ids, self.weight, out)

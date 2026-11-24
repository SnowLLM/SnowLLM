# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from snowllm import ops
from snowllm._capi import GEO_GLM53_FLASH
from snowllm.models.glm5_next.layers import Projection


class GlmBetaTests(unittest.TestCase):
    def test_decode_and_verification_preserve_prefill_reduction(self):
        ops.select_geometry(GEO_GLM53_FLASH)
        torch.manual_seed(427)
        x = torch.randn(33, 4096, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(64, 4096, dtype=x.dtype, device=x.device)
        beta = Projection(64, 4096, dense=weight)
        expected = beta(x)
        for m in (1, 2, 16, 32):
            with self.subTest(rows=m):
                torch.testing.assert_close(beta(x[:m]), expected[:m], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

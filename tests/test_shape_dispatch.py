# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest

import torch

from snowllm import _capi, ops
from snowllm.models.mineru.qwen2_vl import BiasedProjection


class ProjectionAndCopyTests(unittest.TestCase):
    def test_vision_projection(self):
        torch.manual_seed(184)
        ops.select_geometry(_capi.GEO_MINERU_12B)
        n, k = 1280, 896
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        packed = ops.gemm_bf16_shuffle_b(weight, n, k)
        bias = torch.randn(n, device="cuda", dtype=weight.dtype)
        for rows in (1, 93, 211):
            for offset in (None, bias):
                with self.subTest(rows=rows, bias=offset is not None):
                    x = torch.randn(rows, k, device="cuda", dtype=weight.dtype)
                    expected = x.float() @ weight.float().T
                    if offset is not None:
                        expected += offset.float()
                    actual = torch.empty(rows, n, device="cuda", dtype=weight.dtype)
                    BiasedProjection(n, k, packed, offset)(SimpleNamespace(M=rows, arena=ops.Arena()), x, actual)
                    torch.testing.assert_close(actual.float(), expected, rtol=0.012, atol=0.012)

    def test_copy_layout_and_aliasing(self):
        src = torch.arange(120, device='cuda', dtype=torch.float32).view(4, 5, 6)
        for view in (src[:, :3], src.transpose(1, 2), src[:, :, 0], src[0, 0, 0], src[:0],
                     src[:1].expand(4, -1, -1)):
            dst = torch.empty_like(view, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            ops.copy_tensor(dst, view)
            torch.testing.assert_close(dst, view.bfloat16(), rtol=0, atol=0)
        dst = torch.zeros_like(src)[:, ::2]
        ops.copy_tensor(dst, src[:, ::2])
        torch.testing.assert_close(dst, src[:, ::2], rtol=0, atol=0)
        self.assertIs(ops.copy_tensor(src, src), src)
        with self.assertRaisesRegex(ops.SnowLLMError, "aliased"):
            ops.copy_tensor(src[1:], src[:-1])


if __name__ == '__main__':
    unittest.main()

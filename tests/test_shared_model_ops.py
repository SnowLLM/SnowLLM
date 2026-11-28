# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from snowllm import ops
from snowllm.ops import image


class SharedModelOpsTests(unittest.TestCase):
    def test_index_copy_replay_keeps_unselected_embeddings(self):
        values = torch.randn(3, 896, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([4, 0, 6], device="cuda")
        original = torch.randn(8, 896, device="cuda", dtype=values.dtype)
        output = original.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ops.index_copy(output, 0, indices, values)
        values.neg_()
        graph.replay()
        expected = original.clone()
        expected[indices] = values
        self.assertTrue(torch.equal(output, expected))

    def test_vision_rope_with_class_token(self):
        positions = torch.cartesian_prod(torch.arange(4), torch.arange(6))
        positions = (positions + 1).flip(-1)
        inv = 10000 ** (-torch.arange(0, 32, 2).float() / 32)
        angles = (positions[:, :, None] * inv).reshape(24, 32).repeat(1, 2)
        angles = torch.nn.functional.pad(angles, (0, 0, 1, 0))
        cos, sin = ops.vision_rope_table(positions, inv, cls_token=True)
        self.assertTrue(torch.equal(cos, angles.cos()))
        self.assertTrue(torch.equal(sin, angles.sin()))

    def test_packed_attention_preserves_post_matmul_scaling(self):
        torch.manual_seed(91)
        for dtype, width in ((torch.float16, 15), (torch.float32, 32)):
            with self.subTest(dtype=dtype):
                qkv = torch.randn(2, 24, 3, 8, width, device="cuda", dtype=dtype)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                scores = image.multiply(image.matmul(q, k.transpose(-1, -2)), width**-0.5)
                expected = image.matmul(image.softmax(scores), v)
                arena = ops.Arena()
                actual = arena.execute(("attention",), 24, lambda: image.scaled_attention(q, k, v))
                self.assertTrue(torch.equal(actual, expected))

    def test_matmul_broadcasts_distinct_batch_axes(self):
        left = torch.arange(30, dtype=torch.float32).reshape(2, 1, 3, 5)
        right = torch.arange(140, dtype=torch.float32).reshape(1, 4, 5, 7)
        self.assertTrue(torch.equal(image.matmul(left, right), left @ right))


if __name__ == "__main__":
    unittest.main()

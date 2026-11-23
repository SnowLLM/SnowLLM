# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from snowllm.models.inkling2.decoder import convolve
from snowllm.ops.inkling import short_conv


class InklingConvTest(unittest.TestCase):
    def test_ragged_resumed_and_reused_slots(self):
        gen = torch.Generator(device="cuda").manual_seed(714)
        def rand(*shape):
            return torch.randn(shape, device="cuda", generator=gen)
        def i32(values):
            return torch.tensor(values, device="cuda", dtype=torch.int32)
        for lengths in ([1, 1], [1, 7], [15, 2]):
            for heads in (0, 4):
                with self.subTest(lengths=lengths, heads=heads):
                    n, width = sum(lengths), 256
                    backing = rand(3, 2, 3, 512).bfloat16()
                    history = backing[:, 1, :, 64:64+width]
                    ref_history = history.clone()
                    x, w = rand(n, width).bfloat16(), rand(width, 4)
                    dynamic = rand(n, heads*4).bfloat16() if heads else None
                    seq = i32([0]*lengths[0] + [1]*lengths[1])
                    pos = i32([*range(39, 39+lengths[0]), *range(lengths[1])])
                    cu, slots = i32([0, lengths[0], n]), i32([2, 0])
                    actual = short_conv(x, w, dynamic, history, seq, pos, cu, slots)
                    ref_history[0].zero_()
                    expected = []
                    at = 0
                    for count, slot in zip(lengths, [2, 0]):
                        rows = slice(at, at+count)
                        expected.append(convolve(x[rows], ref_history[slot], w,
                                                 dynamic[rows] if heads else None))
                        at += count
                    torch.testing.assert_close(actual.float(), torch.cat(expected).float(), atol=.016, rtol=.008)
                    torch.testing.assert_close(history, ref_history, atol=0, rtol=0)

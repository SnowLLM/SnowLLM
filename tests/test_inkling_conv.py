# SPDX-License-Identifier: Apache-2.0
import math
import unittest

import torch

from snowllm.ops.inkling import short_conv


def reference_conv(x, history, weight, dynamic):
    taps = weight.size(1)
    sequence = torch.cat((history, x), dim=0)
    history.copy_(sequence[-history.size(0):])
    windows = sequence.unfold(0, taps, 1).float()
    out = (windows * weight).sum(-1) + x
    if dynamic is not None:
        taps_by_head = dynamic.view(x.size(0), -1, 1, taps).flip((-1,)).float()
        by_head = windows.reshape(x.size(0), taps_by_head.size(1), -1, taps)
        out.add_((by_head * taps_by_head).sum(-1).flatten(1))
    return out.mul_(1 / math.sqrt((2 if dynamic is not None else 1) * taps + 1)).bfloat16()


class InklingConvTest(unittest.TestCase):
    def test_expert_combine_post_conv_preserves_output_and_history(self):
        from snowllm.ops.inkling import combine

        gen = torch.Generator(device="cuda").manual_seed(327)
        def rand(*shape):
            return torch.randn(shape, generator=gen, device="cuda")
        for positions, slots in (([8192], [2]), ([0, 1, 7], [2, 0, 1])):
            rows, width = len(slots), 2304
            states = rand(rows, 8, width).bfloat16()
            weights, kernel, residual = rand(rows, 8), rand(width, 4), rand(rows, width)
            plan = tuple(torch.tensor(a, device="cuda", dtype=torch.int32)
                         for a in (range(rows), positions, range(rows + 1), slots))
            history = rand(3, 2, 3, width + 64).bfloat16()
            separate, fused = history.clone(), history.clone()
            def run(source, backing, **extra):
                return short_conv(source, kernel, None, backing[:, 1, :, 32:32 + width], *plan,
                                  residual=residual, out_dtype=torch.bfloat16, **extra)
            expected = run(combine(states, weights), separate)
            actual = run(states, fused, expert_weights=weights)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(fused, separate, atol=0, rtol=0)

    def test_decode_heads_preserve_rounded_outputs_and_state(self):
        from snowllm.ops.inkling import prepare_attention_heads

        gen = torch.Generator(device="cuda").manual_seed(925)
        def rand(*shape):
            return torch.randn(shape, device="cuda", generator=gen)
        def ids(values):
            return torch.tensor(values, device="cuda", dtype=torch.int32)
        for rows, pos, slots in ((1, [8192], [2]), (3, [0, 1, 8191], [2, 0, 1])):
            for global_layer in (False, True):
                q, r = (rand(rows, 24, width).bfloat16() for width in (128, 16))
                k, v = (rand(rows, 1024).bfloat16() for _ in range(2))
                taps = tuple(rand(rows, 32).bfloat16() if global_layer else None for _ in range(2))
                weights = tuple(rand(1024, 4) for _ in range(2))
                args = (q, r, k, v, rand(24, 128), rand(24, 16), weights, taps)
                plan = ids(range(rows)), ids(pos), None, ids(slots)
                mapping = ids([11, 9, 3][:rows])
                initial = rand(3, 2, 3, 2112).bfloat16()
                cache = rand(2, 1, 16, 8, 128).bfloat16()
                outputs = []
                for separate in (True, False):
                    state = initial.clone()
                    pages = cache.clone()
                    histories = state[:, 1].split((1024, 1024, 64), dim=-1)[:2]
                    result = prepare_attention_heads(*args, histories, pages.unbind(0), plan, mapping,
                            3.5, 4., 1.e-5, 128., .2, global_layer, separate)
                    outputs.append((*result, state, pages))
                for old, fused in zip(*outputs):
                    torch.testing.assert_close(fused, old, atol=0, rtol=0)

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
                    before = history.clone()
                    actual = short_conv(x, w, dynamic, history, seq, pos, cu, slots)
                    if dynamic is None:
                        for out_dtype, residual in ((torch.float32, None), (torch.float32, x), (torch.bfloat16, x.float())):
                            fused_history = before.clone()
                            fused = short_conv(x, w, None, fused_history, seq, pos, cu, slots,
                                               residual=residual, out_dtype=out_dtype)
                            expected = actual.float()
                            if residual is not None:
                                expected = expected + residual.float()
                            torch.testing.assert_close(fused, expected.to(out_dtype), atol=0, rtol=0)
                            torch.testing.assert_close(fused_history, history, atol=0, rtol=0)
                    ref_history[0].zero_()
                    expected = []
                    at = 0
                    for count, slot in zip(lengths, [2, 0]):
                        rows = slice(at, at+count)
                        expected.append(reference_conv(x[rows], ref_history[slot], w,
                                                 dynamic[rows] if heads else None))
                        at += count
                    torch.testing.assert_close(actual.float(), torch.cat(expected).float(), atol=.016, rtol=.008)
                    torch.testing.assert_close(history, ref_history, atol=0, rtol=0)

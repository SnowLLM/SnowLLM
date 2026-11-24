# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from snowllm.ops.inkling import attention


def reference(q, k, v, bias, sinks, sequences, positions, table, window):
    _, heads, dim = q.shape
    pages, page_size, kv_heads, _ = k.shape
    groups = heads // kv_heads
    output = torch.zeros_like(q, dtype=torch.float32)
    for seq in range(table.size(0)):
        rows, = torch.where((sequences == seq) & (positions >= 0))
        if rows.numel() == 0:
            continue
        pos = positions[rows].long()
        key_pos = torch.arange(pos.max().item() + 1, device=q.device)
        page = table[seq, (key_pos // page_size) % table.size(1)].long()
        valid_page = (page >= 0) & (page < pages)
        key_pos, page = key_pos[valid_page], page[valid_page]
        keys, values = (cache[page, key_pos % page_size].float() for cache in (k, v))
        queries = q[rows].float().reshape(-1, kv_heads, groups, dim)
        scores = torch.einsum('qkgd,tkd->qkgt', queries, keys).flatten(1, 2) / dim**.5
        distance = pos[:, None] - key_pos[None, :]
        index = distance.clamp(0, bias.size(2) - 1)[:, None].expand(-1, heads, -1)
        scores += bias[rows].gather(2, index) * (distance[:, None] < bias.size(2))
        visible = distance >= 0
        if window:
            visible &= distance < window
        scores.masked_fill_(~visible[:, None], -torch.inf)
        probs = torch.cat((scores, sinks[rows, :, None]), dim=2).softmax(2)[:, :, :-1]
        probs = probs.unflatten(1, (kv_heads, groups))
        output[rows] = torch.einsum('qkgt,tkd->qkgd', probs, values).flatten(1, 2)
    return output


class InklingAttentionTest(unittest.TestCase):
    def check_attention(self, seq, pos, width, extent, windows, seed, bias_dtype=torch.float32, position_shift=0):
        device = 'cuda'
        gen = torch.Generator(device=device).manual_seed(seed)
        def randn(*shape):
            return torch.randn(shape, device=device, generator=gen)
        heads, kv_heads, dim, page_size = 24, 8, 128, 16
        pages = 2 * width
        table = torch.randperm(pages, device=device, generator=gen).int().view(2, width)
        table[1, 1], table[1, 2] = -1, pages
        k, v = (randn(pages, page_size, kv_heads, dim).bfloat16() for _ in range(2))
        q = randn(len(seq), heads, dim).bfloat16()
        bias, sinks = randn(len(seq), heads, extent).to(bias_dtype), randn(len(seq), heads) * 5
        sequences = torch.tensor(seq, dtype=torch.int32, device=device)
        positions = torch.tensor(pos, dtype=torch.int32, device=device)
        for window in windows:
            with self.subTest(window=window):
                expected = reference(q, k, v, bias, sinks, sequences, positions, table, window)
                actual = attention(q, k, v, bias, sinks, sequences, positions + position_shift, table, window)
                diff = actual.float() - expected
                self.assertLess((diff.norm() / expected.norm()).item(), .005)
                self.assertLess(diff.abs().max().item(), .032)
                empty = (expected == 0).all(-1)
                self.assertTrue((actual[empty] == 0).all().item())

    def test_gated_output_projection(self):
        from snowllm.models.projection import pack_bf16
        from snowllm.ops.inkling import dense, sigmoid_gate

        gen = torch.Generator(device="cuda").manual_seed(916)
        def matrix(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
        weight = pack_bf16(matrix(2304, 3072))
        k, v = (matrix(65, 16, 8, 128) for _ in range(2))
        table = torch.arange(65, device="cuda", dtype=torch.int32).view(1, -1)
        for rows, prefill in ((1, False), (47, True)):
            sequence = torch.zeros(rows, device="cuda", dtype=torch.int32)
            positions = torch.arange(980, 980 + rows, device="cuda", dtype=torch.int32)
            args = (matrix(rows, 24, 128), k, v, matrix(rows, 24, 1024), matrix(rows, 24).float(),
                    sequence, positions, table, 1024)
            gate = matrix(rows, 3072)
            expected = dense(sigmoid_gate(attention(*args).flatten(1), gate), *weight, prefill)
            actual = attention(*args, output_projection=(gate, weight[0], prefill))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_paged_global_and_sliding(self):
        self.check_attention([0, 0, 1, 0, 1, 0, -1, 1], [0, 15, 16, 1023, 1024, 1041, 2, -1],
                             width=66, extent=1024, windows=(0, 1024), seed=234)

    def test_ragged_resumed_prefill_with_wrapped_pages(self):
        self.check_attention([0] * 83 + [1] * 54, [*range(58, 141), *range(159, 213)],
                             width=9, extent=81, windows=(0, 67), seed=473)

    def test_sliding_prefill_near_int32_limit(self):
        # Shifting by complete KV rings preserves the visible keys and their relative distances.
        self.check_attention([1] * 79, list(range(177, 256)), width=8, extent=37,
                             windows=(1, 31, 67), seed=827, position_shift=2**31 - 256)

    def test_split_decode_with_ring_and_sink(self):
        self.check_attention([1, 0], [8192, -1], width=65, extent=1024, windows=(1024, 0),
                             seed=132, bias_dtype=torch.bfloat16)

    def test_single_decode_sliding_window(self):
        for position in (8192, 8207):
            with self.subTest(position=position):
                self.check_attention([1], [position], width=65, extent=1024, windows=(1024,),
                                     seed=132, bias_dtype=torch.bfloat16)

    def test_batched_decode(self):
        self.check_attention([0, 1], [130, 32], width=7, extent=60, windows=(0, 1), seed=429)


if __name__ == '__main__':
    unittest.main()

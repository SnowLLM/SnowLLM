# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from snowllm.ops.inkling import attention


class InklingAttentionTest(unittest.TestCase):
    def test_paged_global_and_sliding(self):
        device = "cuda"
        gen = torch.Generator(device=device).manual_seed(234)
        def randn(*shape):
            return torch.randn(shape, device=device, generator=gen)
        size, heads, kv_heads, dim, extent = 16, 24, 8, 128, 1024
        width = 66
        table = torch.randperm(2 * width, device=device, generator=gen).int().view(2, width)
        k, v = [randn(2 * width, size, kv_heads, dim).bfloat16() for _ in range(2)]
        positions = torch.tensor([0, 15, 16, 1023, 1024, 1041], device=device, dtype=torch.int32)
        seq = torch.tensor([0, 0, 1, 0, 1, 0], device=device, dtype=torch.int32)
        q = randn(len(seq), heads, dim).bfloat16()
        bias = randn(len(seq), heads, extent)
        sinks = randn(len(seq), heads) * 5
        for window in (0, extent):
            got = attention(q, k, v, bias, sinks, seq, positions, table, window)
            for i, (s, pos) in enumerate(zip(seq.tolist(), positions.tolist())):
                start = max(0, pos + 1 - window) if window else 0
                p = torch.arange(start, pos + 1, device=device)
                pages = table[s, p // size]
                key, value = [a[pages, p % size].repeat_interleave(heads // kv_heads, 1).float()
                              for a in (k, v)]
                score = torch.einsum("hd,thd->ht", q[i].float(), key) / dim**.5
                dist = pos - p
                recent = dist < extent
                score[:, recent] += bias[i, :, dist[recent]]
                probs = torch.cat((score, sinks[i, :, None]), dim=1).softmax(1)[:, :-1]
                want = torch.einsum("ht,thd->hd", probs, value)
                torch.testing.assert_close(got[i].float(), want, rtol=.005, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
import pathlib
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from snowllm import ops
from snowllm.checkpoint.gguf import GGUF, TYPES
from snowllm.checkpoint.gguf.dequant import dequantize
from snowllm.checkpoint.gguf.source import GGUFWeightSource
from snowllm.errors import SnowLLMError
from test_qwen4exp_fused_gguf import checkpoint


def blocks(rows, k, device='cpu'):
    codes = torch.randint(0, 4, (rows * k // 64, 64), device=device)
    scales = torch.linspace(-.03, .03, codes.shape[0], device=device).half().unsqueeze(1)
    packed = sum(codes[:, shift::4] << (2 * shift) for shift in range(4)).byte()
    raw = torch.cat((scales.view(torch.uint8), packed), dim=1).flatten()
    return raw, (codes.float() - 1) * scales.float()


class Q20Tests(unittest.TestCase):
    def test_old_native_library_is_rejected_before_allocation(self):
        tensor = SimpleNamespace(quant=SimpleNamespace(name='Q2_0'))
        rd = SimpleNamespace(gguf=SimpleNamespace(tensors={'weight': tensor},
                                                 path=pathlib.Path('model.gguf')))
        for library in (SimpleNamespace(), SimpleNamespace(snowllm_kquant_supported=lambda _: 0)):
            with self.subTest(library=library), \
                 patch('snowllm.checkpoint.gguf.source.lib', library), \
                 self.assertRaisesRegex(SnowLLMError, 'upgrade or rebuild'):
                GGUFWeightSource._check_formats(rd)

    def test_dequantization_matches_independent_scalar_reference(self):
        raw = b''.join(struct.pack('<e', scale) + bytes([0xe4, 0x1b] * 8)
                       for scale in (0., 1., -1., 2 ** -24, 65504.))
        expected = [((byte >> shift) & 3) - 1
                    for byte in [0xe4, 0x1b] * 8 for shift in (0, 2, 4, 6)]
        expected = torch.tensor([value * scale
                                for scale in (0., 1., -1., 2 ** -24, 65504.)
                                for value in expected], dtype=torch.float32)
        actual = dequantize(torch.tensor(list(raw), dtype=torch.uint8), 'Q2_0', len(expected),
                            torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_standard_type_42_and_legacy_vq_blob_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'model.gguf'
            raw, _ = blocks(2, 64)
            blob = b'DQVL' + bytes(60)
            checkpoint(path, [('blk.8.ffn_down_exps.weight', (64, 2), 42, bytes(raw.tolist())),
                              ('blk.0.ffn_exps_vq.blob', (len(blob),), 42, blob)])
            g = GGUF(path)
            weight = g['blk.8.ffn_down_exps.weight']
            self.assertEqual(weight.quant.name, 'Q2_0')
            self.assertEqual(weight.nbytes, 36)
            self.assertEqual(g['blk.0.ffn_exps_vq.blob'].quant.name, 'VQ')
            self.assertEqual(g.read('blk.0.ffn_exps_vq.blob'), blob)
            self.assertEqual(TYPES[42].block, 64)


class Q20GPUTests(unittest.TestCase):
    def test_native_packed_gemm_matches_reference_and_planned_sizes(self):
        torch.manual_seed(7)
        fmt = ops.KQUANT_FORMATS['Q2_0']
        for k in (64, 320, 640, 1024):
            with self.subTest(k=k):
                n = 384
                raw, values = blocks(n, k, 'cuda')
                self.assertEqual(ops.kquant_gguf_bytes(fmt, n, k), raw.numel())
                with ops.dry_load():
                    planned = ops.gemm_kquant_shuffle_b(fmt, raw, n, k)
                quant, meta = ops.gemm_kquant_shuffle_b(fmt, raw, n, k)
                self.assertEqual([t.numel() for t in planned], [quant.numel(), meta.numel()])
                ref = values.reshape(n, k).bfloat16()
                for m in (1, 17, 257):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
                    out = torch.empty(m, n, dtype=torch.float32, device='cuda')
                    ws = ops.empty_bytes(ops.gemm_kquant_a_ws_bytes(m, k))
                    ops.gemm_kquant_a(fmt, x, quant, meta, out, m, n, k, ws)
                    expected = x.float() @ ref.float().T
                    relative = (out - expected).norm() / expected.norm()
                    self.assertLess(relative.item(), .003)


if __name__ == '__main__':
    unittest.main()

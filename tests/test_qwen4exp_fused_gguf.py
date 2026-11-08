import pathlib
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from snowllm._capi import SnowLLMError
from snowllm.checkpoint.gguf import GGUF, TYPES
from snowllm.checkpoint.gguf.qwen4exp import projection_views
from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource


META = {
    'general.architecture': 'qwen4exp',
    'qwen4exp.embedding_length': 256,
    'qwen4exp.hyper_connection.count': 2,
    'qwen4exp.hyper_connection.low_rank': 3,
    'qwen4exp.ssm.group_count': 2,
    'qwen4exp.ssm.state_size': 4,
    'qwen4exp.ssm.inner_size': 16,
}


def string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def checkpoint(path, specs, meta=None):
    meta = META if meta is None else meta
    header = b'GGUF' + struct.pack('<IQQ', 3, len(specs), len(meta))
    for key, value in meta.items():
        header += string(key) + (struct.pack('<I', 8) + string(value) if isinstance(value, str)
                                 else struct.pack('<Ii', 5, value))
    payload = bytearray()
    for name, dims, kind, raw in specs:
        header += string(name) + struct.pack('<I', len(dims))
        header += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, len(payload))
        payload.extend(raw)
        payload.extend(b'\0' * (-len(payload) % 32))
    path.write_bytes(header + b'\0' * (-len(header) % 32) + payload)


def payload(kind, rows, k):
    n = TYPES[kind].nbytes(rows * k)
    return bytes((i * 13 + i // 17) % 256 for i in range(n))


class FusedProjectionTests(unittest.TestCase):
    def test_hc_views_preserve_payload_and_total_size(self):
        for kind in (6, 8, 14, 20, 21, 23, 30):
            for branch in ('attn', 'ffn'):
                with self.subTest(kind=kind, branch=branch), tempfile.TemporaryDirectory() as directory:
                    path = pathlib.Path(directory) / 'model.gguf'
                    name = f'blk.0.hc_{branch}_down_inject.weight'
                    raw = payload(kind, 5, 512)
                    checkpoint(path, [(name, (512, 5), kind, raw)])
                    g = GGUF(path)
                    start = g.file_offset(name)
                    projection_views(g)
                    down = name.replace('_down_inject.', '_down.')
                    inject = name.replace('_down_inject.', '_inject.')
                    self.assertEqual(g[down].shape, (3, 512))
                    self.assertEqual(g[inject].shape, (2, 512))
                    self.assertEqual(g.read(down) + g.read(inject), raw)
                    self.assertEqual(g.file_offset(down), start)
                    self.assertEqual(g.file_offset(inject), start + g[down].nbytes)
                    self.assertEqual(sum(t.nbytes for t in g.tensors.values()), len(raw))
                    projection_views(g)
                    self.assertEqual(len(g.tensors), 2)

    def test_fused_linear_retains_qkv_and_gate_head_permutations(self):
        for kind in (6, 8, 14, 20, 21, 23):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                qkv, gate = payload(kind, 32, 256), payload(kind, 16, 256)
                checkpoint(root / 'fused.gguf', [('blk.0.attn_qkv_gate.weight', (256, 48), kind, qkv + gate)])
                checkpoint(root / 'split.gguf', [('blk.0.attn_qkv.weight', (256, 32), kind, qkv),
                                                ('blk.0.attn_gate.weight', (256, 16), kind, gate)])
                sources = []
                for path in (root / 'fused.gguf', root / 'split.gguf'):
                    with patch('snowllm.checkpoint.gguf.source.FileReader.__init__', return_value=None):
                        rd = GGUFReader(path)
                    rd.raw = lambda name, reader=rd: torch.tensor(list(reader.gguf.read(name)), dtype=torch.uint8)
                    sources.append(GGUFWeightSource(rd, dict(linear_num_key_heads=2,
                        linear_num_value_heads=4, linear_key_head_dim=4)))
                keys = ['model.language_model.layers.0.linear_attn.' + suffix
                        for suffix in ('in_proj_qkv.weight', 'in_proj_z.weight')]
                actual, expected = [s.kquant_concat(keys) for s in sources]
                self.assertEqual(actual[1], expected[1])
                self.assertTrue(torch.equal(actual[0], expected[0]))
                for key in keys:
                    self.assertTrue(sources[0].has(key))
                    self.assertEqual(sources[0].rows(key), sources[1].rows(key))
                    self.assertEqual(sources[0].quant_names(key), sources[1].quant_names(key))

    def test_views_cross_byte_range_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / 'whole.gguf'
            raw = payload(21, 5, 512)
            name = 'blk.0.hc_attn_down_inject.weight'
            checkpoint(path, [(name, (512, 5), 21, raw)])
            cut = GGUF(path).file_offset(name) + TYPES[21].nbytes(3 * 512) + 13
            data = path.read_bytes()
            parts = [root / f'model.gguf.part{i}-of-2' for i in (1, 2)]
            parts[0].write_bytes(data[:cut])
            parts[1].write_bytes(data[cut:])
            g = GGUF(parts[0])
            projection_views(g)
            inject = 'blk.0.hc_attn_inject.weight'
            self.assertEqual(len(list(g.ranges(inject))), 2)
            self.assertEqual(g.read('blk.0.hc_attn_down.weight') + g.read(inject), raw)

    def test_rejects_wrong_shapes_ambiguous_members_and_partial_quant_rows(self):
        name = 'blk.0.hc_attn_down_inject.weight'
        cases = [([(name, (512, 6), 21, payload(21, 6, 512))], META, 'shape'),
                 ([(name, (512, 5), 21, payload(21, 5, 512)),
                   ('blk.0.hc_attn_inject.weight', (512, 2), 21, payload(21, 2, 512))], META,
                  'also carries'),
                 ([(name, (128, 6), 21, payload(21, 6, 128))],
                  dict(META, **{'qwen4exp.embedding_length': 64,
                                'qwen4exp.hyper_connection.low_rank': 4}), 'cannot be split')]
        for specs, meta, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / 'model.gguf'
                checkpoint(path, specs, meta)
                g = GGUF(path)
                names = set(g.tensors)
                with self.assertRaisesRegex(SnowLLMError, error):
                    projection_views(g)
                self.assertEqual(set(g.tensors), names)


if __name__ == '__main__':
    unittest.main()

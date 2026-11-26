import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from snowllm import _capi, ops
from snowllm.engine import Engine, Request, SamplingParams
from snowllm.engine.inputs import synthetic_batch
from snowllm.engine.prefix_cache import LinearResidue, PrefixCache, PrefixStore
from snowllm.engine.qwen4exp_runner import Qwen4ExpRunner
from snowllm.engine.qwen_runner import QwenRunner


class Qwen4ExpPrefillPlanTest(unittest.TestCase):
    def setUp(self):
        _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
        self.runner = Qwen4ExpRunner.__new__(Qwen4ExpRunner)
        self.runner.max_num_seqs = 4
        self.runner.max_blocks_per_seq = 8

    def probe(self, rows):
        def cpu_batch(*args, **kwargs):
            return synthetic_batch(*args, **kwargs,
                tensor=lambda name, values, dtype: torch.tensor(values, dtype=dtype))
        with patch('snowllm.engine.qwen4exp_runner.synthetic_batch', cpu_batch):
            return self.runner._probe_batch(rows, True)

    def test_prefill_covers_ragged_batches(self):
        for rows in (1, 3, 63, 64, 65, 1025, 6145, 8192):
            with self.subTest(rows=rows):
                b = self.probe(rows)
                self.assertEqual(b.batch_size, min(rows, 4))
                self.assertEqual(b.num_tokens, rows)
                self.assertEqual(b.host_cu_seqlens[-1], rows)
                self.assertEqual(b.last_row.numel(), b.batch_size)
                self.assertEqual(b.state_indices.numel(), b.batch_size)
                self.assertEqual(b.has_state.numel(), b.batch_size)
                self.assertTrue(b.is_prefill)
                self.assertFalse(b.need_logits)
                for count in range(1, min(rows, 4) + 1):
                    lengths = [rows // count] * count
                    lengths[-1] += rows % count
                    needed, _ = ops.prefill_q_plan(lengths)
                    self.assertGreaterEqual(b.total_q_blocks, needed)
        b = self.probe(6145)
        needed, _ = ops.prefill_q_plan([3073, 3072])
        self.assertGreaterEqual(b.total_q_blocks, needed)

    def test_qsa_job_descriptors_preserve_job_and_null_plan(self):
        self.assertIsNone(ops.qsa_job_descriptors(None))
        self.assertEqual(len(ops.qsa_job_descriptors([])), 0)
        with patch('snowllm.ops.ple._p', side_effect=lambda value: value):
            jobs = ops.qsa_job_descriptors([(1280, 512, 1, 130560, 10, 20, 32, 30)])
        job = jobs[0]
        self.assertEqual((job.at, job.rows, job.request, job.first, job.q_blocks),
                         (1280, 512, 1, 130560, 32))
        self.assertEqual((job.cu, job.length, job.qmap), (10, 20, 30))

    def test_short_uniform_qsa_and_ragged_verify_have_separate_plans(self):
        runner = self.runner
        runner._plan_short_drafts = True
        runner.geo = SimpleNamespace(index_topk=256)
        runner.full_mods = [object()]
        def key(tokens, ragged):
            batch = SimpleNamespace(batch_size=4, tokens_per_req=tokens, varlen_attn=ragged)
            return runner._decode_plan_key(batch)
        self.assertEqual(key(1, False), key(4, False))
        self.assertNotEqual(key(4, False), key(4, True))
        self.assertNotEqual(key(4, False), key(16, False))

    def test_decode_uses_existing_probe(self):
        sentinel = object()
        with patch.object(QwenRunner, '_probe_batch', return_value=sentinel) as base:
            self.assertIs(self.runner._probe_batch(12, False), sentinel)
        base.assert_called_once_with(12, False)

    def test_prefetch_tracks_loaded_ple_layers(self):
        self.runner.ple_mods = []
        self.assertFalse(self.runner.prefetch_ple)
        self.runner.ple_mods.append(object())
        self.assertTrue(self.runner.prefetch_ple)


class PleChunkOrderTest(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.__new__(Engine)
        self.engine.prefill_chunk = 8192
        self.engine.runner = SimpleNamespace(prefetch_ple=True)
        residue = LinearResidue([(0, SimpleNamespace(state=(), ckpt=()))])
        self.engine.cache = PrefixCache(PrefixStore(SimpleNamespace(align=16), residue, 8))

    def chunks(self, length, cached=0, system_end=0):
        r = Request([0] * length, SamplingParams(), num_prefilled=cached, cached_tokens=cached)
        chunks = []
        predicted = None
        while r.num_prefilled < length:
            hi, next_hi = self.engine._prefill_ends(r, system_end)
            self.assertGreater(hi, r.num_prefilled)
            if predicted is not None:
                self.assertEqual(hi, predicted)
            chunks.append(hi - r.num_prefilled)
            predicted = next_hi if next_hi > hi else None
            r.num_prefilled = hi
        return chunks

    def test_only_move_a_remainder_that_preserves_two_chunks(self):
        for length, expected in [(8000, [8000]), (8196, [8192, 4]), (8224, [8192, 32]),
                                 (8256, [8192, 64]), (8320, [128, 8192]),
                                 (8704, [512, 8192]), (8736, [8192, 544]),
                                 (16384, [8192, 8192]), (16448, [8192, 8192, 64])]:
            with self.subTest(length=length):
                self.assertEqual(self.chunks(length), expected)

    def test_respects_cached_prefix_and_system_checkpoint(self):
        self.assertEqual(self.chunks(9216, cached=512), [512, 8192])
        self.assertEqual(self.chunks(8704, system_end=1024), [1024, 7680])
        self.assertEqual(self.chunks(9000, system_end=1024), [1024, 7968, 8])
        self.assertEqual(self.chunks(8464, cached=32), [480, 7952])

    def test_without_prefix_cache_or_prefetch(self):
        self.engine.cache = None
        self.assertEqual(self.chunks(8224), [8192, 32])
        self.assertEqual(self.chunks(8736), [544, 8192])
        self.engine.runner.prefetch_ple = False
        self.assertEqual(self.chunks(8704), [8192, 512])


if __name__ == '__main__':
    unittest.main()

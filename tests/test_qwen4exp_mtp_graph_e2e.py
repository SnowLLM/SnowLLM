# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import unittest
from unittest.mock import patch

import torch

from snowllm.benchmark import context as bc
from snowllm.cli import build
from snowllm.engine import SamplingParams


@unittest.skipUnless(os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_MTP_MODEL'),
                     'set SNOWLLM_QWEN38_FLASH_NEXT_MTP_MODEL')
class Qwen4ExpMtpGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        cls.k = int(os.environ.get('SNOWLLM_TEST_NUM_SPEC', '2'))
        model = os.environ['SNOWLLM_QWEN38_FLASH_NEXT_MTP_MODEL']
        with patch.dict(os.environ, SNOWLLM_MTP_DRAFT_GRAPH='1'):
            cls.state = build(model_path=model, max_num_seqs=4, max_model_len=409600,
                num_kv_blocks=None, default_max_tokens=32, seed=73, profile_dir=None,
                num_spec=cls.k, prefill_chunk=8192, prefix_memory_ratio=.08,
                exact_stats=True, max_gpu_mem_util=.9)
        cls.engine = cls.state.engine.engine
        cls.prompts = []
        for rep in range(4):
            text = bc.prompt_of(1024, rep, model, thinking=False)
            ids = cls.state.tokenizer.apply_chat_template(bc._messages(text), tokenize=True,
                add_generation_prompt=True, thinking=False, enable_thinking=False, tools=None)
            cls.prompts.append(ids['input_ids'] if hasattr(ids, 'keys') else ids)

    @staticmethod
    def digest(value):
        bits = value.contiguous().view(torch.int16).cpu().numpy().tobytes()
        return hashlib.sha256(bits).hexdigest()

    def compare(self, count, prefix, sampling=False, graph_available=True):
        engine, k = self.engine, self.k
        spec = engine.spec
        records, arms = [], []
        original = spec.propose_first

        def record(batch, pos, cu, total, width, tokens, accepted, seq):
            kept = accepted.tolist()
            original(batch, pos, cu, total, width, tokens, accepted, seq)
            # Rejected first-pass rows have slot=-1 and no defined output contract.
            valid = ([i * (width + 1) + j for i, n in enumerate(kept) for j in range(n)]
                     if width == 1 else list(range(len(batch))))
            records.append((kept, spec.d_drafts[:len(batch), :width].cpu().tolist(),
                self.digest(engine.runner.mtp_streams[valid]),
                self.digest(engine.runner.draft_logits[valid])))

        rng = engine.sampler.rng.get_state()
        for graph in (False, True):
            while engine.cache.evict():
                pass
            spec.use_graphs = False
            if prefix:
                for ids in self.prompts[:count]:
                    engine.add(ids, SamplingParams(temperature=0, max_new_tokens=8))
                engine.run()
            spec.use_graphs = graph
            spec.accept_hist = []
            records.clear()
            old = engine.runner.n_draft_replays
            engine.sampler.rng.set_state(rng)
            params = (SamplingParams(temperature=.7, top_k=20, top_p=.9,
                        repetition_penalty=1.05, no_repeat_ngram_size=3, max_new_tokens=32)
                      if sampling else SamplingParams(temperature=0, max_new_tokens=32))
            requests = [engine.add(ids, params) for ids in self.prompts[:count]]
            with patch.object(spec, 'propose_first', record):
                engine.run()
            torch.cuda.synchronize()
            replays = engine.runner.n_draft_replays - old
            self.assertEqual(replays > 0, graph and graph_available)
            if prefix:
                self.assertTrue(any(r.cached_tokens for r in requests))
            arms.append(([r.out for r in requests], [r.cached_tokens for r in requests],
                         list(spec.accept_hist), list(records)))
        self.assertEqual(arms[0], arms[1], (k, count, prefix, sampling))

    def test_full_proposal_graph(self):
        for count in (1, 2, 3, 4):
            self.assertIn(('mtp-proposals', count, self.k + 1), self.engine.runner.graphs)
        for count in (1, 2, 4):
            for prefix in (False, True):
                with self.subTest(batch=count, prefix=prefix):
                    self.compare(count, prefix)
        self.compare(2, True, sampling=True)

    def test_missing_graph_falls_back(self):
        graphs = {key: value for key, value in self.engine.runner.graphs.items()
                  if key[0] != 'mtp-proposals'}
        with patch.object(self.engine.runner, 'graphs', graphs):
            self.compare(1, False, graph_available=False)


if __name__ == '__main__':
    unittest.main()

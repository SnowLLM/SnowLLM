# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from pathlib import Path

import torch

from snowllm.ops.inkling import commit_history, draft_gates, draft_mix, draft_scores


class TestInklingDraft(unittest.TestCase):
    def test_verify_policy_ignores_replaced_drafts(self):
        from types import SimpleNamespace
        from snowllm.engine.dspark_schedule import InklingVerifyPolicy

        policy = InklingVerifyPolicy(8)
        request = SimpleNamespace(drafts=[10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(policy.choose(request, [.8] * 7), 2)
        request.drafts[1] = 44
        policy.observe(request, 4, 3)
        self.assertEqual(policy.by_confidence[8], [1, 1])
        self.assertIsNone(policy.pending)
        for width in policy.widths:
            policy.times[width].extend([width * width] * 2)
        self.assertEqual(policy.choose(request, [.8] * 7), 2)
        request.drafts.clear()
        self.assertEqual(policy.choose(request, [.8] * 7), 8)
        self.assertIsNone(policy.pending)

    def test_recurrent_head(self):
        torch.manual_seed(14)
        device = "cuda"
        rows, hidden, rank = 3, 2304, 256
        taps = torch.randn(rows,4,hidden,device=device,dtype=torch.bfloat16)
        alpha = torch.randn(4,device=device).softmax(0)
        gamma = torch.randn(hidden,device=device)
        expected = (taps.float()*alpha.view(1,4,1)).sum(1)
        expected = expected * (expected.square().mean(1,keepdim=True)+1.e-6).rsqrt() * gamma
        torch.testing.assert_close(draft_mix(taps.flatten(1),alpha,gamma,1.e-6),expected.bfloat16(),
                                   atol=.001,rtol=.008)
        p = torch.randn(rows,3*rank,device=device)
        b = torch.randn(3*rank,device=device)
        prev = torch.randn(rows,rank,dtype=torch.bfloat16,device=device)
        gate, candidate, output = (p+b).bfloat16().chunk(3,dim=1)
        gate=gate.sigmoid()
        updated=gate*prev+(1-gate)*candidate.tanh()
        latent=draft_gates(p,b,prev)
        torch.testing.assert_close(prev,updated)
        torch.testing.assert_close(latent,output.tanh())
        vocab=107
        base = torch.randn(rows,8,vocab+5,device=device)[:,2]
        fold = torch.randn(rows,vocab+5,device=device)
        bias=torch.randn(vocab,device=device)
        torch.testing.assert_close(draft_scores(base,fold,bias,30),
                                   30*((base[:,:vocab]+(fold[:,:vocab]+bias))/30).tanh())

    def test_search_logits_preserve_matches_and_zero_padding(self):
        from snowllm.ops.inkling import pack_search, search_logits

        generator = torch.Generator(device="cuda").manual_seed(17)
        weights = torch.randn(17, 256, generator=generator, device="cuda").sign().bfloat16()
        weights[0].zero_()
        queries = weights[[3, 7, 16]]
        logits = search_logits(queries, pack_search(weights, 200064))
        expected = queries.float() @ weights.float().T
        torch.testing.assert_close(logits[:, :17], expected)
        self.assertEqual(logits[:, 17:].count_nonzero().item(), 0)

    def test_sampled_recurrent_step(self):
        from snowllm.ops.inkling import sample_draft_step

        scores = torch.tensor([[2., 3., 1., -1.], [1., 0., 2., 4.]], device="cuda")
        temperature = torch.tensor([0., .8], device="cuda")
        slots = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
        ids = torch.full((3, 2, 3), -1, dtype=torch.int64, device="cuda")
        q = torch.full_like(ids, -1, dtype=torch.float32)
        uniform = torch.tensor([.5, .999], device="cuda")
        picked = sample_draft_step(scores, temperature, uniform, slots, ids[:, 1], q[:, 1])
        self.assertEqual(picked.tolist(), [1, 0])
        self.assertEqual(ids[:, 1].tolist(), [[3, 2, 0], [-1, -1, -1], [1, 0, 2]])
        expected = torch.softmax(scores[1, [3, 2, 0]] / .8, 0)
        torch.testing.assert_close(q[0, 1], expected)
        self.assertEqual(q[2, 1].tolist(), [1., 0., 0.])
        self.assertTrue((q[:, 0] == -1).all())

    def test_commit_only_accepted_prefix(self):
        device = "cuda"
        h = torch.randn(4,3,3,32,device=device,dtype=torch.bfloat16)
        inputs = torch.randn(9,3,32,device=device,dtype=torch.bfloat16)
        expected = h.clone()
        for slot,start,kept in ((2,0,1),(0,4,3),(3,7,2)):
            expected[slot]=torch.cat((h[slot],inputs[start:start+kept].transpose(0,1)),dim=1)[:,-3:]
        def i32(values):
            return torch.tensor(values,dtype=torch.int32,device=device)
        commit_history(inputs,h,i32([2,0,3]),i32([1,3,2]),4,i32([0,4,7,9]))
        torch.testing.assert_close(h,expected,rtol=0,atol=0)

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF") and os.getenv("SNOWLLM_INKLING_DSPARK"),
                         "set the Inkling target and DSpark draft paths")
    def test_batched_graph_and_prefix_restore(self):
        self._check_batched_restore(8)

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF") and os.getenv("SNOWLLM_INKLING_DSPARK"),
                         "set the Inkling target and DSpark draft paths")
    def test_batched_short_verification(self):
        self._check_batched_restore(4)

    def _check_batched_restore(self, width):
        from snowllm import cli
        from snowllm.engine import SamplingParams

        root = Path(os.environ["SNOWLLM_INKLING_GGUF"]).parent
        state = cli.build(str(root), 2, 4096, None, 32, 777, None,
                          dflash=os.environ["SNOWLLM_INKLING_DSPARK"],
                          dflash_block=width, dflash_adaptive=width == 8, prefill_chunk=128, prefix_memory_ratio=0.08, batch_prefill=False)
        engine, tokenizer = state.engine.engine, state.tokenizer
        window = engine.dflash.geo.sliding_window
        text = "The files contain lists of integers. Read the table before computing the result. "
        shared = tokenizer.encode(text * 100)[:window + 64]
        self.assertGreater(len(shared), window)
        self.assertGreater(engine.runner.draft_ring_pages, 0)

        def run(suffixes):
            params = SamplingParams(temperature=0, max_new_tokens=24)
            requests = [engine.add(shared + tokenizer.encode(suffix), params) for suffix in suffixes]
            engine.run()
            return requests

        question = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 2? Answer with a single number."}],
            tokenize=False, add_generation_prompt=True, reasoning_effort="none")
        prompt = tokenizer.encode(question, add_special_tokens=False)
        request = engine.add(prompt, SamplingParams(temperature=0, max_new_tokens=8))
        engine.run()
        self.assertEqual(request.out, [200001, 200008, 371, 150409, 171960, 149878, 150744, 169292])

        first, second = run([" Question 1", " Question 2"])
        before = engine.stats().cache_hits
        restored, other = run([" Question 1", " Question 2"])
        self.assertTrue(second.out)
        self.assertEqual(other.out, second.out)
        self.assertEqual(restored.out, first.out)
        self.assertGreater(restored.cached_tokens, window)
        self.assertGreater(engine.stats().cache_hits, before)
        self.assertGreater(engine.stats().graph_replays, 0)
        self.assertGreater(engine.stats().draft_replays, 0)
        self.assertGreater(sum(engine.dflash.accept_hist[1:]), 0)

        greedy = engine.add(prompt, SamplingParams(temperature=0, max_new_tokens=8))
        sampled = engine.add(prompt, SamplingParams(temperature=.8, top_k=12, max_new_tokens=24))
        engine.run()
        self.assertEqual(greedy.out, request.out)
        self.assertTrue(sampled.out)
        self.assertNotIn(-1, sampled.out)
        self.assertEqual(engine.dflash.sampling.proposals, [None, None])

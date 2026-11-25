# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from pathlib import Path

import torch

from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.dequant import dequantize
from snowllm.checkpoint.tokenizer import load_tokenizer
from snowllm.models.inkling2 import InklingTextModel


class InklingDecoderTest(unittest.TestCase):
    def test_qkv_input_norm_strides(self):
        from snowllm.ops.inkling import normalize_qkv, normalize_rows

        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(17, 3, 2304, dtype=dtype, device="cuda")
            inputs = x.unbind(1)
            weights = tuple(torch.randn(2304, device="cuda") for _ in range(3))
            actual = normalize_qkv(inputs, weights, 1.e-6)
            for y, v, w in zip(actual, inputs, weights):
                torch.testing.assert_close(y, normalize_rows(v, 1.e-6, w), atol=0, rtol=0)

    def test_grouped_attention_projection_sources(self):
        from snowllm.models.projection import pack_bf16
        from snowllm.ops.inkling import dense, input_projections

        widths = (3072, 3072, 1024, 1024, 384, 24, 32, 32)
        sources = (0, 0, 1, 2, 0, 0, 1, 2)
        gen = torch.Generator(device="cuda").manual_seed(915)
        def matrix(rows):
            return torch.randn(rows, 2304, device="cuda", dtype=torch.bfloat16, generator=gen)
        weights = [pack_bf16(matrix(width)) for width in widths]
        for rows, count in ((1, 6), (2, 8)):
            inputs = tuple(matrix(rows) for _ in range(3))
            outputs = input_projections(inputs, [w[0] for w in weights[:count]], widths[:count], False)
            self.assertEqual(len(outputs), count)
            for j, result in enumerate(outputs):
                expected = dense(inputs[sources[j]], *weights[j], False)
                torch.testing.assert_close(result, expected.float() if j == 5 else expected, atol=0, rtol=0)

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF"), "set SNOWLLM_INKLING_GGUF to the Inkling GGUF")
    def test_packed_q8_matches_expanded_weights(self):
        from snowllm.checkpoint.gguf.source import GGUFReader
        from snowllm.models.projection import pack_bf16, pack_kquant
        from snowllm.ops.inkling import input_projections, packed_logits

        generator = torch.Generator(device="cuda").manual_seed(493)
        with GGUFReader(Path(os.environ["SNOWLLM_INKLING_GGUF"])) as rd:
            def pack(name):
                weight = rd.tensor(name)
                bf16, n, k, rows = pack_bf16(weight)
                if rd.gguf[name].quant.name == "Q8_0" and rows % 128 == 0:
                    return bf16, pack_kquant(rd.raw(name), 8, n, k), rows
                return bf16, bf16, rows

            if rd.gguf["output.weight"].quant.name != "Q8_0":
                self.skipTest("requires a Q8_0 output head")
            vocab = rd.gguf.need("{arch}.vocab_size_unpadded")
            expanded, n, k, _ = pack_bf16(rd.tensor("output.weight")[:vocab])
            compact = pack_kquant(rd.raw("output.weight"), 8, n, k)
            x = torch.randn(5, k, device="cuda", dtype=torch.bfloat16, generator=generator)
            torch.testing.assert_close(packed_logits(x, expanded, n)[:, :vocab],
                                       packed_logits(x, compact, n)[:, :vocab], atol=0, rtol=0)
            del expanded, compact
            names = ("q", "gate", "k", "v", "rel", "sink_proj", "k_conv_proj", "v_conv_proj")
            names = [f"blk.0.attn_{key}.weight" for key in names]
            expanded, mixed, widths = zip(*(pack(name) for name in names if name in rd.gguf))
            for rows, prefill in ((5, False), (256, True)):
                inputs = tuple(torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=generator)
                               for _ in range(3))
                expected = input_projections(inputs, expanded, widths, prefill)
                actual = input_projections(inputs, mixed, widths, prefill)
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, atol=0, rtol=0)

    def test_attention_clip_and_norm_rounding(self):
        from snowllm.ops.inkling import normalize_rows

        gen = torch.Generator(device="cuda").manual_seed(1846)
        rows, heads = 2048, 24
        pos = torch.arange(6900, 6900 + rows, dtype=torch.int32, device="cuda")
        tau = (((pos.float() + 1) / 128).clamp_min(1).log() * .2 + 1).view(-1, 1, 1)
        for width in (16, 128):
            x = torch.randn(rows, heads, width, device="cuda", generator=gen).bfloat16()
            weight = torch.randn(heads, width, device="cuda", generator=gen)
            values = x.float() * weight
            values *= (3.5 / (values.norm(dim=-1, keepdim=True) + 1e-6)).clamp_max(1)
            if width == 16:
                values = values.bfloat16().float()
            expected = (values * tau).bfloat16()
            actual = normalize_rows(x, 3.5, weight, clip=True, positions=pos, floor=128, alpha=.2,
                                    round_relative=width == 16)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        x = torch.randn(rows, 2, 2304, device="cuda", generator=gen)[:, 1]
        weight = torch.randn(2304, device="cuda", generator=gen)
        expected = torch.nn.functional.rms_norm(x, (2304,), weight, 1e-5).bfloat16()
        torch.testing.assert_close(normalize_rows(x, 1e-5, weight), expected, atol=0, rtol=0)

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF"), "set SNOWLLM_INKLING_GGUF to the Inkling GGUF")
    def test_chunked_prefill_and_greedy_decode(self):
        path = Path(os.environ["SNOWLLM_INKLING_GGUF"])
        tokenizer, _ = load_tokenizer(path.parent)
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with a single number."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               reasoning_effort="none")
        ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), device="cuda")
        model = InklingTextModel.from_gguf(path)
        gguf = GGUF(path)
        rows = [0, 127, model.geo.vocab_size - 1]
        width = model.geo.hidden
        weight = []
        quant = gguf["output.weight"].quant
        stride = quant.nbytes(width)
        for row in rows:
            raw = torch.frombuffer(bytearray(gguf.read("output.weight", row * stride, stride)), dtype=torch.uint8)
            weight.append(dequantize(raw, quant.name, width))
        weight = torch.stack(weight).cuda().double()
        gen = torch.Generator(device="cuda").manual_seed(19)
        hidden = torch.randn(33, width, device="cuda", generator=gen).bfloat16()
        cap = model.config["final_logit_softcapping"]
        for count in (1, 32, 33):
            projected = model.logits(hidden[:count])
            self.assertEqual(projected.dtype, torch.float32)
            self.assertEqual(projected.shape, (count, model.geo.vocab_size))
            reference = (hidden[:count].double() @ weight.T / cap).tanh() * cap
            torch.testing.assert_close(projected[:, rows], reference.float(), rtol=1e-5, atol=1e-5)
        cache = model.cache(capacity=64, chunk_size=32)
        logits = model.forward(ids, cache)
        split = model.cache(capacity=64, chunk_size=32)
        chunked = torch.cat((model.forward(ids[:5], split), model.forward(ids[5:], split)))
        self.assertLess(((chunked - logits).norm() / logits.norm()).item(), .02)
        self.assertEqual(chunked[-1].argmax().item(), logits[-1].argmax().item())
        # llama.cpp b12648, Unsloth UD-Q4_K_M and UD-Q8_K_XL, greedy decoding.
        expected = [200001, 200008, 371, 150409, 171960, 149878, 150744, 169292]
        verify_cache = model.cache(capacity=64, chunk_size=32)
        model.forward(ids, verify_cache)
        block = torch.tensor(expected[:4], device="cuda")
        verified = model.forward(block, verify_cache, prefill=False)
        for i, token in enumerate(expected):
            predicted = logits[-1].argmax().view(1)
            self.assertEqual(predicted.item(), token, f"greedy token {i}")
            if i + 1 < len(expected):
                logits = model.forward(predicted, cache)
                if i < len(block):
                    torch.testing.assert_close(verified[i], logits[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import unittest

import torch
import torch.nn.functional as F

from snowllm.checkpoint.gguf.inkling2 import config
from snowllm.checkpoint.gguf.source import GGUFReader
from snowllm.models.inkling2.experts import InklingMoE


def dense_expert(projection, expert, gguf, name):
    if projection.meta is not None:
        from snowllm.checkpoint.gguf.dequant import dequantize

        t = gguf[name]
        _, n, k = t.shape
        size = t.quant.nbytes(n * k)
        raw = torch.frombuffer(bytearray(gguf.read(name, expert * size, size)), dtype=torch.uint8)
        return dequantize(raw, t.quant.name, n * k).view(n, k).float()
    weight = projection.weight[expert].cpu()
    if projection.scale is None:
        return weight.float()
    row_tiles, k_blocks, _, _ = weight.shape
    n = row_tiles * 16
    raw = torch.empty(row_tiles, 16, k_blocks, 36, dtype=torch.uint8)
    raw[..., :4] = weight[..., :16].permute(0, 3, 1, 2)
    raw[..., 4:] = weight[..., 16:].reshape(row_tiles, k_blocks, 4, 16, 8).permute(0, 3, 1, 2, 4).flatten(3)
    blocks = raw.reshape(-1, 36)
    scales = blocks[:, :4].contiguous().view(torch.float8_e4m3fn).float()
    packed = blocks[:, 4:].reshape(-1, 4, 8)
    codes = torch.cat((packed & 15, packed >> 4), dim=-1)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    decoded = levels[(codes & 7).long()] * torch.where(codes < 8, 1., -1.)
    return (decoded * scales.unsqueeze(-1)).reshape(n, -1) * projection.scale[expert].item()


class InklingExpertsTest(unittest.TestCase):
    def test_float_projections_and_packed_router(self):
        from snowllm.ops.gemm import gemm_bf16_shuffle_b
        from snowllm.ops.inkling import float_projection, relative_projection, route

        gen = torch.Generator(device="cuda").manual_seed(758)
        for k, n in ((16, 1024), (7, 141)):
            x = torch.randn(11, 3, k, generator=gen, device="cuda")
            w = torch.randn(k, n, generator=gen, device="cuda")
            actual = float_projection(x, w)
            torch.testing.assert_close(actual, (x.double() @ w.double()).float(), rtol=2e-6, atol=3e-6)

        for n in (137, 1024):
            x = torch.randn(5, 7, 16, generator=gen, device="cuda").bfloat16()
            w = torch.randn(n, 16, generator=gen, device="cuda").bfloat16()
            y = relative_projection(x, w)
            expected = x.double() @ w.double().T
            self.assertEqual(y.dtype, torch.bfloat16)
            torch.testing.assert_close(y.float(), expected.bfloat16().float(), rtol=.008, atol=1e-6)

        x = torch.randn(65, 256, generator=gen, device="cuda").bfloat16()
        w = torch.randn(128, 256, generator=gen, device="cuda").mul_(.03).bfloat16()
        bias = torch.randn(128, generator=gen, device="cuda").mul_(.03)
        packed = gemm_bf16_shuffle_b(w, 128, 256)
        for weight, layout in ((w.float(), packed), (w.float() + .00017, None)):
            selected, weights = route(x, weight, bias, 8, 1.25, True, packed=layout)
            scores = (x.double() @ weight.double().T).float().sigmoid()
            ids = (scores + bias).topk(8).indices
            torch.testing.assert_close(selected, ids.int())
            prob = scores.gather(1, ids)
            torch.testing.assert_close(weights, prob / prob.sum(1, keepdim=True) * 1.25)

    def test_packed_decode_router_preserves_ranking_and_weights(self):
        from snowllm.models.projection import pack_bf16
        from snowllm.ops.inkling import packed_logits, route, lib, check, _p, _stream

        gen = torch.Generator(device="cuda").manual_seed(781)
        matrix = torch.randn(128, 2304, generator=gen, device="cuda").mul_(.02).bfloat16()
        packed = pack_bf16(matrix)[0]
        bias = torch.randn(128, generator=gen, device="cuda")
        for rows in (1, 2, 11):
            x = torch.randn(rows, 2304, generator=gen, device="cuda").bfloat16()
            for b in (bias, None):
                if b is None:
                    x.zero_()
                logits = packed_logits(x, packed, 128)
                ids = torch.empty(rows, 8, dtype=torch.int32, device="cuda")
                weights = torch.empty_like(ids, dtype=torch.float32)
                check(lib.snowllm_inkling_select(_p(logits), _p(b), _p(ids), _p(weights), rows,
                      128, 8, 1, 0, 1.25, _stream()), "inkling_select")
                selected, probs = route(x, matrix.float(), b, 8, 1.25, packed=packed)
                torch.testing.assert_close(selected, ids, rtol=0, atol=0)
                torch.testing.assert_close(probs, weights, rtol=0, atol=0)

    def test_decode_routing_and_residuals(self):
        from snowllm.ops.inkling import combine, residual_mix, route, softcapped_swiglu

        gen = torch.Generator(device="cuda").manual_seed(311)
        def random(*shape):
            return torch.randn(shape, device="cuda", generator=gen)

        x = random(2, 256).bfloat16()
        router = random(128, 256) * .05
        for bias in (None, random(128) * .125):
            selected, weights = route(x, router, bias, 8, 1.25)
            scores = (x.float() @ router.T).sigmoid()
            ids = (scores if bias is None else scores + bias).topk(8).indices
            expected = scores.gather(1, ids)
            torch.testing.assert_close(selected, ids.int())
            torch.testing.assert_close(weights, expected / expected.sum(1, keepdim=True) * 1.25)
        gate_up = random(2, 8, 128).bfloat16()
        gate, up = gate_up.float().chunk(2, dim=-1)
        values = softcapped_swiglu(gate_up, 4., 25.)
        expected = ((gate/4.).tanh()*4.*gate.sigmoid() * ((up/25.).tanh()*25.)).bfloat16()
        torch.testing.assert_close(values, expected)
        torch.testing.assert_close(combine(values, weights), (values.float()*weights[..., None]).sum(1).bfloat16())
        for count, dtype in ((4, torch.bfloat16), (33, torch.float32)):
            states = random(2, count, 128).to(dtype)
            q = random(3, 128)
            decoded = residual_mix(states, q, 1e-5, False)
            ref = residual_mix(states, q, 1e-5, True)
            torch.testing.assert_close(decoded, ref, rtol=1e-5, atol=2e-5)

    def test_grouped_decode_controller_with_strided_messages(self):
        from snowllm.ops.inkling import (controller_tanh_table, pack_controller, packed_loop_control,
                                        qkv_loop_control, residual_mix_blocks)

        gen = torch.Generator(device="cuda").manual_seed(622)
        def random(*shape):
            return torch.randn(shape, device="cuda", generator=gen)
        width, eps = 2304, 1e-5
        queries = random(3, 4, width)
        table = controller_tanh_table("cuda")
        downs, gates = zip(*(pack_controller((random(64, 2*width) * .01).bfloat16(),
                                             (random(width, 64) * .125).bfloat16(),
                                             (random(width, 64) * .125).bfloat16())
                             for _ in range(3)))
        for rows in (1, 2):
            history, current = (random(rows, width).bfloat16() for _ in range(2))
            messages = residual_mix_blocks([history], current, queries.flatten(0, 1), eps, False)
            inputs = random(rows, 3, width).unbind(1)
            for query, messages_for_layer in zip(queries, messages.unflatten(1, (3, 4)).unbind(1)):
                separate = residual_mix_blocks([history], current, query, eps, False)
                self.assertTrue(torch.equal(messages_for_layer, separate))
                msg = messages_for_layer.unbind(1)[:3]
                grouped = qkv_loop_control(inputs, msg, downs, gates, table, eps, False)
                for result, x, m, down, gate in zip(grouped, inputs, msg, downs, gates):
                    self.assertTrue(torch.equal(result, packed_loop_control(x, m, down, gate, table, eps, False)))

    def test_prefill_residual_blocks_and_controller(self):
        from snowllm._capi import check, lib
        from snowllm.ops._common import _p, _stream
        from snowllm.ops.inkling import controller_tanh_table, pack_controller, packed_loop_control, residual_mix_blocks

        gen = torch.Generator(device="cuda").manual_seed(29)
        def random(*shape):
            return torch.randn(shape, generator=gen, device="cuda")

        rows, width = 17, 2304
        for count, heads, dtype in ((2, 4, torch.bfloat16), (3, 1, torch.bfloat16),
                                    (4, 3, torch.bfloat16), (5, 1, torch.float32)):
            with self.subTest(count=count, heads=heads):
                blocks = [random(rows, width).bfloat16() for _ in range(count - 1)]
                current = random(rows, width).to(dtype)
                query = random(heads, width)
                values = F.rms_norm(torch.stack([*blocks, current], 1).float(), (width,), eps=1e-5)
                scores = F.linear(values, query).transpose(1, 2) * width**-.5
                expected = torch.bmm(scores.softmax(-1), values)
                actual = residual_mix_blocks(blocks, current, query, 1e-5, True)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)

        x, msg = random(rows, 3, width)[:, 1], random(rows, 4, width)[:, 2]
        down = (random(64, width * 2) / (width * 2)**.5).bfloat16()
        forget, gain = [(random(width, 64) / 8).bfloat16() for _ in range(2)]
        source = torch.cat([F.rms_norm(v, (width,), eps=1e-5) for v in (x, msg)], -1)
        z = F.linear(source.bfloat16(), down)
        expected = x * (1 - F.linear(z, forget).float().tanh()) + msg * F.linear(z, gain).float().tanh()
        down_w, gates = pack_controller(down, forget, gain)
        table = controller_tanh_table(x.device)
        actual = packed_loop_control(x, msg, down_w, gates, table, 1e-5, True)
        self.assertLess(((actual - expected).norm() / expected.norm()).item(), 1e-3)

        packed = torch.empty_like(source, dtype=torch.bfloat16)
        hidden = torch.empty_like(z)
        output = torch.empty_like(actual)
        check(lib.snowllm_inkling_controller_gemm(_p(packed), _p(down_w), _p(gates), _p(table),
              _p(x), _p(msg), _p(hidden), _p(output), rows, width, x.stride(0), msg.stride(0),
              1e-5, True, _stream()), "inkling_controller_gemm")
        self.assertTrue(torch.equal(packed, source.bfloat16()))
        self.assertTrue(torch.equal(output, actual))

        def rounded_bounds(a, w):
            a, w = a.double(), w.double()
            exact = a @ w.T
            # FP32 dot products can straddle a BF16 midpoint when their reduction order differs.
            error = (a.abs() @ w.abs().T) * (a.size(1) * torch.finfo(torch.float32).eps)
            return (exact - error).bfloat16().float(), (exact + error).bfloat16().float()

        z_lo, z_hi = rounded_bounds(packed, down)
        self.assertTrue(torch.all((hidden >= z_lo) & (hidden <= z_hi)).item())
        f_lo, f_hi = rounded_bounds(hidden, forget)
        g_lo, g_hi = rounded_bounds(hidden, gain)
        a, b = x * (1 - f_lo.tanh()), x * (1 - f_hi.tanh())
        c, d = msg * g_lo.tanh(), msg * g_hi.tanh()
        lower, upper = a.minimum(b) + c.minimum(d), a.maximum(b) + c.maximum(d)
        torch.testing.assert_close(actual, actual.clamp(lower, upper), rtol=1e-5, atol=2e-5)

    def test_prefill_postprocessing_rounding(self):
        from snowllm.ops.inkling import combine, softcapped_swiglu

        gen = torch.Generator(device="cuda").manual_seed(17)
        gu = torch.randn(96, 8, 2304, device="cuda", dtype=torch.bfloat16, generator=gen)
        weights = torch.rand(96, 8, device="cuda", generator=gen)
        g, u = gu.float().chunk(2, dim=-1)
        expected = (g.div(4).tanh().mul(4) * g.sigmoid()) * u.div(25).tanh().mul(25)
        self.assertTrue(torch.equal(softcapped_swiglu(gu, 4., 25., True), expected.bfloat16()))
        self.assertTrue(torch.equal(combine(gu, weights, True),
                                    (gu.float() * weights[..., None]).sum(1).bfloat16()))

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF"), "set SNOWLLM_INKLING_GGUF to the model file")
    def test_grouped_prefill(self):
        from snowllm.ops.inkling import expert_plan

        generator = torch.Generator(device="cuda").manual_seed(31)
        for layer in (0, 20):
            with GGUFReader(pathlib.Path(os.environ["SNOWLLM_INKLING_GGUF"])) as reader:
                moe = InklingMoE(reader, config(reader.gguf), layer)
            for rows in (769, 2049):
                with self.subTest(rows=rows, layer=layer):
                    ids = torch.randint(128, (rows, 8), device="cuda", dtype=torch.int32, generator=generator)
                    ids[:17] = 0
                    ids[-1, :2] = -1
                    ids[-1, 2] = 128
                    packed = moe.gate_up.scale is not None
                    plan = expert_plan(ids, 128, 2304, 1152, True, packed=packed)
                    self.assertIsNotNone(plan)
                    self.assertIsNone(expert_plan(ids, 128, 2304, 1152, False, packed=packed))
                    for proj, groups, k in ((moe.gate_up, 1, 2304), (moe.down, 8, 1152)):
                        x = torch.randn(rows, groups, k, device="cuda", dtype=torch.bfloat16, generator=generator)
                        expected = proj(x, ids, True)
                        actual = proj(x, ids, True, plan)
                        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF"), "set SNOWLLM_INKLING_GGUF to the model file")
    def test_real_experts(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(31)
        with GGUFReader(pathlib.Path(os.environ["SNOWLLM_INKLING_GGUF"])) as reader:
            cfg = config(reader.gguf)
            for layer in (0, 20):
                with self.subTest(layer=layer):
                    moe = InklingMoE(reader, cfg, layer)
                    x = torch.randn(2, cfg["hidden_size"], generator=generator).bfloat16()
                    got = moe(x.cuda()).cpu().float()
                    scores = F.linear(x.float(), moe.router.cpu()).sigmoid()
                    ranked = scores if moe.bias is None else scores + moe.bias.cpu()
                    ids = ranked.topk(moe.topk).indices
                    probs = scores.gather(1, ids)
                    probs = probs / probs.sum(dim=1, keepdim=True) * cfg["route_scale"]
                    gate_name, down_name = (f"blk.{layer}.ffn_{p}_exps.weight" for p in ("gate_up", "down"))
                    want = torch.zeros_like(got)
                    for token, experts in enumerate(ids.tolist()):
                        for slot, expert in enumerate(experts):
                            fused = dense_expert(moe.gate_up, expert, reader.gguf, gate_name) @ x[token].float()
                            gate, up = fused.bfloat16().float().chunk(2)
                            act = 4 * (gate / 4).tanh() * gate.sigmoid() * 25 * (up / 25).tanh()
                            down = dense_expert(moe.down, expert, reader.gguf, down_name) @ act.bfloat16().float()
                            want[token] += probs[token, slot] * down.bfloat16().float()
                    error = ((got - want.bfloat16().float()).norm() / want.norm()).item()
                    print(f"Inkling layer {layer} MoE relative L2 {error:.6g}", flush=True)
                    self.assertLess(error, .005)
                    # Cross a WMMA routing segment with repeated and invalid expert IDs.
                    rows = 131
                    ids = torch.randint(cfg["num_experts"], (rows, moe.topk), generator=generator,
                                        dtype=torch.int32)
                    ids[:, :2] = 0
                    ids[-1, -1] = -1
                    ids = ids.cuda()
                    for proj, groups, size in ((moe.gate_up, 1, cfg["hidden_size"]),
                                               (moe.down, moe.topk, cfg["moe_intermediate_size"])):
                        inputs = torch.randn(rows, groups, size, generator=generator).bfloat16().cuda()
                        result = proj(inputs, ids)
                        for row in (0, 127, 128, 130):
                            ref = proj(inputs[row:row+1], ids[row:row+1])[0].float()
                            rel = (result[row].float() - ref).norm() / ref.norm()
                            self.assertLess(rel.item(), .002)
                        self.assertEqual(result[-1, -1].count_nonzero().item(), 0)
                    del moe


if __name__ == "__main__":
    unittest.main()

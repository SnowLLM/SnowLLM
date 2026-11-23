# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import unittest

import torch
import torch.nn.functional as F

from snowllm.checkpoint.gguf.inkling2 import config
from snowllm.checkpoint.gguf.source import GGUFReader
from snowllm.models.inkling2.experts import InklingMoE


def dense_expert(projection, expert):
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

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF"), "set SNOWLLM_INKLING_GGUF to the model file")
    def test_grouped_bf16_prefill(self):
        from snowllm.ops.inkling import bf16_expert_plan

        with GGUFReader(pathlib.Path(os.environ["SNOWLLM_INKLING_GGUF"])) as reader:
            moe = InklingMoE(reader, config(reader.gguf), 20)
        generator = torch.Generator(device="cuda").manual_seed(31)
        for rows in (769, 2049):
            with self.subTest(rows=rows):
                ids = torch.randint(128, (rows, 8), device="cuda", dtype=torch.int32, generator=generator)
                ids[:17] = 0
                ids[-1, :2] = -1
                ids[-1, 2] = 128
                plan = bf16_expert_plan(ids, 128, 2304, 1152, True)
                self.assertIsNotNone(plan)
                self.assertIsNone(bf16_expert_plan(ids, 128, 2304, 1152, False))
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
                    want = torch.zeros_like(got)
                    for token, experts in enumerate(ids.tolist()):
                        for slot, expert in enumerate(experts):
                            fused = dense_expert(moe.gate_up, expert) @ x[token].float()
                            gate, up = fused.bfloat16().float().chunk(2)
                            act = 4 * (gate / 4).tanh() * gate.sigmoid() * 25 * (up / 25).tanh()
                            down = dense_expert(moe.down, expert) @ act.bfloat16().float()
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

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
    n = weight.shape[0]
    blocks = weight.reshape(-1, 36)
    scales = blocks[:, :4].contiguous().view(torch.float8_e4m3fn).float()
    packed = blocks[:, 4:].reshape(-1, 4, 8)
    codes = torch.cat((packed & 15, packed >> 4), dim=-1)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    decoded = levels[(codes & 7).long()] * torch.where(codes < 8, 1., -1.)
    return (decoded * scales.unsqueeze(-1)).reshape(n, -1) * projection.scale[expert].item()


class InklingExpertsTest(unittest.TestCase):
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
                    del moe


if __name__ == "__main__":
    unittest.main()

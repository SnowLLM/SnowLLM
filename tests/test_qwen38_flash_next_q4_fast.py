# SPDX-License-Identifier: Apache-2.0
import os
import struct
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from snowllm import ops
from snowllm.checkpoint.gguf.dequant import CHUNK_BLOCKS, dequantize
from snowllm.models.qwen4exp.qwen4exp_weights import _shared


class Qwen38FlashNextQ4FastTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'), 'set SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF')
    def test_checkpoint_moe_long_prefill_matches_generic(self):
        from snowllm.checkpoint.gguf.names import config
        from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource
        from snowllm.models.geometry import Qwen4ExpGeometry
        from snowllm.models.qwen4exp.qwen4exp_weights import load_moe, validate_config
        with GGUFReader(Path(os.environ['SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'])) as rd:
            cfg = config(rd.gguf)['text_config']
            validate_config(cfg)
            geo, weights = Qwen4ExpGeometry.from_config(cfg), GGUFWeightSource(rd, cfg)
            representatives = {}
            for layer in range(cfg['num_hidden_layers']):
                p = f'model.language_model.layers.{layer}.mlp.'
                formats = tuple(weights.quant_names(p + key)[0] for key in (
                    'experts.gate_up_proj', 'experts.down_proj',
                    'shared_expert.gate_proj.weight', 'shared_expert.down_proj.weight'))
                representatives.setdefault(formats, layer)
            try:
                for formats, layer in representatives.items():
                    moe = load_moe(weights, f'model.language_model.layers.{layer}.', geo)
                    for rows in (8192, 8193):
                        with self.subTest(layer=layer, formats=formats, rows=rows):
                            torch.manual_seed(1743 + layer + rows)
                            x = torch.randn(rows, geo.hidden, device='cuda').bfloat16() * .1
                            out = torch.empty_like(x)
                            ws = torch.empty(ops.inference_moe_workspace_bytes(rows),
                                             device='cuda', dtype=torch.uint8)
                            def run():
                                ops.inference_moe(x, moe.router_w, moe.gate_up_w, moe.down_w,
                                    moe.shared_gate_up_w, moe.shared_down_w, out, ws)
                            ops.moe_variant_force(2)
                            run()
                            want = out.clone()
                            ops.moe_variant_force(-1)
                            out.fill_(float('nan'))
                            run()
                            self.assertTrue(torch.isfinite(out).all().item())
                            torch.testing.assert_close(out, want, rtol=0, atol=0)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                run()
                            out.fill_(float('nan'))
                            graph.replay()
                            torch.testing.assert_close(out, want, rtol=0, atol=0)
                            del graph, x, out, want, ws
                    del moe
            finally:
                ops.moe_variant_force(-1)

    @unittest.skipUnless(os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'), 'set SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF')
    def test_checkpoint_moe_graph(self):
        from snowllm.checkpoint.gguf.names import config
        from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource
        from snowllm.models.geometry import Qwen4ExpGeometry
        from snowllm.models.qwen4exp.qwen4exp_weights import load_moe, validate_config
        with GGUFReader(Path(os.environ['SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'])) as rd:
            cfg = config(rd.gguf)['text_config']
            validate_config(cfg)
            geo, w = Qwen4ExpGeometry.from_config(cfg), GGUFWeightSource(rd, cfg)
            for layer in (3, 7, 23, 43):
                moe = load_moe(w, f'model.language_model.layers.{layer}.', geo)
                self.assertIsInstance(moe.shared_gate_up_w, ops.KQuantExpertWeight)
                self.assertIsInstance(moe.shared_down_w, ops.KQuantExpertWeight)
                for rows in (1, 17):
                    with self.subTest(layer=layer, rows=rows):
                        arena, key = ops.Arena(), ('moe', layer, rows)
                        x = torch.randn(rows, geo.hidden, device='cuda').bfloat16() * .1
                        out = torch.empty_like(x)
                        def run():
                            with arena.frame():
                                ws = arena.flat(ops.inference_moe_workspace_bytes(rows), torch.uint8)
                                ops.inference_moe(x, moe.router_w, moe.gate_up_w, moe.down_w,
                                    moe.shared_gate_up_w, moe.shared_down_w, out, ws)
                        arena.execute(key, rows, run)
                        self.assertTrue(torch.isfinite(out).all().item())
                        self.assertGreater(out.float().norm().item(), 0.)
                        want = out.clone()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            arena.execute(key, rows, run)
                        out.zero_()
                        graph.replay()
                        torch.testing.assert_close(out, want, rtol=0, atol=0)
                        del graph
                del moe

    @unittest.skipUnless(os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'), 'set SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF')
    def test_checkpoint_attention_projection_and_graph(self):
        from snowllm.checkpoint.gguf.names import config
        from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource
        from snowllm.engine.forward_context import Qwen4ExpContext
        from snowllm.models.geometry import Qwen4ExpGeometry
        from snowllm.models.qwen3_5.qwen3_5_weights import Staging, load_full_attn
        from snowllm.models.qwen4exp.layers import Qwen4ExpFullAttention
        from snowllm.models.qwen4exp.qwen4exp_weights import _hyper, _indexer, validate_config
        with GGUFReader(Path(os.environ['SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'])) as rd:
            cfg = config(rd.gguf)['text_config']
            validate_config(cfg)
            geo = Qwen4ExpGeometry.from_config(cfg)
            w, staging = GGUFWeightSource(rd, cfg), Staging(geo, kquant=True)
            for layer in (3, 7, 23, 47):
                prefix = f'model.language_model.layers.{layer}.'
                base = load_full_attn(w, prefix, staging, geo)
                attn = Qwen4ExpFullAttention(base.qkv_proj, base.qk_norm_rope, base.o_proj,
                    geo, _indexer(w, prefix, geo, base.qkv_proj))
                attn.hc = _hyper(w, prefix + 'attn_hyper_connection.', geo, True)
                for rows in (1, 17):
                    with self.subTest(layer=layer, rows=rows):
                        arena = ops.Arena()
                        ctx = Qwen4ExpContext(batch=SimpleNamespace(is_prefill=rows > 1),
                            M=rows, eps=geo.eps, path=ops.Path.PREFILL if rows > 1 else ops.Path.DECODE,
                            arena=arena, plan_key=('qwen38-flash-next-q4-fast', layer, rows), geo=geo)
                        x = torch.randn(rows, geo.hc_count, geo.hidden, device='cuda').bfloat16()
                        out = torch.empty(rows, geo.qkv_proj_n, device='cuda', dtype=torch.bfloat16)
                        index = torch.empty(rows, attn.qkv_proj.index_n,
                                            device='cuda', dtype=torch.bfloat16)
                        attn._lo = torch.empty(rows, attn.hc.down.n,
                                               device='cuda', dtype=torch.bfloat16)
                        mixed = torch.empty(rows, geo.hidden, device='cuda', dtype=torch.bfloat16)
                        want_lo = torch.empty_like(attn._lo)
                        ops.inference_hc(x, attn.hc.gamma, attn.hc.down, attn.hc.up, want_lo,
                            mixed, geo.hc_lowrank, geo.eps,
                            torch.empty(ops.qwen4exp_hc_mix_ws_bytes(rows, geo.hc_count,
                                geo.hidden, geo.hc_lowrank), dtype=torch.uint8, device='cuda'),
                            inject=attn.hc.inject_weight)
                        want, want_index = torch.empty_like(out), torch.empty_like(index)
                        attn.qkv_proj(ctx, mixed, want, (), want_index)
                        run = lambda: attn._project(ctx, x, out, (), index)
                        arena.execute(ctx.plan_key, rows, run)
                        torch.testing.assert_close(out, want, rtol=.02, atol=.02)
                        torch.testing.assert_close(index, want_index, rtol=.02, atol=.02)
                        torch.testing.assert_close(attn._lo, want_lo, rtol=.02, atol=.02)
                        eager, eager_index, eager_lo = out.clone(), index.clone(), attn._lo.clone()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            arena.execute(ctx.plan_key, rows, run)
                        out.zero_()
                        index.zero_()
                        graph.replay()
                        self.assertTrue(torch.equal(out, eager))
                        self.assertTrue(torch.equal(index, eager_index))
                        self.assertTrue(torch.equal(attn._lo, eager_lo))
                        del graph

    @unittest.skipUnless(os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'), 'set SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF')
    def test_complete_checkpoint_planning(self):
        from snowllm.checkpoint.gguf.names import config
        from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource
        from snowllm.models.qwen4exp.qwen4exp_weights import load_weights, validate_config
        with GGUFReader(Path(os.environ['SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF'])) as rd:
            cfg = config(rd.gguf)['text_config']
            validate_config(cfg)
            with ops.dry_load() as planned:
                model = load_weights(GGUFWeightSource(rd, cfg), cfg,
                                     range(cfg['num_hidden_layers']), False, False)
                self.assertEqual(len(model.model.layers), cfg['num_hidden_layers'])
                self.assertGreater(planned.peak_device, planned.device)
                print(f'Qwen3.8-Flash-Next-Q4-Fast.GGUF weights: {planned.device} bytes; load peak: {planned.peak_device}')

    def test_q5_0_signed_scale_high_bits_and_chunk_boundary(self):
        blocks, expected = [], []
        for i in range(32):
            scale = (0., .125, -.25, 2.)[i % 4]
            high = (0, 0xffffffff, 0x80000001, 0x5aa5a55a)[i % 4]
            low = bytes((i * 7 + j * 13) % 256 for j in range(16))
            blocks.append(struct.pack('<eI', scale, high) + low)
            expected.extend(scale * (((low[j % 16] >> (4 * (j // 16))) & 15)
                                      + 16 * ((high >> j) & 1) - 16) for j in range(32))
        raw = torch.tensor(list(b''.join(blocks)), dtype=torch.uint8)
        want = torch.tensor(expected, dtype=torch.float32)
        for device in ('cpu', 'cuda'):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    got = dequantize(raw.to(device), 'Q5_0', want.numel(), dtype)
                    self.assertTrue(torch.equal(got.cpu(), want.to(dtype)))
            count = CHUNK_BLOCKS // 32 + 1
            got = dequantize(raw.repeat(count).to(device), 'Q5_0', want.numel() * count)
            self.assertTrue(torch.equal(got.cpu(), want.bfloat16().repeat(count)))

    def test_shared_slabs_use_matching_metadata_or_bf16(self):
        for gate_fmt, down_fmt in (('Q5_0', 'IQ4_NL'), ('IQ4_XS', 'Q5_0'),
                                   ('BF16', 'BF16'), ('IQ4_XS', 'IQ4_NL')):
            with self.subTest(gate=gate_fmt, down=down_fmt):
                formats = {'gate_proj': gate_fmt, 'up_proj': gate_fmt, 'down_proj': down_fmt}
                w = SimpleNamespace(
                    quant_names=lambda key: [formats[key.split('.')[-2]]],
                    blocks=lambda key: [key],
                    read=lambda key: torch.zeros(32, 32, dtype=torch.bfloat16))
                with patch.object(ops, 'moe_kquant_shuffle_gate_up', return_value='quant_gu') as qgu, \
                     patch.object(ops, 'moe_kquant_shuffle_down', return_value='quant_down') as qdn, \
                     patch.object(ops, 'moe_shuffle_gate_up', return_value='bf16_gu') as bgu, \
                     patch.object(ops, 'moe_shuffle_down', return_value='bf16_down') as bdn:
                    gu, dn = _shared(w, 'mlp.')
                    compressed_gu = compressed_dn = (
                        gate_fmt in ops.KQUANT_FORMATS and down_fmt in ops.KQUANT_FORMATS)
                    self.assertEqual(gu, 'quant_gu' if compressed_gu else 'bf16_gu')
                    self.assertEqual(dn, 'quant_down' if compressed_dn else 'bf16_down')
                    self.assertEqual((qgu.call_count, bgu.call_count),
                                     (int(compressed_gu), int(not compressed_gu)))
                    self.assertEqual((qdn.call_count, bdn.call_count),
                                     (int(compressed_dn), int(not compressed_dn)))


if __name__ == '__main__':
    unittest.main()

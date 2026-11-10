# SPDX-License-Identifier: Apache-2.0
import gc
import os
from pathlib import Path
import unittest

import torch

from snowllm import ops, _capi


class Qwen38FlashNextUD31Tests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_UD31_GGUF'),
                         'set SNOWLLM_QWEN38_FLASH_NEXT_UD31_GGUF')
    def test_checkpoint_moe_formats_match_generic_and_graph(self):
        from snowllm.checkpoint.gguf.names import config
        from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource
        from snowllm.models.geometry import Qwen4ExpGeometry
        from snowllm.models.qwen4exp.qwen4exp_weights import load_moe

        _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
        with GGUFReader(Path(os.environ['SNOWLLM_QWEN38_FLASH_NEXT_UD31_GGUF'])) as rd:
            cfg = config(rd.gguf)['text_config']
            geo, weights = Qwen4ExpGeometry.from_config(cfg), GGUFWeightSource(rd, cfg)
            representatives = {}
            for layer in range(cfg['num_hidden_layers']):
                prefix = f'model.language_model.layers.{layer}.mlp.'
                formats = tuple(weights.quant_names(prefix + key)[0] for key in (
                    'experts.gate_up_proj', 'experts.down_proj',
                    'shared_expert.gate_proj.weight', 'shared_expert.down_proj.weight'))
                representatives.setdefault(formats, layer)
            try:
                for formats, layer in representatives.items():
                    moe = load_moe(weights, f'model.language_model.layers.{layer}.', geo)
                    for rows in (1, 3, 4, 8, 17, 1024, 8192, 8193):
                        with self.subTest(layer=layer, formats=formats, rows=rows):
                            torch.manual_seed(1743 + rows)
                            x = torch.randn(rows, geo.hidden, device='cuda').bfloat16() * .1
                            out = torch.empty_like(x)
                            ops.moe_variant_force(2)
                            reference_bytes = ops.inference_moe_workspace_bytes(rows)
                            ops.moe_variant_force(-1)
                            scratch = ops.empty_bytes(max(reference_bytes,
                                                          ops.inference_moe_workspace_bytes(rows)))

                            def run():
                                ops.inference_moe(x, moe.router_w, moe.gate_up_w, moe.down_w,
                                    moe.shared_gate_up_w, moe.shared_down_w, out, scratch)

                            ops.moe_variant_force(2)
                            run()
                            want = out.clone()
                            ops.moe_variant_force(-1)
                            out.fill_(float('nan'))
                            run()
                            self.assertTrue(torch.isfinite(out).all().item())
                            error = ((out.float() - want.float()).norm() / want.float().norm()).item()
                            self.assertLess(error, .006)
                            eager = out.clone()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                run()
                            out.fill_(float('nan'))
                            graph.replay()
                            torch.testing.assert_close(out, eager, rtol=0, atol=0)
                            del graph, x, out, want, eager, scratch
                    del moe
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                ops.moe_variant_force(-1)


if __name__ == '__main__':
    unittest.main()

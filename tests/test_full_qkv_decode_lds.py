import unittest

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf.dequant import dequantize


def quantized_weight(fmt, rows):
    name, block, size = {21: ('IQ3_S', 256, 110), 23: ('IQ4_XS', 256, 136),
                          50: ('Q5_0', 32, 22), 8: ('Q8_0', 32, 34)}[fmt]
    raw = torch.randint(0, 256, (rows * 2560 // block, size), device='cuda', dtype=torch.uint8)
    scale = torch.empty(raw.shape[0], 1, device='cuda', dtype=torch.float16).uniform_(-.003, .003)
    raw[:, :2] = scale.view(torch.uint8)
    dense = dequantize(raw.flatten(), name, rows * 2560, torch.bfloat16).view(rows, 2560)
    return raw.flatten(), dense


class FullQkvDecodeLdsTests(unittest.TestCase):
    def test_quantized_projection_index_and_graph(self):
        _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
        torch.manual_seed(6137)
        for fmt, n0, tail_fmt in ((23, 13312, 0), (21, 12288, 0),
                                  (23, 12288, 0), (23, 12800, 50), (8, 13312, 0)):
            raw, dense = quantized_weight(fmt, n0)
            tail, dense_tail = None, None
            if n0 == 13312:
                main = ops.qkv_proj_shuffle_w_kquant(raw, fmt)
            elif n0 == 12800:
                main = ops.qkv_proj_qk_shuffle_w_kquant(raw, fmt)
                tail_raw, dense_tail = quantized_weight(tail_fmt, 512)
                tail = ops.qkv_proj_v_shuffle_w_kquant(tail_raw, tail_fmt)
            else:
                main = ops.qkv_proj_q_shuffle_w_kquant(raw, fmt)
                dense_tail = torch.randn(1024, 2560, device='cuda', dtype=torch.bfloat16) * .01
                tail = ops.qkv_proj_kv_shuffle_w(dense_tail)
            index_dense = torch.randn(640, 2560, device='cuda', dtype=torch.bfloat16) * .01
            index = ops.qkv_proj_index_shuffle_w(index_dense)
            for rows, path in ((1, ops.Path.DECODE), (7, ops.Path.DECODE),
                               (16, ops.Path.DECODE), (17, ops.Path.DECODE),
                               (17, ops.Path.PREFILL)):
                x = torch.randn(rows, 2560, device='cuda', dtype=torch.bfloat16) * .1
                want = (x.float() @ dense.float().T).flatten()
                if dense_tail is not None:
                    want = torch.cat((want, (x.float() @ dense_tail.float().T).flatten()))
                want_index = x.float() @ index_dense.float().T
                for with_index in (False, True):
                    with self.subTest(fmt=fmt, n0=n0, rows=rows, path=path, index=with_index):
                        out = torch.empty(rows, 13312, device='cuda', dtype=torch.bfloat16)
                        ix = torch.empty(rows, 640, device='cuda', dtype=torch.bfloat16)
                        need = ops.qkv_proj_scratch_bytes(rows)
                        guard = torch.full((need + 256,), 165, device='cuda', dtype=torch.uint8)
                        ws = guard[:need]

                        def run():
                            ops.inference_qkv(x, main, None, tail if tail_fmt else None,
                                tail if n0 == 12288 else None, ws, out, path,
                                index_w=index if with_index else None,
                                index_out=ix if with_index else None)

                        run()
                        relative = ((out.flatten().float() - want).norm() / want.norm()).item()
                        self.assertLess(relative, .004)
                        if with_index:
                            relative = ((ix.float() - want_index).norm() / want_index.norm()).item()
                            self.assertLess(relative, .004)
                        self.assertTrue(bool(torch.all(guard[need:] == 165)))
                        if rows in (1, 16):
                            saved, saved_ix = out.clone(), ix.clone()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                run()
                            for _ in range(3):
                                out.fill_(float('nan')); ix.fill_(float('nan'))
                                graph.replay()
                                self.assertTrue(torch.equal(saved, out))
                                if with_index:
                                    self.assertTrue(torch.equal(saved_ix, ix))
                            del graph


if __name__ == '__main__':
    unittest.main()

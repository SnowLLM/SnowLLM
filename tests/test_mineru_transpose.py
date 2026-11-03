# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


def main():
    import torch
    from snowllm.ops import image as op

    torch.manual_seed(128)
    with torch.inference_mode():
        for batch, rows, cols, offset, gap in (
            (1, 10000, 256, 0, 3125),
            (8, 2500, 256, 10000, 625),
            (2, 625, 256, 12500, 0),
            (3, 37, 65, 7, 11),
            (2, 1, 33, 2, 3),
            (2, 33, 1, 1, 4),
        ):
            storage = torch.randn(batch, offset + rows + gap, cols, device="cuda")
            source = storage[:, offset:offset + rows].transpose(1, 2)
            original = storage.clone()
            expected = source.contiguous()
            torch.testing.assert_close(op.dense(source), expected, atol=0, rtol=0)
            torch.testing.assert_close(storage, original, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            for _ in range(3):
                op.dense(source)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                actual = op.dense(source)
            source.add_(1)
            graph.replay()
            torch.testing.assert_close(actual, source.contiguous(), atol=0, rtol=0)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            source = torch.randn(2, 43, 67, dtype=dtype, device="cuda")
            for value in (source.transpose(1, 2), source[:, ::2, ::2], source[:0]):
                torch.testing.assert_close(op.dense(value), value, atol=0, rtol=0)
        source = torch.randn(2, 43, 67).transpose(1, 2)
        torch.testing.assert_close(op.dense(source), source, atol=0, rtol=0)
        source = torch.empty(2, 43, 67, device="meta").transpose(1, 2)
        with patch.object(op.lib, "snowllm_auxiliary_transpose_f32", side_effect=AssertionError):
            actual = op.dense(source)
        assert actual.shape == source.shape and actual.is_contiguous()
    print("PASS: FP32 sliced transpose, tails, graph replay and fallback layouts", flush=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


def main():
    import torch
    from snowllm.ops import image as op

    torch.manual_seed(133)
    torch.set_num_threads(2)
    weight = torch.randn(8, 64, 1, 1, device="cuda") / 8
    with torch.inference_mode():
        for batch, height, width in ((1, 1, 1), (2, 1, 31), (3, 3, 11), (2, 7, 13), (1, 302, 302), (8, 302, 302)):
            storage = torch.randn(batch + 1, height, width, 64, device="cuda")
            x = storage[1:].permute(0, 3, 1, 2)
            for bias in (None, torch.randn(8, device="cuda")):
                expected = op.pointwise_f32(x.contiguous(), weight, bias, config=7)
                actual = op.convolution(x, weight, bias)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                assert actual.is_contiguous()
                for _ in range(3):
                    op.convolution(x, weight, bias)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = op.convolution(x, weight, bias)
                x.add_(0.25)
                graph.replay()
                expected = op.pointwise_f32(x.contiguous(), weight, bias, config=7)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
        x = torch.randn(2, 5, 7, 64, device="cuda").permute(0, 3, 1, 2)
        scale = torch.randn(8, device="cuda")
        unaligned = torch.randn(x.numel() + 1, device="cuda")[1:].view(2, 5, 7, 64).permute(0, 3, 1, 2)
        unaligned_weight = torch.randn(513, device="cuda")[1:].view(8, 64, 1, 1)
        with patch.object(op, "pointwise_64_8_f32", side_effect=AssertionError("unexpected specialized dispatch")):
            for value, filt, kwargs in (
                (x.contiguous(), weight, {}),
                (x, weight, {"scale": scale}),
                (x, weight, {"act": 4}),
                (unaligned, weight, {}),
                (x, unaligned_weight, {}),
            ):
                actual = op.pointwise_f32(value, filt, **kwargs)
                expected = op.pointwise_f32(value.contiguous(), filt, config=7, **kwargs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        x = torch.empty(2, 5, 7, 64, device="meta").permute(0, 3, 1, 2)
        w = torch.empty(8, 64, 1, 1, device="meta")
        with patch.object(op.lib, "snowllm_auxiliary_pointwise_64_8_f32", side_effect=AssertionError):
            out = op.convolution(x, w)
        assert out.shape == (2, 8, 5, 7) and out.is_contiguous()
    print("PASS: channels-last 64→8 projection, tails, bias, graph replay and fallback paths", flush=True)


if __name__ == "__main__":
    main()

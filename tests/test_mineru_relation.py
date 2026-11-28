# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


def main():
    import torch
    from snowllm.ops import image as op
    from snowllm.ops.arena import Arena

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
        with patch.object(op.lib, "snowllm_convolution_execute", side_effect=AssertionError):
            out = op.convolution(x, w)
        assert out.shape == (2, 8, 5, 7) and out.is_contiguous()
        for dtype, channels, columns, groups, k, stride, dilation, transpose in (
            (torch.float32, 4, 8, 1, 1, 1, 1, False),
            (torch.float32, 4, 4, 4, 5, 2, 1, False),
            (torch.float32, 4, 6, 2, 3, 1, 2, False),
            (torch.float16, 4, 8, 1, 3, 2, 1, False),
            (torch.float16, 4, 6, 2, 3, 2, 1, True),
        ):
            x = torch.randn(2, channels, 9, 11, device="cuda", dtype=dtype)
            shape = (channels, columns // groups) if transpose else (columns, channels // groups)
            w = torch.randn(*shape, k, k, device="cuda", dtype=dtype) / k
            bias = torch.randn(columns, device="cuda", dtype=torch.float32 if transpose else dtype)
            scale, shift = torch.randn(2, columns, device="cuda").unbind()
            geometry = dict(groups=groups, stride=stride, dilation=dilation, padding=k//2)
            fn = torch.nn.functional.conv_transpose2d if transpose else torch.nn.functional.conv2d
            if transpose:
                geometry["output_padding"] = 1
            actual = op.convolution(x, w, bias, scale=scale, shift=shift, act="silu",
                                    transpose=transpose, **geometry)
            # A CPU reference also checks MIOpen solution selection independently of the GPU backend.
            ref = fn(x.cpu().float(), w.cpu().float(), **geometry).to(dtype)
            ref = (ref + bias.cpu()[None, :, None, None]).to(dtype)
            ref = (ref.float() * scale.cpu()[None, :, None, None] + shift.cpu()[None, :, None, None]).to(dtype)
            expected = torch.nn.functional.silu(ref.float()).to(dtype)
            tol = 5e-3 if dtype == torch.float16 else 2e-5
            torch.testing.assert_close(actual.cpu(), expected, atol=tol, rtol=tol)
        kernel = torch.randn(4, 4, 3, 3, device="cuda")
        value = torch.randn(2, 4, 7, 9, device="cuda")

        def two_layers():
            return op.convolution(op.convolution(value, kernel, padding=1), kernel, padding=1)

        arena = Arena()
        key = ("test-convolution",)
        torch.testing.assert_close(arena.execute(key, 2, two_layers), two_layers())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = arena.execute(key, 2, two_layers)
        value.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(captured, two_layers())
    print("PASS: channels-last 64→8 projection, tails, bias, graph replay and fallback paths", flush=True)


if __name__ == "__main__":
    main()

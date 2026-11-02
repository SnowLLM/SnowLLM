# SPDX-License-Identifier: Apache-2.0
import itertools


def main():
    import torch
    import torch.nn.functional as F

    from snowllm import ops
    from snowllm._capi import GEO_MINERU_12B

    ops.select_geometry(GEO_MINERU_12B)
    torch.manual_seed(113)
    cases = itertools.product(
        (72, 80),
        (
            (1,),
            (8,),
            (9,),
            (16,),
            (17,),
            (62,),
            (63,),
            (64,),
            (65,),
            (80,),
            (126,),
            (129,),
            (257,),
            (1023,),
            (1024,),
            (1041,),
            (4096,),
            (4524,),
            (1, 0, 127, 257),
            (1025, 0, 1041),
            (1, 3073, 1),
        ),
        (0.0, 0.1, 1.0, 3.0),
    )
    with torch.inference_mode():
        for dim, lengths, scale in cases:
            rows, hidden = sum(lengths), 16 * dim
            qkv = (
                torch.randn(rows, 3 * hidden, dtype=torch.bfloat16, device="cuda")
                * scale
            )
            out = torch.empty(rows, hidden, dtype=qkv.dtype, device=qkv.device)
            offsets = [0, *itertools.accumulate(lengths)]
            cu = torch.tensor(offsets, dtype=torch.int32, device=qkv.device)
            expected = torch.empty_like(out)
            for start, stop in itertools.pairwise(offsets):
                if start == stop:
                    continue
                q, k, v = (
                    qkv[start:stop].view(-1, 3, 16, dim).permute(1, 2, 0, 3).float()
                )
                expected[start:stop] = (
                    F.scaled_dot_product_attention(q, k, v)
                    .transpose(0, 1)
                    .reshape(-1, hidden)
                )
            ops.vision_attention(qkv, None, None, cu, out, 16, dim)
            torch.testing.assert_close(out, expected, atol=0.006 * scale, rtol=0.03)
            relative_l2 = (
                out.float() - expected.float()
            ).norm() / expected.float().norm().clamp_min(1e-30)
            assert relative_l2 < 0.006, (dim, lengths, scale, relative_l2.item())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ops.vision_attention(qkv, None, None, cu, out, 16, dim)
            qkv.copy_(qkv.flip(0)).mul_(3)
            ops.vision_attention(qkv, None, None, cu, expected, 16, dim)
            graph.replay()
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            misaligned = torch.empty(
                rows * hidden + 1, device=qkv.device, dtype=qkv.dtype
            )[1:].view(rows, hidden)
            ops.vision_attention(qkv, None, None, cu, misaligned, 16, dim)
            torch.testing.assert_close(
                misaligned.view(torch.int16), out.view(torch.int16), atol=0, rtol=0
            )
            if len(lengths) == 1:
                ops.vision_attn(
                    qkv,
                    qkv[:, hidden:],
                    qkv[:, 2 * hidden :],
                    expected,
                    rows,
                    hidden,
                    16,
                    dim,
                )
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
            print(
                f"PASS: dim={dim} lengths={lengths} scale={scale}, FP32 reference and graph replay",
                flush=True,
            )


if __name__ == "__main__":
    main()

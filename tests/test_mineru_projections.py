# SPDX-License-Identifier: Apache-2.0
import math


def main():
    import torch
    import torch.nn.functional as F
    from snowllm import ops
    from snowllm._capi import GEO_MINERU_12B, lib
    from snowllm.ops.image import linear_half as gemm_half

    torch.manual_seed(91)
    ops.select_geometry(GEO_MINERU_12B)
    with torch.inference_mode():
        for columns, contraction in (
            (128, 896),
            (1024, 896),
            (896, 1024),
            (896, 4864),
            (4864, 896),
            (3840, 1280),
            (5120, 1280),
            (1280, 1280),
            (1280, 5120),
            (1280, 1184),
            (5120, 5120),
            (896, 5120),
            (256, 96),
        ):
            weight = torch.randn(
                columns, contraction, dtype=torch.bfloat16, device="cuda"
            ) / math.sqrt(contraction)
            packed = ops.gemm_bf16_shuffle_b(weight, columns, contraction)
            bias = torch.randn(columns, dtype=torch.bfloat16, device="cuda")
            for rows in (1, 2, 8, 16, 32, 33, 64, 128, 256, 512, 1024):
                value = torch.randn(
                    rows, contraction, dtype=torch.bfloat16, device="cuda"
                )
                residual = torch.randn(
                    rows, columns, dtype=torch.bfloat16, device="cuda"
                )
                workspace = torch.empty(
                    ops.vision_gemm_workspace_bytes(rows, columns, contraction),
                    dtype=torch.uint8,
                    device="cuda",
                )
                projection = F.linear(value.float(), weight.float(), bias.float())
                for act in range(4):
                    activated = (
                        projection
                        if act == 0
                        else F.gelu(projection, approximate="tanh")
                        if act == 1
                        else F.gelu(projection)
                        if act == 2
                        else projection * torch.sigmoid(1.702 * projection)
                    )
                    expected = (activated + residual.float()).to(torch.bfloat16)
                    actual = residual.clone()
                    ops.vision_gemm_bias_act(
                        value,
                        packed,
                        workspace,
                        actual,
                        bias,
                        actual,
                        rows,
                        columns,
                        contraction,
                        act,
                    )
                    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.008)
            print(
                "PASS: BF16 fused bias, activations and aliased residual",
                columns,
                contraction,
                flush=True,
            )
        for columns, contraction in ((512, 512), (1536, 512), (2048, 512), (512, 2048)):
            weight = torch.randn(
                columns, contraction, dtype=torch.float16, device="cuda"
            ) / math.sqrt(contraction)
            packed = (
                ops.gemm_bf16_shuffle_b(
                    weight.view(torch.bfloat16), columns, contraction
                )
                .view(torch.float16)
                .view_as(weight)
            )
            for rows in (1, 2, 8, 16, 32):
                value = torch.randn(
                    rows, contraction, dtype=torch.float16, device="cuda"
                )
                assert (16 * rows * columns if columns <= 1024 else 0) == lib.snowllm_auxiliary_gemm_half_workspace_bytes(
                    rows, columns, contraction
                )
                for bias in (
                    None,
                    torch.randn(columns, dtype=torch.float16, device="cuda"),
                ):
                    torch.testing.assert_close(
                        gemm_half(value, packed, bias),
                        F.linear(value, weight, bias),
                        atol=0.002,
                        rtol=0.002,
                    )
            print(
                "PASS: native FP16 projection and shared scratch plan",
                columns,
                contraction,
                flush=True,
            )


if __name__ == "__main__":
    main()

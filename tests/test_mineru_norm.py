# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


def main():
    import torch
    from snowllm import ops
    from snowllm.ops.trace import dummy_run

    torch.manual_seed(173)
    with torch.inference_mode():
        for rows in (1, 2, 8, 32, 128, 512):
            for stride in (896, 1024):
                gamma = torch.randn(896, dtype=torch.bfloat16, device="cuda") * 2
                value = torch.randn(rows, stride, dtype=gamma.dtype, device="cuda")[
                    :, :896
                ]
                expected = torch.empty(
                    rows, 896, dtype=value.dtype, device=value.device
                )
                actual = torch.empty_like(expected)
                for scale in (1e-20, 1.0, 64.0, 1e20):
                    value.copy_(torch.randn_like(value) * scale)
                    ops.dsv4_rmsnorm(value, None, expected, 1e-5)
                    torch.mul(expected, gamma, out=expected)
                    with patch(
                        "torch.mul",
                        side_effect=AssertionError(
                            "normalization must fuse the multiply"
                        ),
                    ):
                        ops.rmsnorm_rows(
                            value, gamma, actual, 1e-5, round_before_weight=True
                        )
                    assert torch.equal(
                        actual.view(torch.int16), expected.view(torch.int16)
                    )
                value.normal_()
                ops.dsv4_rmsnorm(value, None, expected, 1e-5)
                torch.mul(expected, gamma, out=expected)
                copied = value.clone()
                ops.rmsnorm_rows(copied, gamma, copied, 1e-5, round_before_weight=True)
                assert torch.equal(copied.view(torch.int16), expected.view(torch.int16))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    ops.rmsnorm_rows(
                        value, gamma, actual, 1e-5, round_before_weight=True
                    )
                value.normal_()
                ops.dsv4_rmsnorm(value, None, expected, 1e-5)
                torch.mul(expected, gamma, out=expected)
                graph.replay()
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                actual.fill_(7)
                with dummy_run():
                    ops.rmsnorm_rows(
                        value, gamma, actual, 1e-5, round_before_weight=True
                    )
                torch.cuda.synchronize()
                assert bool((actual == 7).all()), "planning must not launch the kernel"
        print(
            "PASS: fused rounded RMSNorm is bitexact across rows, strides, aliases and graph replay",
            flush=True,
        )


if __name__ == "__main__":
    main()

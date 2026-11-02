# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path


def main():
    import numpy as np
    import onnxruntime as ort
    import torch
    from snowllm.mineru.onnx import TableSession

    root = Path(
        os.environ.get("SNOWLLM_MINERU_DIR", "~/models/MinerU-4.1")
    ).expanduser()
    root = root / "small-models/Table"
    torch.set_num_threads(2)
    cases = [
        ("PP-LCNet_x1_0_table_cls.onnx", (2, 3, 224, 224)),
        ("unet.onnx", (1, 3, 96, 128)),
        ("slanet-plus.onnx", (1, 3, 64, 96)),
    ]
    for name, shape in cases:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        reference = ort.InferenceSession(
            str(root / name), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
        expected = reference.run(None, {reference.get_inputs()[0].name: inputs})
        actual = TableSession(root / name).run(
            None, {reference.get_inputs()[0].name: inputs}
        )
        for received, wanted in zip(actual, expected):
            np.testing.assert_allclose(received, wanted, rtol=5e-3, atol=5e-3)
            print(
                "PASS",
                name,
                received.shape,
                "maximum error",
                np.max(np.abs(received - wanted)),
                flush=True,
            )


if __name__ == "__main__":
    main()

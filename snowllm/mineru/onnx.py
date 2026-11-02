# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import torch

from ..checkpoint.loader import load_auxiliary


MODELS = {
    "PP-LCNet_x1_0_table_cls.onnx": "classifier",
    "unet.onnx": "unet",
    "slanet-plus.onnx": "slanet",
}


class TableSession:
    def __init__(self, path, *, device="cuda", graphs=None, load_model=None):
        if device != "cuda":
            raise ValueError("MinerU table models require the shared GPU worker")
        kind = MODELS.get(Path(path).name)
        if kind is None:
            raise ValueError(f"unsupported MinerU table checkpoint: {Path(path).name}")
        load = load_model or load_auxiliary
        self.model = load(path, "MinerUTableModel", kind=kind)
        self.model._native_graph_cache = graphs
        self.admit = graphs.admit if graphs else None
        if self.admit:
            self.model._memory_admit = self.admit
        self.device = device

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.model.metadata)

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.model.inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self.model.outputs]

    def get_providers(self):
        return ["SnowLLMGPUExecutionProvider"]

    @torch.inference_mode()
    def run(self, output_names, input_feed, run_options=None):
        name = self.model.inputs[0]
        value = input_feed[name]
        if self.admit:
            self.admit((value.size * value.dtype.itemsize + 511) // 512 * 512)
        x = torch.as_tensor(value, device=self.device)
        result = self.model(x)
        values = result if isinstance(result, tuple) else (result,)
        outputs = dict(zip(self.model.outputs, values))
        return [
            outputs[name].cpu().numpy() for name in output_names or self.model.outputs
        ]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import pathlib
import sys

from snowllm import models
from snowllm.checkpoint import loader
from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import find_gguf

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))
LAYERS = 2


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        return 0
    ck = _harness.Checks()

    ck("the loader sees a GGUF tree here", loader.is_gguf(MODEL_DIR), str(MODEL_DIR))
    cfg = config(GGUF(find_gguf(MODEL_DIR)))
    arch = cfg["architectures"]
    ck("the container names an architecture the registry carries", arch[0] in models.supported(),
       f"{arch[0]} in {models.supported()}")

    mod, load_weights = models.resolve(arch)
    ck("it resolves to the deepseek_v4 module", mod.__name__.endswith("deepseek_v4"), mod.__name__)
    ck("with no safetensors loader, and a reader-taking hook instead",
       load_weights is None and hasattr(mod, "load_gguf_weights"), f"load_weights={load_weights}")
    mod.validate_config(cfg)

    n = min(LAYERS, cfg["num_hidden_layers"])
    model = loader.load(MODEL_DIR, layers=range(n))
    ck(f"loader.load builds the model ({n} of {cfg['num_hidden_layers']} layers)",
       type(model).__name__ == "DeepSeekV4ForCausalLM" and len(model.layers) == n,
       f"{type(model).__name__} with {len(model.layers)} layers")
    ck("and stamps the multimodal attributes the server reads",
       model.vision_config is None and model.visual is None,
       f"vision_config={model.vision_config} visual={model.visual}")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
from ... import ops
from ..._capi import GEO_INKLING2_MINI
from ...errors import SnowLLMError
from .decoder import InklingTextModel

__all__ = ["InklingTextModel"]


def validate_config(cfg):
    shape = (cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"],
             cfg["head_dim"], cfg["sconv_kernel_size"], cfg["num_experts"], cfg["num_experts_per_tok"],
             cfg["moe_intermediate_size"], cfg["unpadded_vocab_size"])
    if shape != (2304, 24, 8, 128, 4, 128, 8, 1152, 200058):
        raise SnowLLMError(f"unsupported Inkling-2-Mini text geometry {shape}")
    ops.select_geometry(GEO_INKLING2_MINI)


def load_gguf_weights(reader, cfg, layers, *, device_map, mtp, fast_draft=False):
    if tuple(layers) != tuple(range(cfg["num_hidden_layers"])):
        raise SnowLLMError("the fixed Inkling loop requires every weight layer")
    from ...ops.inkling import candidate_tiles
    fast_draft = fast_draft and bool(candidate_tiles(1, 1, 200064, cfg["hidden_size"], 256))
    model = InklingTextModel(reader, device_map, fast_draft)
    if not model.embedding.is_meta:
        from ...ops.disk_moe import planning_copy
        with ops.dry_load(), planning_copy():
            model.planning_model = InklingTextModel(reader, device_map, fast_draft)
    return model

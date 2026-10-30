# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from .errors import SnowLLMError


# Official model cards specify decimal token limits.
_YARN_LIMITS = {
    'qwen3_5_text': 1_010_000,
    'qwen3_5_moe_text': 1_010_000,
    'qwen4_exp_text': 1_000_000,
}


@dataclass(frozen=True)
class ContextCapacity:
    tokens: int
    per_request: int
    model_limit: int
    original: int
    auto_yarn: bool

    @classmethod
    def for_model(cls, model, tokens):
        cfg = getattr(model, 'config', {})
        kind = getattr(model, 'multimodal_type', None) or cfg.get('model_type')
        native = int(cfg.get('max_position_embeddings', 262144))
        if tokens <= 0 or native <= 0:
            raise SnowLLMError('context capacity and model context length must be positive')
        auto_yarn = kind in _YARN_LIMITS
        maximum = _YARN_LIMITS.get(kind, native)
        rp = cfg.get('rope_parameters') or {}
        original = int(rp.get('original_max_position_embeddings', native)) if auto_yarn else native
        if original <= 0:
            raise SnowLLMError('original model context length must be positive')
        return cls(tokens, min(tokens, maximum), maximum, original, auto_yarn)

    @property
    def rope_factor(self):
        return max(1., self.per_request / self.original) if self.auto_yarn else 1.

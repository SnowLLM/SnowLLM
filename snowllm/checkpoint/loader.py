# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

import contextlib
import json
import math
import pathlib
from typing import TYPE_CHECKING

import torch

from .. import models, term
from .._capi import SnowLLMError
from .reader import Reader

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    from ..models.deepseek_v4.deepseek4 import DeepSeekV4ForCausalLM
    from ..models.placement import DeviceMap
    from ..models.qwen3_5.qwen3_5 import Qwen3_5MoeForCausalLM
    from ..models.vision import VisionModel
    from .gguf import GGUF

    TargetModel = Qwen3_5MoeForCausalLM | DeepSeekV4ForCausalLM


class WeightSource:
    def __init__(self, rd: Reader) -> None:
        self.rd = rd

    def has(self, key: str) -> bool:
        return key in self.rd.keys()

    def visual_keys(self) -> list[str]:
        return [k for k in self.rd.keys() if k.startswith(VISUAL_PREFIX)]

    def is_fp8(self, key: str) -> bool:
        return self.rd.spec(key)[0] is torch.uint8

    def rows(self, key: str) -> int:
        return self.rd.spec(key)[1][0]

    def read(self, key: str, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.rd.to_device(key, out=out) if out is not None else self.rd.to_device(key)

    def read_dequant(self, key: str) -> torch.Tensor:
        if self.is_fp8(key):
            return block_dequant(self.rd.to_device(key), self.rd.to_device(key + "_scale_inv"))
        return self.rd.to_device(key)

    def concat(self, dst: torch.Tensor, prefix: str, names: list[str]) -> int:
        off = 0
        for n in names:
            key = prefix + n
            rows = self.rows(key)
            if self.is_fp8(key):
                dst[off:off + rows] = block_dequant(self.rd.to_device(key),
                                                    self.rd.to_device(key + "_scale_inv"))
            else:
                self.rd.to_device(key, out=dst[off:off + rows])
            off += rows
        return off

    def concat_fp8(self, dst8: torch.Tensor, dst_s: torch.Tensor, prefix: str,
                   names: list[str]) -> int:
        off = soff = 0
        for n in names:
            key = prefix + n
            rows = self.rows(key)
            self.rd.to_device(key, out=dst8[off:off + rows])
            s = self.rd.to_device(key + "_scale_inv")
            dst_s[soff:soff + s.shape[0]] = s
            off += rows
            soff += s.shape[0]
        return off


def block_dequant(w8: torch.Tensor, scale: torch.Tensor, bn: int = 128,
                  bk: int = 128) -> torch.Tensor:
    n, k = w8.shape
    wf = w8.view(torch.float8_e4m3fn).float()
    s = scale.float().repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:n, :k]
    return (wf * s).to(torch.bfloat16).contiguous()


def yarn_rope_table(cfg: dict, factor: float, orig_max_pos: int | None = None,
                    beta_fast: float = 32.0, beta_slow: float = 1.0) -> tuple[torch.Tensor, float]:
    rp = cfg["rope_parameters"]
    dr = int(cfg["head_dim"] * rp["partial_rotary_factor"])
    base = float(rp["rope_theta"])
    i = torch.arange(0, dr, 2, dtype=torch.float32)
    if factor <= 1.0:
        return 1.0 / (base ** (i / dr)), 1.0
    L = int(orig_max_pos if orig_max_pos is not None else cfg["max_position_embeddings"])
    pos_freqs = base ** (i / dr)
    inv_extrap, inv_interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)

    def corr_dim(rot: float) -> float:
        return (dr * math.log(L / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), dr - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dr // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    ext = 1.0 - ramp
    inv_freq = inv_interp * (1 - ext) + inv_extrap * ext
    return inv_freq, 0.1 * math.log(factor) + 1.0


VISUAL_PREFIX = "model.visual."


def load_vision(w: WeightSource, vision_config: dict | None) -> "VisionModel | None":
    keys = w.visual_keys()
    if not keys:
        return None
    if not vision_config:
        raise SnowLLMError("the checkpoint carries model.visual.* weights but no vision_config in "
                           "config.json, so there is nothing to read the tower's shape from")
    from ..models.vision import VisionModel
    return VisionModel.from_hf_state({k: w.read_dequant(k) for k in keys}, vision_config,
                                     prefix=VISUAL_PREFIX)


def is_gguf(root: pathlib.Path) -> bool:
    from .gguf.source import gguf_dir
    return not (root / "config.json").exists() and any(gguf_dir(root).glob("*.gguf"))


def load_tokenizer(root: pathlib.Path) -> "tuple[PreTrainedTokenizerBase, tuple[int, ...]]":
    if is_gguf(root):
        from .gguf import GGUF
        from .gguf.source import find_gguf
        from .gguf.tokenizer import build, stop_token_ids
        container = GGUF(find_gguf(root))
        return build(container), stop_token_ids(container)

    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if (root / "tokenizer.json").exists():
        tok = PreTrainedTokenizerFast.from_pretrained(str(root))
    else:
        tok = AutoTokenizer.from_pretrained(str(root))
    path = root / "generation_config.json"
    cfg = json.loads(path.read_text()) if path.exists() else {}
    eos = cfg.get("eos_token_id", tok.eos_token_id)
    return tok, tuple(eos) if isinstance(eos, list) else (eos,)


def sampling_defaults(root: pathlib.Path) -> dict[str, float | int]:
    if is_gguf(root):
        from .gguf import GGUF
        from .gguf.source import find_gguf
        g = GGUF(find_gguf(root))
        raw = {n: g.get(f"general.sampling.{k}") for n, k in
               (("temperature", "temp"), ("top_p", "top_p"), ("top_k", "top_k"))}
    else:
        path = root / "generation_config.json"
        cfg = json.loads(path.read_text()) if path.exists() else {}
        raw = {n: cfg.get(n) for n in ("temperature", "top_p", "top_k")}
    return {k: int(v) if k == "top_k" else round(float(v), 6) for k, v in raw.items()
            if v is not None}


PROCESSOR_CONFIGS = ("preprocessor_config.json", "video_preprocessor_config.json")
PROCESSOR_SOURCE = "Qwen/Qwen3.6-35B-A3B"


def load_processor(root: pathlib.Path, tok: "PreTrainedTokenizerBase") -> "ProcessorMixin":
    from transformers import AutoImageProcessor, AutoProcessor, AutoVideoProcessor

    if not is_gguf(root):
        return AutoProcessor.from_pretrained(str(root))

    missing = [n for n in PROCESSOR_CONFIGS if not (root / n).exists()]
    if missing:
        raise SnowLLMError(
            f"{root.name} carries a vision tower but not {', '.join(missing)}, so its images "
            f"cannot be preprocessed. `snowllm pull` fetches them; otherwise copy them from "
            f"{PROCESSOR_SOURCE} (775 bytes for both). Text-only use needs neither.")

    import transformers
    raw = json.loads((root / PROCESSOR_CONFIGS[0]).read_text())
    name = raw.get("processor_class")
    if not name or not hasattr(transformers, name):
        raise SnowLLMError(f"{root.name}/{PROCESSOR_CONFIGS[0]} names the processor class "
                           f"{name!r}, which this transformers does not have")
    return getattr(transformers, name)(
        image_processor=AutoImageProcessor.from_pretrained(str(root)),
        video_processor=AutoVideoProcessor.from_pretrained(str(root)),
        tokenizer=tok, chat_template=getattr(tok, "chat_template", None))


def load(path: str | pathlib.Path, layers: range | None = None, shard_cache: int | None = None,
         mtp: bool = True, vision: bool = True, device_map: str = "",
         kv: dict | None = None) -> "TargetModel":
    root = pathlib.Path(path)
    if is_gguf(root):
        return load_gguf(root, layers=layers, mtp=mtp, vision=vision, device_map=device_map,
                         kv=kv)
    if device_map:
        raise SnowLLMError(f"--device-map is a GGUF-only placement today and {root} is a "
                           f"safetensors tree")
    raw = json.loads((root / "config.json").read_text())
    cfg = raw.get("text_config", raw)
    arch = raw.get("architectures") or cfg.get("architectures")
    if not arch:
        raise SnowLLMError(f"{root}/config.json names no architectures")
    mod, load_weights = models.resolve(arch)
    if load_weights is None:
        raise SnowLLMError(f"{arch[0]} is served from GGUF only in this build, and {root} is a "
                           f"safetensors tree. Convert it, or point at the GGUF.")
    mod.validate_config(cfg)

    want = layers if layers is not None else range(cfg["num_hidden_layers"])
    fp8 = (raw.get("quantization_config") or cfg.get("quantization_config")
           or {}).get("quant_method") == "fp8"

    if shard_cache is None:
        shard_cache = 6 if fp8 else 0
    with Reader(root, shard_cache=shard_cache) as rd:
        w = WeightSource(rd)
        model = load_weights(w, cfg, want, fp8, mtp)
        model.vision_config = raw.get("vision_config")
        model.visual = load_vision(w, model.vision_config) if vision else None
        model.image_token_id = raw.get("image_token_id")
    torch.cuda.empty_cache()
    return model


def geo_context(cfg: dict) -> int:
    return int(cfg.get("max_position_embeddings") or 1 << 62)


def dry_need(build: "Callable[[], TargetModel]", cfg: dict, kv: dict | None,
             budget: int) -> int:
    from .. import ops
    from ..engine import carve_out_need
    if not kv:
        return 0
    with ops.dry_load() as d:
        model = build()
        return carve_out_need(model, cfg, kv, budget, d.device)


def _device_map(gguf: "GGUF", spec: str, cfg: dict, kv: dict | None,
                build: "Callable[[], TargetModel] | None" = None) -> "DeviceMap | None":
    if not spec:
        return None
    from ..models import placement
    budget = max(0, placement.auto_budget() - int((kv or {}).get("host_taken") or 0))
    need = dry_need(build, cfg, kv, budget)
    dmap = placement.check_budget(
        placement.parse(spec, placement.group_bytes(gguf), budget=budget, need=need,
                        per_layer=placement.layer_bytes(gguf)), budget)
    print(f"{term.stamp()} {dmap.describe()}", flush=True)
    return dmap


def load_gguf(root: pathlib.Path, layers: range | None = None, mtp: bool = True,
              vision: bool = True, device_map: str = "",
              kv: dict | None = None) -> "TargetModel":
    from .gguf.names import config, vision_config
    from .gguf.source import (
        GGUFReader,
        GGUFWeightSource,
        find_gguf,
        find_mmproj,
        find_mtp_gguf,
        vision_state,
    )
    from .gguf.tokenizer import image_token_id

    side = find_mtp_gguf(root) if mtp else None
    tower = find_mmproj(root) if vision else None
    with GGUFReader(find_gguf(root)) as rd, \
            (GGUFReader(side) if side else contextlib.nullcontext()) as mtp_rd:
        raw = config(rd.gguf)
        cfg = raw.get("text_config", raw)
        mod, load_weights = models.resolve(raw["architectures"])
        mod.validate_config(cfg)
        want = layers if layers is not None else range(cfg["num_hidden_layers"])
        def build(dmap: "DeviceMap | None" = None) -> "TargetModel":
            reads_raw = getattr(mod, "load_gguf_weights", None)
            if reads_raw is not None:
                m = (reads_raw(rd, cfg, want, device_map=dmap) if dmap is not None
                     else reads_raw(rd, cfg, want))
            else:
                m = load_weights(GGUFWeightSource(rd, cfg, mtp_rd), cfg, want, False, mtp,
                                 device_map=dmap)
            m.vision_config = None
            m.visual = None
            m.image_token_id = image_token_id(rd.gguf)
            if tower is not None:
                from ..models.vision import VisionModel
                with GGUFReader(tower) as vrd:
                    m.vision_config = vision_config(vrd.gguf)
                    m.visual = VisionModel.from_hf_state(vision_state(vrd), m.vision_config)
            return m

        dmap = _device_map(rd.gguf, device_map, cfg, kv, build)
        model = build(dmap)
    torch.cuda.empty_cache()
    return model

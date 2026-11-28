# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

import contextlib
import dataclasses
import json
import math
import pathlib
from typing import TYPE_CHECKING

import torch

from .. import models, term
from .._capi import SnowLLMError, lib
from .._platform import release_device_cache
from ..capacity import placement_options
from ..weight_map import WeightMap
from .reader import Reader, TensorMapping
from .tokenizer import is_gguf, load_tokenizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    from ..models.deepseek_v4.deepseek4 import DeepSeekV4ForCausalLM
    from ..models.glm5_next.layers import Glm5NextForConditionalGeneration
    from ..models.placement import DeviceMap
    from ..models.qwen3_5.qwen3_5 import Qwen3_5MoeForCausalLM
    from ..models.vision import VisionModel

    TargetModel = Qwen3_5MoeForCausalLM | DeepSeekV4ForCausalLM | Glm5NextForConditionalGeneration


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

def load_vision(w: WeightSource, vision_config: dict | None, **options) -> "VisionModel | None":
    keys = w.visual_keys()
    if not keys:
        return None
    if not vision_config:
        raise SnowLLMError("the checkpoint carries model.visual.* weights but no vision_config in "
                           "config.json, so there is nothing to read the tower's shape from")
    from ..models.vision import VisionModel
    return VisionModel.from_hf_state(TensorMapping(keys, w.read_dequant), vision_config,
                                     prefix=VISUAL_PREFIX, **options)


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


def load_processor(root: pathlib.Path, tok: "PreTrainedTokenizerBase",
                   vision_head: str | None = None) -> "ProcessorMixin | PreTrainedTokenizerBase":
    from transformers import AutoImageProcessor, AutoProcessor, AutoVideoProcessor

    if not is_gguf(root):
        return AutoProcessor.from_pretrained(str(root))

    from .gguf import GGUF, minicpmv
    from .gguf.source import find_mmproj
    tower = find_mmproj(root, vision_head)
    if tower is not None:
        g = GGUF(tower)
        if (g.get("clip.vision.projector_type") == "inkling2v" or g.get("clip.projector_type") == "glm5v"
                or g.get("{arch}.sidecar") == "vision" and g.get("{arch}.variant") == "v41_flash"):
            return tok
        if g.get("clip.projector_type") == "minicpmv4_7":
            from ..models.minicpmv.processing import processor
            return processor(tok, minicpmv.vision_config(g))

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


def _resolve_weight_map(device_map, cfg, kv, count, target, host_offload):
    explicit = WeightMap.parse(device_map)
    if explicit is None:
        if (count and device_map in ('', 'auto') and host_offload is None and kv
                and 'host_offload_allowed' in kv and kv.get('offload_ratio') is None):
            device_map = 'auto' if kv['host_offload_allowed'] else 'none'
        if host_offload == 0 and (kv or {}).get('max_disk_bytes') is not None and not count:
            kv = dict(kv, fixed_weight_map='none', memory_auto=True)
        return device_map, kv, count
    kv = dict(kv or {}, fixed_weight_map=device_map, memory_auto=False)
    if explicit.disk_experts:
        topk = cfg.get('num_experts_per_tok', 0)
        if not topk:
            raise SnowLLMError('experts=disk requires a model with routed experts')
        if not count and not target:
            if kv.get('max_disk_bytes') is None:
                count = topk
            else:
                kv['memory_auto'] = True
    return device_map, kv, count


def load(path: str | pathlib.Path, layers: range | None = None, shard_cache: int | None = None,
         mtp: bool = True, vision: bool = True, device_map: str = "",
         kv: dict | None = None, moe_cache_experts: int = 0,
         moe_cache_mode: str = 'all', disk_offload: int | str | None = None,
         host_offload: int | str | None = None, vision_head: str | None = None,
         vision_placement: str = "auto") -> "TargetModel":
    root = pathlib.Path(path)
    if is_gguf(root):
        return load_gguf(root, layers=layers, mtp=mtp, vision=vision, device_map=device_map,
                         kv=kv, moe_cache_experts=moe_cache_experts, moe_cache_mode=moe_cache_mode,
                         disk_offload=disk_offload, host_offload=host_offload, vision_head=vision_head,
                         vision_placement=vision_placement)
    if vision_head is not None:
        raise SnowLLMError("an external vision head requires a GGUF model")
    from ..host_alloc import enabled
    host_offload, disk_offload = placement_options(host_alloc=enabled(), device_map=device_map,
        host_offload=host_offload, disk_offload=disk_offload,
        offload_ratio=(kv or {}).get('offload_ratio'), max_disk_bytes=(kv or {}).get('max_disk_bytes'),
        moe_cache_experts=moe_cache_experts, moe_cache_mode=moe_cache_mode)
    raw = json.loads((root / "config.json").read_text())
    cfg = raw.get("text_config", raw)
    device_map, kv, moe_cache_experts = _resolve_weight_map(
        device_map, cfg, kv, moe_cache_experts, disk_offload, host_offload)
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
    if raw.get("model_type") == "minicpmv4_7" and (raw.get("quantization_config") or cfg.get("quantization_config")):
        raise SnowLLMError("MiniCPM-V quantized safetensors are not supported; use BF16 or GGUF")

    if shard_cache is None:
        shard_cache = 6 if fp8 else 0
    from .. import ops
    with Reader(root, shard_cache=shard_cache) as rd:
        w = WeightSource(rd)
        from ..models.vision_layers import RESIDENT_VISION
        vc = raw.get("vision_config")
        model_type = raw.get("model_type")
        if vc and model_type == "minicpmv4_7":
            vc = {**vc, "downsample_mode": raw.get("downsample_mode", "16x"),
                  "insert_layer_id": raw.get("insert_layer_id", 6), "out_hidden_size": cfg["hidden_size"],
                  "scale_resolution": 448, "max_slice_nums": 9}
        def visual(weights):
            if model_type == "qwen2_vl":
                from ..models.mineru.qwen2_vl.vision import Qwen2Vision
                return Qwen2Vision(w, vc, weights)
            if model_type == "minicpmv4_7":
                from ..models.minicpmv.vision import MiniCPMVision
                return MiniCPMVision(TensorMapping(rd.keys(), w.read_dequant), vc, weights)
            return load_vision(w, vc, weights=weights)

        cache = None
        if vision and vc and (vision_placement == "disk" or
                              (vision_placement == "auto" and model_type != "qwen2_vl")):
            from .vision_cache import TowerWeightCache
            prefixes = {"qwen2_vl": ("visual.",), "minicpmv4_7": ("model.vision_tower.", "model.merger.")}
            keys = [k for k in rd.keys() if k.startswith(prefixes.get(model_type, (VISUAL_PREFIX,)))]
            if keys:
                files = sorted({root / rd.weight_map[k] for k in keys})
                files.append(root / "config.json")
                index = root / "model.safetensors.index.json"
                if index.exists():
                    files.append(index)
                def streamed_visual(weights):
                    with rd.individual_tensors():
                        return visual(weights)
                cache = TowerWeightCache(files, model_type, streamed_visual)
        def build(dmap=None, *, offload=None):
            if cache is not None and ops.drying() is None:
                cache.prepare()
            with rd.loading():
                model = load_weights(w, cfg, want, fp8, mtp, device_map=dmap)
                model.multimodal_config = raw
                model.vision_config = vc
                model.visual = (cache.tower() if cache else visual(RESIDENT_VISION)) if vision else None
                model.image_token_id = raw.get("image_token_id")
                return model

        model = _place_model(build, cfg, kv, device_map, host_offload, moe_cache_experts,
                             moe_cache_mode, disk_offload)
    release_device_cache()
    return model


def _place_model(materialize, cfg, kv, device_map, host_offload, count, mode, target):
    from ..models import placement
    from ..ops.disk_moe import Catalog, Store, building
    catalog = Catalog()
    explicit = WeightMap.parse(device_map)
    requested = bool(count or target or (kv or {}).get('max_disk_bytes') or (explicit and explicit.disk_experts))
    with building(catalog if requested else None):
        total, sizes, by = placement.measure_weights(materialize)
    unspecified = object()

    def build(dmap=None, *, offload=unspecified):
        amount = target if offload is unspecified else offload
        if not count and not amount:
            return materialize(dmap)
        from ..defaults import DEFAULT_PREFILL_CHUNK
        chunk = (kv or {}).get('chunk', 'auto')
        chunk = DEFAULT_PREFILL_CHUNK if chunk == 'auto' else int(chunk)
        ctx = (kv or {}).get('ctx') or chunk
        store = Store(catalog, dmap, cfg.get('num_experts_per_tok', 0), mode, count, amount,
                      min(chunk, ctx))
        try:
            with building(store):
                model = materialize(dmap)
        except BaseException:
            store.close()
            raise
        model.moe_offload_plan = store
        return model

    eligibility = None
    if (kv or {}).get('memory_auto'):
        dmap, target, eligibility = _memory_device_map(
            cfg, kv, build, sizes, by, total, disk_candidates=catalog.layers)
    else:
        dmap = _device_map(device_map, cfg, kv, build, sizes, by, total_bytes=total,
                           host_offload=host_offload, disk_candidates=catalog.layers if requested else None)
    if dmap is None:
        dmap = placement.DeviceMap(frozenset(), sizes, per_layer=by, total_bytes=total)
    else:
        dmap = dataclasses.replace(dmap, total_bytes=total)
    if count or target:
        from .. import ops
        with ops.dry_load() as dry:
            probe = build(dmap)
            savings = total - dmap.host_bytes - dry.device
            if target and savings < target:
                raise SnowLLMError(f'disk placement saves {savings} bytes, requested {target}')
            dmap = dataclasses.replace(dmap, disk_savings=savings)
        del probe
    return _load_placed(build, cfg, kv, dmap, device_map, host_offload=host_offload, eligibility=eligibility)


def dry_need(build: "Callable[[], TargetModel]", cfg: dict, kv: dict | None,
             budget: int) -> int:
    from .. import ops
    from ..engine import carve_out_need
    if not kv:
        return 0
    with ops.dry_load() as d:
        model = build()
        return carve_out_need(model, cfg, kv, budget, d.device, load_peak=d.peak_device,
                              host_weights=d.host)


def _check_weight_capacity(total, kv, movable):
    if kv.get('max_disk_bytes') is not None:
        return
    from .._platform import gpu_mem_info
    from ..defaults import DEFAULT_GPU_UTIL
    from ..memory_report import current, size
    from ..models import placement
    free, physical = gpu_mem_info()
    gpu = max(0, int(physical * float(kv.get('util') or DEFAULT_GPU_UTIL)) - (physical-free))
    host = (min(movable, max(0, placement.auto_budget() - int(kv.get('host_taken') or 0)))
            if kv.get('host_offload_allowed', False) else 0)
    weights = total + int(kv.get('draft_weight_bytes', kv.get('draft_resident')) or 0)
    missing = weights - gpu - host
    if missing <= 0:
        return
    report = current()
    if report is not None:
        report.weight_lower_bound = weights, gpu, host
    else:
        print(f'[planner] weights need at least {size(weights)}; available: '
              f'GPU {size(gpu)} + host up to {size(host)}', flush=True)
    host_help = ' or --max-host-mem' if kv.get('host_offload_allowed', False) and movable > host else ''
    raise SnowLLMError(
        f'weights alone need at least {size(missing)} more memory '
        '(KV cache and workspace not included); '
        f'increase --max-gpu-mem{host_help} if more memory is available, '
        'or use a smaller or more heavily quantized checkpoint')


def _device_map(spec: str, cfg: dict, kv: dict | None,
                build: "Callable[[], TargetModel]", sizes: dict[str, int],
                per_layer: dict[str, dict[int, int]], *, total_bytes=None,
                disk_candidates=None, host_reserve=0, host_offload=None) -> "DeviceMap | None":
    if not spec and host_offload is None:
        return None
    from ..models import placement
    budget = max(0, placement.auto_budget() - int((kv or {}).get("host_taken") or 0) - host_reserve)
    if host_offload is not None:
        planner = placement.plan_moe if disk_candidates is not None else placement.plan
        dmap = placement.check_budget(planner(sizes, budget, host_offload, per_layer), budget)
        if dmap.short:
            raise SnowLLMError('--host-offload exceeds movable weights or available host memory')
        print(f'{term.stamp()} {dmap.describe()}', flush=True)
        return dmap
    if disk_candidates is not None and spec == 'all':
        dmap = placement.plan_moe(sizes, budget, budget, per_layer)
        dmap = dataclasses.replace(dmap, short=0)
        print(f'{term.stamp()} {dmap.describe()}', flush=True)
        return dmap
    if kv and kv.get('offload_ratio') is not None:
        total = sum(sizes.values()) if total_bytes is None else total_bytes
        need = math.ceil(total * float(kv['offload_ratio']))
        dmap = placement.check_budget(placement.plan(sizes, budget, need, per_layer), budget)
        if dmap.short:
            raise SnowLLMError('the requested offload ratio exceeds movable weights or available host memory')
        print(f'{term.stamp()} {dmap.describe()}', flush=True)
        return dmap
    need = dry_need(build, cfg, kv, budget) if spec in ('auto', 'all') else 0
    dmap = (placement.plan_moe(sizes, budget, need, per_layer)
            if disk_candidates is not None and spec == 'auto' else
            placement.parse(spec, sizes, budget=budget, need=need, per_layer=per_layer,
                            disk_candidates=disk_candidates))
    dmap = placement.check_budget(dmap, budget)
    print(f"{term.stamp()} {dmap.describe()}", flush=True)
    return dmap


def _memory_device_map(cfg, kv, build, sizes, per_layer, total, *, hash_layers=(), disk_candidates=None):
    from .. import ops
    from ..capacity import check_host, memory_shortage
    from ..models import placement
    eligible, by = dict(sizes), {g: dict(v) for g, v in per_layer.items()}
    if cfg.get('model_type') == 'deepseek_v4':
        for group in ('shared', 'shared.gate_up', 'shared.down'):
            eligible[group], by[group] = 0, {}
        for group in ('experts', 'experts.gate_up', 'experts.down'):
            by[group] = {i: n for i, n in by.get(group, {}).items() if i not in hash_layers}
            eligible[group] = sum(by[group].values())
    empty = placement.DeviceMap(frozenset(), sizes, per_layer=per_layer, total_bytes=total)
    _check_weight_capacity(total, kv, sum(eligible.values()))
    from ..memory_report import current
    report = current()
    if report is not None:
        report.max_host_offload = placement.max_host_bytes(empty, (eligible, by))
    print('[planner] planning KV cache and workspace', flush=True)
    needed = max(dry_need(lambda: build(empty, offload=None), cfg, kv, 0),
                 kv.get('minimum_host_savings', 0))
    fixed = kv.get('fixed_weight_map')
    disk_floor = kv.get('minimum_disk_savings', 0)
    if needed <= 0 and not fixed and not disk_floor:
        return empty, None, (eligible, by)
    fixed_map = (placement.parse(fixed, sizes, per_layer=by if hash_layers else per_layer,
                                 disk_candidates=disk_candidates) if fixed else None)
    allow_host = kv.get('host_offload_allowed', False)
    disk_limit = kv.get('max_disk_bytes')
    reserve, target = 0, 0
    for _ in range(8):
        host_room = max(0, placement.auto_budget() - int(kv.get('host_taken') or 0))
        host_budget = max(0, host_room - reserve) if allow_host else 0
        planner = (placement.plan_moe if cfg.get('model_type') == 'deepseek_v4' or
                   (disk_limit and cfg.get('num_experts_per_tok')) else placement.plan)
        if fixed_map is not None:
            dmap = placement.check_budget(fixed_map, host_room)
        else:
            dmap = planner(eligible, host_budget, needed, by)
        dmap = dataclasses.replace(dmap, sizes=sizes, per_layer=per_layer, total_bytes=total)
        padding = 0
        def placed_build():
            nonlocal padding
            model = build(dmap, offload=None)
            if (layout := ops.drying()) is not None:
                padding = max(0, layout.mapped_host_bytes - dmap.host_bytes)
            return model
        print('[planner] checking host placement and workspace', flush=True)
        remaining = dry_need(placed_build, cfg, kv, 0)
        if dmap.host_bytes:
            if fixed and dmap.host_bytes + padding > host_room:
                check_host(dmap.host_bytes + padding + int(kv.get('host_taken') or 0), ops.host_pinned_bytes())
            if not fixed and dmap.host_bytes + padding > host_room and padding > reserve:
                reserve = padding
                continue
        if remaining <= 0 and not disk_floor:
            return dmap, None, (eligible, by)
        host_help = not fixed and planner(eligible, sum(eligible.values()), needed, by).host_bytes > dmap.host_bytes
        if disk_limit is None:
            if report is not None:
                report.shortfall(remaining)
            raise memory_shortage(remaining, host_help=host_help)
        if not cfg.get('num_experts_per_tok') and cfg.get('model_type') != 'deepseek_v4':
            raise SnowLLMError('this model has no routed experts to offload')
        target = max(target, remaining, disk_floor)
        try:
            with ops.dry_load():
                probe = build(dmap, offload=target)
                store = probe.moe_offload_plan
                stage, weights = store.host_staging_bytes, store.disk_weight_bytes
                saved = store.disk_saved_bytes
            del probe
        except SnowLLMError as exc:
            raise SnowLLMError(f'{memory_shortage(remaining)}; {exc}') from exc
        if weights > disk_limit:
            raise SnowLLMError(f'--max-disk-offload permits {disk_limit} bytes of offloaded expert '
                               f'weights, but this layout needs {weights} bytes; '
                               f'{memory_shortage(remaining, host_help=host_help, disk_help=True)}')
        if not fixed and dmap.host_bytes + padding + stage > host_room and allow_host and padding + stage > reserve:
            reserve = padding + stage
            continue
        check_host(dmap.host_bytes + padding + stage + int(kv.get('host_taken') or 0), ops.host_pinned_bytes())
        shortage = dry_need(lambda: build(dmap, offload=target), cfg, kv, 0)
        if shortage <= 0:
            return dmap, target, (eligible, by)
        # Expert slots are discrete. Advance beyond the savings of the current layout.
        target = max(target, saved) + shortage
    raise memory_shortage(max(needed, target))


def _load_placed(build, cfg, kv, dmap, device_map, *, host_offload=None, eligibility=None):
    from .. import ops
    from ..host_alloc import enabled
    from ..models import placement
    memory_auto = bool((kv or {}).get('memory_auto'))
    automatic = not (kv or {}).get('fixed_weight_map') and ((memory_auto and kv.get('host_offload_allowed', False)) or
                 (not memory_auto and host_offload is None and device_map.strip() == 'auto'
                  and bool(kv) and kv.get('offload_ratio') is None))
    if enabled() or kv or dmap.host_bytes or dmap.disk_savings:
        from .._platform import gpu_mem_info
        from ..capacity import check_host
        with ops.dry_load() as planned:
            probe = build(dmap)
            peak = planned.peak_device
            host = planned.host + planned.mapped_host_bytes - sum(planned.placed_bytes().values())
            store = getattr(probe, 'moe_offload_plan', None)
            host += getattr(store, 'host_staging_bytes', 0)
            disk_limit = (kv or {}).get('max_disk_bytes')
            if store is not None and disk_limit is not None and store.disk_weight_bytes > disk_limit:
                raise SnowLLMError(f'expert placement needs {store.disk_weight_bytes} bytes on disk, '
                                   f'over --max-disk-offload {disk_limit}')
            from ..memory_report import current
            if current() is not None and kv:
                frozen = ('experts', 'experts.gate_up', 'experts.down') if store is not None else ()
                current().max_host_offload = placement.max_host_bytes(dmap, eligibility, frozen=frozen)
                from ..engine import carve_out_need
                carve_out_need(probe, cfg, kv, 0, planned.device, load_peak=peak,
                               host_weights=planned.host)
        del probe
        check_host(host, ops.host_pinned_bytes())
        free, physical = gpu_mem_info()
        room = (free if enabled() else
                int(physical * float((kv or {}).get('util', 1.))) - (physical-free))
        if peak > room:
            if memory_auto:
                from ..capacity import memory_shortage
                raise memory_shortage(peak-room)
            kind = '--host-alloc: loading' if enabled() else 'loading'
            raise SnowLLMError(f'{kind} needs {peak} bytes, but only {max(0, room)} bytes '
                               'remain within the memory budget')
    with (placement.capture_allocations() if automatic else contextlib.nullcontext(())) as allocations:
        model = build(dmap)
    model.device_map = dmap
    model._automatic_placement = automatic
    if eligibility is not None:
        model._placement_eligibility = eligibility
    if automatic:
        model._initial_host_bytes = dmap.host_bytes
        model._placement_allocations = tuple(allocations)
        plan = getattr(model, 'moe_offload_plan', None)
        # Allocated I/O staging is already reflected in MemAvailable.
        model._placement_host_reserve = int(kv.get('host_taken') or 0)
        if plan is not None:
            model._placement_frozen_groups = ('experts', 'experts.gate_up', 'experts.down')
        release_device_cache()
        budget = placement.host_budget(model)
        need = dry_need(lambda: build(dmap), cfg, dict(kv, loaded_weight_bytes=dmap.device_bytes),
                        max(0, budget - dmap.host_bytes))
        placement.ensure_room(model, need, budget=budget)
    return model


def load_gguf(root: pathlib.Path, layers: range | None = None, mtp: bool = True,
              vision: bool = True, device_map: str = "",
              kv: dict | None = None, moe_cache_experts: int = 0,
              moe_cache_mode: str = 'all', disk_offload: int | str | None = None,
              host_offload: int | str | None = None, vision_head: str | None = None,
         vision_placement: str = "auto") -> "TargetModel":
    from ..host_alloc import enabled
    host_offload, disk_offload = placement_options(host_alloc=enabled(), device_map=device_map,
        host_offload=host_offload, disk_offload=disk_offload,
        offload_ratio=(kv or {}).get('offload_ratio'), max_disk_bytes=(kv or {}).get('max_disk_bytes'),
        moe_cache_experts=moe_cache_experts, moe_cache_mode=moe_cache_mode)
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
    tower = find_mmproj(root, vision_head) if vision else None
    with (
        GGUFReader(find_gguf(root)) as rd,
        GGUFReader(side) if side else contextlib.nullcontext() as mtp_rd,
        GGUFReader(tower) if tower else contextlib.nullcontext() as vrd,
    ):
        raw = config(rd.gguf)
        cfg = raw.get("text_config", raw)
        device_map, kv, moe_cache_experts = _resolve_weight_map(
            device_map, cfg, kv, moe_cache_experts, disk_offload, host_offload)
        mod, load_weights = models.resolve(raw["architectures"])
        mod.validate_config(cfg)
        want = layers if layers is not None else range(cfg["num_hidden_layers"])
        unspecified = object()
        from .gguf.sequential import WeightReads
        from .. import ops
        reads = WeightReads()
        vision_budgets = {}
        from ..models.vision_layers import RESIDENT_VISION
        def visual(weights):
            if raw.get("model_type") == "inkling2_loop":
                from ..models.inkling2.vision import InklingVision
                return InklingVision(vrd, vc, weights)
            if raw.get("model_type") == "minicpmv4_7":
                from ..models.minicpmv.vision import MiniCPMVision
                return MiniCPMVision.from_gguf(vrd, weights)
            from ..models.vision import VisionModel
            return VisionModel.from_hf_state(vision_state(vrd), vc, weights=weights)

        vision_cache = tower_cache = vc = None
        vision_cls = None
        if vrd is not None:
            if raw.get('model_type') == 'glm5_next':
                from ..models.glm5_next.vision import Glm5NextVision
                vision_cls = Glm5NextVision
            elif rd.gguf.get('{arch}.variant') == 'v41_flash':
                from ..models.deepseek_v4.vision import DeepSeekV41Vision
                vision_cls = DeepSeekV41Vision
            if vision_cls is not None:
                if vision_placement != 'gpu':
                    from .vision_cache import VisionWeightCache
                    vision_cache = VisionWeightCache(vrd.gguf, vision_cls)
            else:
                from .gguf import minicpmv
                vc = (minicpmv.vision_config if raw.get('model_type') == 'minicpmv4_7'
                      else vision_config)(vrd.gguf)
                if vision_placement != 'gpu':
                    from .vision_cache import TowerWeightCache
                    tower_cache = TowerWeightCache(vrd.gguf.parts, raw.get('model_type'), visual)

        def build(dmap=None, **options):
            if ops.drying() is None:
                if vision_cache is not None:
                    vision_cache.prepare(vrd)
                if tower_cache is not None:
                    tower_cache.prepare()
            if enabled() or disk_offload or moe_cache_experts or (dmap and dmap.host_bytes):
                return materialize(dmap, **options)
            with reads.loading(ops.drying() is not None):
                return materialize(dmap, **options)

        def materialize(dmap: "DeviceMap | None" = None, *, cache_experts=moe_cache_experts,
                  cache_mode=moe_cache_mode, offload=unspecified) -> "TargetModel":
            offload = disk_offload if offload is unspecified else offload
            reads_raw = getattr(mod, "load_gguf_weights", None)
            if reads_raw is not None:
                extra = (dict(moe_cache_experts=cache_experts, moe_cache_mode=cache_mode,
                              disk_offload=offload) if cache_experts or offload else {})
                if getattr((kv or {}).get("draft_geo"), "inkling", False) and (kv or {}).get("dflash_adaptive") and cfg.get("model_type") == "inkling2_loop":
                    extra["fast_draft"] = True
                m = reads_raw(rd, cfg, want, device_map=dmap, mtp=mtp, **extra)
            else:
                m = load_weights(GGUFWeightSource(rd, cfg, mtp_rd), cfg, want, False, mtp,
                                 device_map=dmap)
            m.vision_config = None
            m.visual = None
            m.multimodal_config = raw
            m.image_token_id = image_token_id(rd.gguf)
            if vrd is not None:
                if vision_cls is not None:
                    if m.image_token_id is None:
                        raise SnowLLMError("the main GGUF tokenizer is missing its image marker")
                    vision_cls.attach(m, vrd, weights=vision_cache, budgets=vision_budgets,
                                      max_images=(kv or {}).get("max_images", 4))
                else:
                    m.vision_config = vc
                    m.visual = tower_cache.tower() if tower_cache else visual(RESIDENT_VISION)
            return m

        from ..models import placement
        import os
        legacy = (cfg.get('model_type') == 'deepseek_v4' and
                  (os.environ.get('SNOWLLM_MOE_CACHE_DIRS') or
                   0 < moe_cache_experts < cfg.get('num_experts_per_tok', 0) or
                   not hasattr(lib, 'snowllm_moe_disk_execute')) and
                  not any(name.endswith('ffn_exps_vq.blob') for name in rd.gguf.tensors))
        if not legacy:
            def resident(dmap):
                return build(dmap, cache_experts=0, offload=None)
            model = _place_model(resident, cfg, kv, device_map, host_offload, moe_cache_experts,
                                 moe_cache_mode, disk_offload)
        else:
            memory_auto = bool((kv or {}).get('memory_auto'))
            eligibility = None
            if memory_auto:
                total, sizes, by = placement.measure_weights(lambda dm: build(dm, offload=None))
                hashed = tuple(i for i in range(cfg['num_hidden_layers']) if
                               i < int(cfg.get('num_hash_layers', 0)) or
                               f'blk.{i}.ffn_gate_tid2eid.weight' in rd.gguf)
                dmap, disk_offload, eligibility = _memory_device_map(
                    cfg, kv, build, sizes, by, total, hash_layers=hashed, disk_candidates=set(want) - set(hashed))
            else:
                total, sizes, by = placement.measure_weights(
                    (lambda dm: build(dm, cache_experts=1, cache_mode='all', offload=None))
                    if moe_cache_experts or disk_offload else build)
            moe_plan = None
            if moe_cache_experts or disk_offload:
                with ops.dry_load():
                    probe = build()
                    moe_plan = probe.moe_offload_plan
                del probe
            if not memory_auto:
                dmap = _device_map(device_map, cfg, kv, build, sizes, by, total_bytes=total,
                                   disk_candidates=moe_plan.specs if moe_plan is not None else None,
                                   host_offload=host_offload)
            dmap = (placement.DeviceMap(frozenset(), sizes, per_layer=by, total_bytes=total)
                    if dmap is None else dataclasses.replace(dmap, total_bytes=total))
            if moe_plan is not None:
                if device_map not in ('', 'auto', 'all', 'off', 'none', 'device') and any(
                        group in dmap.host for group in ('shared', 'shared.gate_up', 'shared.down')):
                    raise SnowLLMError('shared experts must stay on GPU while MoE offload is enabled')
                with ops.dry_load() as planned:
                    probe = build(dmap)
                    stage = probe.moe_offload_plan.host_staging_bytes
                    planned_device = planned.device
                del probe
                budget = max(0, placement.auto_budget() - int((kv or {}).get('host_taken') or 0) - stage)
                if dmap.host_bytes > budget:
                    if device_map in ('auto', 'all') or host_offload is not None:
                        dmap = _device_map(device_map, cfg, kv, build, sizes, by, total_bytes=total,
                                           disk_candidates=moe_plan.specs, host_reserve=moe_plan.host_staging_bytes,
                                           host_offload=host_offload)
                        dmap = dataclasses.replace(dmap, total_bytes=total)
                        with ops.dry_load() as planned:
                            probe = build(dmap)
                            planned_device = planned.device
                        del probe
                    else:
                        placement.check_budget(dmap, budget)
                dmap = dataclasses.replace(dmap, disk_savings=total - dmap.host_bytes - planned_device)
            model = _load_placed(build, cfg, kv, dmap, device_map,
                                 host_offload=host_offload, eligibility=eligibility)
    release_device_cache()
    return model


def load_auxiliary(path, architecture, *, admit=None, **options):
    from .reader import TorchReader, ONNXReader
    from .. import ops
    module, load_weights = models.resolve([architecture])
    reader_type = {'.safetensors': Reader, '.onnx': ONNXReader}.get(pathlib.Path(path).suffix, TorchReader)
    reader = reader_type(path)
    with reader as source:
        weights = WeightSource(source)
        with ops.dry_load() as planned:
            probe = load_weights(weights, **options)
            peak = planned.peak_device
        del probe
        if admit is not None:
            admit(peak)
        model = load_weights(weights, **options)
    model.load_peak_bytes = peak
    return model

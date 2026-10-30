# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch

from .. import ops
from .._capi import SnowLLMError
from .._platform import gpu_mem_info, release_device_cache
from ..defaults import DEFAULT_GPU_UTIL, DEFAULT_MAX_NUM_SEQS, DEFAULT_PREFILL_CHUNK
from ..models.geometry import DeepSeekV4Geometry, DFlashGeometry, Glm5NextGeometry, Qwen4ExpGeometry
from .block_manager import BlockAllocator, StateSlots
from .forward_context import Batch, plain_decode_shapes
from .prefix_cache import CKPT_EVERY, Entry, PrefixCache, thin
from .request import Request, SamplingParams
from .dspark_decode import DEFAULT_DSPARK_P_MIN, DSparkDecoder
from .runner import (
    DraftGeometry,
    BaseRunner,
)
from .resources import DraftWorkspace, allocate_draft_pool, draft_pool_bytes, draft_table_blocks
from .qwen_runner import QwenRunner
from .sampler import Sampler
from .spec_decode import SPEC_MAX_STEP_ROWS, DFlashDecoder, DSparkScheduling, SpecDecoder

if TYPE_CHECKING:
    from ..checkpoint.gguf.dflash import DictDraftSource, GGUFDraftSource
    from ..models.deepseek_v4.deepseek4 import DeepSeekV4ForCausalLM
    from ..models.glm5_next.layers import Glm5NextForConditionalGeneration
    from ..models.qwen3_5.qwen3_5 import Qwen3_5MoeForCausalLM
    from ..models.qwen4exp.qwen4exp import Qwen4ExpForConditionalGeneration

    TargetModel = (Qwen3_5MoeForCausalLM | DeepSeekV4ForCausalLM
                   | Qwen4ExpForConditionalGeneration | Glm5NextForConditionalGeneration)
    DraftSource = GGUFDraftSource | DictDraftSource


@dataclass
class StepAccounting:
    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    prefill_steps: int = 0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    decode_steps: int = 0
    decode_rows: int = 0

    def add(self, kind: str, tokens: int, seconds: float = 0.0, rows: int = 0) -> None:
        if kind == "prefill":
            self.prefill_tokens += tokens
            self.prefill_seconds += seconds
            self.prefill_steps += 1
        elif kind == "decode":
            self.decode_tokens += tokens
            self.decode_seconds += seconds
            self.decode_steps += 1
            self.decode_rows += rows

    @property
    def prefill_tok_s(self) -> float:
        return self.prefill_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def accept_len(self) -> float:
        return self.decode_tokens / self.decode_rows if self.decode_rows else 0.0

    def snapshot(self) -> dict:
        return dict(prefill_tokens=self.prefill_tokens, prefill_seconds=self.prefill_seconds,
                    prefill_steps=self.prefill_steps, prefill_tok_s=self.prefill_tok_s,
                    decode_tokens=self.decode_tokens, decode_seconds=self.decode_seconds,
                    decode_steps=self.decode_steps, decode_rows=self.decode_rows,
                    decode_tok_s=self.decode_tok_s, accept_len=self.accept_len)


@dataclass
class EngineStats:
    running: int
    waiting: int
    free_slots: int
    free_kv_blocks: int
    preemptions: int
    cache_hits: int
    cache_misses: int
    cached_prefixes: int
    cached_blocks: int
    prefill_tokens_saved: int
    graph_sizes: list[int]
    graph_replays: int
    eager_forwards: int
    draft_replays: int = 0
    draft_eager: int = 0
    accounting: dict = field(default_factory=dict)


def _draft_geometry(path: str) -> DFlashGeometry:
    import json
    import pathlib

    from ..checkpoint.gguf import GGUF
    from ..checkpoint.gguf import dflash as dflash_gguf
    from ..checkpoint.gguf import dspark as dspark_gguf
    from ..checkpoint.gguf.source import find_dspark_gguf
    from ..models.geometry import DSparkGeometry

    p = pathlib.Path(path).expanduser()
    gguf = find_dspark_gguf(p) if p.is_dir() else (p if p.suffix == ".gguf" else None)
    if gguf is None and p.is_dir():
        gguf = dflash_gguf.find_gguf(p)
    if gguf is not None:
        g = GGUF(gguf)
        if not dflash_gguf.is_dflash(g):
            return DSparkGeometry.from_config(dspark_gguf.config(g))
        geo = dflash_gguf.geometry(g)
        ops.dflash.select(geo)
        return geo

    root = p if p.is_dir() else p.parent
    cfg = root / "config.json"
    if not cfg.is_file():
        raise SnowLLMError(
            f"no config.json beside the DFlash draft at {path}. It carries the taps, the mask "
            f"token and the window, none of which can be read off the weights.")
    geo = DFlashGeometry.from_config(json.loads(cfg.read_text()))
    ops.dflash.select(geo)
    return geo


def _check_dflash_target(model: "TargetModel", draft: DraftGeometry) -> None:
    geo = model.geo
    if _dspark(draft):
        from ..checkpoint.gguf import dspark as dspark_gguf
        if not isinstance(geo, DeepSeekV4Geometry):
            raise SnowLLMError("a DSpark drafter reads a DeepSeek-V4 target through three of its "
                               "layers; this model is not one.")
        if draft.stack.v41 and (not geo.v41 or draft.stack.hidden != geo.hidden
                                or draft.stack.vocab_size != geo.vocab_size):
            raise SnowLLMError("native V4.1 DSpark requires a V4.1 target with matching hidden size and vocabulary")
        dspark_gguf.taps_in(draft.tap_layers, geo.num_layers)
        return
    if draft.dspark and draft.vocab_size != model.config["vocab_size"]:
        raise SnowLLMError(f"DSpark vocabulary {draft.vocab_size} differs from the target's "
                           f"{model.config['vocab_size']}")
    if draft.hidden != geo.hidden:
        raise SnowLLMError(
            f"this DFlash draft reads target hidden states of width {draft.hidden}, "
            f"but the loaded model has width {geo.hidden}.")
    if not draft.num_target_layers:
        bad = [i for i in draft.tap_layers if not 0 <= i < geo.num_layers]
        if bad:
            raise SnowLLMError(
                f"this DFlash draft taps the target's hidden states at layers "
                f"{list(draft.tap_layers)}, and {bad} name no depth on a {geo.num_layers}-layer "
                f"model. Drop --dflash, or use --num-spec for the checkpoint's own MTP head.")
        return
    if geo.num_layers != draft.num_target_layers:
        raise SnowLLMError(
            f"this DFlash draft reads a {draft.num_target_layers}-layer target and the loaded "
            f"model has {geo.num_layers}. Its fc consumes the target's hidden states at layers "
            f"{list(draft.tap_layers)}, which name other depths on a model of this shape. Drop "
            f"--dflash, or use --num-spec for the checkpoint's own MTP head.")


def _dspark(geo: object) -> bool:
    from ..models.geometry import DSparkGeometry
    return isinstance(geo, DSparkGeometry)


def draft_block(geo: DraftGeometry, block: int) -> int:
    if _dspark(geo):
        if geo.stack.v41 and block not in (0, geo.block_size):
            raise SnowLLMError(f"native V4.1 DSpark requires its checkpoint block size {geo.block_size}")
        return max(block or 0, geo.block_size)
    if geo is not None and geo.dspark:
        if block not in (0, 1, 4, 8, 15) or block > geo.block_size:
            raise SnowLLMError("DSpark block must be 1, 4, 8 or 15 and fit the checkpoint")
        return block or geo.block_size
    return block or 8


@dataclass(frozen=True)
class DraftPlan:
    geo: DraftGeometry
    device_bytes: int = 0
    load_peak_bytes: int = 0
    workspace: DraftWorkspace | None = None
    host_load_peak_bytes: int = 0


def plan_draft(path: str | None, geo: DraftGeometry = None) -> DraftPlan:
    if not path:
        return DraftPlan(None)
    geo = _draft_geometry(path) if geo is None else geo
    weights = _dspark_weight_plan(path, geo) if _dspark(geo) else _dflash_weight_plan(path, geo)
    return DraftPlan(geo, *weights)


def draft_device_weight_bytes(path: str | None, geo: DraftGeometry = None) -> int:
    return plan_draft(path, geo).device_bytes


def draft_load_peak_bytes(path: str | None, geo: DraftGeometry = None) -> int:
    return plan_draft(path, geo).load_peak_bytes


def _dspark_weight_plan(path: str, geo: DraftGeometry) -> tuple[int, int, DraftWorkspace | None, int]:
    import pathlib

    from ..checkpoint.gguf.source import GGUFReader, find_dspark_gguf
    from ..models.deepseek_v4 import dspark

    p = pathlib.Path(path).expanduser()
    side = find_dspark_gguf(p) if p.is_dir() else (p if p.suffix == ".gguf" else None)
    if side is None:
        return 0, 0, None, 0
    with GGUFReader(side) as rd:
        return dspark.weight_plan(rd, geo)


def _draft_source(path: str, dry: bool = True) -> "DraftSource | None":
    import pathlib

    from ..checkpoint.gguf import dflash as dflash_gguf
    from ..checkpoint.reader import Shard
    from ..checkpoint.gguf.source import find_dspark_gguf

    p = pathlib.Path(path).expanduser()
    gguf = ((find_dspark_gguf(p) or dflash_gguf.find_gguf(p)) if p.is_dir()
            else (p if p.suffix == ".gguf" else None))
    if gguf is not None:
        return dflash_gguf.GGUFDraftSource(gguf)
    f = p / "model.safetensors" if p.is_dir() else p
    if not f.is_file():
        return None
    shard = Shard(f)
    if not dry:
        from safetensors.torch import load_file
        return dflash_gguf.DictDraftSource(load_file(str(f)))
    sd = {}
    for k in shard.meta:
        dtype, shape, _, _ = shard.spec(k)
        sd[k] = torch.empty(shape, dtype=dtype, device="meta")
    return dflash_gguf.DictDraftSource(sd)


def _dflash_weight_plan(path: str, geo: DFlashGeometry) -> tuple[int, int, DraftWorkspace | None]:
    from .. import ops
    from ..models.qwen3_5 import dflash_draft

    src = _draft_source(path)
    if src is None:
        return 0, 0, None
    none = torch.empty(0, dtype=torch.bfloat16, device="meta")
    try:
        with ops.dry_load() as d:
            draft = dflash_draft.load(src, geo, [(none, none)] * geo.num_layers,
                                      ops.Arena(), ops.KV_BLOCK_SIZES[0])
            sizes = d.device, d.peak_device
        return *sizes, dflash_draft.Workspace(draft)
    finally:
        src.close()


def draft_num_spec(geo: DraftGeometry, block: int) -> int:
    b = draft_block(geo, block)
    return b if _dspark(geo) or geo.dspark else b - 1


def draft_carve_out(path: str | None, block: int, max_num_seqs: int,
                    max_model_len: int) -> tuple[int, tuple[int, ...], int, DraftGeometry]:
    if not path:
        return 0, (), 0, None
    draft = plan_draft(path)
    return (draft.load_peak_bytes, draft.geo.tap_layers,
            draft_num_spec(draft.geo, block), draft.geo)


def _runner_for(model: "TargetModel") -> type[BaseRunner]:
    if isinstance(model.geo, Glm5NextGeometry):
        from .glm5_next_runner import Glm5NextRunner
        return Glm5NextRunner
    if isinstance(model.geo, DeepSeekV4Geometry):
        from .dsv4_runner import Dsv4Runner
        return Dsv4Runner
    if isinstance(model.geo, Qwen4ExpGeometry):
        from .qwen4exp_runner import Qwen4ExpRunner
        return Qwen4ExpRunner
    return QwenRunner


def carve_out_need(model: "TargetModel", cfg: dict, kv: dict, budget: int,
                   weights: int, *, load_peak: int = 0) -> int:
    ctx = int(kv.get("ctx") or 0)
    if ctx <= 0:
        return 0
    from ..context import ContextCapacity
    capacity = ContextCapacity.for_model(model, ctx)
    chunk = kv.get('chunk', 'auto')
    if chunk != 'auto':
        chunk = min(int(chunk), capacity.per_request)
    free, total = gpu_mem_info()
    resident = int(kv.get("draft_resident", kv.get("reserve")) or 0)
    available = int(total * float(kv.get("util") or DEFAULT_GPU_UTIL)) - (total - free)
    room = available - resident + int(kv.get("loaded_weight_bytes") or 0)
    r = _runner_for(model)(
        model, kv.get('blocks'), ops.kv_blocks_for(capacity.per_request, ops.KV_BLOCK_SIZES[0]), max(1, int(kv.get("slots") or 1)),
        max_prefill_tokens=chunk, num_spec=int(kv.get("num_spec") or 0),
        kv_int8=bool(kv.get("kv_int8")), gpu_memory_utilization=1.0, reserve_bytes=resident,
        load_peak_bytes=int(kv.get("reserve") or 0) if "draft_resident" in kv else 0,
        tap_layers=tuple(kv.get("taps") or ()), draft_geo=kv.get("draft_geo"),
        prefix_memory_ratio=float(kv.get("prefix_ratio") or 0.0), plan_activations=True,
        draft_workspace=kv.get("draft_workspace"), context_capacity=capacity)
    loading = 0 if kv.get("loaded_weight_bytes") is not None else max(0, load_peak - available)
    from .capacity import auxiliary_bytes
    aux = auxiliary_bytes(r, kv)
    got = loading
    for rows in r.candidates():
        got = max(loading, weights + r.minimum_bytes(rows) + r.act_bytes_for(rows) + aux - room)
        if got <= budget:
            break
    return got


class Engine:
    def __init__(self, model: torch.nn.Module, num_kv_blocks: int | None = None,
                 max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                 max_model_len: int = 8192, stop_token_ids: Sequence[int] = (),
                 seed: int | None = None, enforce_eager: bool = False, num_spec: int = 0,
                 kv_int8: bool = False, mtp_window: int = 0,
                 mtp_sinks: int = 64,
                 dflash_path: str | None = None, dflash_block: int = 0,
                 dflash_p_min: float = DEFAULT_DSPARK_P_MIN,
                 dspark_adaptive: bool = False,
                 prefill_chunk: int | str = DEFAULT_PREFILL_CHUNK, batch_prefill: bool = False,
                 plan_activations: bool = True,
                 account: bool = True, exact_stats: bool = False,
                 gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
                 preempt: bool = True, prefix_memory_ratio: float = 0.0,
                 draft_plan: DraftPlan | None = None) -> None:
        draft_plan = plan_draft(dflash_path) if draft_plan is None else draft_plan
        self.dflash_geo = draft_plan.geo
        if dspark_adaptive and not (_dspark(self.dflash_geo)
                                    or getattr(self.dflash_geo, 'dspark', False)):
            raise ValueError('--dspark-adaptive requires a DSpark checkpoint')
        if dspark_adaptive and dflash_p_min != 0:
            raise ValueError('--dspark-adaptive requires --dflash-p-min 0')
        if dflash_path:
            _check_dflash_target(model, self.dflash_geo)
        from ..context import ContextCapacity
        capacity = ContextCapacity.for_model(model, max_model_len)
        self.kv_capacity_tokens = capacity.tokens
        self.model_context_limit = capacity.model_limit
        max_model_len = capacity.per_request
        self.max_blocks = ops.kv_blocks_for(max_model_len, ops.KV_BLOCK_SIZES[0])
        mpt = "auto" if prefill_chunk == "auto" else min(int(prefill_chunk), max_model_len)
        self.dflash_block = dflash_block = draft_block(self.dflash_geo, dflash_block)
        cls = _runner_for(model)
        draft_peak = draft_plan.load_peak_bytes
        draft_resident = (draft_plan.device_bytes
                          if _dspark(self.dflash_geo) else draft_peak)
        self.runner = cls(
                             model, num_kv_blocks, self.max_blocks, max_num_seqs,
                             max_prefill_tokens=mpt,
                             num_spec=(self._draft_n() if dflash_path else num_spec),
                             kv_int8=kv_int8,
                             gpu_memory_utilization=gpu_memory_utilization,
                             reserve_bytes=draft_resident,
                             load_peak_bytes=draft_peak if _dspark(self.dflash_geo) else 0,
                             tap_layers=(self.dflash_geo.tap_layers if dflash_path
                                         else ()),
                             draft_geo=self.dflash_geo,
                             draft_workspace=draft_plan.workspace,
                             prefix_memory_ratio=prefix_memory_ratio,
                             plan_activations=plan_activations, context_capacity=capacity)
        try:
            self.prefill_chunk = self.runner.max_prefill_tokens
            self.dflash_blocks = self.runner.draft_blocks
            self.ring_blocks = self.runner.ring_blocks

            self.num_spec = self.runner.num_spec
            self.T = self.num_spec + 1
            self.max_num_seqs = max_num_seqs
            self.max_model_len = max_model_len

            self.slots = StateSlots(max_num_seqs, self.runner.slots_per_request,
                                    self.runner.dummy_slot)
            self.sampler = Sampler(frozenset(stop_token_ids),seed,
                                   native_rng=isinstance(model.geo, DeepSeekV4Geometry) and model.geo.v41)
            self.sampler.reserve(max_num_seqs*self.T,self.runner.geo.vocab_size,
                                 max_tokens=max_model_len+self.T,
                                 max_requests=max_num_seqs,arena=self.runner.arena)
            self.tables, self.stage = self.runner.create_inputs()
            self.dflash = None
            if dflash_path and _dspark(self.dflash_geo):
                self.dflash = self._build_dspark(dflash_path, dflash_block, max_model_len,
                                                 dflash_p_min, dspark_adaptive)
            elif dflash_path:
                self.dflash = self._build_dflash(dflash_path, dflash_block, max_model_len,
                                                dflash_p_min, dspark_adaptive)
            if self.dflash is not None:
                model._placement_host_reserve = 0
                self.runner.load_peak_bytes = 0
                self.runner.reserve_bytes = (self.dflash.workspace_bytes()
                                             if isinstance(self.dflash, DFlashDecoder) else 0)
            self.spec = self.dflash or (
                getattr(self.runner,"spec_decoder_type",SpecDecoder)(self.runner, self.sampler, self.slots, self.tables, self.stage,
                            num_spec=self.num_spec, window=mtp_window, sinks=mtp_sinks)
                if self.num_spec else None)

            self.waiting: deque[Request] = deque()
            self.running: list[Request] = []
            self._prefill_turn = True
            self.batch_prefill = batch_prefill
            self.n_batched_prefills = 0
            self.n_preemptions = 0
            self.preempt = preempt
            self.cache = None
            self.acct = StepAccounting() if account else None
            self.exact_stats = exact_stats

            self._active_factor: float | None = None
            self._loaded_rope_factor: float | None = None
            self._rope_cache: dict[float, tuple] = {}
            self._rope_orig_max = capacity.original
            self.default_rope_factor = capacity.rope_factor if self.runner.auto_yarn else 1.0

            self.runner.resources.add(lambda slabs: None, [(self, ('dflash', 'spec', 'sampler',
                'cache', 'tables', 'stage'))]).allocate()
            if self.runner.auto_yarn:
                self._activate_rope(self.default_rope_factor)

            shapes = (self.spec.decode_shapes(self.runner.max_num_seqs) if self.spec is not None
                      else plain_decode_shapes(self.runner.max_num_seqs))
            each, n_shapes = (0, 0) if enforce_eager else self.runner.probe_graph_bytes(shapes)
            if self.dflash is not None:
                self._draft_pool.release()
            num_kv_blocks = self.runner.finish_pools(each * n_shapes)
            self.dflash_blocks = self.runner.draft_blocks
            if self.dflash is not None:
                if not self.dflash.cache_window:
                    self.dflash_blocks = max(self.dflash_blocks, self.dflash.max_blocks_per_seq)
                self._retake_draft(self.dflash_blocks)
            need = min(self.max_blocks, self.ring_blocks) if self.ring_blocks else self.max_blocks
            if num_kv_blocks < need:
                raise SnowLLMError(
                    f"the KV pool holds {num_kv_blocks} blocks but one sequence needs {need}. A "
                    f"request could be admitted and then never finish, so this is refused here: "
                    + (f"lower max_num_seqs, or lower --max-num-batched-tokens to shrink the window "
                       f"each request holds."
                       if self.ring_blocks else
                       f"--max-model-len {num_kv_blocks * self.runner.block_size} is what this pool "
                       f"holds at max_model_len={max_model_len}, or lower max_num_seqs to leave it "
                       f"more room."))
            self.blocks = BlockAllocator(num_kv_blocks, self.runner.block_size, self.ring_blocks)
            if self.runner.prefix_slots:
                store = self.runner.prefix_store(
                    self.blocks, self.dflash.blocks if self.dflash is not None else None,
                    lookahead=getattr(self.spec, "prefix_lookahead", 0))
                if self.ring_blocks and store.pages.is_ring:
                    raise SnowLLMError(
                        "prefix caching cannot share a KV axis with a ring: a cached prefix names "
                        "pages that the request which wrote them has since reused. Pass "
                        "prefix_memory_ratio=0.")
                self.cache = PrefixCache(store)
            if dspark_adaptive and _dspark(self.dflash_geo):
                from .dspark_profile import profile
                # Reconcile measured startup memory before graphs retain weight addresses.
                profile(self.dflash, warmup=True)
                self.runner.ensure_capacity()
            self.graph_sizes = []
            if not enforce_eager:
                self.graph_sizes = self.runner.capture(shapes)
            if self.dflash and not enforce_eager:
                self.dflash.capture(self.runner)
            if dspark_adaptive:
                from .dspark_profile import profile
                profile(self.dflash)

        except BaseException:
            self.runner._discard_pools()
            raise

    def _retake_draft(self, blocks: int) -> None:
        self._draft_pool.allocate(blocks)
        self.dflash.blocks.widen(blocks)

    def _draft_n(self) -> int:
        return draft_num_spec(self.dflash_geo, self.dflash_block)

    def _dflash_pool_bytes(self, blocks: int) -> int:
        return draft_pool_bytes(self.dflash_geo, blocks, self.runner.block_size)

    def _build_dspark(self, path: str, block: int, max_model_len: int,
                      p_min: float = 0.0, adaptive: bool = False) -> DSparkDecoder:
        import pathlib

        from ..checkpoint.gguf.source import GGUFReader, find_dspark_gguf
        from ..models.deepseek_v4 import dspark
        from .block_manager import BlockAllocator
        from .dspark_decode import DSparkDecoder

        p = pathlib.Path(path).expanduser()
        f = find_dspark_gguf(p) if p.is_dir() else p
        with GGUFReader(f) as rd:
            draft = dspark.load(rd, self.dflash_geo, self.runner.model, self.runner.walker)
        release_device_cache()
        need = self.dflash_blocks
        def allocate(slabs, blocks):
            draft.cache = allocate_draft_pool(self.dflash_geo, blocks, self.runner.block_size, slabs)
            return draft.cache
        self._draft_pool = self.runner.resources.add(allocate, [(draft, ('cache',))])
        self._draft_pool.allocate(need)
        return DSparkDecoder(self.runner, self.sampler, self.slots, draft,
                             BlockAllocator(need, self.runner.block_size), self.tables, self.stage,
                             max_blocks_per_seq=draft_table_blocks(
                                 self.dflash_geo, max_model_len, block, self.runner.block_size),
                             block=block, p_min=p_min, adaptive=adaptive)

    def _build_dflash(self, path: str, block: int, max_model_len: int,
                      p_min: float = 0.0, adaptive: bool = False) -> DFlashDecoder:
        from ..models.qwen3_5 import dflash_draft
        from .block_manager import BlockAllocator

        src = _draft_source(path, dry=False)
        if src is None:
            raise ValueError(f"no DFlash draft at {path}: expected a .gguf or .safetensors file, "
                             f"or a directory holding one")
        g = self.dflash_geo
        per_seq = ops.kv_blocks_for(max_model_len + block, self.runner.block_size)
        need = self.dflash_blocks
        page = self.runner.block_size
        def allocate(slabs, blocks):
            pools = allocate_draft_pool(g, blocks, page, slabs)
            if self.dflash is not None:
                self.dflash.draft.pools = pools
            return pools
        self._draft_pool = self.runner.resources.add(allocate)
        pools = self._draft_pool.allocate(need)
        try:
            draft = dflash_draft.load(src, g, pools, self.runner.arena, self.runner.block_size)
        finally:
            src.close()
        self._draft_pool.bindings = ((draft, ('pools',)),)
        return DFlashDecoder(self.runner, self.sampler, self.slots, draft,
                             BlockAllocator(need, self.runner.block_size), self.tables, self.stage,
                             block=block,
                             max_blocks_per_seq=per_seq, p_min=p_min, adaptive=adaptive)

    def add(self, prompt: list[int], params: SamplingParams | None = None,
            rope_factor: float | None = None, mrope: torch.Tensor | None = None, pos_delta: int = 0,
            embeds: torch.Tensor | None = None,
            embed_rows: list[int] | None = None, queue: bool = True,
            system_prefix_len: int = 0) -> Request:
        p = params or SamplingParams()
        rope_factor = self.default_rope_factor if rope_factor is None else rope_factor
        cap = min(self.max_model_len, int(rope_factor * self._rope_orig_max))
        if len(prompt) + p.max_new_tokens > cap:
            raise SnowLLMError(f"prompt {len(prompt)} + {p.max_new_tokens} new > {cap} "
                               f"(factor {rope_factor}, max_model_len {self.max_model_len})")
        if not 0 <= system_prefix_len <= len(prompt):
            raise ValueError('system_prefix_len must be within the prompt')
        r = Request(prompt=list(prompt), params=p, rope_factor=rope_factor, mrope=mrope,
                    pos_delta=pos_delta, embeds=embeds, embed_rows=list(embed_rows or []),
                    system_prefix_len=system_prefix_len)
        if queue:
            self.waiting.append(r)
        return r

    def step(self) -> str:
        if self.acct is None:
            return self._step()[0]
        t0 = time.perf_counter()
        kind, tokens, rows = self._step()
        if kind == "idle":
            return kind
        if self.exact_stats:
            torch.cuda.synchronize()
        self.acct.add(kind, tokens, time.perf_counter() - t0, rows)
        return kind

    def _step(self) -> tuple[str, int, int]:
        self._retire_finished()
        if not self.running:
            self._active_factor = None

        pending = self._pending_prefill()
        if pending is None:
            group = self._batchable_prefill_group()
            if group:
                self._prefill_turn = False
                started = time.perf_counter()
                self._prefill_batch(group)
                elapsed = time.perf_counter() - started
                for r in group:
                    r.timing.add("prefill", len(r.prompt), elapsed)
                return "prefill", sum(len(r.prompt) for r in group), 0
            if len(self.running) < self.max_num_seqs:
                cand = self._next_admissible()
                if cand is not None and self._admit(cand):
                    self.waiting.remove(cand)
                    self.running.append(cand)
                    pending = self._pending_prefill()
                elif cand is not None and not self.running:
                    raise SnowLLMError(
                        f"this engine cannot admit a {cand.prefill_len}-token prompt with nothing "
                        f"else running and the prefix cache already evicted: the pools have "
                        f"{len(self.blocks.free)} of {self.blocks.total} KV blocks free. It was "
                        f"built for max_model_len={self.max_model_len} at "
                        f"max_num_seqs={self.max_num_seqs}, so this is a sizing bug rather than "
                        f"a load one -- it would not have got smaller by waiting.")

        decodable = [r for r in self.running if r.prefilled]
        if pending is not None and (not decodable or pending.num_prefilled == 0
                                    or self._prefill_turn):
            self._prefill_turn = False
            before = pending.num_prefilled
            started = time.perf_counter()
            self._prefill_chunk(pending)
            pending.timing.add("prefill", pending.num_prefilled - before,
                               time.perf_counter() - started)
            return "prefill", pending.num_prefilled - before, 0
        if decodable:
            self._prefill_turn = True
            before = [(r, len(r.out)) for r in decodable]
            started = time.perf_counter()
            emitted, rows = self._decode_or_verify(len(decodable))
            elapsed = time.perf_counter() - started
            for r, count in before:
                r.timing.add("decode", len(r.out) - count, elapsed)
            return "decode", emitted, rows
        return "idle", 0, 0

    def run(self) -> None:
        while self.waiting or self.running:
            self.step()

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return self.sampler.stop_token_ids

    def stats(self) -> EngineStats:
        r, c = self.runner, self.cache
        return EngineStats(
            running=len(self.running), waiting=len(self.waiting),
            free_slots=len(self.slots.free), free_kv_blocks=len(self.blocks.free),
            preemptions=self.n_preemptions,
            cache_hits=c.hits if c else 0, cache_misses=c.misses if c else 0,
            cached_prefixes=len(c.entries) if c else 0,
            cached_blocks=c.held_blocks() if c else 0,
            prefill_tokens_saved=c.saved_tokens if c else 0,
            graph_sizes=self.graph_sizes,
            graph_replays=r.n_graph_replays, eager_forwards=r.n_eager_forwards,
            draft_replays=r.n_draft_replays, draft_eager=r.n_draft_eager,
            accounting=self.acct.snapshot() if self.acct else {})

    def _alloc_blocks(self, n: int, keep: Entry | None = None) -> list[int] | None:
        while True:
            got = self.blocks.alloc(n)
            if got is not None or self.cache is None or not self.cache.evict(keep):
                return got

    def _reserve_pools(self, r: Request, covered: int, keep: Entry | None = None) -> bool:
        while not self.runner.admit(r, r.prefill_len, covered):
            if self.cache is None or not self.cache.evict(keep):
                return False
        return True

    def _admit(self, r: Request, use_cache: bool = True) -> bool:
        r.system_entry = r.prefix_entry = None
        if self._active_factor != r.rope_factor:
            self._activate_rope(r.rope_factor)
            self._active_factor = r.rope_factor
        hit = (self.cache.lookup(r.prefill_src, r.prefill_len)
               if self.cache is not None and use_cache and r.embeds is None else None)
        r.cached_tokens = len(hit.tokens) if hit is not None else 0
        ring, pools = self.cache.pages.covered(hit) if hit is not None else (0, 0)
        blocks = self._alloc_blocks(
            ops.kv_blocks_for(r.prefill_len, self.runner.block_size)
            - ops.kv_blocks_for(ring, self.runner.block_size), hit)
        if blocks is None:
            return False
        self.slots.take(r)
        if not self._reserve_pools(r, pools, hit):
            self.slots.give_back(r)
            self.blocks.release(blocks)
            return self._admit(r, use_cache=False) if hit is not None else False
        r.blocks = self.cache.pages.own(self.blocks, hit.blocks, blocks, r) \
            if hit is not None else blocks
        if hit is not None:
            r.num_prefilled = len(hit.tokens)
        if self.dflash is not None and not self._reserve_draft(
                r, r.prefill_len - r.num_prefilled, hit):
            self.blocks.release(r.blocks)
            self.dflash.release(r)
            self.runner.release(r.slot)
            self.slots.give_back(r)
            r.blocks, r.num_prefilled, r.cached_tokens = [], 0, 0
            return False
        if self.cache is not None and use_cache:
            self.cache.took(hit)
            system_end = self._system_cache_end(r)
            system = self.cache.lookup(r.prefill_src, system_end + 1) if system_end else None
            r.system_entry = system if system is not None and len(system.tokens) == system_end else None
            if r.system_entry is not None:
                r.system_entry.priority = True
            r.prefix_entry = hit if hit is not None and not hit.priority else None
        if hit is not None:
            self.cache.store.residue.load(hit.ckpt, self.slots.name(r, 1)[0], r.blocks,
                                         self.cache.pages.target(hit.blocks))
        return True

    def _retire_finished(self) -> None:
        for r in [r for r in self.running if r.done]:
            self.blocks.release(r.blocks)
            if self.dflash:
                self.dflash.release(r)
            self.runner.release(r.slot)
            self.slots.give_back(r)
            self.running.remove(r)

    def _next_admissible(self) -> Request | None:
        return next((r for r in self.waiting
                     if self._active_factor is None or r.rope_factor == self._active_factor), None)

    def _pending_prefill(self) -> Request | None:
        return next((r for r in self.running if not r.prefilled and not r.done), None)

    def _decodable_batch(self, rows: int | dict[Request, int]) -> list[Request]:
        while True:
            decodable = [r for r in self.running if r.prefilled]
            batch, short = [], ""
            for r in decodable:
                n = rows[r] if isinstance(rows, dict) else rows
                if not self.blocks.grow(r, n):
                    short = short or f"the KV pool, which has {self.blocks.total} blocks,"
                elif not self.runner.grow(r, n):
                    short = short or "the pools this model pages itself"
                elif self.dflash is not None and not self.dflash.grow(r, n):
                    short = short or "the drafter's own KV pool"
                else:
                    batch.append(r)
            if not short:
                return batch
            if self.cache is not None and self.cache.evict():
                continue
            if not self.preempt:
                raise SnowLLMError(
                    f"{short} ran out mid-decode over {len(decodable)} requests and this engine "
                    f"was built with preempt=False, which is a promise that it would not have to: "
                    f"it is too small for this workload")
            if self._preempt() is None:
                raise SnowLLMError(
                    f"{short} ran out mid-decode with one request running and nothing left to "
                    f"preempt: it cannot carry that request to max_model_len="
                    f"{self.max_model_len}")

    def _preempt(self) -> Request | None:
        if len(self.running) < 2:
            return None
        r = self.running.pop()
        self.blocks.release(r.blocks)
        if self.dflash:
            self.dflash.release(r)
        self.runner.release(r.slot)
        self.slots.give_back(r)
        r.blocks = []
        r.slot = r.state_head = -1
        r.replay = len(r.tokens) - 1 if r.out else 0
        r.num_prefilled = 0
        r.drafts.clear()
        r.n_accepted = 1
        self.waiting.appendleft(r)
        self.n_preemptions += 1
        return r

    def _rope_table(self, factor: float) -> tuple[torch.Tensor, float]:
        if factor not in self._rope_cache:
            from ..checkpoint import loader
            inv, mscale = loader.yarn_rope_table(self.runner.model.config, factor,
                                                 orig_max_pos=self._rope_orig_max)
            self._rope_cache[factor] = (inv.cuda(), mscale)
        return self._rope_cache[factor]

    def _activate_rope(self, factor: float) -> None:
        if not self.runner.tunable_rope:
            if factor != 1.0:
                raise SnowLLMError(
                    f"rope_factor {factor} was asked for and this model's context cannot be "
                    f"extended that way: {type(self.runner.model).__name__} builds its rope tables "
                    f"at load and the engine has no handle on them. Serve it without a =FACTOR.")
            return
        self.runner.set_rope(*self._rope_table(factor))
        if self._loaded_rope_factor != factor and self.cache is not None:
            while self.cache.evict():
                pass
        self._loaded_rope_factor = factor

    def _prefill(self, spans: list[tuple[Request, int, int]]) -> Batch:
        return self.runner.prefill_batch(spans, self.slots)

    def _reserve_draft(self, r: Request, rows: int, keep: Entry | None = None) -> bool:
        while not self.dflash.grow(r, rows):
            if self.cache is None or not self.cache.evict(keep):
                return False
        return True

    def _system_cache_end(self, r: Request) -> int:
        if self.cache is None or r.embeds is not None:
            return 0
        store = self.cache.store
        # Keep shifted MTP dependencies inside the shared system prefix.
        end = max(0, r.system_prefix_len - store.lookahead)
        return end // store.pages.align * store.pages.align

    def _prefill_chunk(self, r: Request) -> None:
        lo = r.num_prefilled
        hi = min(lo + self.prefill_chunk, r.prefill_len)
        system_end = self._system_cache_end(r)
        if lo < system_end < hi:
            hi = system_end
        if self.cache is not None and r.embeds is None:
            hi = self.cache.store.prefill_end(lo, hi)
            aligned = hi // self.cache.pages.align * self.cache.pages.align
            if aligned > lo:
                hi = aligned
        marks, idx = [], []
        latest_end = 0
        backed = self.dflash is None or self._reserve_draft(r, hi - lo)
        if self.cache is not None and backed and r.embeds is None:
            marks = self.cache.store.marks(lo, hi)
            keep = {entry for other in self.running if other is not r
                    for entry in (other.system_entry, other.prefix_entry) if entry is not None}
            if r.system_entry is not None:
                keep.add(r.system_entry)
            priority = self.cache.reserve(1, priority=True, keep=keep) if lo < system_end == hi else []
            if priority:
                marks = [m for m in marks if m != system_end]
            latest_end = hi if hi % self.cache.pages.align == 0 and hi != system_end else 0
            latest = (self.cache.reserve(1, latest=True, replace=r.prefix_entry, keep=keep)
                      if latest_end else [])
            if latest:
                marks = [m for m in marks if m != latest_end]
            idx = self.cache.reserve(len(marks), keep=keep)
            pairs = list(zip(thin(marks, len(idx)), idx))
            if priority:
                pairs.append((system_end, priority[0]))
            if latest:
                pairs.append((latest_end, latest[0]))
            pairs.sort()
            marks, idx = [p[0] for p in pairs], [p[1] for p in pairs]
        b = self._prefill([(r, lo, hi)])
        if marks:
            self.cache.store.residue.arm(b, marks, idx, lo)
        M = b.input_ids.numel()

        out = self.runner.forward(b)
        r.num_prefilled = hi

        def publish() -> None:
            # Draft caches must be complete before their shared pages are visible.
            for m, i in zip(marks, idx):
                entry = self.cache.insert(r.prefill_src[:m],
                                  self.cache.store.took(r, m, i, self.slots.name(r, 1)[0]), i,
                                  priority=m == system_end, latest=m == latest_end)
                if entry is not None:
                    if m == system_end:
                        r.system_entry = entry
                    elif m == latest_end:
                        if r.prefix_entry is not None and r.prefix_entry is not entry:
                            r.prefix_entry.latest = False
                        r.prefix_entry = entry

        if not b.need_logits:
            if self.dflash:
                self.dflash.after_prefill(r, lo, hi, M, propose=False)
            elif self.spec:
                self.spec.propose_after_prefill(r, b, lo, hi, M)
            publish()
            return
        self.sampler.emit(out, [r])
        if self.dflash and not r.done:
            self.dflash.after_prefill(r, lo, hi, M, propose=True)
            r.n_accepted = 1
            publish()
            self._retire_finished()
            return
        if self.spec and not r.done:
            logits, hid = self.spec.propose_after_prefill(r, b, lo, hi, M, r.out[-1])
            r.drafts = self.sampler.greedy_list(logits)
            r.n_accepted = 1
            if (self.num_spec > 1 and r.num_cached + self.num_spec <= self.max_model_len
                    and self.blocks.grow(r, self.num_spec)):
                self.spec.propose_rest([r], hid, self.num_spec)
        publish()
        self._retire_finished()

    def _batchable_prefill_group(self) -> list[Request]:
        if not self.batch_prefill or self.num_spec:
            return []
        cands = [r for r in self.waiting
                 if r.num_prefilled == 0 and not r.replay and len(r.prompt) <= self.prefill_chunk
                 and not (self.cache is not None and r.system_prefix_len)
                 and (self._active_factor is None or r.rope_factor == self._active_factor)]
        if len(cands) < 2:
            return []
        f = cands[0].rope_factor
        group, tokens = [], 0
        for r in (c for c in cands if c.rope_factor == f):
            if len(self.running) >= self.max_num_seqs:
                break
            if tokens + len(r.prompt) > self.runner.max_prefill_tokens:
                break
            if not self._admit(r, use_cache=False):
                break
            self.waiting.remove(r)
            self.running.append(r)
            group.append(r)
            tokens += len(r.prompt)
        return group

    def _prefill_batch(self, batch: list[Request]) -> None:
        out = self.runner.forward(self._prefill([(r, 0, len(r.prompt)) for r in batch]))
        self.n_batched_prefills += 1
        for r in batch:
            r.num_prefilled = len(r.prompt)
        self.sampler.emit(out, batch)
        self._retire_finished()

    def _decode_or_verify(self, n: int) -> tuple[int, int]:
        if isinstance(self.spec, DSparkScheduling) and self.spec.adaptive and n <= SPEC_MAX_STEP_ROWS:
            while True:
                batch = [r for r in self.running if r.prefilled]
                if not batch:
                    return 0, 0
                lengths = self.spec.select_lengths(batch, self.max_model_len, remember=False)
                if self._decodable_batch(dict(zip(batch, lengths))) == batch:
                    break
            T = self.spec.select_lengths(batch, self.max_model_len)
        else:
            T = self.spec.verify_rows(n) if self.spec else 0
            if T and any(r.num_cached + self.spec.rows_needed(T) > self.max_model_len
                         for r in self.running if r.prefilled):
                T = 0
            if T:
                batch = self._decodable_batch(self.spec.rows_needed(T))
        if T:
            self.spec.verify_step(batch, T)
            self.runner.commit(batch)
            emitted, rows = sum(r.n_accepted for r in batch), len(batch)
        else:
            emitted = rows = self._decode()
        self._retire_finished()
        return emitted, rows

    def _decode(self) -> int:
        batch = self._decodable_batch(1)
        p = [r.num_cached for r in batch]
        k, bs = self.runner.prev_context, self.runner.block_size
        ids = [r.last for r in batch]
        d_ids, pos, slots, seq, sidx = self.stage(
            (ids, [q + r.pos_delta for q, r in zip(p, batch)] * 3),
            ([r.blocks[q // bs] * bs + q % bs for q, r in zip(p, batch)], [q + 1 for q in p],
             [self.slots.name(r, 1)[0] for r in batch]))
        b = Batch(
            input_ids=d_ids, positions=pos.view(3, -1), slot_mapping=slots,
            block_tables=self.tables([r.blocks for r in batch]),
            seq_lens=seq, host_seq_lens=[q + 1 for q in p],
            is_prefill=False, num_tokens=len(batch),
            state_indices=sidx,
            prev_ids=[r.tail(k + 1)[:-1] for r in batch] if k else None,
            host_ids=ids,
        )
        self.sampler.emit(self.runner.forward(b), batch)
        if self.dflash:
            self.dflash.after_decode(batch, p)
        elif self.spec and hasattr(self.spec,"after_decode"):
            self.spec.after_decode(batch,p)
        for r in batch:
            r.drafts.clear()
        return len(batch)

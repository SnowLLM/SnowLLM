# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import gc
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, NamedTuple, NoReturn

import torch

from ..defaults import DEFAULT_GPU_UTIL, DEFAULT_MAX_NUM_SEQS, MAX_NUM_SEQS, DEFAULT_PREFILL_CHUNK

from .. import ops, term
from .._capi import SnowLLMError
from .._platform import gpu_mem_info, prepare_gpu_commit, release_device_cache, retain_device_cache
from ..ops._common import dry_alloc
from ..models.geometry import DFlashGeometry, DSparkGeometry
from ..trace import span
from .host_io import SchedulerInputs
from .graphs import CapturedGraph, GraphBudget, GraphInputs, capture_graph
from .inputs import prefill_batch
from .resources import DraftWorkspace, ProbeGroup, plan_activations
from .forward_context import (
    Batch,
    DecodeShape,
)


class CapacityError(SnowLLMError):
    pass


if TYPE_CHECKING:
    from .block_manager import BlockAllocator
    from .prefix_cache import Pages, PrefixStore, Residue
    from .request import Request

DraftGeometry = DFlashGeometry | DSparkGeometry | None


class ForwardResult(NamedTuple):
    output: torch.Tensor
    hidden: torch.Tensor


class BaseRunner:
    host_messages = True
    text_only = False
    defer_model_end = False
    inputs = None

    max_num_seqs: int
    gpu_util: float

    def __init__(self, model, num_kv_blocks, max_blocks_per_seq,
                 max_num_seqs=DEFAULT_MAX_NUM_SEQS, max_prefill_tokens=DEFAULT_PREFILL_CHUNK, num_spec=0,
                 kv_int8=False, block_size=None, gpu_memory_utilization=DEFAULT_GPU_UTIL,
                 reserve_bytes=0, tap_layers=(), draft_geo=None, prefix_memory_ratio=0.,
                 plan_activations=True, load_peak_bytes=0, draft_workspace: DraftWorkspace | None = None,
                 context_capacity=None):
        self._check_seqs(max_num_seqs)
        if num_spec < 0:
            raise SnowLLMError('num_spec must be nonnegative')
        self.model, self.geo = model, model.geo
        self.max_num_seqs, self.max_blocks_per_seq = max_num_seqs, max_blocks_per_seq
        self.block_size = self._pick_block_size(block_size)
        self.ctx_cap = max_blocks_per_seq * self.block_size
        self.context_capacity = context_capacity
        self.total_kv_blocks = (max_blocks_per_seq if context_capacity is None else
                                ops.kv_blocks_for(context_capacity.tokens, self.block_size))
        if context_capacity is not None:
            self.auto_yarn = self.tunable_rope and context_capacity.auto_yarn
        self.num_spec = num_spec
        self.kv_int8, self.plan_activations = kv_int8, plan_activations
        self.gpu_util, self.reserve_bytes = gpu_memory_utilization, int(reserve_bytes)
        from ..host_alloc import enabled
        if enabled():
            self.gpu_util = 1.
        self.load_peak_bytes = int(load_peak_bytes)
        self.prefix_memory_ratio, self.prefix_slots = prefix_memory_ratio, 0
        self.tap_layers, self.draft_geo = tuple(tap_layers), draft_geo
        self.draft_workspace = draft_workspace
        self.num_taps = len(self.tap_layers)
        self._want_blocks = None if num_kv_blocks is None else int(num_kv_blocks)
        self._want_chunk, self._act = max_prefill_tokens, {}
        self._configure()
        self.T = self.num_spec + 1
        self.dummy_slot = max_num_seqs * self.slots_per_request
        self._init_graphs()
        previous = getattr(model, "runtime_resources", None)
        if previous is not None:
            previous.close()
            model.runtime_resources = model.kv_slabs = None
            del previous
            gc.collect()
            release_device_cache()
        try:
            self._initialize()
        except BaseException:
            self._discard_pools()
            raise

    def _initialize(self):
        self._setup_resources()
        self.model.runtime_resources = self.resources
        self.resources.add(lambda slabs: self,
            [(self, ('graphs', 'last_hidden', 'inputs', '_pool'))],
            dispose=lambda r: r.arena.release() if r.arena is not None else None).allocate()
        self.slabs = self.model.kv_slabs = self.resources
        self._allow = self._allowance()
        persistent = sum(part.bytes_for() for part in self.resources.support)
        if ops.drying() is None:
            self._persistent_capacity(persistent)
        for part in self.resources.support:
            part.allocate()
        chunk = self._check_chunk(self.candidates()[0])
        self.max_prefill_tokens = chunk
        with dry_alloc(), self.resources.planning():
            self.resources.prepare(chunk)
        self.resources.prepare(chunk)
        if self.arena is not None:
            self._reserve_activations(chunk)
        else:
            self.act_bytes = self.act_bytes_for(chunk)
        plan = self._pool_plan(0)
        plan.future_bytes += self._loading_extra(chunk)
        if ops.drying() is None:
            self._pool_capacity(plan)
        self._allocate_pools(self.resources.probe_units())

    def create_inputs(self):
        rows = max(self.max_prefill_tokens, self.max_num_seqs*self.T)
        capacity = 8*rows + 4*self.max_num_seqs
        if not self.host_messages:
            from .block_manager import Staging, TableMirror
            self.inputs = None
            stage = Staging()
            stage.reserve(capacity)
            return TableMirror(self.max_num_seqs, self.max_blocks_per_seq), stage
        self.inputs = SchedulerInputs(self.max_num_seqs, self.max_blocks_per_seq)
        self.inputs.reserve('stage64', capacity)
        self.inputs.reserve('stage32', capacity, torch.int32)
        self.inputs.reserve('prefill_qmap', 2*rows, torch.int32)
        return self.inputs.tables, self.inputs

    def _input_scope(self):
        return self.inputs.reading() if self.inputs is not None else contextlib.nullcontext()

    def prefill_batch(self, spans, slots):
        if self.text_only and any(r.embeds is not None or r.mrope is not None for r, _, _ in spans):
            raise SnowLLMError(f"{type(self).__name__} accepts text inputs only")
        return prefill_batch(spans, slots, self.block_size, self.inputs, self.prev_context)

    def _init_stats(self) -> None:
        self.n_graph_replays = 0
        self.n_eager_forwards = 0
        self.n_draft_replays = 0
        self.n_draft_eager = 0

    def _discard_pools(self):
        resources = getattr(self, "resources", None)
        if resources is not None:
            resources.close()

    def capture(self, shapes: Sequence[DecodeShape]) -> list[int]:
        return []

    def probe_graph_bytes(self, shapes: Sequence[DecodeShape]) -> tuple[int, int]:
        return 0, 0

    def admit(self, r: "Request", length: int, covered: int = 0) -> bool:
        return True

    def grow(self, r: "Request", n: int = 1) -> bool:
        return True

    def release(self, slot: int) -> None:
        pass

    def commit(self, batch: "list[Request]") -> None:
        pass

    def prefix_store(self, alloc: "BlockAllocator",
                     draft: "BlockAllocator | None" = None, *,
                     lookahead: int = 0) -> "PrefixStore":
        from .prefix_cache import PrefixStore, WithDraft
        pages = self.prefix_pages(alloc)
        if draft is not None:
            from .resources import draft_cache_window
            pages = WithDraft(pages, draft, self.block_size, draft_cache_window(self.draft_geo))
        return PrefixStore(pages, self.prefix_residue(), self.prefix_slots, lookahead=lookahead)

    def prefix_pages(self, alloc: "BlockAllocator") -> "Pages":
        from .prefix_cache import BlockPages
        return BlockPages(alloc, self.block_size)

    def prefix_residue(self) -> "Residue":
        from .prefix_cache import NoResidue
        return NoResidue()

    def _allowance(self, util: float | None = None) -> int:
        free, total = gpu_mem_info()
        return (int(total * (self.gpu_util if util is None else util)) - (total - free)
                - self.reserve_bytes)

    def _persistent_capacity(self, required):
        from ..models.placement import ensure_room
        while required > self._allow:
            if not ensure_room(self.model, required - self._allow):
                from ..memory_report import current
                if current() is not None:
                    current().shortfall(required - self._allow)
                raise SnowLLMError(f'persistent state and workspace need {required/(1<<30):.2f} GiB; '
                                   'lower --max-num-seqs/--num-spec or offload more weights')
            self._allow = self._allowance()

    def ensure_capacity(self):
        from ..capacity import memory_shortage
        from ..models.placement import ensure_room
        release_device_cache()
        shortage = -self._allowance()
        while shortage > 0:
            if not ensure_room(self.model, shortage):
                from ..memory_report import current
                if current() is not None:
                    current().shortfall(shortage)
                raise memory_shortage(shortage)
            release_device_cache()
            shortage = -self._allowance()

    def _pool_capacity(self, plan, *, prefix_budget=0):
        release_device_cache()
        from ..host_alloc import enabled
        from ..models.placement import ensure_room
        floor = plan.minimum if plan.requested is None else plan.requested

        def allowance():
            room = self._allowance()
            if not enabled() and not retain_device_cache():
                # Tensor pools can reuse inactive blocks in segments still held by live weights.
                device = torch.cuda.current_device()
                stream = torch.cuda.current_stream().cuda_stream
                cached = sum(s['total_size'] - s['active_size']
                    for s in torch.cuda.memory_snapshot()
                    if s['device'] == device and s['stream'] == stream
                    and s.get('segment_pool_id', (0, 0)) == (0, 0))
                room += min(cached, plan.bytes_for(floor))
            return room

        while True:
            shortage = (plan.bytes_for(floor) + plan.snapshot_floor + plan.future_bytes
                        - allowance())
            if shortage <= 0 or not ensure_room(self.model, shortage):
                break
        try:
            units, snapshots = plan.choose(allowance(), prefix_budget)
        except ValueError as exc:
            from ..capacity import largest_fitting, memory_shortage
            from ..memory_report import current
            if floor < plan.minimum or (plan.maximum is not None and floor > plan.maximum):
                raise CapacityError(
                    f'{type(self).__name__}: the requested KV pool cannot carry one sequence '
                    f'at max_model_len={self.ctx_cap}, or exceeds the supported pool capacity') from exc
            room = allowance()
            missing = max(0, plan.bytes_for(floor) + plan.snapshot_floor + plan.future_bytes - room)
            report = current()
            if report is not None:
                report.shortfall(missing)
                if getattr(self, '_want_blocks', None) is None:
                    scale = self.ctx_cap // plan.minimum
                    unit = self.block_size if scale == 1 else 1
                    try:
                        fitting = largest_fitting(
                            lambda n: plan.bytes_for(n) + plan.snapshot_floor + plan.future_bytes <= room,
                            floor - 1, unit=unit)
                        report.suggestion = min(fitting * scale, report.values['context'] - 1)
                    except SnowLLMError:
                        pass
            raise CapacityError(
                f'{type(self).__name__}: '
                f'{memory_shortage(missing, target="host" if enabled() else "GPU")}') from exc
        future = plan.future_bytes + snapshots * plan.checkpoint_bytes
        prepare_gpu_commit(plan.bytes_for(units) + future)
        return units, snapshots

    def _cache_blocks(self):
        # All request slots share this total KV pool.
        return (getattr(self, 'total_kv_blocks', self.max_blocks_per_seq)
                if self._want_blocks is None else self._want_blocks)

    def _draft_pool_blocks(self, blocks, *, prefix_slots=0):
        if self.draft_geo is None:
            return 0
        width = self.num_spec if isinstance(self.draft_geo, DSparkGeometry) else self.T
        from .resources import draft_cache_window
        window = draft_cache_window(self.draft_geo)
        if window:
            window = min(window, self.ctx_cap)
            each = ops.kv_blocks_for(window + self.T + width + self.block_size - 1, self.block_size)
            active = self.max_num_seqs * each + ops.kv_blocks_for(self.max_prefill_tokens, self.block_size)
            return active + prefix_slots * ops.kv_blocks_for(window, self.block_size)
        return max(blocks, ops.kv_blocks_for(self.ctx_cap+width, self.block_size))

    def _draft_checkpoint_bytes(self):
        from .resources import draft_cache_window, draft_pool_bytes
        window = draft_cache_window(self.draft_geo)
        if not window:
            return 0
        blocks = self._draft_pool_blocks(0)
        extra = ops.kv_blocks_for(min(window, self.ctx_cap), self.block_size)
        return (draft_pool_bytes(self.draft_geo, blocks + extra, self.block_size)
                - draft_pool_bytes(self.draft_geo, blocks, self.block_size))

    def _prepare_pools(self):
        self.resources.checkpoints.release()
        self.resources.cache.release()

    def finish_pools(self, graph_bytes=0):
        try:
            plan = self._pool_plan(graph_bytes)
            self._prepare_pools()
            while True:
                units, self.prefix_slots = self._pool_capacity(
                    plan, prefix_budget=self._prefix_budget(plan))
                self._allocate_pools(units)
                release_device_cache()
                shortage = graph_bytes - self._allowance()
                if shortage <= 0:
                    break
                # Measure allocator/driver overhead of the actual pools before graph capture.
                self._prepare_pools()
                plan.future_bytes += shortage
            if self.draft_geo is not None:
                self.draft_blocks = self._draft_pool_blocks(self.draft_blocks, prefix_slots=self.prefix_slots)
            return self.num_kv_blocks
        except BaseException:
            self._discard_pools()
            raise

    def _prefix_budget(self, plan):
        floor = plan.minimum if plan.requested is None else plan.requested
        ratio = self.prefix_memory_ratio
        if not 0 <= ratio < 1:
            raise SnowLLMError('--prefix-memory-ratio must be in [0, 1)')
        return int(plan.bytes_for(floor) * ratio)

    def candidates(self) -> list:
        rows = (DEFAULT_PREFILL_CHUNK if self._want_chunk == 'auto' else int(self._want_chunk),)
        if any(c <= 0 for c in rows):
            raise SnowLLMError("prefill chunk must be positive")
        return list(dict.fromkeys(min(c, self.ctx_cap) for c in rows))

    def _loading_cache_bytes(self, rows: int) -> int:
        return self.fixed_bytes(rows)

    def _loading_extra(self, rows: int) -> int:
        if not self.load_peak_bytes:
            return 0
        return max(0, self._loading_cache_bytes(rows) + self.load_peak_bytes
                   - self.reserve_bytes - self.fixed_bytes(rows))

    def minimum_bytes(self, rows: int) -> int:
        return self.fixed_bytes(rows) + self._loading_extra(rows)

    def act_bytes_for(self, rows: int) -> int:
        got = self._act.get(rows)
        if got is None:
            got = self._act[rows] = self._walk_high(rows) + self._act_statics(rows)
        return got

    def _act_statics(self, rows: int) -> int:
        return 0

    def _walk_high(self, rows: int) -> int:
        previous = self.arena
        arena = self.arena = ops.Arena()
        hidden, eager = getattr(self, 'last_hidden', None), self.n_eager_forwards
        draft_eager = self.n_draft_eager
        try:
            with ops.dry_load(), self._activation_scope(rows) as allocation:
                self._plan_activations(rows, allocation)
            return arena.bytes_after(allocation.extra_bytes) + allocation.outside_bytes
        finally:
            self.arena = previous
            self.last_hidden, self.n_eager_forwards = hidden, eager
            self.n_draft_eager = draft_eager
            release_device_cache()

    def _plan_activations(self, rows, allocation):
        from ..ops.sampling import sample_workspace_bytes
        sampling = sample_workspace_bytes(self.max_num_seqs*self.T,self.geo.vocab_size,
                                         min(self.geo.vocab_size,ops.SAMPLING_MAX_K))
        head = ops.lm_head_scratch_bytes(self.max_num_seqs*self.T)
        allocation.extra_bytes = max(allocation.extra_bytes,sampling,head)
        plan_activations(self.arena,allocation.groups,after=allocation.extra_bytes)
        if self.draft_workspace is not None:
            probes = self.draft_workspace.probes(self.arena, rows, self.max_num_seqs,
                self.num_spec, self.block_size, self.max_num_seqs * self.T)
            plan_activations(self.arena, [ProbeGroup(probes)])

    def _pick_block_size(self, block_size: int | None) -> int:
        if block_size is None:
            return ops.KV_BLOCK_SIZES[0]
        if block_size not in ops.KV_BLOCK_SIZES:
            raise ValueError(f"block_size={block_size}: this build has KV cache kernels for "
                             f"{', '.join(str(n) for n in ops.KV_BLOCK_SIZES)}")
        return block_size

    def _check_seqs(self, max_num_seqs: int) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if max_num_seqs > MAX_NUM_SEQS:
            raise ValueError(f"max_num_seqs={max_num_seqs} > {MAX_NUM_SEQS}, this engine's "
                             f"policy cap (runner.py). No kernel stops there; the state pool is "
                             f"what does, at this many requests x (num_spec + 1) state slots.")

    def _check_chunk(self, chunk: int) -> int:
        if chunk > ops.paged_prefill_max_tokens():
            raise ValueError(
                f"max_prefill_tokens={chunk} exceeds this kernel's "
                f"{ops.paged_prefill_max_tokens()}-token ceiling. Chunk the prefill instead -- an "
                f"engine ABLATION(2026-07-18) puts one-shot only 3% ahead of a 32768 chunk.")
        return chunk

    def _freeze_arena(self, extra: int = 0) -> int:
        a = self.arena
        n = a.freeze(a.bytes_after(extra))
        if self.plan_activations:
            a.planner.buf = a.buf
        release_device_cache()
        return n

    def _reserve_activations(self, rows=None):
        rows = self.max_prefill_tokens if rows is None else rows
        with self._activation_scope(rows) as allocation:
            self._plan_activations(rows, allocation)
        self.n_eager_forwards = 0
        self.arena_after = allocation.extra_bytes
        self.arena_bytes = self._freeze_arena(self.arena_after)
        if allocation.zero:
            self.arena.buf.zero_()
        self.act_bytes = self.arena_bytes + allocation.outside_bytes
        return self.arena_bytes

    def _why(self, what: str, gate: str) -> str:
        return (f"{type(self).__name__} has no {what}: this geometry does not carry it, and the "
                f"caller was supposed to have checked {gate} first.")

    def _no(self, what: str, gate: str) -> NoReturn:
        raise SnowLLMError(self._why(what, gate))

    def set_rope(self, inv_freq: torch.Tensor, mscale: float) -> None:
        self._no("tunable rope table", "runner.tunable_rope")

    def mtp_draft(self, hidden: torch.Tensor, next_ids: torch.Tensor,
                  b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        self._no("MTP head to draft with", "an mtp.* tree in the checkpoint")

    _ABSENT = {
        "draft_logits": ("buffer for a draft model's logits", "runner.num_spec"),
        "mtp_h": ("buffer for the MTP head's hidden state", "an mtp.* tree in the checkpoint"),
    }

    class Absent(SnowLLMError, AttributeError):
        pass

    def __getattr__(self, name: str) -> None:
        hit = BaseRunner._ABSENT.get(name)
        if hit is None:
            raise AttributeError(name)
        raise BaseRunner.Absent(self._why(*hit))

    tunable_rope = False
    auto_yarn = False
    prev_context = 0
    ring_blocks = 0
    tap_at: dict[int, int] | None = None
    tap_layers: tuple[int, ...] = ()
    num_taps = 0
    reserve_bytes = 0
    draft_blocks = 0
    draft_block_bytes = 0
    PROBE_BLOCKS = 8
    draft_geo: DraftGeometry = None
    prefix_slots = 0


class GraphRunner(BaseRunner):
    graphs: dict
    GRAPH_FIELDS = ("input_ids", "positions", "slot_mapping", "block_tables", "seq_lens",
                    "state_indices", "num_accepted", "cu_seqlens", "last_row")

    def _init_graphs(self) -> None:
        self.graphs = {}
        self.budget = GraphBudget()
        self._pool = None
        self._said_miss = False
        self._init_stats()

    # Arena tracing relies on view ownership, which inference_mode disables.
    @torch.no_grad()
    def forward(self, batch):
        with self._input_scope():
            prepared, key = self._prepare_batch(batch)
            record = self.graphs.get(key) if key is not None else None
            if record is not None:
                return self._replay(prepared, record)
            if not batch.is_prefill:
                self._miss((batch.batch_size, batch.input_ids.numel()))
            return self._forward_eager(prepared)

    def _prepare_batch(self, batch):
        return batch, self._graph_key(batch)

    def _forward_eager(self, batch):
        ctx = self._ctx(batch)
        ctx.defer_end = self.defer_model_end
        try:
            self.n_eager_forwards += 1
            scope = ctx.arena.activate() if ctx.arena is not None else contextlib.nullcontext()
            with scope, span(f"forward {'prefill' if batch.is_prefill else 'decode'} M={ctx.M}"):
                result = self._execute(ctx)
            self.last_hidden = result.hidden
            if ctx.arena is not None:
                ctx.arena.retain(result.hidden)
            return result.output
        finally:
            ctx.defer_end = False
            ctx.close()

    @torch.no_grad()
    def mtp_draft(self, hidden, next_ids, batch):
        if getattr(getattr(self, "model", None), "mtp", None) is None:
            self._no("MTP head to draft with", "an mtp.* tree in the checkpoint")
        with self._input_scope():
            key = self._graph_key(batch, draft=True)
            record = self.graphs.get(key) if key is not None else None
            if record is not None:
                return self._replay(batch, record, draft=True,
                                    extra={"input_ids": next_ids, "draft_hidden": hidden})
            return self._draft_eager(hidden, next_ids, batch)

    def _draft_eager(self, hidden, next_ids, batch):
        ctx = self._draft_context(batch, next_ids)
        ctx.defer_end = self.defer_model_end
        try:
            self.n_draft_eager += 1
            scope = ctx.arena.activate() if ctx.arena is not None else contextlib.nullcontext()
            with scope, span(f"mtp draft M={ctx.M}"):
                return self._execute_draft(ctx, hidden, next_ids)
        finally:
            ctx.defer_end = False
            ctx.close()

    def _graph_values(self, batch):
        return GraphInputs.batch_values(batch, self.GRAPH_FIELDS)

    def _replay(self, batch, record, *, draft=False, extra=None):
        values = self._graph_values(batch)
        if extra is not None:
            values.update(extra)
        with span(f"{'mtp' if draft else 'decode'} graph replay M={batch.input_ids.numel()}"):
            output = record.replay(values)
        if draft:
            self.n_draft_replays += 1
        else:
            self.n_graph_replays += 1
            self.last_hidden = record.hidden
        return output

    def capture(self, shapes: Sequence[DecodeShape]) -> list[int]:
        if not self.graphs_enabled:
            return []
        self._arm()
        got = set()
        for shape in shapes:
            if not self._graph_room():
                self._report()
                return sorted(got)
            if self._capture_shape(shape):
                got.add(shape.B)
        self._report()
        self.n_eager_forwards = 0
        return sorted(got)

    def probe_graph_bytes(self, shapes: Sequence[DecodeShape]) -> tuple[int, int]:
        if not self.graphs_enabled:
            return 0, 0
        with self._graph_probe_scope(shapes):
            return self._probe_graph_bytes(shapes)

    def _probe_graph_bytes(self, shapes):
        self._arm()
        widest = max(shapes)
        each = 0
        for _ in range(2):
            was = self.budget._used = self.budget._in_use()
            if not self._capture_shape(widest):
                break
            each = self.budget._in_use() - was
            self.graphs.clear()
            self._pool = None
            gc.collect()
        print(f"{term.stamp(time.strftime('%H:%M:%S'))}   a decode graph costs "
              f"{each >> 20} MiB, {len(shapes)} of them to capture", flush=True)
        self._init_graphs()
        return each, len(shapes)

    graphs_enabled = True

    @contextlib.contextmanager
    def _graph_probe_scope(self, shapes):
        yield

    def _capture_shape(self, shape):
        captured = False
        try:
            with contextlib.closing(self._graph_cases(shape)) as cases:
                for case in cases:
                    if case.key in self.graphs:
                        captured = True
                        continue
                    if not self._graph_room():
                        break
                    graph, out = self._capture(case.label, case.run)
                    self.graphs[case.key] = CapturedGraph(
                        graph, case.inputs, out, None if case.draft else self.last_hidden)
                    captured = True
        except torch.OutOfMemoryError:
            self.budget.full = True
            release_device_cache()
        finally:
            self.n_eager_forwards = 0
        return captured

    def _report(self) -> None:
        print(f"{term.stamp(time.strftime('%H:%M:%S'))}   {len(self.graphs)} decode graphs, "
              f"{self.budget.measured >> 20} MiB apiece", flush=True)

    def _arm(self) -> None:
        pass

    def _graph_room(self) -> bool:
        return not self.budget.full and self.budget.room(self.gpu_util, len(self.graphs))

    @property
    def _graphs_full(self) -> bool:
        return self.budget.full

    def _graph_pool(self) -> object:
        if self._pool is None:
            from ..host_alloc import graph_pool
            self._host_graph_pool = graph_pool()
            self._pool = (self._host_graph_pool.id if self._host_graph_pool is not None
                          else torch.cuda.graph_pool_handle())
        return self._pool

    def _capture(self, label: str,
                 walk: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, object]:
        t0 = time.time()
        cur = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            for _ in range(2):
                walk()
        cur.wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with capture_graph(g, pool=self._graph_pool()):
            out = walk()
        torch.cuda.synchronize()
        self.budget.charge()
        print(f"{term.stamp(time.strftime('%H:%M:%S'))}   capture {label}  "
              f"{(time.time() - t0) * 1e3:5.0f} ms", flush=True)
        return g, out

    def _miss(self, key: tuple[int, int]) -> None:
        if self._graphs_full or self._said_miss or not self.graphs:
            return
        self._said_miss = True
        print(f"{term.stamp()} {term.paint('no graph', term.YELLOW)} for decode shape {key}; "
              f"it and every step like it run eager.",
              flush=True)

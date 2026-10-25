# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import ctypes
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch

from .. import ops
from .._capi import SnowLLMError
from ..trace import span
from .block_manager import BlockAllocator, slot_mapping
from .forward_context import (
    Batch,
    DecodeShape,
    Dsv4Walker,
    i32,
    positions,
)
from .host_io import HostArena, host_view
from .inputs import synthetic_batch
from .graphs import GraphCase, GraphInputs
from .resources import ActivationProbe, ActivationReservation, PoolPlan, ProbeGroup, draft_pool_bytes, draft_table_blocks, plan_activations
from .runner import (
    GraphRunner,
    ForwardResult,
)

if TYPE_CHECKING:
    from .cache import CacheStorage
    from .prefix_cache import Pages, Residue
    from .request import Request

BF16 = 2


class Dsv4Runner(GraphRunner):
    @property
    def graphs_enabled(self):
        return not self.geo.v41 and getattr(self.model, 'moe_cache_store', None) is None

    @property
    def host_messages(self) -> bool:
        return not (self.num_spec or self.num_taps)

    @contextlib.contextmanager
    def _input_scope(self):
        if self.inputs is not None and self.cache.plan.messages is None:
            self.cache.plan.messages = HostArena()
            for ratio, width in self.cache.plan.table_cap.items():
                self.cache.plan.messages.reserve(("compression_table", ratio),
                                                 self.max_num_seqs * max(1, width), torch.int32)
        messages = self.cache.plan.messages
        with super()._input_scope(), (messages.reading() if messages is not None
                                     else contextlib.nullcontext()):
            yield

    def _configure(self):
        if self.geo.v41:
            if self.max_num_seqs != 1:
                raise SnowLLMError("DeepSeek V4.1 currently requires --max-num-seqs 1")
            if (self.num_spec or self.tap_layers) and not getattr(getattr(self.draft_geo, "stack", None), "v41", False):
                raise SnowLLMError("DeepSeek V4.1 speculation requires its native DSpark towers")
            if any(i < max(self.geo.kv_source_layers) or i >= self.geo.num_layers for i in self.tap_layers):
                raise SnowLLMError("DeepSeek V4.1 DSpark taps must be decoder layer inputs")
            if self.prefix_memory_ratio:
                raise SnowLLMError("DeepSeek V4.1 currently requires --prefix-memory-ratio 0")
        if self.num_spec and not self.tap_layers:
            raise SnowLLMError(
                f"num_spec={self.num_spec}: this checkpoint ships no MTP head of its own. Its drafter "
                f"is DSpark, which arrives as a separate dspark-*.gguf beside the model and reads "
                f"{len(self.tap_layers) or 'no'} taps out of the target.")
        if self.kv_int8:
            raise SnowLLMError(
                "kv_int8 has nothing to quantize here: the MLA latent already carries an fp8 tail "
                "of its own (dsv4_fp8_kv_quantize), and the int8 paged kernels are the other "
                "geometry's.")
        self.slots_per_request = 1

    def _setup_resources(self):
        from .cache_resources import Dsv4Resources
        self.walker = Dsv4Walker()
        self.spec_rows = self.T
        self.roll_forward = False
        self.mtp_layer = None
        self.last_hidden = None
        self.walker.set_decode_max_m(self.max_num_seqs*self.T)
        self.lm_rows = self.max_num_seqs*self.T
        self.resources = Dsv4Resources(self)

    def _take_cache(self) -> None:
        self.resources.cache.allocate(self.tokens)
        self.num_kv_blocks = self.cache.plan.raw_blocks
        self.kv_bytes = self.cache.bytes()

    def _capture_tokens(self) -> int:
        return self.max_num_seqs * (self.block_size + self.spec_rows)

    def _cache_bytes(self, tokens):
        blocks = self._draft_pool_blocks(ops.kv_blocks_for(tokens, self.block_size))
        return (self.resources.cache.bytes_for(tokens, self.max_prefill_tokens)
                + draft_pool_bytes(self.draft_geo, blocks, self.block_size))

    def _pool_plan(self, graph_bytes):
        return PoolPlan(self._cache_bytes, self.ctx_cap,
            requested=self._cache_blocks()*self.block_size,
            maximum=self.max_num_seqs*self.ctx_cap if self._want_blocks is None else None,
            checkpoint_bytes=self.resources.checkpoint_bytes() if self.prefix_memory_ratio else 0,
            future_bytes=graph_bytes, minimum_snapshots=2*self.max_num_seqs)

    def _allocate_pools(self, tokens):
        self.tokens = tokens
        self._take_cache()
        self.resources.checkpoints.allocate(self.prefix_slots)
        self.draft_blocks = ops.kv_blocks_for(tokens, self.block_size) if self.draft_geo else 0
        if getattr(getattr(self.draft_geo, "stack", None), "v41", False):
            self.draft_blocks = self._draft_pool_blocks(0)

    def _draft_pool_blocks(self, blocks):
        if getattr(getattr(self.draft_geo, "stack", None), "v41", False):
            return self.max_num_seqs * draft_table_blocks(
                self.draft_geo, self.ctx_cap, self.num_spec, self.block_size)
        return super()._draft_pool_blocks(blocks)

    @property
    def arena(self) -> ops.Arena:
        return self.walker.arena

    def _act_statics(self, chunk: int) -> int:
        return self._arena_after() + self.resources.outputs.bytes_for(chunk)

    def act_bytes_for(self, rows: int) -> int:
        if not self.geo.v41:
            return super().act_bytes_for(rows)
        if rows not in self._act:
            previous = self.walker.arena
            arena = self.walker.arena = ops.Arena()
            try:
                with self._activation_scope(rows) as allocation:
                    plan_activations(arena, allocation.groups)
                self._act[rows] = (max(arena.high, arena.planner.high) + self._act_statics(rows)
                                   + allocation.outside_bytes)
            finally:
                self.walker.arena = previous
        return self._act[rows]

    def _arena_after(self) -> int:
        return self.lm_rows * self.geo.hidden * BF16

    def fixed_bytes(self, chunk: int) -> int:
        tokens = self._cache_blocks()*self.block_size
        return (self.resources.cache.bytes_for(tokens, chunk)
                + draft_pool_bytes(self.draft_geo,
                    self._draft_pool_blocks(ops.kv_blocks_for(tokens, self.block_size)), self.block_size)
                + (self.resources.checkpoints.bytes_for(2*self.max_num_seqs)
                   if self.prefix_memory_ratio else 0))

    def _loading_cache_bytes(self, rows: int) -> int:
        return self.resources.cache.bytes_for(self._capture_tokens(), rows)

    @contextlib.contextmanager
    def _probe_cache(self, chunk: int, sweep: int = 1) -> Iterator["CacheStorage"]:
        was = self.walker.tap_at
        if self.tap_layers:
            self.walker.tap_at = {layer: j for j, layer in enumerate(self.tap_layers)}
        try:
            tokens = max(chunk + sweep, 3 * chunk, self.geo.sliding_window) if self.geo.v41 else chunk + sweep
            with self.resources.cache.temporary(tokens, chunk, sweep, probe=True):
                yield self.cache
        finally:
            self.walker.tap_at = was

    @contextlib.contextmanager
    def _walk_scope(self, chunk: int) -> Iterator[Callable[[], torch.Tensor]]:
        with self._probe_cache(chunk) as cache:
            yield lambda: self.model(self.walker.context(
                self.model,
                cache.plan.batch(torch.zeros(chunk, dtype=torch.int64, device="cuda"),
                            torch.arange(chunk, dtype=torch.int64, device="cuda"),
                            [0], [chunk], [0]), cache))

    def _ring_blocks(self, chunk: int) -> int:
        return ops.dsv4_mla_raw_ring_blocks(self.geo.sliding_window, chunk, self.block_size)

    @contextlib.contextmanager
    def _activation_scope(self, chunk):
        widths = self.walker.plan_widths(self.max_num_seqs * self.T)
        with self._probe_cache(chunk, self.DECODE_PHASE_STEPS * sum(widths) + 1) as cache:
            yield ActivationReservation(self._activation_groups(cache, chunk, widths),
                                        extra_bytes=self._arena_after())

    DECODE_PHASE_STEPS = 64
    DECODE_PHASE_PATIENCE = 8

    def _activation_groups(self, cache, chunk, widths):
        pos = 0
        def probes(rows, steps, picked=None, stage="full"):
            nonlocal pos
            for _ in range(steps):
                b = cache.plan.batch(torch.zeros(rows, dtype=torch.int64, device='cuda'),
                    torch.arange(pos, pos+rows, dtype=torch.int64, device='cuda'),
                    [pos], [rows], [0], last_row=picked)
                b.ced_stage = stage
                b.is_prefill = stage != "full" if self.geo.v41 else b.is_prefill
                ctx = self.walker.context(self.model, b, cache)
                pos += rows
                yield ActivationProbe(ctx.plan_key, rows, lambda: self.model(ctx))
        picked = torch.full((self.max_num_seqs,), chunk-1, dtype=torch.int32, device='cuda')
        for last in (None, picked):
            cache.reset(0)
            pos = 0
            yield ProbeGroup(probes(chunk, 3 if self.geo.v41 else 1, last,
                                    "encoder" if self.geo.v41 else "full"))
        if self.geo.v41:
            cache.reset(0)
            last = self.geo.sliding_window - 1
            cache.plan.batch(torch.zeros(last, dtype=torch.int64, device="cuda"),
                torch.arange(last, dtype=torch.int64, device="cuda"), [0], [last], [0])
            pos = last
            yield ProbeGroup(probes(1, 1, torch.zeros(1, dtype=torch.int32, device="cuda"), "encoder"))
            for rows in (1, self.geo.sliding_window):
                cache.reset(0)
                pos = 0
                selected = torch.full((1,), rows - 1, dtype=torch.int32, device="cuda")
                yield ProbeGroup(probes(rows, 1, selected, "replay"))
            cache.reset(0)
            cache.plan.batch(torch.zeros(chunk, dtype=torch.int64, device="cuda"),
                torch.arange(chunk, dtype=torch.int64, device="cuda"), [0], [chunk], [0])
        pos = chunk
        for rows in widths:
            yield ProbeGroup(probes(rows, self.DECODE_PHASE_STEPS if self.plan_activations else 1),
                             self.DECODE_PHASE_PATIENCE)
        cache.reset(0)

    def prefix_pages(self, alloc: BlockAllocator) -> "Pages":
        from .prefix_cache import HistoryPages
        return HistoryPages(self.cache.history)

    def prefix_residue(self) -> "Residue":
        from .prefix_cache import SnapshotResidue
        return SnapshotResidue(self.cache, self.checkpoint_buffer, self.cache.history.align)

    def release(self, slot: int) -> None:
        self.cache.reset(slot)

    def grow(self, r: "Request", n: int = 1) -> bool:
        return self.cache.history.reserve(r.slot, r.num_cached + n)

    def admit(self, r: "Request", length: int, covered: int = 0) -> bool:
        return self.cache.history.reserve(r.slot, length, covered)

    def commit(self, batch: "list[Request]") -> None:
        for r in batch:
            self.cache.plan.commit(r.slot, r.num_cached)

    def _prepare_batch(self, b: Batch) -> tuple:
        firsts, lens, slots = self._spans(b)
        n = sum(lens)
        spec = not b.is_prefill and max(lens) > 1
        if spec and max(lens) > self.spec_rows:
            raise SnowLLMError(f"a verify step of {max(lens)} rows needs a compressor carry built "
                               f"for it; this cache was built for {self.spec_rows}")
        db = self.cache.plan.batch(b.input_ids[:n], b.positions[0][:n], firsts, lens, slots,
                              b.block_tables, b.slot_mapping[:n], b.last_row, spec=spec,
                              is_prefill=b.is_prefill)
        db.need_logits = b.need_logits
        db.host_ids = b.host_ids
        if self.geo.v41 and b.is_prefill:
            db.ced_stage = "encoder"
        if b.is_prefill:
            return db, None
        return db, (len(lens), n, db.total_q_blocks,
                    tuple((pl.n_new, int(pl.keep_dst is not None))
                          for _, pl in sorted(db.plans.items())))

    def _ctx(self, batch):
        return self.walker.context(self.model, batch, self.cache)

    def _execute(self, ctx):
        if self.plan_activations:
            self.walker.plan_for(self.model, ctx, self.arena_after)
        hidden = self.model(ctx)
        if ctx.batch.ced_stage == "encoder" and ctx.batch.need_logits:
            batch = self.cache.plan.ced.batch(ctx.batch)
            ctx = self.walker.context(self.model, batch, self.cache)
            if self.plan_activations:
                self.walker.plan_for(self.model, ctx, self.arena_after)
            hidden = self.model(ctx)
        out = self.logits[:hidden.shape[0]]
        if ctx.batch.need_logits:
            with span("lm_head"):
                out = self.model.lm_head(self.arena, hidden, self.logits)
        return ForwardResult(out, hidden)

    GRAPH_FIELDS = ("input_ids", "positions", "seq_of_row", "cu_seqlens", "seq_lens",
                    "block_tables", "slot_mapping", "q_block_map", "last_row")

    def _graph_values(self, batch):
        values = super()._graph_values(batch)
        for ratio, plan in batch.plans.items():
            values.update({(ratio, name): value for name, value in vars(plan).items()
                           if isinstance(value, torch.Tensor)})
        return values

    def _freeze_graph_inputs(self, batch):
        if self.inputs is None:
            return None
        return GraphInputs.freeze_batch(batch, self.GRAPH_FIELDS)

    def _spans(self, b: Batch) -> tuple[list[int], list[int], list[int]]:
        state = b.state_indices[:b.batch_size]
        host = host_view(state)
        if host is None:
            slots = state.tolist()
        else:
            scalar = {torch.int32: ctypes.c_int32, torch.int64: ctypes.c_int64}[host.dtype]
            slots = [scalar.from_address(host.data_ptr()+i*host.stride(0)*host.element_size()).value
                     for i in range(host.numel())]
        bad = [s for s in slots if not 0 <= s < self.max_num_seqs]
        if bad:
            raise SnowLLMError(f"state slot {bad[0]} is outside [0, {self.max_num_seqs}); this "
                               f"runner keys the compressors' carries by it")
        ends = b.host_seq_lens if b.host_seq_lens is not None else b.seq_lens.tolist()
        if b.is_prefill or b.host_cu_seqlens is not None:
            host = host_view(b.cu_seqlens)
            cu = b.host_cu_seqlens
            if cu is None:
                cu = (b.cu_seqlens if host is None else host).tolist()
            lens = [cu[i + 1] - cu[i] for i in range(len(ends))]
        else:
            lens = [b.input_ids.numel() // len(ends)] * len(ends)
        return [e - n for e, n in zip(ends, lens)], lens, slots

    def _graph_cases(self, shape: DecodeShape):
        B, T = shape.B, shape.T
        c, ring = self.block_size, self.ring_blocks
        nb = ops.kv_blocks_for(c + T, self.block_size)
        if nb > ring or B * nb > self.cache.plan.raw_blocks:
            return
        bt = i32([[i * nb + j for j in range(nb)] for i in range(B)])
        try:
            for i in range(B):
                self.cache.reset(i)
                if not self.cache.history.reserve(i, c + T):
                    return
            for lo in range(0, B, max(1, self.max_prefill_tokens // c)):
                hi = min(B, lo + max(1, self.max_prefill_tokens // c))
                self.forward(self._warm_prefill(lo, c, bt[lo:hi].contiguous()))
            db, key = self._prepare_batch(self._warm_decode(B, T, c, bt, shape.lengths))
            messages = self._freeze_graph_inputs(db)
            inputs = GraphInputs(self._graph_values(db), messages)
            yield GraphCase(key, f'decode graph B={B:2d} T={T} M={db.num_tokens:3d}',
                            lambda: self._forward_eager(db), inputs)
        finally:
            for i in range(B):
                self.cache.reset(i)
            self.n_eager_forwards = 0

    def _warm_prefill(self, slot: int, c: int, bt: torch.Tensor) -> Batch:
        B = bt.shape[0]
        rows = [(i, t) for i in range(B) for t in range(c)]
        return synthetic_batch([c]*B, bt.shape[1],
            positions=positions([p for _, p in rows]), slot_mapping=slot_mapping(bt, rows, self.block_size),
            block_tables=bt, state_indices=list(range(slot, slot+B)),
            last_row=[(i+1)*c-1 for i in range(B)], is_prefill=True, need_logits=self.geo.v41)

    def _warm_decode(self, B, T, c, bt, lengths=()):
        lengths = list(lengths) if lengths else [T]*B
        rows = [(i, c+t) for i in range(B) for t in range(lengths[i])]
        return synthetic_batch(lengths, bt.shape[1],
            positions=positions([p for _, p in rows]), slot_mapping=slot_mapping(bt, rows, self.block_size),
            block_tables=bt, seq_lens=[c+n for n in lengths], state_indices=list(range(B)),
            num_accepted=[1]*B, host_seq_lens=[c+n for n in lengths],
            total_q_blocks=ops.prefill_q_plan(lengths)[0])

    def _arm(self) -> None:
        self.cache.plan.arm()

    tunable_rope = False

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch

from .. import ops
from .._capi import SnowLLMError
from ..models.deepseek_v4.deepseek4 import DeepSeekV4ForCausalLM
from ..models.geometry import COFF, DeepSeekV4Geometry
from ..trace import span
from .block_manager import BlockAllocator, slot_mapping
from .dsv4_cache import Cache, make_cache
from .forward_context import (
    Batch,
    DecodeShape,
    Dsv4Context,
    Dsv4Walker,
    i32,
    i64,
    positions,
)
from .runner import (
    DEFAULT_GPU_UTIL,
    DEFAULT_MAX_NUM_SEQS,
    KV_AUTOSIZE_RESERVE,
    DraftGeometry,
    GraphRunner,
    draft_token_bytes,
)

if TYPE_CHECKING:
    from .prefix_cache import Pages, Residue
    from .request import Request

# EMPIRICAL, widest first: 16384 is a measured knee, and 32768 was measured and refused. The
# ladder and the numbers behind it are in 1c3a9c1.
PREFILL_CHUNK_CANDIDATES = (16384, 8192, 4096, 2048, 1024, 512)

BF16 = 2
F32 = 4


def _raw_token_bytes(geo: DeepSeekV4Geometry) -> int:
    return geo.num_layers * geo.kv_dim * BF16


def _comp_token_bytes(geo: DeepSeekV4Geometry) -> int:
    n = 0
    for i, ratio in enumerate(geo.compress_ratios[:geo.num_layers]):
        if not ratio:
            continue
        n += geo.kv_dim * BF16 // ratio
        if geo.is_indexed(i):
            n += geo.index_head_dim * BF16 // ratio
    return n


def _carry_bytes(geo: DeepSeekV4Geometry, slots: int) -> int:
    n = 0
    for i, ratio in enumerate(geo.compress_ratios[:geo.num_layers]):
        if not ratio:
            continue
        window = 2 * COFF[ratio] * ratio
        n += slots * window * COFF[ratio] * geo.kv_dim * F32 * 2
        if geo.is_indexed(i):
            n += slots * window * 2 * geo.index_head_dim * F32 * 2
    return n


def _slack_bytes(geo: DeepSeekV4Geometry, slots: int, block_size: int) -> int:
    page = slots * block_size
    n = 0
    for i, ratio in enumerate(geo.compress_ratios[:geo.num_layers]):
        if not ratio:
            continue
        n += page * 2 * geo.kv_dim * BF16
        if geo.is_indexed(i):
            n += page * geo.index_head_dim * BF16
    return n


def ckpt_bytes(geo: DeepSeekV4Geometry, block_size: int) -> int:
    window = -(-geo.sliding_window // block_size) * block_size
    return window * _raw_token_bytes(geo) + _carry_bytes(geo, 1)


def kv_fixed(geo: DeepSeekV4Geometry, ctx_cap: int, slots: int, chunk: int, block_size: int,
             draft: int = 0) -> int:
    ring = ops.dsv4_mla_raw_ring_blocks(geo.sliding_window, chunk, block_size)
    return (KV_AUTOSIZE_RESERVE + _carry_bytes(geo, slots) + _slack_bytes(geo, slots, block_size)
            + slots * ring * block_size * _raw_token_bytes(geo)
            + ctx_cap * (_comp_token_bytes(geo) + draft))


class Dsv4Runner(GraphRunner):
    CHUNK_CANDIDATES = PREFILL_CHUNK_CANDIDATES

    def __init__(self, model: DeepSeekV4ForCausalLM, num_kv_blocks: int | None,
                 max_blocks_per_seq: int, max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                 max_prefill_tokens: int | str = "auto", num_spec: int = 0,
                 kv_int8: bool = False, block_size: int | None = None,
                 gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
                 reserve_bytes: int = 0, tap_layers: tuple[int, ...] = (),
                 draft_geo: DraftGeometry = None, prefix_memory_ratio: float = 0.0,
                 plan_activations: bool = True) -> None:
        geo = self.geo = model.geo
        self.plan_activations = plan_activations
        self.model = model
        self.walker = Dsv4Walker()
        self.block_size = self._pick_block_size(block_size)
        self._check_seqs(max_num_seqs)
        if num_spec and not tap_layers:
            raise SnowLLMError(
                f"num_spec={num_spec}: this checkpoint ships no MTP head of its own. Its drafter "
                f"is DSpark, which arrives as a separate dspark-*.gguf beside the model and reads "
                f"{len(tap_layers) or 'no'} taps out of the target.")
        if kv_int8:
            raise SnowLLMError(
                "kv_int8 has nothing to quantize here: the MLA latent already carries an fp8 tail "
                "of its own (dsv4_fp8_kv_quantize), and the int8 paged kernels are the other "
                "geometry's.")

        self.reserve_bytes = int(reserve_bytes)
        self.draft_geo = draft_geo
        self.draft_token_bytes = draft_token_bytes(draft_geo)
        self.tap_layers = tuple(tap_layers)
        self.num_taps = len(self.tap_layers)
        self.gpu_util = gpu_memory_utilization
        self.prefix_slots = 0
        self.prefix_memory_ratio = prefix_memory_ratio
        self.max_num_seqs = max_num_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.kv_int8 = False

        self.num_spec = num_spec
        self.T = num_spec + 1
        self.spec_rows = self.T
        self.slots_per_request = 1
        self.roll_forward = False
        self.dummy_slot = max_num_seqs
        self.mtp_layer = None
        self._init_graphs()
        self.last_hidden = None

        self.ctx_cap = max_blocks_per_seq * self.block_size
        self._want_chunk = max_prefill_tokens
        self._act: dict[int, int] = {}
        allow = self._allowance()

        self.walker.set_decode_max_m(max_num_seqs * self.spec_rows)
        self.lm_rows = max_num_seqs * self.T

        def taps_at(rows: int) -> torch.Tensor:
            return torch.zeros(max(rows, max_num_seqs * self.T), self.num_taps * geo.hidden,
                               dtype=torch.bfloat16, device="cuda")

        if self.num_taps:
            self.walker.taps = taps_at(max(self.candidates()))
        chunk = self._check_chunk(self._chunk_for(allow))

        self.logits = torch.empty(self.lm_rows, geo.vocab_size, dtype=torch.float32,
                                  device="cuda")

        for chunk in [c for c in self.candidates() if c <= chunk] or [chunk]:
            self.ring_blocks = self._ring_blocks(chunk)
            self.max_prefill_tokens = chunk
            if self.num_taps and self.walker.taps.shape[0] != max(chunk,
                                                                   max_num_seqs * self.T):
                self.walker.taps = None
                torch.cuda.empty_cache()
                self.walker.taps = taps_at(chunk)
            self.act_bytes = self._reserve_activations(chunk)
            tokens, fixed = self._capacity(self.gpu_util, max_num_seqs, chunk)
            if num_kv_blocks is not None or tokens >= self.ctx_cap:
                break
            self.arena.release()
            torch.cuda.empty_cache()
        if num_kv_blocks is not None:
            tokens = num_kv_blocks * self.block_size
        elif tokens < self.ctx_cap:
            raise ValueError(self._too_small(max_num_seqs, fixed))
        else:
            tokens = min(tokens, max_num_seqs * self.ctx_cap)
        self._allow = allow
        self.tokens_top = tokens
        self.tokens = tokens if num_kv_blocks is not None else self._capture_tokens()
        self._pinned = num_kv_blocks is not None
        self._take_cache()
        self.draft_blocks = (ops.kv_blocks_for(self.tokens, self.block_size)
                             if draft_geo is not None else 0)

    def _take_cache(self) -> None:
        self.cache = make_cache(self.geo, self.tokens, self.block_size, self.max_num_seqs,
                                      raw_blocks=self.max_num_seqs * self.ring_blocks,
                                      spec_rows=self.spec_rows, ctx_cap=self.ctx_cap)
        self.num_kv_blocks = self.cache.raw_blocks
        self.kv_bytes = self.cache.bytes()

    def _capture_tokens(self) -> int:
        return self.max_num_seqs * (self.block_size + self.spec_rows)

    def finish_pools(self, graph_bytes: int = 0) -> int:
        if not self._pinned:
            self.cache.slabs.release()
            self.cache = None
            self.tokens = min(self.tokens_top, self._afford(self._per_token(), graph_bytes))
            self._take_cache()
        self.draft_blocks = (ops.kv_blocks_for(self.tokens, self.block_size)
                             if self.draft_geo is not None else 0)
        self.prefix_slots = self._prefix_slots(self._allow)
        return self.num_kv_blocks

    def _prefix_slots(self, allow: int) -> int:
        if self.prefix_memory_ratio <= 0:
            return 0
        each = ckpt_bytes(self.geo, self.block_size)
        return max(0, min(int(self.prefix_memory_ratio * allow), self._allowance())) // each

    @property
    def arena(self) -> ops.Arena:
        return self.walker.arena

    def _act_statics(self, chunk: int) -> int:
        return self._arena_after() + chunk * self.num_taps * self.geo.hidden * BF16

    def _arena_after(self) -> int:
        return self.lm_rows * self.geo.hidden * BF16

    def fixed_bytes(self, chunk: int) -> int:
        return kv_fixed(self.geo, self.ctx_cap, self.max_num_seqs, chunk, self.block_size,
                        self.draft_token_bytes)

    @contextlib.contextmanager
    def _probe_cache(self, chunk: int, sweep: int = 1) -> Iterator[Cache]:
        cache = make_cache(self.geo, chunk + sweep, self.block_size, self.max_num_seqs,
                           raw_blocks=ops.kv_blocks_for(chunk + sweep, self.block_size),
                           spec_rows=1, ctx_cap=max(self.ctx_cap, chunk + sweep))
        was = self.walker.tap_at
        if self.tap_layers:
            self.walker.tap_at = {layer: j for j, layer in enumerate(self.tap_layers)}
        try:
            yield cache
        finally:
            self.walker.tap_at = was
            cache.slabs.release()
            del cache

    @contextlib.contextmanager
    def _walk_scope(self, chunk: int) -> Iterator[Callable[[], torch.Tensor]]:
        with self._probe_cache(chunk) as cache:
            yield lambda: self.model(self.walker.context(
                self.model,
                cache.batch(torch.zeros(chunk, dtype=torch.int64, device="cuda"),
                            torch.arange(chunk, dtype=torch.int64, device="cuda"),
                            [0], [chunk], [0]), cache))

    def _ring_blocks(self, chunk: int) -> int:
        return ops.dsv4_mla_raw_ring_blocks(self.geo.sliding_window, chunk, self.block_size)

    def _reserve_activations(self, chunk: int) -> int:
        widths = self.walker.plan_widths(self.max_num_seqs * self.T)
        with self._probe_cache(chunk, self.DECODE_PHASE_STEPS * sum(widths) + 1) as cache:
            self._plan_activations(cache, chunk, widths)
        cache = None
        self.arena_after = self._arena_after()
        self.arena_bytes = self._freeze_arena(self.arena_after)
        return self.arena_bytes

    DECODE_PHASE_STEPS = 64
    DECODE_PHASE_PATIENCE = 8

    def _plan_activations(self, cache: Cache, chunk: int, widths: tuple[int, ...]) -> None:
        model = self.model
        planner = self.arena.planner

        def walk(rows: int, pos: int, last_row: torch.Tensor | None) -> bool:
            b = cache.batch(torch.zeros(rows, dtype=torch.int64, device="cuda"),
                            torch.arange(pos, pos + rows, dtype=torch.int64, device="cuda"),
                            [pos], [rows], [0], last_row=last_row)
            ctx = self.walker.context(model, b, cache)
            key = ctx.plan_key
            with ops.dummy_run() as t:
                hidden = model(ctx)
            if key in planner.plans:
                return False
            planner.install(key, ops.pack(t, rows))
            del hidden
            return True

        picked = torch.full((self.max_num_seqs,), chunk - 1, dtype=torch.int32, device="cuda")
        for last_row in (None, picked):
            cache.reset(0)
            walk(chunk, 0, last_row)
        pos = chunk
        for rows in widths:
            stale = 0
            for _ in range(self.DECODE_PHASE_STEPS if self.plan_activations else 1):
                stale = 0 if walk(rows, pos, None) else stale + 1
                pos += rows
                if stale >= self.DECODE_PHASE_PATIENCE:
                    break
        cache.reset(0)

    def prefix_pages(self, alloc: BlockAllocator) -> "Pages":
        from .prefix_cache import RatioPages
        return RatioPages(self.cache)

    def prefix_residue(self) -> "Residue":
        from .prefix_cache import Dsv4Residue
        return Dsv4Residue(self.cache, self.prefix_slots)

    def _capacity(self, util: float, slots: int, chunk: int) -> tuple[int, int]:
        fixed = kv_fixed(self.geo, 0, slots, chunk, self.block_size)
        free, total = torch.cuda.mem_get_info()
        self._decided_used = total - free
        return self._afford(self._per_token(), fixed, util), fixed

    def _per_token(self) -> int:
        return _comp_token_bytes(self.geo) + self.draft_token_bytes

    def _too_small(self, slots: int, fixed: int) -> str:
        _, total = torch.cuda.mem_get_info()
        used = self._decided_used
        per_token = self._per_token()
        one_seq = self.ctx_cap * per_token
        want = (used + fixed + one_seq + self.reserve_bytes) / total + 0.005
        why = (f"the KV pools cannot carry one sequence at max_model_len={self.ctx_cap}: "
               f"{used / (1 << 30):.1f} GiB of this device's {total / (1 << 30):.1f} GiB "
               f"is already taken by the weights and by the "
               f"{self.act_bytes / (1 << 30):.1f} GiB of activations a "
               f"{self.max_prefill_tokens}-token prefill chunk needs "
               f"({self.arena_bytes / (1 << 20):.0f} MiB arena), and this "
               f"stack wants "
               f"{fixed / (1 << 30):.1f} GiB more at max_num_seqs={slots} for the compressor "
               f"state, the raw window each request holds and the allocator's headroom -- against "
               f"a gpu_memory_utilization={self.gpu_util} ceiling of "
               f"{total * self.gpu_util / (1 << 30):.1f} GiB. ")
        return why + self._ceiling_advice(
            want, "--max-num-seqs / --max-num-batched-tokens",
            f"{one_seq / (1 << 30):.1f} GiB of compressed pool at {per_token / 1024:.0f} KiB a "
            f"token",
            lambda: self._capacity(0.98, slots, self.max_prefill_tokens)[0])


    def release(self, slot: int) -> None:
        self.cache.release(slot)

    def grow(self, r: "Request", n: int = 1) -> bool:
        return self.cache.reserve(r.slot, r.num_cached + n)

    def admit(self, r: "Request", length: int, covered: int = 0) -> bool:
        return self.cache.reserve(r.slot, length, covered)

    def commit(self, batch: "list[Request]") -> None:
        for r in batch:
            self.cache.commit(r.slot, r.num_cached)

    def forward(self, b: Batch) -> torch.Tensor:
        with span(f"forward {'prefill' if b.is_prefill else 'decode'} M={b.input_ids.numel()}"):
            db, key = self._plan(b)
            hit = self.graphs.get(key) if key is not None else None
            if hit is not None:
                return self._replay(hit, db, b.need_logits)
            if key is not None:
                self._miss(key[:2])
            self.n_eager_forwards += 1
            return self._run(db, b.need_logits)

    def _plan(self, b: Batch) -> tuple:
        firsts, lens, slots = self._spans(b)
        n = sum(lens)
        spec = not b.is_prefill and max(lens) > 1
        if spec and max(lens) > self.spec_rows:
            raise SnowLLMError(f"a verify step of {max(lens)} rows needs a compressor carry built "
                               f"for it; this cache was built for {self.spec_rows}")
        db = self.cache.batch(b.input_ids[:n], b.positions[0][:n], firsts, lens, slots,
                              b.block_tables, b.slot_mapping[:n], b.last_row, spec=spec,
                              is_prefill=b.is_prefill)
        if b.is_prefill:
            return db, None
        return db, (len(lens), max(lens), db.total_q_blocks,
                    tuple((pl.n_new, int(pl.keep_dst is not None))
                          for _, pl in sorted(db.plans.items())))

    def _plan_for(self, ctx: Dsv4Context) -> None:
        self.walker.plan_for(self.model, ctx, self.arena_after)

    def _run(self, db: Batch, need_logits: bool) -> torch.Tensor:
        ctx = self.walker.context(self.model, db, self.cache)
        if self.plan_activations:
            self._plan_for(ctx)
        hidden = self.model(ctx)
        self.last_hidden = hidden
        if not need_logits:
            return self.logits[:hidden.shape[0]]
        with span("lm_head"):
            return self.model.lm_head(self.arena, hidden, self.logits)

    GRAPH_FIELDS = ("input_ids", "positions", "seq_of_row", "cu_seqlens", "seq_lens",
                    "block_tables", "slot_mapping", "q_block_map", "last_row")

    @classmethod
    def _tensors(cls, db: Batch) -> list[torch.Tensor]:
        out = [t for t in (getattr(db, f) for f in cls.GRAPH_FIELDS)
               if isinstance(t, torch.Tensor)]
        for plan in db.plans.values():
            out += [v for v in vars(plan).values() if isinstance(v, torch.Tensor)]
        return out

    def _replay(self, hit: tuple[torch.cuda.CUDAGraph, list[torch.Tensor], int], db: Batch,
                need_logits: bool) -> torch.Tensor:
        g, static, rows = hit
        with span(f"decode graph replay B={len(db.plans) and db.seq_lens.numel()}"):
            src = self._tensors(db)
            if len(src) != len(static):
                raise SnowLLMError(f"the plan for this shape has {len(src)} tensors where the "
                                   f"graph captured {len(static)}")
            torch._foreach_copy_(static, src)
            g.replay()
        self.n_graph_replays += 1
        return self.logits[:rows]

    def _spans(self, b: Batch) -> tuple[list[int], list[int], list[int]]:
        slots = b.state_indices[:b.batch_size].tolist()
        bad = [s for s in slots if not 0 <= s < self.max_num_seqs]
        if bad:
            raise SnowLLMError(f"state slot {bad[0]} is outside [0, {self.max_num_seqs}); this "
                               f"runner keys the compressors' carries by it")
        ends = b.seq_lens.tolist()
        if b.is_prefill:
            cu = b.cu_seqlens.tolist()
            lens = [cu[i + 1] - cu[i] for i in range(len(ends))]
        else:
            lens = [b.input_ids.numel() // len(ends)] * len(ends)
        return [e - n for e, n in zip(ends, lens)], lens, slots

    def _capture_shape(self, shape: DecodeShape) -> bool:
        B, T = shape.B, shape.T
        c, ring = self.block_size, self.ring_blocks
        nb = ops.kv_blocks_for(c + T, self.block_size)
        if nb > ring or B * ring > self.cache.raw_blocks:
            return False
        bt = i32([[i * ring + j for j in range(nb)] for i in range(B)])
        try:
            for i in range(B):
                self.cache.reset(i)
                if not self.cache.reserve(i, c + T):
                    return False
                self.forward(self._warm_prefill(i, c, bt[i:i + 1].contiguous()))
            db, key = self._plan(self._warm_decode(B, T, c, bt))
            if key not in self.graphs:
                g, rows = self._capture(f"decode graph B={B:2d} T={T} M={B * T:3d}",
                                        lambda: self._run(db, True).shape[0])
                self.graphs[key] = (g, self._tensors(db), rows)
        finally:
            for i in range(B):
                self.cache.reset(i)
            self.n_eager_forwards = 0
        return True

    def _warm_prefill(self, slot: int, c: int, bt: torch.Tensor) -> Batch:
        rows = [(0, t) for t in range(c)]
        return Batch(
            input_ids=torch.zeros(c, dtype=torch.int64, device="cuda"),
            positions=positions([p for _, p in rows]),
            slot_mapping=slot_mapping(bt, rows, self.block_size), block_tables=bt,
            seq_lens=i32([c]), is_prefill=True, num_tokens=c,
            state_indices=i32([slot]), cu_seqlens=i32([0, c]),
            last_row=i64([c - 1]), need_logits=False)

    def _warm_decode(self, B: int, T: int, c: int, bt: torch.Tensor) -> Batch:
        rows = [(i, c + t) for i in range(B) for t in range(T)]
        return Batch(
            input_ids=torch.zeros(B * T, dtype=torch.int64, device="cuda"),
            positions=positions([p for _, p in rows]),
            slot_mapping=slot_mapping(bt, rows, self.block_size), block_tables=bt,
            seq_lens=i32([c + T] * B), is_prefill=False, num_tokens=B * T,
            state_indices=i32(list(range(B))),
            num_accepted=torch.ones(B, dtype=torch.int32, device="cuda"),
            cu_seqlens=i32([i * T for i in range(B + 1)]),
            total_q_blocks=ops.prefill_q_plan([T] * B)[0])

    def _arm(self) -> None:
        self.cache.arm()

    tunable_rope = False

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import dataclasses
import gc
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, NoReturn

import torch

from .. import ops, term
from .._capi import SnowLLMError
from ..models.geometry import DFlashGeometry, DSparkGeometry
from ..models.qwen3_5.layers import FullAttention, GatedDeltaNet
from ..trace import span
from .forward_context import (
    PLAN_DECODE,
    PLAN_MAIN,
    PLAN_MTP,
    PLAN_MTP_DECODE,
    Batch,
    DecodeShape,
    ForwardContext,
)

if TYPE_CHECKING:
    from ..models.qwen3_5.qwen3_5 import Qwen3_5MoeForCausalLM
    from .block_manager import BlockAllocator
    from .prefix_cache import Pages, PrefixStore, Residue
    from .request import Request

DraftGeometry = DFlashGeometry | DSparkGeometry | None


PREFILL_CHUNK_CANDIDATES = (32768, 16384, 8192)

KV_AUTOSIZE_RESERVE = 512 << 20

DEFAULT_GPU_UTIL = 0.9

MAX_NUM_SEQS = 256
DEFAULT_MAX_NUM_SEQS = 16

def kv_block_bytes(pools: int, block_size: int, kv_int8: bool = False) -> int:
    n = sum(ops.kv_pool_bytes(1, kv_int8, block_size))
    if kv_int8:
        n += sum(ops.kv_scale_bytes(1, block_size))
    return pools * n


def draft_token_bytes(geo: DraftGeometry) -> int:
    if geo is None:
        return 0
    st = geo.stack if isinstance(geo, DSparkGeometry) else geo
    return st.num_layers * 2 * st.kv_dim * 2


def draft_pool_bytes(geo: DraftGeometry, blocks: int, block_size: int) -> int:
    return blocks * block_size * draft_token_bytes(geo)


POOL_FLOOR_SLOTS = 1

@dataclasses.dataclass(frozen=True)
class KvPlan:
    geo: object
    ctx_cap: int
    block_size: int
    slots: int = 1
    full: int = 0
    linear: int = 0
    T: int = 1
    taps: int = 0
    kv_int8: bool = False
    mtp: bool = False
    draft: int = 0
    index: int = 0
    @property
    def state_slots(self) -> int:
        return self.slots * (1 if self.taps else self.T) + 1

    @property
    def state_rows(self) -> int:
        return self.state_slots

    def state_bytes(self) -> int:
        return self.state_rows * self.state_slot_bytes()

    def state_slot_bytes(self) -> int:
        geo = self.geo
        per = (geo.lin_conv_state * geo.lin_conv_dim * 2
               + geo.lin_num_v_heads * geo.lin_head_k * geo.lin_head_v * 4)
        return per * self.linear

    def pool_bytes(self, slots: int = POOL_FLOOR_SLOTS) -> int:
        return slots * ops.kv_blocks_for(self.ctx_cap, self.block_size) * self.block_bytes()

    def block_bytes(self) -> int:
        return kv_block_bytes(self.full, self.block_size, self.kv_int8) + self.draft + self.index

    def fixed(self, slots: int = POOL_FLOOR_SLOTS) -> int:
        return KV_AUTOSIZE_RESERVE + self.pool_bytes(slots) + self.state_bytes()


class GraphBudget:
    def __init__(self) -> None:
        self.measured = 0
        self.full = False
        self._used = 0

    @staticmethod
    def _in_use() -> int:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        return total - free

    def room(self, gpu_util: float, have: int) -> bool:
        self._used = self._in_use()
        total = torch.cuda.mem_get_info()[1]
        if not self.measured or self._used + self.measured <= int(total * gpu_util):
            return True
        if not self.full:
            self.full = True
            print(f"{term.stamp()} {term.paint('graphs stop', term.YELLOW)} at {have}: "
                  f"another {self.measured >> 20} MiB would cross "
                  f"the "
                  f"gpu_memory_utilization={gpu_util} ceiling. Steps on shapes that have no graph "
                  f"run eager.", flush=True)
        return False

    def charge(self) -> None:
        self.measured = max(self.measured, self._in_use() - self._used)


class GraphRunner:
    graphs: dict
    max_num_seqs: int
    gpu_util: float

    def _init_graphs(self) -> None:
        self.graphs = {}
        self.budget = GraphBudget()
        self._pool = None
        self._said_miss = False
        self.n_graph_replays = 0
        self.n_eager_forwards = 0
        self.n_draft_replays = 0
        self.n_draft_eager = 0

    def capture(self, shapes: Sequence[DecodeShape]) -> list[int]:
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

    def _report(self) -> None:
        print(f"{term.stamp(time.strftime('%H:%M:%S'))}   {len(self.graphs)} decode graphs, "
              f"{self.budget.measured >> 20} MiB apiece", flush=True)

    def _arm(self) -> None:
        pass

    def _index_block_bytes(self, full: int) -> int:
        return 0

    def _extra_pools(self) -> None:
        pass

    def prefix_store(self, alloc: "BlockAllocator",
                     draft: "BlockAllocator | None" = None) -> "PrefixStore":
        from .prefix_cache import PrefixStore, WithDraft
        pages = self.prefix_pages(alloc)
        if draft is not None:
            pages = WithDraft(pages, draft, self.block_size)
        return PrefixStore(pages, self.prefix_residue(), self.prefix_slots)

    def prefix_pages(self, alloc: "BlockAllocator") -> "Pages":
        from .prefix_cache import BlockPages
        return BlockPages(alloc, self.block_size)

    def prefix_residue(self) -> "Residue":
        from .prefix_cache import LinearResidue, NoResidue
        return (LinearResidue(self.linear_mods) if self.linear_mods else NoResidue())

    def _graph_room(self) -> bool:
        return self.budget.room(self.gpu_util, len(self.graphs))

    def _allowance(self, util: float | None = None) -> int:
        free, total = torch.cuda.mem_get_info()
        return (int(total * (self.gpu_util if util is None else util)) - (total - free)
                - self.reserve_bytes)

    def _afford(self, per_unit: int, fixed: int = 0, util: float | None = None) -> int:
        return max(0, int((self._allowance(util) - fixed) // per_unit))


    CHUNK_CANDIDATES: tuple = ()

    def candidates(self) -> list:
        if self._want_chunk != "auto":
            return [min(int(self._want_chunk), self.ctx_cap)]
        return [min(c, self.ctx_cap) for c in self.CHUNK_CANDIDATES]

    def _chunk_for(self, allowance: int) -> int:
        cands = self.candidates()
        for rows in cands:
            if self.fixed_bytes(rows) + self.act_bytes_for(rows) <= allowance:
                return rows
        return cands[-1]

    def act_bytes_for(self, rows: int) -> int:
        got = self._act.get(rows)
        if got is None:
            got = self._act[rows] = self._walk_high(rows) + self._act_statics(rows)
        return got

    def _act_statics(self, rows: int) -> int:
        return 0

    def _walk_high(self, rows: int) -> int:
        a = self.arena
        was, a.high = a.high, 0
        walk = None
        try:
            with self._walk_scope(rows) as walk, ops.dummy_run():
                walk()
            return a.high
        finally:
            a.high, a.at = was, 0
            self.n_eager_forwards = 0
            walk = None
            torch.cuda.empty_cache()

    @contextlib.contextmanager
    def _walk_scope(self, rows: int) -> Iterator[Callable[[], torch.Tensor]]:
        raise NotImplementedError


    def _pick_block_size(self, block_size: int | None) -> int:
        if block_size is None:
            return ops.KV_BLOCK_SIZES[0]
        if block_size not in ops.KV_BLOCK_SIZES:
            raise ValueError(f"block_size={block_size}: this build has KV cache kernels for "
                             f"{', '.join(str(n) for n in ops.KV_BLOCK_SIZES)}")
        return block_size

    def _check_seqs(self, max_num_seqs: int) -> None:
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
        n = a.freeze(max(a.high, a.planner.high) + extra)
        if self.plan_activations:
            a.planner.buf = a.buf
        torch.cuda.empty_cache()
        return n

    def _ceiling_advice(self, want: float, lower: str, needs: str,
                        fits: Callable[[], int]) -> str:
        if want <= 0.99:
            return (f"Try --gpu-memory-utilization {want:.2f} if this device is yours alone, or "
                    f"lower {lower}.")
        n = fits()
        return (f"Raising it will not be enough on its own: that context needs {needs}. Even at "
                f"0.98 this device holds {n} tokens, so --max-model-len {n // 1024 * 1024} "
                f"--gpu-memory-utilization 0.98 is about the shape of it.")

    @property
    def _graphs_full(self) -> bool:
        return self.budget.full

    def _graph_pool(self) -> object:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
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
        with torch.cuda.graph(g, pool=self._graph_pool()):
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

    def linear_advance(self, state_indices: torch.Tensor, num_accepted: torch.Tensor, B: int,
                       T: int) -> None:
        self._no("linear-attention state to advance", "runner.linear_mods")

    _ABSENT = {
        "draft_logits": ("buffer for a draft model's logits", "runner.num_spec"),
        "mtp_h": ("buffer for the MTP head's hidden state", "an mtp.* tree in the checkpoint"),
    }

    class Absent(SnowLLMError, AttributeError):
        pass

    def __getattr__(self, name: str) -> None:
        hit = GraphRunner._ABSENT.get(name)
        if hit is None:
            raise AttributeError(name)
        raise GraphRunner.Absent(self._why(*hit))

    tunable_rope = True
    prev_context = 0
    ring_blocks = 0
    linear_mods: list[tuple[int, GatedDeltaNet]] = []
    tap_at: dict[int, int] | None = None
    tap_layers: tuple[int, ...] = ()
    num_taps = 0
    reserve_bytes = 0
    draft_blocks = 0
    draft_block_bytes = 0
    index_block_bytes = 0
    PROBE_BLOCKS = 8
    draft_geo: DraftGeometry = None
    prefix_slots = 0


class Runner(GraphRunner):
    CHUNK_CANDIDATES = PREFILL_CHUNK_CANDIDATES

    def _alloc(self, sizes: Callable[[int], Sequence[int]], nb: int) -> list[torch.Tensor]:
        return [self.slabs.take(n).span() for n in sizes(nb)]

    def _alloc_kv(self, nb: int) -> tuple[torch.Tensor, ...]:
        pools = self._alloc(lambda b: ops.kv_pool_bytes(b, self.kv_int8, self.block_size), nb)
        return tuple(t.view(torch.int8) if self.kv_int8 else t for t in pools)

    def _alloc_kv_scale(self, nb: int) -> tuple[torch.Tensor, ...]:
        return tuple(t.view(torch.bfloat16)
                     for t in self._alloc(lambda b: ops.kv_scale_bytes(b, self.block_size), nb))

    def _autosize_kv_blocks(self, pools: int, off_top: int = 0) -> int:
        per_block = (kv_block_bytes(pools, self.block_size, self.kv_int8)
                     + self.draft_block_bytes + self.index_block_bytes)
        blocks = self._afford(per_block, off_top)
        need = self.max_blocks_per_seq
        if blocks >= need:
            return blocks
        free, total = torch.cuda.mem_get_info()
        held = (f" and {self.reserve_bytes / (1 << 30):.1f} GiB reserved for the draft model's "
                f"weights" if self.reserve_bytes else "")
        want = ((total - free) + off_top + need * per_block
                + self.reserve_bytes) / total + 0.005
        why = (f"the KV pool cannot carry one sequence at max_model_len={self.ctx_cap}: it holds "
               f"{blocks} of the {need} blocks that needs. "
               f"{(total - free) / (1 << 30):.1f} GiB of this device's "
               f"{total / (1 << 30):.1f} GiB is already taken by the weights, the "
               f"{self.max_prefill_tokens}-token prefill chunk's activations and the state pool "
               f"sized off max_num_seqs={self.max_num_seqs}{held}, against a "
               f"gpu_memory_utilization={self.gpu_util} ceiling of "
               f"{total * self.gpu_util / (1 << 30):.1f} GiB. ")
        raise ValueError(why + self._ceiling_advice(
            want, "--max-num-seqs",
            f"{need * per_block / (1 << 30):.1f} GiB of pool at {per_block / 1024:.0f} KiB a "
            f"block",
            lambda: self._afford(per_block, off_top, 0.98) * self.block_size))

    @staticmethod
    def _release_previous(model: "Qwen3_5MoeForCausalLM",
                          pooled_types: tuple[type, ...]) -> None:
        found = False
        if (slabs := getattr(model, "kv_slabs", None)) is not None:
            slabs.release()
            found = True
        for m in model.modules():
            if isinstance(m, pooled_types) and (getattr(m, "kv", None) is not None
                                                or getattr(m, "state", None) is not None):
                m.kv = m.kv_scale = m.state = m.retain = m.ckpt = None
                found = True
        if getattr(model.lm_head, "scratch", None) is not None:
            model.lm_head.scratch = None
            found = True
        if found:
            gc.collect()
            torch.cuda.empty_cache()

    def _check_state_pool(self) -> None:
        allow = self._allowance()
        need = self.plan.state_bytes()
        if need <= allow:
            return
        per_slot = need / max(1, self.plan.state_rows)
        mul = 1 if self.roll_forward else self.T
        fits = max(1, int(allow / 2 / (mul * per_slot)))
        wide = ("" if mul == 1 else
                f" x (num_spec + 1) = {self.T}")
        knob = ("" if not self.drafting else
                f" (or a narrower --dflash-block, which is what sets the width here)"
                if self.roll_forward else
                f" (or lower --num-spec, which divides this pool by (num_spec + 1))")
        raise ValueError(
            f"the linear-attn state pool for max_num_seqs={self.max_num_seqs}{wide} wants "
            f"{need / (1 << 30):.1f} GiB and only {allow / (1 << 30):.1f} GiB is "
            f"left after the weights. Try --max-num-seqs {fits}, which leaves about as much again "
            f"for the KV pool{knob}.")

    def __init__(self, model: "Qwen3_5MoeForCausalLM", num_kv_blocks: int | None,
                 max_blocks_per_seq: int, max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                 max_prefill_tokens: int | str = "auto", num_spec: int = 0,
                 kv_int8: bool = False, block_size: int | None = None,
                 gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
                 reserve_bytes: int = 0, tap_layers: tuple[int, ...] = (),
                 draft_geo: DraftGeometry = None, prefix_memory_ratio: float = 0.0,
                 plan_activations: bool = True) -> None:
        self.geo = model.geo
        self.prefix_memory_ratio = prefix_memory_ratio
        self.plan_activations = plan_activations
        self.reserve_bytes = int(reserve_bytes)
        self.draft_geo = draft_geo
        self.tap_layers = tuple(tap_layers)
        self.num_taps = len(self.tap_layers)
        self.block_size = self._pick_block_size(block_size)
        self.draft_block_bytes = draft_pool_bytes(draft_geo, 1, self.block_size)
        self.drafting = model.mtp is not None or bool(self.num_taps)
        self.num_spec = num_spec if self.drafting else 0
        self.T = self.num_spec + 1
        self.roll_forward = bool(self.num_taps)
        self.slots_per_request = 1 if self.roll_forward else self.T
        self._check_seqs(max_num_seqs)
        self.decode_rows = max_num_seqs * self.T
        self.attn_q_tokens = min(self.T, ops.PAGED_DECODE_MAX_Q_TOKENS)
        if num_spec and not self.drafting:
            raise ValueError(
                "num_spec > 0 needs something to propose with, and this runner has neither: no "
                "mtp.* tree in the checkpoint and no DFlash draft beside it (Engine's "
                "dflash_path, which brings its own weights and passes num_taps through).")
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.gpu_util = gpu_memory_utilization
        self.dummy_slot = max_num_seqs * self.slots_per_request

        H = self.geo.hidden
        B = max_num_seqs
        self.kv_int8 = kv_int8

        self.kv = {}
        self.kv_scale = {}
        self.state = {}

        POOLED = (FullAttention, GatedDeltaNet)
        self._release_previous(model, POOLED)
        self.slabs = model.kv_slabs = ops.Slabs()
        pooled: list[tuple[int, object]] = []
        bound = set()
        for i, layer in enumerate(model.layers):
            for m in layer.modules():
                if isinstance(m, POOLED) and id(m) not in bound:
                    bound.add(id(m))
                    pooled.append((i, m))
        nxt = len(model.layers)
        self.mtp_layer = nxt if model.mtp is not None else None
        for m in model.modules():
            if isinstance(m, POOLED) and id(m) not in bound:
                bound.add(id(m))
                pooled.append((nxt, m))
                nxt += 1
        full = [(i, m) for i, m in pooled if isinstance(m, FullAttention)]
        linear = [(i, m) for i, m in pooled if isinstance(m, GatedDeltaNet)]
        self.linear_mods = linear
        self.full_mods = full

        geo = self.geo
        self.ctx_cap = max_blocks_per_seq * self.block_size
        self.index_block_bytes = self._index_block_bytes(len(full))
        self.plan = KvPlan(geo, self.ctx_cap, self.block_size, max_num_seqs, len(full),
                           len(linear), self.T,
                           self.num_taps, self.kv_int8, model.mtp is not None,
                           self.draft_block_bytes, self.index_block_bytes)

        self._allow = self._allowance()
        self._want_chunk = max_prefill_tokens
        self._act: dict[int, int] = {}

        self._check_state_pool()
        rows = self.plan.state_rows
        for idx, attn in linear:
            attn.state = self.state[idx] = (
                torch.zeros(rows, geo.lin_conv_state, geo.lin_conv_dim,
                            dtype=torch.bfloat16, device="cuda"),
                torch.zeros(rows, geo.lin_num_v_heads, geo.lin_head_k, geo.lin_head_v,
                            dtype=torch.float32, device="cuda"),
            )
        if self.roll_forward:
            for _, attn in linear:
                attn.retain = tuple(ops.empty_bytes(n)
                                    for n in ops.linear_attn_retain_bytes(self.decode_rows))
        self._extra_pools()

        M = max(self._probe_rows(), self.decode_rows)

        if self.drafting:
            self.draft_logits = torch.empty(self.decode_rows, self.geo.vocab_size,
                                            dtype=torch.float32, device="cuda")
        if model.mtp is not None:
            self.mtp_h = torch.zeros(max_num_seqs, H, dtype=torch.bfloat16, device="cuda")

        self.arena = ops.Arena()
        self.x = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
        self.taps = (torch.zeros(M, self.num_taps * H, dtype=torch.bfloat16, device="cuda")
                     if self.num_taps else None)

        self.d_mscale = torch.ones(1, dtype=torch.float32, device="cuda")
        self.logits = torch.empty(self.decode_rows, self.geo.vocab_size, dtype=torch.float32,
                                  device="cuda")
        scratch = ops.lm_head_scratch_bytes(self.decode_rows)
        if scratch:
            model.lm_head.scratch = ops.empty_bytes(scratch)

        self.num_slots = ops.paged_decode_num_slots(B)
        self.decode_ws = ops.empty_bytes(
                ops.paged_decode_workspace_size(self.num_slots, self.attn_q_tokens)).zero_()
        self.decode_plan = torch.zeros(ops.paged_decode_plan_elems(B, self.num_slots), dtype=torch.int32,
                                       device="cuda")

        self.d_ids = torch.zeros(self.decode_rows, dtype=torch.int64, device="cuda")
        self.d_pos: dict[int, torch.Tensor] = {}
        self.d_slot = torch.full((self.decode_rows,), -1, dtype=torch.int32, device="cuda")
        self.d_seq = torch.ones(B, dtype=torch.int32, device="cuda")
        self.d_bt = torch.zeros(B, max_blocks_per_seq, dtype=torch.int32, device="cuda")
        self.d_sidx = torch.full((self.decode_rows,), self.dummy_slot, dtype=torch.int32,
                                 device="cuda")
        self.d_nacc = torch.ones(B, dtype=torch.int32, device="cuda")

        self._init_graphs()
        self.max_prefill_tokens = self._check_chunk(self._chunk_for(self._allow))
        M = max(self.max_prefill_tokens, self.decode_rows)
        if self.x.shape[0] != M:
            self.x = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
            if self.taps is not None:
                self.taps = torch.zeros(M, self.num_taps * H, dtype=torch.bfloat16, device="cuda")
            torch.cuda.empty_cache()
        self.arena_bytes = self._reserve_activations()

        self._pools, self._pooled = len(full), full
        self._want_blocks = None if num_kv_blocks is None else int(num_kv_blocks)
        if num_kv_blocks is None:
            self._autosize_kv_blocks(len(full))
        self.num_kv_blocks = self.PROBE_BLOCKS if num_kv_blocks is None else int(num_kv_blocks)
        self.kv_bytes = self.num_kv_blocks * kv_block_bytes(len(full), self.block_size,
                                                           self.kv_int8)
        self.draft_blocks = self.num_kv_blocks if self.draft_geo is not None else 0
        self._take_pools()

    def _take_pools(self) -> None:
        for idx, attn in self._pooled:
            attn.kv = self.kv[idx] = self._alloc_kv(self.num_kv_blocks)
            if self.kv_int8:
                attn.kv_scale = self.kv_scale[idx] = self._alloc_kv_scale(self.num_kv_blocks)

    def finish_pools(self, graph_bytes: int = 0) -> int:
        self.slabs.release()
        for idx, attn in self._pooled:
            attn.kv = self.kv[idx] = None
            if self.kv_int8:
                attn.kv_scale = self.kv_scale[idx] = None
        each = self.plan.state_slot_bytes()
        if each and self.prefix_memory_ratio > 0:
            self.prefix_slots = max(0, min(int(self.prefix_memory_ratio * self._allow),
                                           self._allowance())) // each
            self._alloc_ckpt(self.prefix_slots)
        if self._want_blocks is None:
            self.num_kv_blocks = self._autosize_kv_blocks(self._pools, graph_bytes)
        self.kv_bytes = self.num_kv_blocks * kv_block_bytes(self._pools, self.block_size,
                                                           self.kv_int8)
        self.draft_blocks = self.num_kv_blocks if self.draft_geo is not None else 0
        self._take_pools()
        return self.num_kv_blocks

    def _alloc_ckpt(self, rows: int) -> None:
        geo = self.geo
        for _, attn in self.linear_mods:
            attn.ckpt = (
                torch.zeros(rows, geo.lin_conv_state, geo.lin_conv_dim,
                            dtype=torch.bfloat16, device="cuda"),
                torch.zeros(rows, geo.lin_num_v_heads, geo.lin_head_k, geo.lin_head_v,
                            dtype=torch.float32, device="cuda"),
            ) if rows else None

    def release(self, slot: int) -> None:
        pass

    def grow(self, r: "Request", n: int = 1) -> bool:
        return True

    def admit(self, r: "Request", length: int, covered: int = 0) -> bool:
        return True

    def commit(self, batch: "list[Request]") -> None:
        pass

    def _ctx(self, b: Batch, plan_key: tuple = PLAN_MAIN) -> ForwardContext:
        M = b.input_ids.numel()
        if not b.is_prefill:
            plan_key = PLAN_MTP_DECODE if plan_key is PLAN_MTP else PLAN_DECODE
        return ForwardContext(
            batch=b, M=M, eps=self.model.eps,
            path=ops.Path.PREFILL if b.is_prefill else ops.Path.DECODE,
            arena=self.arena, plan_key=plan_key,
            decode_plan=self.decode_plan, decode_ws=self.decode_ws, num_slots=self.num_slots,
            block_size=self.block_size,
            kv_int8=self.kv_int8, mscale=self.d_mscale, tap_at=self.tap_at,
            x=self.x[:M],
            taps=self.taps[:M] if self.taps is not None else None,
        )

    def _probe_rows(self) -> int:
        return max(self.candidates())

    def _act_statics(self, rows: int) -> int:
        return rows * self.geo.hidden * 2 * (1 + self.num_taps)

    def fixed_bytes(self, chunk: int = 0) -> int:
        return self.plan.fixed()

    @contextlib.contextmanager
    def _walk_scope(self, rows: int) -> Iterator[Callable[[], torch.Tensor]]:
        with self._probe_pools():
            yield lambda: self._forward_eager(self._probe_batch(rows, True))

    @contextlib.contextmanager
    def _probe_pools(self) -> Iterator[None]:
        was = [(m, m.kv, m.kv_scale) for _, m in self.full_mods]
        for m, _, _ in was:
            m.kv = self._alloc_kv(0)
            m.kv_scale = self._alloc_kv_scale(0) if self.kv_int8 else m.kv_scale
        try:
            yield
        finally:
            for m, kv, scale in was:
                m.kv, m.kv_scale = kv, scale

    def _probe_batch(self, rows: int, prefill: bool) -> Batch:
        B = 1 if prefill else self.max_num_seqs
        M, T = rows, rows // (1 if prefill else self.max_num_seqs)
        z = torch.zeros(M, dtype=torch.int64, device="cuda")
        total, qmap = ops.prefill_q_plan([M] if prefill else [T] * B)
        return Batch(
            input_ids=z, positions=z.expand(3, M).contiguous(),
            slot_mapping=torch.zeros(M, dtype=torch.int32, device="cuda"),
            block_tables=torch.zeros(B, self.max_blocks_per_seq, dtype=torch.int32, device="cuda"),
            seq_lens=torch.full((B,), max(T, 1), dtype=torch.int32, device="cuda"),
            state_indices=torch.zeros(B if prefill or self.roll_forward else M,
                                      dtype=torch.int32, device="cuda"),
            has_state=torch.zeros(B, dtype=torch.int32, device="cuda"),
            cu_seqlens=torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32,
                                    device="cuda"),
            total_q_blocks=total, q_block_map=qmap,
            num_accepted=None if prefill else torch.ones(B, dtype=torch.int32, device="cuda"),
            roll_forward=self.roll_forward and not prefill,
            last_row=torch.zeros(1, dtype=torch.int64, device="cuda") if prefill else None,
            is_prefill=prefill, num_tokens=M, need_logits=False,
        )

    def _reserve_activations(self) -> int:
        a = self.arena
        d, chunk = self.decode_rows, self.max_prefill_tokens
        with self._probe_pools():
            self._plan_walk(PLAN_MAIN, chunk, lambda b: self._forward_eager(b), True)
            self._plan_walk(PLAN_DECODE, d, lambda b: self._forward_eager(b), False)
            if self.model.mtp is not None:
                self._plan_walk(PLAN_MTP_DECODE, d,
                                lambda b: self.mtp_draft(self.x[:d], b.input_ids, b), False)
                self._plan_walk(PLAN_MTP, chunk,
                                lambda b: self.mtp_draft(self.x[:chunk], b.input_ids, b), True)
        self.n_eager_forwards = 0
        n = self._freeze_arena()
        a.buf.zero_()
        return n

    def _plan_walk(self, key: tuple, rows: int, walk: Callable[[Batch], object],
                   prefill: bool) -> None:
        if rows <= 0:
            return
        with ops.dummy_run() as t:
            walk(self._probe_batch(rows, prefill))
        plan = ops.pack(t, rows)
        have = self.arena.planner.plans.get(key)
        if have is None or rows > have.rows:
            self.arena.planner.install(key, plan)

    def forward(self, b: Batch) -> torch.Tensor:
        if not b.is_prefill:
            B, M = b.batch_size, b.input_ids.numel()
            g = self.graphs.get((B, M // B)) if M % B == 0 else None
            if g is not None:
                return self._replay(b, g)
            self._miss((B, M))
        return self._forward_eager(b)

    def _forward_eager(self, b: Batch) -> torch.Tensor:
        self.n_eager_forwards += 1
        ctx = self._ctx(b)
        with span(f"forward {'prefill' if b.is_prefill else 'decode'} M={ctx.M}"):
            self._plan_attn(ctx)
            x = self.model(ctx)
            last = x if not b.is_prefill else x[b.last_row]
            self.last_hidden = x
            out = self.logits[: last.shape[0]]
            if b.need_logits:
                with span("lm_head"):
                    out = self.model.lm_head(last, self.logits)
            return out

    def linear_advance(self, state_indices: torch.Tensor, num_accepted: torch.Tensor, B: int,
                       T: int) -> None:
        for _, attn in self.linear_mods:
            ops.linear_attn_advance(attn.retain, attn.w, state_indices, num_accepted, *attn.state,
                                    B, T)

    def _plan_attn(self, ctx: ForwardContext) -> None:
        if not ctx.batch.varlen_attn:
            with span("attn_plan"):
                ops.paged_attn_decode_plan(ctx.batch.seq_lens, ctx.decode_plan,
                                           ctx.num_slots, ctx.block_size)

    def mtp_draft(self, hidden: torch.Tensor, next_ids: torch.Tensor,
                  b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self._ctx(b, PLAN_MTP)
        with span(f"mtp draft M={ctx.M}"):
            self._plan_attn(ctx)
            x = self.model.mtp(ctx, self.model, hidden, next_ids)
            last = x if not b.is_prefill else x[b.last_row]
            with span("mtp lm_head"):
                return self.model.lm_head(last, self.draft_logits), last

    def _capture_shape(self, shape: DecodeShape) -> bool:
        B, T = shape.B, shape.T
        M = B * T
        self.d_pos[M] = torch.zeros(3, M, dtype=torch.int64, device="cuda")
        if T > 1:
            self.d_seq[:B] = T
        b = self._pad_batch(B, T, shape.chunk if T > 1 else False)
        self.graphs[(B, T)] = self._capture(f"decode graph B={B:2d} T={T} M={M:3d}",
                                            lambda: self._forward_eager(b))[0]
        return True

    def _pad_batch(self, B: int, T: int, chunk: bool = False) -> Batch:
        M = B * T
        if T == 1:
            return Batch(
                input_ids=self.d_ids[:B], positions=self.d_pos[B],
                slot_mapping=self.d_slot[:B], block_tables=self.d_bt[:B],
                seq_lens=self.d_seq[:B], state_indices=self.d_sidx[:B],
                is_prefill=False, num_tokens=B,
            )
        return Batch(
            input_ids=self.d_ids[:M], positions=self.d_pos[M], slot_mapping=self.d_slot[:M],
            block_tables=self.d_bt[:B], seq_lens=self.d_seq[:B],
            state_indices=self.d_sidx[:B if self.roll_forward else M],
            roll_forward=self.roll_forward,
            is_prefill=False, num_tokens=M, num_accepted=self.d_nacc[:B],
            cu_seqlens=torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32,
                                    device="cuda"),
            total_q_blocks=ops.prefill_q_plan([T] * B)[0],
            chunk_decode=chunk,
        )

    def _replay(self, b: Batch, g: torch.cuda.CUDAGraph) -> torch.Tensor:
        B, M = b.batch_size, b.input_ids.numel()
        self.d_ids[:M] = b.input_ids
        self.d_pos[M][:, :M] = b.positions
        self.d_slot[:M] = b.slot_mapping
        self.d_seq[:B] = b.seq_lens
        self.d_bt[:B, : b.block_tables.shape[1]] = b.block_tables
        self.d_sidx[:b.state_indices.numel()] = b.state_indices
        if b.num_accepted is not None:
            self.d_nacc[:B] = b.num_accepted
        with span(f"decode graph replay B={B} M={M}"):
            g.replay()
        self.n_graph_replays += 1
        self.last_hidden = self.x[:M]
        return self.logits[:M]

    def set_rope(self, inv_freq: torch.Tensor, mscale: float) -> None:
        self.model.inv_freq.copy_(inv_freq.to(self.model.inv_freq))
        self.d_mscale.fill_(mscale)

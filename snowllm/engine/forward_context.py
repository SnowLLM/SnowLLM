# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .. import ops

PLAN_MAIN = ("main",)
PLAN_MTP = ("mtp",)
PLAN_MTP_DECODE = ("mtp_decode",)
PLAN_DECODE = ("decode",)
PLAN_DRAFT = ("draft",)


def i32(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32, device="cuda")


def i64(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device="cuda")


def positions(p: list[int]) -> torch.Tensor:
    return i64(p).expand(3, len(p)).contiguous()


class DecodeShape(NamedTuple):
    B: int
    T: int
    chunk: bool = False


def plain_decode_shapes(max_num_seqs: int) -> list[DecodeShape]:
    return [DecodeShape(n, 1) for n in range(1, max_num_seqs + 1)]


@dataclass
class Batch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    is_prefill: bool
    num_tokens: int
    state_indices: torch.Tensor
    num_accepted: torch.Tensor | None = None
    last_row: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    has_state: torch.Tensor | None = None
    total_q_blocks: int = 0
    q_block_map: torch.Tensor | None = None
    need_logits: bool = True
    embeds: torch.Tensor | None = None
    embed_rows: torch.Tensor | None = None
    ckpt_at: torch.Tensor | None = None
    ckpt_slots: torch.Tensor | None = None
    ckpt_n: int = 0
    chunk_decode: bool = False
    roll_forward: bool = False
    seq_of_row: torch.Tensor | None = None
    prev_ids: list[list[int]] | None = None
    host_ids: list[int] | None = None
    plans: dict | None = None
    inplace: bool = False
    split: bool = False
    block_q: bool = False

    @property
    def batch_size(self) -> int:
        return self.seq_lens.numel()

    @property
    def tokens_per_req(self) -> int:
        return self.input_ids.numel() // self.batch_size

    @property
    def varlen_attn(self) -> bool:
        if self.is_prefill:
            return True
        return self.tokens_per_req > ops.PAGED_DECODE_MAX_Q_TOKENS and not self.chunk_decode


@dataclass
class ForwardContext:
    batch: Batch
    M: int
    eps: float
    path: ops.Path
    arena: ops.Arena
    plan_key: tuple

    x: torch.Tensor | None = None
    decode_plan: torch.Tensor | None = None
    decode_ws: torch.Tensor | None = None
    num_slots: int = 0
    block_size: int = 0
    kv_int8: bool = False
    mscale: torch.Tensor | None = None

    residual: torch.Tensor | None = None
    blk: torch.Tensor | None = None
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None

    taps: torch.Tensor | None = None
    tap_at: dict | None = None

    @property
    def decode(self) -> bool:
        return not self.batch.is_prefill

    @property
    def shuffled_attn_out(self) -> bool:
        return self.batch.is_prefill and not self.kv_int8

    def streams(self, a: ops.Arena, M: int) -> tuple[torch.Tensor, torch.Tensor]:
        H = self.x.shape[1]
        return a.new(M, H), a.new(M, H)

    def open(self) -> None:
        self.arena.planner.begin(self.plan_key)
        self.arena.reset()
        a, M = self.arena, self.M
        self.residual, self.blk = self.streams(a, M)
        self.cos = a.new(M, 64, dtype=torch.float32)
        self.sin = a.new(M, 64, dtype=torch.float32)

    def close(self) -> None:
        self.arena.end_plan()

    def apply_mscale(self) -> None:
        self.cos.mul_(self.mscale)
        self.sin.mul_(self.mscale)


class QsaStep(NamedTuple):
    resume: torch.Tensor
    snaps: torch.Tensor
    pos: torch.Tensor
    comp: torch.Tensor
    weights: torch.Tensor
    cells: torch.Tensor


@dataclass
class Qwen4ExpContext(ForwardContext):
    geo: object = None
    ple_emb: torch.Tensor | None = None
    ple_has_state: torch.Tensor | None = None
    stream_buf: torch.Tensor | None = None
    inv_freq: torch.Tensor | None = None
    qsa_max_blocks: int = 0
    qsa_seq_of_row: torch.Tensor | None = None
    qsa_cu_seqlens: torch.Tensor | None = None
    qsa_lens: list | None = None
    qsa: QsaStep | None = None
    hc_xn: torch.Tensor | None = None
    xn_live: bool = False

    def open(self) -> None:
        super().open()
        a, b, g, M = self.arena, self.batch, self.geo, self.M
        self.xn_live = False
        self.hc_xn = None if b.is_prefill else a.new(M, g.hc_count, g.hidden)
        B = b.batch_size
        slots = b.state_indices.view(B, -1)
        cells = a.new(M, dtype=torch.int64)
        if not b.is_prefill:
            T = b.tokens_per_req
            torch.add(b.seq_lens.view(-1, 1).to(torch.int64) - T,
                      torch.arange(T, device="cuda", dtype=torch.int64), out=cells.view(-1, T))
        comp = a.new(B, dtype=torch.int32)
        torch.div(b.seq_lens, g.index_ratio, rounding_mode="floor", out=comp)
        weights = a.new(M, g.index_n_heads, dtype=torch.float32)
        weights.fill_(1.0 / math.sqrt(float(g.index_head_dim)))
        self.qsa = QsaStep(a.copy(slots[:, 0]),
                           a.copy(slots[:, :min(slots.shape[1], M // B)]).view(-1),
                           a.copy(b.positions[:, :M]), comp, weights, cells)

    def take_xn(self) -> torch.Tensor | None:
        live, self.xn_live = self.xn_live, False
        return self.hc_xn if live else None

    def streams(self, a: ops.Arena, M: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.geo
        carrier = (a.new(M, g.hc_count, g.hidden) if self.stream_buf is None
                   else self.stream_buf[:M])
        return carrier, a.new(M, g.hidden)


@dataclass
class Dsv4Context(ForwardContext):
    cache: object = None
    geo: object = None
    decode_max_m: int = 1
    embedded: torch.Tensor | None = None
    pre_head: torch.Tensor | None = None
    rot: dict | None = None

    banks: tuple[torch.Tensor, torch.Tensor] | None = None
    hidden: torch.Tensor | None = None

    def open(self) -> None:
        self.arena.planner.begin(self.plan_key)
        self.arena.reset()
        a, t, g = self.arena, self.M, self.geo
        self.banks = (a.new(t, g.hc_mult, g.hidden), a.new(t, g.hc_mult, g.hidden))
        self.hidden = self.embedded if self.embedded is not None else a.new(t, g.hidden)

    def close(self) -> None:
        self.rot = self.banks = self.hidden = None
        self.arena.end_plan()


class Dsv4Walker:
    def __init__(self, arena: ops.Arena | None = None, decode_max_m: int = 1) -> None:
        self.arena = arena if arena is not None else ops.Arena()
        self.decode_max_m = max(1, int(decode_max_m))
        self.tap_at: dict[int, int] = {}
        self.taps: torch.Tensor | None = None

    def set_decode_max_m(self, rows: int) -> None:
        self.decode_max_m = max(1, int(rows))

    def context(self, stack: object, b: Batch, cache: object, tag: tuple = PLAN_MAIN,
                embedded: torch.Tensor | None = None,
                pre_head: torch.Tensor | None = None) -> Dsv4Context:
        return Dsv4Context(
            batch=b, M=b.input_ids.numel(), eps=stack.geo.eps,
            path=ops.Path.PREFILL if b.is_prefill else ops.Path.DECODE,
            arena=self.arena, plan_key=self.plan_key(b, tag, embedded),
            cache=cache, geo=stack.geo, decode_max_m=self.decode_max_m, embedded=embedded,
            pre_head=pre_head, taps=self.taps, tap_at=self.tap_at)

    def bare_context(self, stack: object, rows: int) -> Dsv4Context:
        return Dsv4Context(batch=None, M=rows, eps=stack.geo.eps, path=ops.Path.DECODE,
                           arena=self.arena, plan_key=(), geo=stack.geo,
                           decode_max_m=self.decode_max_m)

    def plan_key(self, b: Batch, tag: tuple = PLAN_MAIN,
                 embedded: torch.Tensor | None = None) -> tuple:
        return (tag, b.inplace, b.last_row is not None, embedded is not None,
                b.split, tuple(sorted(self.tap_at)),
                b.input_ids.numel() == 1, b.input_ids.numel() <= self.decode_max_m,
                tuple((int(pl.keep_dst is not None), int(pl.n_new > 0), int(pl.max_n_comp > 0))
                      for _, pl in sorted(b.plans.items())))

    def plan_widths(self, widest: int) -> tuple:
        cap = self.decode_max_m
        got = [1]
        if widest > 1:
            got.append(min(widest, cap))
        if widest > cap:
            got.append(widest)
        return tuple(got)

    def plan_for(self, stack: object, ctx: Dsv4Context, after: int = 0) -> None:
        planner = self.arena.planner
        if planner.buf is None:
            return
        key, rows = ctx.plan_key, ctx.M
        have = planner.plans.get(key)
        if have is not None and rows <= have.rows:
            return
        with ops.dummy_run() as t:
            held = stack(ctx)
        plan = ops.pack(t, rows)
        del held
        if plan.high + after <= planner.buf.numel():
            planner.install(key, plan)
        else:
            planner.plans.pop(key, None)

    def run(self, stack: object, tokens: torch.Tensor, positions: torch.Tensor,
            cache: object) -> torch.Tensor:
        first = int(positions[0].item())
        b = cache.batch(tokens, positions, [first], [tokens.numel()], [0])
        return stack(self.context(stack, b, cache))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import torch

from . import ops
from ._capi import SnowLLMError
from .block_manager import BlockAllocator, StateSlots, block_tables, slot_mapping
from .forward_context import Batch, i32, i64, positions, prefill_rows
from .model import DEFAULT_GPU_UTIL, Runner
from .prefix_cache import CKPT_EVERY, PrefixCache, StateStore, checkpoints, thin
from .request import Request, SamplingParams
from .sampler import Sampler
from .spec_decode import SPEC_MAX_STEP_ROWS, SpecDecoder


@dataclass
class StepAccounting:
    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    prefill_steps: int = 0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    decode_steps: int = 0
    decode_rows: int = 0

    def add(self, kind: str, tokens: int, seconds: float, rows: int = 0) -> None:
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
    accounting: dict = field(default_factory=dict)


class Engine:
    def __init__(self, model: torch.nn.Module, num_kv_blocks: "int | None" = None,
                 max_num_seqs: int = 16,
                 max_model_len: int = 8192, stop_token_ids: Sequence[int] = (),
                 seed: int | None = None, enforce_eager: bool = False, num_spec: int = 0,
                 kv_int8: bool = False, mtp_window: int = 0,
                 mtp_sinks: int = 64,
                 prefill_chunk: "int | str" = "auto", batch_prefill: bool = False,
                 account: bool = False, gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
                 preempt: bool = True, prefix_cache_gib: float = 0.0):
        self.max_blocks = ops.kv_blocks_for(max_model_len)
        mpt = "auto" if prefill_chunk == "auto" else min(int(prefill_chunk), max_model_len)
        self.runner = Runner(model, num_kv_blocks, self.max_blocks, max_num_seqs,
                             max_prefill_tokens=mpt, num_spec=num_spec, kv_int8=kv_int8,
                             gpu_memory_utilization=gpu_memory_utilization)
        self.prefill_chunk = self.runner.max_prefill_tokens
        num_kv_blocks = self.runner.num_kv_blocks
        if num_kv_blocks < self.max_blocks:
            raise SnowLLMError(
                f"the KV pool holds {num_kv_blocks} blocks ({num_kv_blocks * self.runner.block_size}"
                f" tokens) but one sequence at max_model_len={max_model_len} needs "
                f"{self.max_blocks}. A request that long could be admitted and then never finish, "
                f"so this is refused here: lower max_model_len, or lower max_num_seqs to leave the "
                f"pool more room.")

        self.num_spec = self.runner.num_spec
        self.T = self.num_spec + 1
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len

        self.blocks = BlockAllocator(num_kv_blocks)
        self.slots = StateSlots(max_num_seqs, self.T, self.runner.dummy_slot)
        self.sampler = Sampler(frozenset(stop_token_ids), seed)
        self.spec = (SpecDecoder(self.runner, self.sampler, self.slots, num_spec=self.num_spec,
                                 window=mtp_window, sinks=mtp_sinks)
                     if self.num_spec else None)

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._prefill_turn = True
        self.batch_prefill = batch_prefill
        self.n_batched_prefills = 0
        self.n_preemptions = 0
        self.preempt = preempt
        self.cache = None
        if prefix_cache_gib > 0 and self.runner.linear_mods:
            each = StateStore.bytes_each(self.runner.linear_mods)
            n = int(prefix_cache_gib * (1 << 30)) // each
            if n:
                self.cache = PrefixCache(self.blocks,
                                         StateStore(self.runner.linear_mods, n),
                                         self.runner.block_size)
        self.acct = StepAccounting() if account else None

        self._active_factor: float | None = None
        self._rope_cache: dict[float, tuple] = {}
        self._rope_orig_max = int(model.config.get("max_position_embeddings", 262144))
        self.graph_sizes = []
        if not enforce_eager:
            self.graph_sizes = (sorted(self.runner.capture_verify(self.spec.depth_for))
                                if self.num_spec else self.runner.capture_decode())

    def add(self, prompt: list[int], params: SamplingParams | None = None,
            rope_factor: float = 1.0, mrope: "torch.Tensor | None" = None, pos_delta: int = 0,
            embeds: "torch.Tensor | None" = None,
            embed_rows: "list[int] | None" = None) -> Request:
        p = params or SamplingParams()
        cap = min(self.max_model_len, int(rope_factor * self._rope_orig_max))
        if len(prompt) + p.max_new_tokens > cap:
            raise SnowLLMError(f"prompt {len(prompt)} + {p.max_new_tokens} new > {cap} "
                               f"(factor {rope_factor}, max_model_len {self.max_model_len})")
        r = Request(prompt=list(prompt), params=p, rope_factor=rope_factor, mrope=mrope,
                    pos_delta=pos_delta, embeds=embeds, embed_rows=list(embed_rows or []))
        self.waiting.append(r)
        return r

    def step(self) -> str:
        if self.acct is None:
            return self._step()[0]
        t0 = time.perf_counter()
        kind, tokens, rows = self._step()
        if kind != "idle":
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
                self._prefill_batch(group)
                return "prefill", sum(len(r.prompt) for r in group), 0
            if len(self.running) < self.max_num_seqs:
                cand = self._next_admissible()
                if cand is not None and self._admit(cand):
                    self.waiting.remove(cand)
                    self.running.append(cand)
                    pending = self._pending_prefill()

        decodable = [r for r in self.running if r.prefilled]
        if pending is not None and (not decodable or pending.num_prefilled == 0
                                    or self._prefill_turn):
            self._prefill_turn = False
            before = pending.num_prefilled
            self._prefill_chunk(pending)
            return "prefill", pending.num_prefilled - before, 0
        if decodable:
            self._prefill_turn = True
            emitted, rows = self._decode_or_verify(len(decodable))
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
            accounting=self.acct.snapshot() if self.acct else {})

    def _alloc_blocks(self, n: int, keep=None) -> "list[int] | None":
        while True:
            got = self.blocks.alloc(n)
            if got is not None or self.cache is None or not self.cache.evict(keep):
                return got

    def _admit(self, r: Request, use_cache: bool = True) -> bool:
        hit = (self.cache.lookup(r.prefill_src, r.prefill_len)
               if self.cache is not None and use_cache else None)
        shared = len(hit.blocks) if hit is not None else 0
        blocks = self._alloc_blocks(ops.kv_blocks_for(r.prefill_len) - shared, hit)
        if blocks is None:
            return False
        if self.cache is not None and use_cache:
            self.cache.took(hit)
        r.blocks = (self.blocks.retain(hit.blocks) + blocks) if hit is not None else blocks
        self.slots.take(r)
        if hit is not None:
            r.num_prefilled = len(hit.tokens)
            self.cache.store.restore(hit.ckpt, self.slots.name(r, 1)[0])
        if self._active_factor != r.rope_factor:
            self._active_factor = r.rope_factor
            self._activate_rope(r.rope_factor)
        return True

    def _retire_finished(self) -> None:
        for r in [r for r in self.running if r.done]:
            self.blocks.release(r.blocks)
            self.slots.give_back(r)
            self.running.remove(r)

    def _next_admissible(self) -> "Request | None":
        return next((r for r in self.waiting
                     if self._active_factor is None or r.rope_factor == self._active_factor), None)

    def _pending_prefill(self) -> "Request | None":
        return next((r for r in self.running if not r.prefilled and not r.done), None)

    def _decodable_batch(self, rows: int) -> list[Request]:
        while True:
            decodable = [r for r in self.running if r.prefilled]
            batch = [r for r in decodable if self.blocks.grow(r, rows)]
            if len(batch) == len(decodable):
                return batch
            if self.cache is not None and self.cache.evict():
                continue
            if not self.preempt:
                raise SnowLLMError(
                    f"KV pool exhausted mid-decode over {len(decodable)} requests and this engine "
                    f"was built with preempt=False, which is a promise that it would not have to: "
                    f"{self.blocks.total} blocks is too few for this workload")
            if self._preempt() is None:
                raise SnowLLMError(
                    f"KV pool exhausted mid-decode with one request running and nothing left to "
                    f"preempt: {self.blocks.total} blocks cannot carry it to max_model_len="
                    f"{self.max_model_len}")

    def _preempt(self) -> "Request | None":
        if len(self.running) < 2:
            return None
        r = self.running.pop()
        self.blocks.release(r.blocks)
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

    def _rope_table(self, factor: float):
        if factor not in self._rope_cache:
            from . import loader
            inv, mscale = loader.yarn_rope_table(self.runner.model.config, factor,
                                                 orig_max_pos=self._rope_orig_max)
            self._rope_cache[factor] = (inv.cuda(), mscale)
        return self._rope_cache[factor]

    def _activate_rope(self, factor: float) -> None:
        self.runner.set_rope(*self._rope_table(factor))

    def _prefill(self, spans: list[tuple[Request, int, int]]) -> Batch:
        n = sum(hi - lo for _, lo, hi in spans)
        M = prefill_rows(n)

        ids = torch.zeros(M, dtype=torch.int64, device="cuda")
        pos = torch.zeros(3, M, dtype=torch.int64, device="cuda")
        rows: list[tuple[int, int]] = []
        cu, last_row, off = [0], [], 0
        erows, eparts = [], []
        for i, (r, lo, hi) in enumerate(spans):
            base, li = off, hi - lo
            ids[base:base + li] = torch.tensor(r.prefill_src[lo:hi], dtype=torch.int64)
            pos[:, base:base + li] = r.prefill_positions(lo, hi)
            rows += [(i, p) for p in range(lo, hi)]
            ks = [k for k, p in enumerate(r.embed_rows) if lo <= p < hi]
            if ks:
                erows += [base + r.embed_rows[k] - lo for k in ks]
                eparts.append(r.embeds[i64(ks)])
            off += li
            cu.append(off)
            last_row.append(off - 1)

        rows += [(-1, 0)] * (M - len(rows))
        total_q, qmap = ops.prefill_q_plan([hi - lo for _, lo, hi in spans])
        resuming = [1 if lo else 0 for _, lo, _ in spans]
        bt = block_tables([r for r, _, _ in spans])
        return Batch(
            input_ids=ids, positions=pos, slot_mapping=slot_mapping(bt, rows),
            block_tables=bt,
            seq_lens=i32([hi for _, _, hi in spans]),
            last_row=i64(last_row),
            is_prefill=True, num_tokens=n,
            state_indices=i32([self.slots.name(r, 1)[0] for r, _, _ in spans]),
            cu_seqlens=i32(cu),
            has_state=i32(resuming) if any(resuming) else None,
            total_q_blocks=total_q,
            q_block_map=qmap,
            need_logits=all(hi == r.prefill_len and not r.replay for r, _, hi in spans),
            embeds=torch.cat(eparts) if eparts else None,
            embed_rows=i64(erows) if erows else None,
        )

    def _prefill_chunk(self, r: Request) -> None:
        lo = r.num_prefilled
        hi = min(lo + self.prefill_chunk, r.prefill_len)
        marks, idx = [], []
        if self.cache is not None:
            marks = checkpoints(lo, hi)
            idx = self.cache.reserve(len(marks))
            marks = thin(marks, len(idx))
        b = self._prefill([(r, lo, hi)])
        if marks:
            b.ckpt_at = i32([[m - lo for m in marks]])
            b.ckpt_slots = i32([idx])
            b.ckpt_n = len(marks)
        M = b.input_ids.numel()

        out = self.runner.forward(b)
        r.num_prefilled = hi
        for m, i in zip(marks, idx):
            self.cache.insert(r.prefill_src[:m], r.blocks, i)
        if not b.need_logits:
            if self.spec:
                self.spec.propose_after_prefill(r, b, lo, hi, M)
            return
        self.sampler.emit(out, [r])
        if self.spec and not r.done:
            logits, hid = self.spec.propose_after_prefill(r, b, lo, hi, M, r.out[-1])
            r.drafts = ops.argmax(logits).tolist()
            r.n_accepted = 1
            if self.num_spec > 1 and self.blocks.grow(r, self.num_spec):
                self.spec.propose_rest([r], hid, self.num_spec)
        self._retire_finished()

    def _batchable_prefill_group(self) -> list[Request]:
        if not self.batch_prefill or self.num_spec:
            return []
        cands = [r for r in self.waiting
                 if r.num_prefilled == 0 and not r.replay and len(r.prompt) <= self.prefill_chunk
                 and (self._active_factor is None or r.rope_factor == self._active_factor)]
        if len(cands) < 2:
            return []
        f = cands[0].rope_factor
        group, tokens = [], 0
        for r in (c for c in cands if c.rope_factor == f):
            if len(self.running) >= self.max_num_seqs:
                break
            if prefill_rows(tokens + len(r.prompt)) > self.runner.max_prefill_tokens:
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
        k = self.spec.depth_for(n) if self.spec else 0
        if k >= 1:
            batch = self._decodable_batch(self.spec.rows_needed(k))
            self.spec.verify_step(batch, k)
            emitted, rows = sum(r.n_accepted for r in batch), len(batch)
        else:
            emitted = rows = self._decode()
        self._retire_finished()
        return emitted, rows

    def _decode(self) -> int:
        batch = self._decodable_batch(1)
        p = [r.num_cached for r in batch]
        b = Batch(
            input_ids=i64([r.tokens[-1] for r in batch]),
            positions=positions([q + r.pos_delta for q, r in zip(p, batch)]),
            slot_mapping=slot_mapping(block_tables(batch), list(enumerate(p))),
            block_tables=block_tables(batch),
            seq_lens=i32([q + 1 for q in p]),
            is_prefill=False, num_tokens=len(batch),
            state_indices=i32([self.slots.name(r, 1)[0] for r in batch]),
        )
        self.sampler.emit(self.runner.forward(b), batch)
        for r in batch:
            r.drafts.clear()
        return len(batch)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import time

import torch

from . import ops
from .forward_context import Batch, ForwardContext, prefill_rows
from .layers import FullAttention, GatedDeltaNet
from .trace import span


PREFILL_CHUNK_CANDIDATES = (32768, 16384, 8192)
PREFILL_AUTOSIZE_RESERVE = 4 << 30

MAX_NUM_SEQS = 256
DEFAULT_MAX_NUM_SEQS = 128


class Runner:
    def _alloc_kv(self, nb: int):
        return tuple(ops.zero_bytes(n) for n in ops.kv_pool_bytes(nb, self.kv_int8))

    def _alloc_kv_scale(self, nb: int):
        return tuple(ops.zero_bytes(n) for n in ops.kv_scale_bytes(nb))

    def _m_buffers(self) -> tuple[dict, dict]:
        geo = self.geo
        H, D, Hq, Hk = geo.hidden, geo.head_size, geo.num_heads, geo.num_kv_heads
        act = {
            "residual": ((H,), torch.bfloat16),
            "x": ((H,), torch.bfloat16),
            "blk": ((H,), torch.bfloat16),
            "proj": ((geo.qkv_proj_n,), torch.bfloat16),
            "q": ((Hq, D), torch.bfloat16),
            "k": ((Hk, D), torch.bfloat16),
            "attn_out": ((Hq, D), torch.bfloat16),
            "cos": ((64,), torch.float32),
            "sin": ((64,), torch.float32),
        }
        scratch = {
            "qkv_scratch": ops.qkv_proj_scratch_bytes,
            "o_scratch": ops.attn_out_scale_oproj_scratch_bytes,
            "moe_ws": ops.moe_workspace_bytes,
            "lin_ws": ops.fused_linear_attn_workspace_bytes,
        }
        if self.model.mtp is not None:
            act["mtp_embed"] = ((H,), torch.bfloat16)
            act["mtp_cat"] = ((geo.mtp_fc_k,), torch.bfloat16)
            scratch["mtp_fc_scratch"] = ops.mtp_fc_scratch_bytes
        return act, scratch

    def _prefill_ws_bytes(self, M: int) -> int:
        act, scratch = self._m_buffers()
        return (sum(M * math.prod(shape) * dt.itemsize for shape, dt in act.values())
                + sum(size(M) for size in scratch.values()))

    def _autosize_prefill_tokens(self, ctx_cap: int) -> int:
        free, total = torch.cuda.mem_get_info()
        budget = free - max(PREFILL_AUTOSIZE_RESERVE, total // 10)
        for c in PREFILL_CHUNK_CANDIDATES:
            M = prefill_rows(min(c, ctx_cap))
            if self._prefill_ws_bytes(M) <= budget:
                return M
        return prefill_rows(min(PREFILL_CHUNK_CANDIDATES[-1], ctx_cap))

    def __init__(self, model, num_kv_blocks: int, max_blocks_per_seq: int,
                 max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                 max_prefill_tokens: "int | str" = "auto", num_spec: int = 0,
                 kv_int8: bool = False, block_size: int | None = None):
        self.geo = model.geo
        self.block_size = ops.KV_BLOCK_SIZE if block_size is None else block_size
        if self.block_size != ops.KV_BLOCK_SIZE:
            raise ValueError(f"block_size={self.block_size}: the KV cache kernels in this build "
                             f"are compiled for {ops.KV_BLOCK_SIZE}")
        self.num_spec = num_spec if model.mtp is not None else 0
        self.T = self.num_spec + 1
        if max_num_seqs > MAX_NUM_SEQS:
            raise ValueError(f"max_num_seqs={max_num_seqs} > {MAX_NUM_SEQS}, this engine's policy "
                             f"cap (model.py). No kernel stops there; the state pool is what does, "
                             f"at this many requests x (num_spec + 1) x 61.4 MiB.")
        self.decode_rows = max_num_seqs * self.T
        self.attn_q_tokens = min(self.T, ops.PAGED_DECODE_MAX_Q_TOKENS)
        self.lm_rows = model.lm_head.padded_rows(self.decode_rows)
        if num_spec and model.mtp is None:
            raise ValueError("num_spec > 0 needs a checkpoint with an mtp.* tree")
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.num_kv_blocks = num_kv_blocks
        self.dummy_slot = max_num_seqs * self.T

        H = self.geo.hidden
        B = max_num_seqs
        self.kv_int8 = kv_int8

        self.kv = {}
        self.kv_scale = {}
        self.state = {}

        def bind(idx: int, attn) -> None:
            if isinstance(attn, FullAttention):
                attn.kv = self.kv[idx] = self._alloc_kv(num_kv_blocks)
                if self.kv_int8:
                    attn.kv_scale = self.kv_scale[idx] = self._alloc_kv_scale(num_kv_blocks)
            elif isinstance(attn, GatedDeltaNet):
                slots = max_num_seqs * self.T + 1
                attn.state = self.state[idx] = (
                    torch.zeros(slots, self.geo.lin_conv_state, self.geo.lin_conv_dim,
                                dtype=torch.bfloat16, device="cuda"),
                    torch.zeros(slots, self.geo.lin_num_v_heads, self.geo.lin_head_k,
                                self.geo.lin_head_v,
                                dtype=torch.float32, device="cuda"),
                )

        POOLED = (FullAttention, GatedDeltaNet)
        bound = set()
        for i, layer in enumerate(model.layers):
            for m in layer.modules():
                if isinstance(m, POOLED) and id(m) not in bound:
                    bound.add(id(m))
                    bind(i, m)
        nxt = len(model.layers)
        self.mtp_layer = nxt if model.mtp is not None else None
        for m in model.modules():
            if isinstance(m, POOLED) and id(m) not in bound:
                bound.add(id(m))
                bind(nxt, m)
                nxt += 1

        if max_prefill_tokens == "auto":
            self.max_prefill_tokens = self._autosize_prefill_tokens(
                max_blocks_per_seq * self.block_size)
        else:
            self.max_prefill_tokens = prefill_rows(max_prefill_tokens)
        if self.max_prefill_tokens > ops.PAGED_PREFILL_MAX_TOKENS:
            raise ValueError(
                f"max_prefill_tokens={self.max_prefill_tokens} exceeds this kernel's "
                f"{ops.PAGED_PREFILL_MAX_TOKENS}-token ceiling. Chunk the prefill instead -- an "
                f"engine ABLATION(2026-07-18) puts one-shot only 3% ahead of a 32768 chunk.")
        M = self.max_prefill_tokens

        if model.mtp is not None:
            self.mtp_logits = torch.empty(self.lm_rows, self.geo.vocab_size, dtype=torch.float32,
                                          device="cuda")
            self.mtp_h = torch.zeros(max_num_seqs, H, dtype=torch.bfloat16, device="cuda")

        act, scratch = self._m_buffers()
        for name, (shape, dt) in act.items():
            setattr(self, name, torch.zeros(M, *shape, dtype=dt, device="cuda"))
        for name, size in scratch.items():
            setattr(self, name, ops.empty_bytes(size(M)).zero_())
        self._act_names, self._scratch_names = tuple(act), tuple(scratch)

        self.d_mscale = torch.ones(1, dtype=torch.float32, device="cuda")
        self.logits = torch.empty(self.lm_rows, self.geo.vocab_size, dtype=torch.float32, device="cuda")
        pad = ops.lm_head_pad_bytes(self.decode_rows)
        if pad:
            model.lm_head.pad_in = ops.zero_bytes(pad)
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
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.vgraphs: dict[int, tuple[int, torch.cuda.CUDAGraph]] = {}
        self._pool = None
        self.n_graph_replays = 0
        self.n_eager_forwards = 0

    @staticmethod
    def _attn_path(b: Batch) -> ops.Path:
        return ops.Path.PREFILL_SHUFFLED_A if b.is_prefill else ops.Path.DECODE

    def _ctx(self, b: Batch) -> ForwardContext:
        M = b.input_ids.numel()
        return ForwardContext(
            batch=b, M=M, eps=self.model.eps, path=self._attn_path(b),
            decode_plan=self.decode_plan, decode_ws=self.decode_ws, num_slots=self.num_slots,
            kv_int8=self.kv_int8, mscale=self.d_mscale,
            **{n: getattr(self, n)[:M] for n in self._act_names},
            **{n: getattr(self, n) for n in self._scratch_names},
        )

    def forward(self, b: Batch) -> torch.Tensor:
        if not b.is_prefill:
            B, M = b.batch_size, b.input_ids.numel()
            if M == B and B in self.graphs:
                return self._replay(b)
            hit = self.vgraphs.get(B)
            if hit is not None and M == B * hit[0]:
                return self._replay_verify(b, hit[1])
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

    def _plan_attn(self, ctx: ForwardContext) -> None:
        if not ctx.batch.varlen_attn:
            with span("attn_plan"):
                ops.paged_attn_decode_plan(ctx.batch.seq_lens, ctx.decode_plan,
                                           ctx.num_slots)

    def mtp_draft(self, hidden: torch.Tensor, next_ids: torch.Tensor,
                  b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self._ctx(b)
        with span(f"mtp draft M={ctx.M}"):
            self._plan_attn(ctx)
            x = self.model.mtp(ctx, self.model, hidden, next_ids)
            last = x if not b.is_prefill else x[b.last_row]
            with span("mtp lm_head"):
                return self.model.lm_head(last, self.mtp_logits), last

    def capture_decode(self) -> list[int]:
        sizes = list(range(1, self.max_num_seqs + 1))

        self._pool = torch.cuda.graph_pool_handle()
        for gb in sizes:
            t0 = time.time()
            self.d_pos[gb] = torch.zeros(3, gb, dtype=torch.int64, device="cuda")
            b = self._pad_batch(gb)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._forward_eager(b)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                self._forward_eager(b)
            self.graphs[gb] = g
            torch.cuda.synchronize()
            print(f"[snowllm {time.strftime('%H:%M:%S')}]   capture decode graph B={gb:2d}  "
                  f"{(time.time() - t0) * 1e3:5.0f} ms", flush=True)
        self.n_eager_forwards = 0
        return sizes

    def capture_verify(self, depth_for) -> dict[int, int]:
        got = {}
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        for gb in range(1, self.max_num_seqs + 1):
            T = depth_for(gb) + 1
            if T < 2:
                continue
            M = gb * T
            t0 = time.time()
            self.d_pos[M] = torch.zeros(3, M, dtype=torch.int64, device="cuda")
            b = self._pad_batch_verify(gb, T)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._forward_eager(b)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                self._forward_eager(b)
            self.vgraphs[gb] = (T, g)
            got[gb] = T
            torch.cuda.synchronize()
            print(f"[snowllm {time.strftime('%H:%M:%S')}]   capture verify graph B={gb:2d} T={T} "
                  f"M={M:3d}  {(time.time() - t0) * 1e3:5.0f} ms", flush=True)
        self.n_eager_forwards = 0
        return got

    def _pad_batch_verify(self, gb: int, T: int) -> Batch:
        M = gb * T
        return Batch(
            input_ids=self.d_ids[:M], positions=self.d_pos[M], slot_mapping=self.d_slot[:M],
            block_tables=self.d_bt[:gb], seq_lens=self.d_seq[:gb], state_indices=self.d_sidx[:M],
            is_prefill=False, num_tokens=M, num_accepted=self.d_nacc[:gb],
            cu_seqlens=torch.tensor([i * T for i in range(gb + 1)], dtype=torch.int32,
                                    device="cuda"),
            total_q_blocks=ops.prefill_q_plan([T] * gb)[0],
        )

    def _replay_verify(self, b: Batch, g: torch.cuda.CUDAGraph) -> torch.Tensor:
        B, M = b.batch_size, b.input_ids.numel()
        self.d_ids[:M] = b.input_ids
        self.d_pos[M][:, :M] = b.positions
        self.d_slot[:M] = b.slot_mapping
        self.d_seq[:B] = b.seq_lens
        self.d_bt[:B, : b.block_tables.shape[1]] = b.block_tables
        self.d_sidx[:M] = b.state_indices
        self.d_nacc[:B] = b.num_accepted
        with span(f"verify graph replay B={B} M={M}"):
            g.replay()
        self.n_graph_replays += 1
        self.last_hidden = self.x[:M]
        return self.logits[:M]

    def _pad_batch(self, gb: int) -> Batch:
        return Batch(
            input_ids=self.d_ids[:gb], positions=self.d_pos[gb], slot_mapping=self.d_slot[:gb],
            block_tables=self.d_bt[:gb], seq_lens=self.d_seq[:gb], state_indices=self.d_sidx[:gb],
            is_prefill=False, num_tokens=gb,
        )

    def _replay(self, b: Batch) -> torch.Tensor:
        B = b.batch_size
        self.d_ids[:B] = b.input_ids
        self.d_pos[B][:, :B] = b.positions
        self.d_slot[:B] = b.slot_mapping
        self.d_seq[:B] = b.seq_lens
        self.d_bt[:B, : b.block_tables.shape[1]] = b.block_tables
        self.d_sidx[:B] = b.state_indices
        with span(f"decode graph replay B={B}"):
            self.graphs[B].replay()
        self.n_graph_replays += 1
        self.last_hidden = self.x[:B]
        return self.logits[:B]

    def set_rope(self, inv_freq: torch.Tensor, mscale: float) -> None:
        self.model.inv_freq.copy_(inv_freq.to(self.model.inv_freq))
        self.d_mscale.fill_(mscale)

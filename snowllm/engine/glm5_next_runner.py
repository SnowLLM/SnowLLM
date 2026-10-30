# SPDX-License-Identifier: Apache-2.0
import copy
import contextlib

import torch

from .. import ops
from ..ops._common import dry_alloc
from .._capi import SnowLLMError
from ..models.glm5_next.layers import LinearAttention, gather_rows
from .runner import ForwardResult, GraphRunner
from .graphs import GraphCase, GraphInputs
from .resources import ActivationProbe, ActivationReservation, PoolPlan, ProbeGroup, plan_activations
from .host_io import HostArena
from .inputs import synthetic_batch
from .glm5_next_spec import Glm5NextSpecDecoder
from .forward_context import Glm5NextContext


class Glm5NextRunner(GraphRunner):
    spec_decoder_type = Glm5NextSpecDecoder
    tunable_rope = False
    text_only = True
    slots_per_request = 1

    def _configure(self):
        if self.draft_geo is not None or self.tap_layers:
            raise SnowLLMError("GLM5-Next currently supports its own MTP head")
        if self.num_spec and self.model.mtp is None:
            raise SnowLLMError("load the GLM checkpoint with mtp=True before requesting drafts")
        if self.kv_int8:
            raise SnowLLMError("GLM5-Next currently needs bf16 latent caches")
        self.slots_per_request = self.num_spec+1

    def _setup_resources(self):
        from .cache_resources import GlmResources
        self.arena = ops.Arena() if self.plan_activations else None
        self._activation_plans, self._planning = {}, False
        self.num_kv_blocks = self._want_blocks
        self.kv_bytes, self.act_bytes = 0, 0
        self.ws = torch.empty(0, dtype=torch.uint8, device="cuda")
        self.resources = GlmResources(self)

    def prefix_residue(self):
        from .prefix_cache import RecurrentResidue
        return RecurrentResidue([((a.history, a.state), a.ckpt)
                                 for layer in self.model.layers
                                 if isinstance(a := layer.attention, LinearAttention)])

    def block_bytes(self):
        return self.resources.cache.bytes_for(1) - self.resources.cache.bytes_for(0)

    def sparse_state_bytes(self):
        return self.resources.cache.bytes_for(0)

    def fixed_bytes(self, chunk):
        return (self.resources.states.bytes_for() + self._cache_bytes(self._cache_blocks())
                + (self.resources.checkpoints.bytes_for(2*self.max_num_seqs)
                   if self.prefix_memory_ratio else 0))

    def _cache_bytes(self, blocks):
        return self.resources.cache.bytes_for(blocks)

    def _walk_high(self, rows):
        arena, workspace, _ = self._activation_plan(rows)
        return arena.planner.high + workspace

    def _act_statics(self, rows):
        return self._activation_plan(rows)[2]

    def host_tensor(self, name, values, dtype=torch.int64):
        self.inputs.begin()
        return self.inputs.tensor(name, values, dtype)

    def gather_hidden(self, hidden, rows):
        with self.inputs.reading(reuse=True):
            ids = self.inputs.tensor("hidden_rows", rows)
            out = self.gathered_hidden[:len(rows)] if self.plan_activations else None
            return gather_rows(hidden,ids,out=out)

    def _pool_plan(self, graph_bytes):
        future = graph_bytes
        if not self.plan_activations:
            future += max(0, self.act_bytes_for(self.max_prefill_tokens) - self.ws.nbytes)
        return PoolPlan(self._cache_bytes, self.max_blocks_per_seq, requested=self._cache_blocks(),
                        checkpoint_bytes=self.resources.checkpoints.bytes_for(1) if self.prefix_memory_ratio else 0,
                        future_bytes=future, minimum_snapshots=2*self.max_num_seqs)

    def _allocate_pools(self, blocks):
        self.num_kv_blocks = blocks
        self.resources.checkpoints.allocate(self.prefix_slots)
        self.resources.cache.allocate(blocks)
        self.kv_bytes = self.resources.cache.slabs.committed() - self.sparse_state_bytes()
        # Serving never grows a succession of old scratch allocations as chunk
        # widths change; all layers and MTP reuse this capacity on one stream.
        if not self.plan_activations:
            self.resources.workspace.allocate(self._activation_plan(self.max_prefill_tokens)[1])
            self.act_bytes = self.act_bytes_for(self.max_prefill_tokens)

    def workspace(self, n):
        if self.ws.numel() < n:
            if self.plan_activations and not self._planning:
                raise SnowLLMError(f"GLM workspace needs {n} bytes, exceeds its startup capacity {self.ws.numel()}")
            self.ws = torch.empty(n, dtype=torch.uint8, device="cuda")
        return self.ws

    def _allocate_outputs(self, rows=None):
        decode = self.max_num_seqs*self.slots_per_request
        rows = max(self.max_prefill_tokens if rows is None else rows, decode)
        h = self.geo.hidden
        self.hidden = torch.empty(rows,h,dtype=torch.bfloat16,device="cuda")
        self.logits = torch.empty(decode,self.geo.vocab_size,dtype=torch.float32,device="cuda")
        outputs = [self.hidden,self.logits]
        if self.num_spec:
            self.draft_hidden = torch.empty_like(self.hidden)
            self.gathered_hidden = torch.empty(decode,h,dtype=torch.bfloat16,device="cuda")
            self.head_hidden = torch.empty(self.max_num_seqs,h,dtype=torch.bfloat16,device="cuda")
            outputs += [self.draft_hidden,self.gathered_hidden,self.head_hidden]
        return sum(t.nbytes for t in outputs)

    def _plan_key(self, batch, draft=False):
        return (draft,batch.is_prefill,batch.last_row is not None)

    def _probe_batch(self, rows, prefill, last, draft):
        B = min(rows, self.max_num_seqs)
        lengths = [rows//B + (i < rows % B) for i in range(B)]
        return synthetic_batch(lengths, self.max_blocks_per_seq,
            is_prefill=prefill, state_indices=[0]*(rows if not prefill and self.num_spec and not draft else B),
            last_row=[0]*B if last else None,
            chunk_decode=not prefill and bool(self.num_spec) and not draft,
            need_logits=last or not prefill, host_seq_lens=[self.ctx_cap]*B)

    def _activation_plan(self, rows):
        cached = self._activation_plans.get(rows)
        if cached is not None:
            return cached
        # Budgeting and startup allocation use the same native queries and arena plans.
        arena = ops.Arena()
        probe = copy.copy(self)
        probe.arena = ops.Arena()
        probe._planning = True
        from .cache_resources import GlmResources
        resources = GlmResources(probe)
        with contextlib.ExitStack() as stack:
            for part in resources.parts:
                stack.enter_context(part.preserve())
            with dry_alloc():
                try:
                    resources.workspace.allocate(0)
                    static = resources.outputs.allocate(rows)
                    resources.states.allocate()
                    resources.cache.allocate(1)
                    plan_activations(arena, [ProbeGroup(probe._activation_cases(rows))])
                    capacity = 0 if self.plan_activations else probe.ws.numel()
                finally:
                    resources.close()
        plan = arena, capacity, static
        self._activation_plans[rows] = plan
        return plan

    def _activation_cases(self, rows):
        for draft in ((False, True) if self.num_spec else (False,)):
            for prefill in (True, False):
                width = rows if prefill else self.max_num_seqs*self.slots_per_request
                for last in (False, True):
                    batch = self._probe_batch(width, prefill, last, draft)
                    walk = (lambda: self._draft_eager(self.hidden[:width], batch.input_ids, batch)
                            if draft else self._forward_eager(batch))
                    yield ActivationProbe(self._plan_key(batch, draft), width, walk,
                        lifetimes='uses' if self.plan_activations else 'references')

    @contextlib.contextmanager
    def _activation_scope(self, rows):
        self.arena, capacity, _ = self._activation_plan(rows)
        self.resources.workspace.allocate(capacity)
        static = self.resources.outputs.allocate()
        yield ActivationReservation(outside_bytes=self.ws.nbytes+static)

    def _execute(self, ctx):
        ctx.open()
        out = self.model(ctx)
        return ForwardResult(out, ctx.last_hidden)

    def _ctx(self,batch,ids=None,*,draft=False):
        if self.arena is not None and not self._planning:
            capacity = self.max_prefill_tokens if batch.is_prefill else self.max_num_seqs*self.slots_per_request
            if batch.input_ids.numel() > capacity or batch.batch_size > self.max_num_seqs:
                raise SnowLLMError("GLM batch exceeds its startup activation capacity")
        # Host lengths are already known to the scheduler; avoid reading GPU
        # metadata back merely to upload five derived tensors on every forward.
        if batch.is_prefill:
            lens = batch.host_seq_lens
            if lens is None:
                lens = batch.seq_lens.tolist()  # Direct Batch callers outside Engine.
            max_pools = max(lens,default=0)//self.geo.index_ratio
        else:
            max_pools = min(self.ctx_cap,batch.block_tables.shape[1]*self.block_size)//self.geo.index_ratio
        return Glm5NextContext(
            batch=batch, M=batch.input_ids.numel(), eps=self.geo.eps,
            path=ops.Path.PREFILL if batch.is_prefill else ops.Path.DECODE,
            arena=self.arena, plan_key=self._plan_key(batch,draft),
            num_spec=self.num_spec, embedding_weight=self.model.embed,
            next_ids=ids, max_pools=max_pools,
            require_plan=self.plan_activations and not self._planning,
            plan_workspace=self.plan_activations,
            hidden=getattr(self,"hidden",None), head_hidden=getattr(self,"head_hidden",None),
            draft_hidden=getattr(self,"draft_hidden",None), logits=getattr(self,"logits",None),
            scratch=self.workspace)

    def _draft_context(self, batch, next_ids):
        return self._ctx(batch, next_ids, draft=True)

    def _execute_draft(self, ctx, hidden, next_ids):
        ctx.open()
        return self.model.mtp(ctx, self.model, hidden, next_ids)

    def _graph_key(self,b,draft=False):
        if b.is_prefill or (b.last_row is not None and not draft) or b.has_state is not None or b.ckpt_n:
            return None
        return (draft,b.batch_size,b.input_ids.numel(),b.chunk_decode,b.num_accepted is not None,
                b.carry_source_offset,b.need_logits,b.cu_seqlens is not None,b.last_row is not None)

    @contextlib.contextmanager
    def _graph_probe_scope(self,shapes):
        blocks = max(self.PROBE_BLOCKS, ops.kv_blocks_for(max(s.T for s in shapes), self.block_size))
        with self.resources.cache.temporary(blocks):
            yield

    @property
    def graphs_enabled(self):
        return self.plan_activations

    def _static_batch(self,b,t,draft=False,*,accepted=False,offset=0,logits=True):
        messages = HostArena()
        m = b*t
        accepted_rows = messages.tensor('accepted', [t]*b, torch.int64) if accepted else [1]*b if not draft and t>1 else None
        batch = synthetic_batch([t]*b, self.max_blocks_per_seq, tensor=messages.tensor,
            slot_mapping=[-1]*m,
            state_indices=[2*self.max_num_seqs-1 if draft else self.dummy_slot]*(b if draft or t==1 else m),
            cu_seqlens=[i*t for i in range(b+1)] if t>1 else None,
            chunk_decode=not draft and t>1, num_accepted=accepted_rows,
            last_row=[(i+1)*t-1 for i in range(b)] if draft and accepted else None,
            carry_source_offset=offset, need_logits=logits)
        return batch,messages

    def _graph_cases(self,shape):
        b,t = shape.B,shape.T
        variants = [(False,t,False,0,True)]
        if t>1:
            variants.append((False,1,False,0,True))
        if self.num_spec:
            variants += [(True,t,True,0,True),(True,1,False,0,False)]
            if self.num_spec>1:
                variants.append((True,1,False,-self.max_num_seqs,True))
            if self.num_spec>2:
                variants.append((True,1,False,0,True))
        for draft,width,accepted,offset,logits in variants:
            batch,messages = self._static_batch(b,width,draft,accepted=accepted,offset=offset,logits=logits)
            hidden = ops.empty_shaped((b*width,self.geo.hidden),torch.bfloat16) if draft else None
            if hidden is not None:
                hidden.zero_()
            def walk():
                return self._draft_eager(hidden,batch.input_ids,batch) if draft else self._forward_eager(batch)
            values = self._graph_values(batch)
            if draft:
                values['draft_hidden'] = hidden
            yield GraphCase(self._graph_key(batch,draft),
                f"GLM {'MTP' if draft else 'decode'} B={b} T={width}",
                walk, GraphInputs(values,messages), draft)

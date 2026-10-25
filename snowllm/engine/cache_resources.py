# SPDX-License-Identifier: Apache-2.0
import torch

from .. import ops
from ..ops._common import dry_alloc
from ..models.qwen3_5.layers import FullAttention, GatedDeltaNet
from ..models.qwen4exp.layers import Ple
from ..models.glm5_next.layers import LinearAttention, SparseAttention
from .resources import ResourceSet
from .cache import CacheStorage, RequestState, paged_buffers, tensor
from .host_io import HostArena
from .inputs import synthetic_batch


def pooled_modules(model, kinds):
    pooled, seen = [], set()
    for i, layer in enumerate(model.layers):
        for m in layer.modules():
            if isinstance(m, kinds) and id(m) not in seen:
                seen.add(id(m))
                pooled.append((i, m))
    nxt = len(model.layers)
    for m in model.modules():
        if isinstance(m, kinds) and id(m) not in seen:
            seen.add(id(m))
            pooled.append((nxt, m))
            nxt += 1
    return pooled



class QwenResources(ResourceSet):
    def __init__(self, runner):
        super().__init__()
        self.r = r = runner
        pooled = pooled_modules(r.model, (FullAttention, GatedDeltaNet))
        r.full_mods = [(i, a) for i, a in pooled if isinstance(a, FullAttention)]
        r.linear_mods = [(i, a) for i, a in pooled if isinstance(a, GatedDeltaNet)]
        r.ple_mods = [m for m in r.model.modules() if isinstance(m, Ple)]
        self.indexers = ([a.indexer for _, a in r.full_mods if getattr(a, 'indexer', None) is not None]
                         if getattr(r.geo, 'index_topk', 0) else [])
        self.states = self.add(self.allocate_states,
            [(a, ('state', 'retain')) for _, a in r.linear_mods]
            + [(a, ('state',)) for a in r.ple_mods]
            + [(a, ('carry', 'carry_pos')) for a in self.indexers]
            + [(a, ('compact', 'compact_table', 'plan', 'plan_ws')) for _, a in r.full_mods
               if getattr(a, 'indexer', None) is not None]
            + [(r, ('state', 'ple_fresh', 'ple_going'))])
        self.cache = self.add(self.allocate_cache,
            [(a, ('kv', 'kv_scale')) for _, a in r.full_mods]
            + [(a, ('pool',)) for a in self.indexers]
            + [(r, ('kv', 'kv_scale', 'cache'))])
        self.checkpoints = self.add(self.allocate_checkpoints,
            [(a, ('ckpt',)) for _, a in r.linear_mods] + [(a, ('ckpt',)) for a in r.ple_mods])
        self.outputs = self.add(self.allocate_outputs, [(r, ('x', 'taps', 'draft_taps', 'ple_emb',
            'trunk_streams', 'mtp_streams'))])
        self.statics = self.add(self.allocate_statics, [(r, ('logits', 'draft_logits', 'mtp_h',
            'd_mscale', 'decode_ws', 'decode_plan')), (r.model.lm_head, ('scratch',))])
        self.messages = self.add(self.allocate_messages, [(r, ('d_ids', 'd_slot', 'd_seq', 'd_bt',
            'd_sidx', 'd_nacc', 'd_pos', 'd_cu', '_graph_inputs'))])
        self.metadata = self.add(self.allocate_metadata, [(r, ('_qsa_of_row', '_qsa_launch'))])
        self.support = (self.states, self.statics, self.messages, self.metadata)
        self.planning_parts = (self.metadata,)

    def prepare(self, rows):
        self.outputs.allocate(max(rows, self.r.decode_rows))

    def probe_units(self):
        return self.r.PROBE_BLOCKS

    def allocate_metadata(self, slabs):
        self.r._qsa_of_row, self.r._qsa_launch = {}, {}

    def allocate_messages(self, slabs):
        r = self.r
        r._graph_inputs = HostArena() if r.host_messages else None
        def message(name, values, dtype):
            if r._graph_inputs is None:
                return torch.tensor(values, dtype=dtype, device='cuda')
            return r._graph_inputs.tensor(name, values, dtype)
        b = synthetic_batch([r.T]*r.max_num_seqs, r.max_blocks_per_seq, tensor=message,
            slot_mapping=[-1]*r.decode_rows, seq_lens=[1]*r.max_num_seqs,
            state_indices=[r.dummy_slot]*r.decode_rows, num_accepted=[1]*r.max_num_seqs)
        r.d_ids, r.d_slot, r.d_seq = b.input_ids, b.slot_mapping, b.seq_lens
        r.d_bt, r.d_sidx, r.d_nacc = b.block_tables, b.state_indices, b.num_accepted
        r.d_pos, r.d_cu = {}, {}

    def allocate_states(self, slabs):
        r, g = self.r, self.r.geo
        rows = r.dummy_slot + 1
        r.state = {}
        for i, a in r.linear_mods:
            a.state = r.state[i] = (
                tensor(slabs, (rows, g.lin_conv_state, g.lin_conv_dim), torch.bfloat16),
                tensor(slabs, (rows, g.lin_num_v_heads, g.lin_head_k, g.lin_head_v), torch.float32))
            if r.roll_forward:
                a.retain = tuple(slabs.take(n).span() for n in ops.linear_attn_retain_bytes(r.decode_rows))
        for a in r.ple_mods:
            a.state = tensor(slabs, (rows, a.hist, g.hc_dim), torch.bfloat16)
        if r.ple_mods:
            r.prev_context = g.ngram_size - 1
            r.ple_fresh = tensor(slabs, (r.max_num_seqs,), torch.int32)
            r.ple_going = tensor(slabs, (r.decode_rows,), torch.int32).fill_(1)
        if not self.indexers:
            return RequestState([t for _, a in r.linear_mods for t in a.state]
                                + [a.state for a in r.ple_mods])
        for a in self.indexers:
            a.carry = tensor(slabs, (rows, g.index_ratio - 1, g.index_head_dim), torch.bfloat16)
            a.carry_pos = tensor(slabs, (rows, g.index_ratio - 1, 3), torch.int64)
        width = ops.qsa_decode_rows(r.decode_rows)
        pages = -(-(g.index_topk + g.index_ratio - 1) // r.block_size)
        compact = tuple(slabs.take(n).span().view(torch.bfloat16)
                        for n in ops.kv_pool_bytes(width*pages, False, r.block_size))
        table = torch.arange(width*pages, dtype=torch.int32, device='cuda').view(width, pages)
        slots = ops.paged_decode_num_slots(width)
        plan = tensor(slabs, (ops.paged_decode_plan_elems(width, slots),), torch.int32)
        ws = slabs.take(ops.paged_decode_workspace_size(slots, 1)).span()
        for _, a in r.full_mods:
            a.compact, a.compact_table, a.plan, a.plan_ws = compact, table, plan, ws
            a.plan_slots = slots
        return RequestState([t for _, a in r.linear_mods for t in a.state]
                            + [a.state for a in r.ple_mods]
                            + [t for a in self.indexers for t in (a.carry, a.carry_pos)])

    def allocate_cache(self, slabs, blocks):
        r = self.r
        r.kv, r.kv_scale = {}, {}
        for i, a in r.full_mods:
            a.kv = r.kv[i] = paged_buffers(slabs, blocks, r.block_size, int8=r.kv_int8)
            if r.kv_int8:
                a.kv_scale = r.kv_scale[i] = tuple(slabs.take(n).span().view(torch.bfloat16)
                                    for n in ops.kv_scale_bytes(blocks, r.block_size))
        if self.indexers:
            rows = blocks * (r.block_size // r.geo.index_ratio) + 1
            for a in self.indexers:
                a.pool = tensor(slabs, (rows, r.geo.index_head_dim), torch.bfloat16)
        r.cache = CacheStorage([a for _, a in r.full_mods], r.block_size, slabs=slabs,
                               state=self.states.value)
        return r.cache

    def allocate_checkpoints(self, slabs, rows):
        r, g = self.r, self.r.geo
        for _, a in r.linear_mods:
            a.ckpt = RequestState(a.state).layout.allocate(rows, slabs) if rows else None
        for a in r.ple_mods:
            a.ckpt = RequestState((a.state,)).layout.allocate(rows, slabs)[0] if rows else None

    def allocate_outputs(self, slabs, rows):
        self.r._alloc_activations(rows)

    def allocate_statics(self, slabs):
        r, g = self.r, self.r.geo
        if r.drafting:
            r.draft_logits = tensor(slabs, (r.decode_rows, g.vocab_size), torch.float32)
        if r.model.mtp is not None:
            r._alloc_mtp_hidden()
        r.d_mscale = tensor(slabs, (1,), torch.float32).fill_(1)
        r.logits = tensor(slabs, (r.decode_rows, g.vocab_size), torch.float32)
        if n := ops.lm_head_scratch_bytes(r.decode_rows):
            r.model.lm_head.scratch = slabs.take(n).span()
        r.num_slots = ops.paged_decode_num_slots(r.max_num_seqs)
        r.decode_ws = slabs.take(ops.paged_decode_workspace_size(r.num_slots, r.attn_q_tokens)).span()
        r.decode_plan = tensor(slabs, (ops.paged_decode_plan_elems(r.max_num_seqs, r.num_slots),), torch.int32)


class GlmResources(ResourceSet):
    def __init__(self, runner):
        super().__init__()
        self.r = r = runner
        self.linear = [a for _, a in pooled_modules(r.model, (LinearAttention,))]
        self.sparse = [layer.attention for layer in r.model.layers if isinstance(layer.attention, SparseAttention)]
        self.sparse += [r.model.mtp.attention] if r.num_spec else []
        self.states = self.add(self.allocate_states, [(a, ('history', 'state')) for a in self.linear])
        self.cache = self.add(self.allocate_cache,
            [(a, ('bs', 'kv', 'index', 'carry_k', 'carry_g', 'sinks')) for a in self.sparse]
            + [(r, ('cache',))])
        self.checkpoints = self.add(self.allocate_checkpoints, [(a, ('ckpt',)) for a in self.linear])
        self.outputs = self.add(lambda slabs, rows=None: r._allocate_outputs(rows),
            [(r, ('hidden', 'logits', 'draft_hidden', 'gathered_hidden', 'head_hidden'))])
        self.workspace = self.add(self.allocate_workspace, [(r, ('ws',))])
        self.support = (self.states,)

    def prepare(self, rows):
        pass

    def probe_units(self):
        return self.r.PROBE_BLOCKS

    def allocate_states(self, slabs):
        for a in self.linear:
            a.allocate(self.r.dummy_slot+1, slabs)
        return RequestState([t for a in self.linear for t in (a.history, a.state)])

    def allocate_cache(self, slabs, blocks):
        r = self.r
        for a in self.sparse:
            slots = 2*r.max_num_seqs if a is getattr(r.model.mtp, 'attention', None) else r.dummy_slot+1
            a.allocate(blocks, r.block_size, slots, slabs)
        r.cache = CacheStorage(list(self.sparse), r.block_size, slabs=slabs, state=self.states.value)
        return r.cache

    def allocate_checkpoints(self, slabs, rows):
        for a in self.linear:
            if rows:
                a.allocate_checkpoint(rows, slabs)
            else:
                a.ckpt = None

    def allocate_workspace(self, slabs, size):
        self.r.ws = slabs.take(size).span()


class Dsv4Resources(ResourceSet):
    def __init__(self, runner):
        super().__init__()
        self.r = r = runner
        self.cache = self.add(self.allocate_cache, [(r, ('cache',))])
        self.outputs = self.add(self.allocate_outputs,
            [(r, ('logits', 'draft_context_ws', 'draft_markov_ws')), (r.walker, ('taps',))])
        self.checkpoints = self.add(self.allocate_checkpoints, [(r, ('checkpoint_buffer',))])
        self.support = ()

    def prepare(self, rows):
        self.r.ring_blocks = self.r._ring_blocks(rows)
        self.outputs.allocate(rows)

    def probe_units(self):
        return self.r._capture_tokens()

    def checkpoint_bytes(self):
        with dry_alloc(), self.cache.temporary(self.r.ctx_cap):
            return self.checkpoints.bytes_for(1)

    def allocate_cache(self, slabs, tokens, chunk=None, sweep=0, probe=False):
        from ..models.deepseek_v4.pools import make_cache
        r = self.r
        chunk = r.max_prefill_tokens if chunk is None else chunk
        ring = ops.dsv4_mla_raw_ring_blocks(r.geo.sliding_window, chunk, r.block_size)
        r.cache = make_cache(r.geo, tokens, r.block_size, r.max_num_seqs,
            raw_blocks=(ops.kv_blocks_for(tokens, r.block_size) if probe else r.max_num_seqs*ring),
            slabs=slabs, spec_rows=1 if probe else r.T, ctx_cap=max(r.ctx_cap, tokens) if probe else r.ctx_cap)
        return r.cache

    def allocate_checkpoints(self, slabs, rows):
        self.r.checkpoint_buffer = self.r.cache.snapshots.allocate(rows, slabs, packed=True)

    def allocate_outputs(self, slabs, chunk):
        r = self.r
        r.logits = tensor(slabs, (r.lm_rows, r.geo.vocab_size), torch.float32)
        tap_rows = max(min(chunk, r.geo.sliding_window) if r.geo.v41 else chunk, r.lm_rows)
        r.walker.taps = (tensor(slabs, (tap_rows, r.num_taps*r.geo.hidden), torch.bfloat16)
                         if r.num_taps else None)
        from ..models.geometry import DSparkGeometry
        r.draft_context_ws = r.draft_markov_ws = None
        if isinstance(r.draft_geo, DSparkGeometry):
            g = r.draft_geo
            context = (0 if g.stack.v41 else
                       ops.dspark_context_workspace_bytes(tap_rows, g.stack.hidden, g.fc_k))
            markov = ops.dspark_markov_workspace_bytes(r.max_num_seqs, g.stack.vocab_size,
                                                       g.markov_rank, r.num_spec)
            r.draft_context_ws = tensor(slabs, (context,), torch.uint8)
            r.draft_markov_ws = tensor(slabs, (markov,), torch.uint8)

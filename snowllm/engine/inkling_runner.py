# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .. import ops
from ..ops.inkling import batch_metadata, converted
from ..errors import SnowLLMError
from .cache import tensor
from .cache_resources import QwenResources
from .inputs import synthetic_batch
from .prefix_cache import Residue
from .qwen_runner import QwenRunner
from .resources import ActivationReservation, ResourceSet, draft_pool_bytes


@dataclass
class InklingPages:
    full: torch.Tensor
    sliding: torch.Tensor | None = None


class InklingResidue(Residue):
    mid_walk = False

    def __init__(self, history, checkpoints, sliding, windows, *, draft_ring=None, draft_windows=None):
        self.history, self.checkpoints = history, checkpoints
        self.sliding, self.windows = sliding, windows
        self.draft_ring, self.draft_windows = draft_ring, draft_windows
        self.positions = {}
        self.copy = torch.Tensor.copy_ if history.device.type == "cpu" else ops.copy_tensor

    def _copy_window(self, source, snapshot, start, count, restore):
        start %= source.size(2)
        first = min(count, source.size(2) - start)
        for offset, ring, rows in ((0, start, first), (first, 0, count - first)):
            if rows:
                live = source.narrow(2, ring, rows)
                saved = snapshot.narrow(2, offset, rows)
                if restore:
                    self.copy(live, saved)
                else:
                    self.copy(saved, live)

    def _window(self, i, slot, position, restore):
        if self.windows is not None:
            source = self.sliding[:, :, slot].flatten(2, 3)
            snapshot = self.windows[i]
            count = min(position, snapshot.size(2))
            self._copy_window(source, snapshot, position-count, count, restore)
        if self.draft_windows is not None:
            # Draft K and V have distinct layouts within a page, so copy whole pages.
            end = (position+self.page_size-1)//self.page_size
            start = max(0, position-self.draft_lookback)//self.page_size
            self._copy_window(self.draft_ring[:,:,slot], self.draft_windows[i], start, end-start, restore)

    def bytes_each(self):
        return self.history[0].nbytes + sum(x[0].nbytes for x in (self.windows,self.draft_windows) if x is not None)

    def save(self, i, slot, m, r):
        self.copy(self.checkpoints[i], self.history[slot])
        self._window(i, slot, m, False)
        self.positions[i] = m
        return True

    def load(self, i, slot, raw_blocks=None, held=None):
        self.copy(self.history[slot], self.checkpoints[i])
        self._window(i, slot, self.positions[i], True)

    def drop(self, i):
        self.positions.pop(i, None)


class InklingResources(ResourceSet):
    def __init__(self, runner):
        super().__init__()
        self.r = r = runner
        self.messages = self.add(self.allocate_messages, [(r, ('d_ids', 'd_slot', 'd_seq', 'd_bt',
            'd_sidx', 'd_nacc', 'd_pos', 'd_cu', '_graph_inputs'))])
        self.states = self.add(self.allocate_history, [(r, ('history', 'verify_history', 'verify_inputs', 'draft_ring'))])
        self.cache = self.add(self.allocate_cache, [(r, ('pages',))])
        self.checkpoints = self.add(self.allocate_checkpoints, [(r, ('saved', 'saved_windows', 'draft_windows'))])
        self.outputs = self.add(self.allocate_taps, [(r, ('taps', 'draft_taps'))])
        self.support = (self.states, self.messages)

    allocate_messages = QwenResources.allocate_messages

    def allocate_history(self, slabs):
        r = self.r
        r.history = self.history_tensor(slabs, r.max_num_seqs)
        r.draft_ring = (self.draft_pages(slabs, (r.max_num_seqs,r.draft_ring_pages))
                        if r.draft_ring_pages else None)
        if r.roll_forward:
            r.verify_history = self.history_tensor(slabs, r.max_num_seqs)
            r.verify_inputs = tensor(slabs, (r.decode_rows, len(r.model.visits), sum(r.channels)), torch.bfloat16)

    def allocate_taps(self, slabs, rows):
        r = self.r
        width = r.num_taps * r.geo.hidden
        r.taps = tensor(slabs, (max(rows, r.decode_rows), width), torch.bfloat16)
        r.draft_taps = tensor(slabs, (r.decode_rows, width), torch.bfloat16)

    def draft_pages(self, slabs, dims, checkpoints=()):
        r, g = self.r, self.r.draft_geo
        return tensor(slabs, (*checkpoints,g.num_sliding_layers,2,*dims,r.block_size*g.kv_dim), torch.bfloat16)

    def allocate_checkpoints(self, slabs, rows):
        r = self.r
        r.saved = self.history_tensor(slabs, rows) if rows else None
        r.draft_windows = None
        if rows and r.draft_ring_pages:
            pages = ops.kv_blocks_for(r.draft_geo.sliding_window-1+r.block_size-1,r.block_size)
            r.draft_windows = self.draft_pages(slabs, (pages,),checkpoints=(rows,))
        r.saved_windows = (tensor(slabs, (rows, r.local_count, 2, r.window - 1, *r.kv_shape), torch.bfloat16)
                           if rows and r.ring_pages else None)

    def history_tensor(self, slabs, rows):
        r = self.r
        shape = (rows, len(r.model.visits), r.history_rows, sum(r.channels))
        return tensor(slabs, shape, torch.bfloat16)

    def allocate_cache(self, slabs, blocks):
        r = self.r
        n = len(r.model.visits)
        groups = n - r.local_count if r.ring_pages else n
        full = tensor(slabs, (groups, 2, blocks, r.block_size, *r.kv_shape), torch.bfloat16)
        sliding = None
        if r.ring_pages:
            count = min(r.ring_pages, blocks)
            sliding = tensor(slabs, (r.local_count, 2, r.max_num_seqs, count, r.block_size, *r.kv_shape),
                             torch.bfloat16)
        r.pages = InklingPages(full, sliding)

    def prepare(self, rows):
        if self.r.num_taps:
            self.outputs.allocate(rows)

    def probe_units(self):
        return 1


class InklingRunner(QwenRunner):
    text_only = True
    tunable_rope = False
    plan_query_tiles = False

    def _check_chunk(self, chunk):
        return chunk

    def _configure(self):
        if self.kv_int8 or (self.draft_geo is not None or self.num_spec) and not getattr(self.draft_geo, 'inkling', False):
            raise SnowLLMError("Inkling requires BF16 KV and its paired DSpark draft for speculation")
        self.tap_at = {visit: idx for idx,visit in enumerate(self.tap_layers)}
        self.drafting = self.roll_forward = bool(self.tap_layers)
        self.slots_per_request = 1

    def _draft_ring_size(self):
        if self.draft_geo is None:
            return 0
        tokens = self.draft_geo.sliding_window-1 + max(self.draft_geo.block_size,self.candidates()[0])
        pages = ops.kv_blocks_for(tokens,self.block_size)
        return pages if self.max_num_seqs*pages < self._draft_pool_blocks(self._cache_blocks()) else 0

    def _setup_resources(self):
        self.arena = ops.Arena() if self.num_taps else None
        self.decode_rows = self.max_num_seqs * self.T
        self.draft_ring_pages = self._draft_ring_size()
        self.draft_block_bytes = draft_pool_bytes(self.draft_geo, 1, self.block_size,
                                                 separate_sliding=bool(self.draft_ring_pages))
        cfg = self.model.config
        width = cfg['num_key_value_heads'] * cfg['head_dim']
        self.channels = (width, width, self.geo.hidden, self.geo.hidden)
        self.history_rows = cfg['sconv_kernel_size'] - 1
        self.window = cfg['sliding_window']
        self.kv_shape = (cfg['num_key_value_heads'], cfg['head_dim'])
        local = [self.model.layers[i].local for i in self.model.visits]
        self.local_count = sum(local)
        # Storing a prefill chunk must leave the first query's visible window intact.
        ring = ops.kv_blocks_for(self.window - 1 + max(self.candidates()[0],self.T), self.block_size)
        self.ring_pages = ring if ring * self.max_num_seqs < self._cache_blocks() else 0
        self.page_indices = []
        counts = [0, 0]
        for sliding in local:
            self.page_indices.append(counts[sliding])
            counts[sliding] += 1
        self.resources = InklingResources(self)

    @contextmanager
    def _activation_scope(self, rows):
        yield ActivationReservation(outside_bytes=self.target_activation_bytes(rows))

    def _plan_activations(self, rows, allocation):
        from ..ops.sampling import sample_workspace_bytes
        from .resources import ProbeGroup, plan_activations
        allocation.extra_bytes = sample_workspace_bytes(self.decode_rows, self.geo.vocab_size,
            min(self.geo.vocab_size,ops.SAMPLING_MAX_K))
        probes = self.draft_workspace.probes(self.arena, rows, self.max_num_seqs,
            self.num_spec, self.block_size, self.decode_rows)
        plan_activations(self.arena, [ProbeGroup(probes)])

    def prefix_residue(self):
        residue = InklingResidue(self.history, self.saved, self.pages.sliding, self.saved_windows,
                                 draft_ring=self.draft_ring, draft_windows=self.draft_windows)
        if self.draft_ring_pages:
            residue.page_size, residue.draft_lookback = self.block_size,self.draft_geo.sliding_window-1
        return residue

    def _cache_bytes(self, blocks):
        return (self.resources.cache.bytes_for(blocks) + draft_pool_bytes(self.draft_geo,
                self._draft_pool_blocks(blocks),self.block_size,separate_sliding=bool(self.draft_ring_pages)))

    def fixed_bytes(self, chunk=0):
        return (self.resources.states.bytes_for() + self.resources.messages.bytes_for()
                + (self.resources.outputs.bytes_for(chunk or self.decode_rows) if self.num_taps else 0)
                + self._cache_bytes(self._cache_blocks())
                + (2 * self.max_num_seqs * self._checkpoint_bytes() if self.prefix_memory_ratio else 0))

    def _pool_plan(self, graph_bytes):
        return super()._pool_plan(graph_bytes + self.target_activation_bytes(self.max_prefill_tokens))

    def act_bytes_for(self, rows):
        if self.draft_workspace is not None:
            return super().act_bytes_for(rows)
        return self.target_activation_bytes(rows)

    def _act_statics(self, rows):
        return 0

    def target_activation_bytes(self, rows):
        if not hasattr(self, "_target_act"):
            self._target_act = {}
        if rows not in self._target_act:
            model = getattr(self.model, 'planning_model', self.model)
            peaks = []
            with ops.dry_load():
                history = self.resources.history_tensor(None, self.max_num_seqs)
                verify = (self.resources.history_tensor(None,self.max_num_seqs),
                          torch.empty(self.decode_rows,len(model.visits),sum(self.channels),device="meta",
                                      dtype=torch.bfloat16)) if self.roll_forward else ()
                with self.resources.cache.temporary(self.ring_pages or 1):
                    pages = self.pages
                for tokens, prefill in ((rows, True), (self.decode_rows, False)):
                    b = min(tokens, self.max_num_seqs)
                    lengths = [tokens // b + (i < tokens % b) for i in range(b)]
                    last = [sum(lengths[:i+1])-1 for i in range(b)]
                    batch = synthetic_batch(lengths, self.max_blocks_per_seq, last_row=last,
                                            is_prefill=prefill, roll_forward=self.roll_forward and not prefill)
                    taps = (torch.empty(max(tokens,self.decode_rows),self.num_taps*self.geo.hidden,
                                        device="meta",dtype=torch.bfloat16) if self.num_taps else None)
                    with ops.dry_load() as measurement:
                        for t in (*model.weight_tensors(), history, pages.full, pages.sliding, taps, *verify, *vars(batch).values()):
                            if isinstance(t, torch.Tensor):
                                measurement.note(t)
                        resident = measurement.device
                        self._run(model, batch, history, pages, taps=taps, verify=verify)
                        peaks.append(measurement.peak_device - resident)
            self._target_act[rows] = max(peaks)
        return self._target_act[rows]

    def _run(self, model, batch, history, pages, *, taps=None, verify=()):
        retained = None
        if batch.roll_forward and self.roll_forward:
            scratch, retained = verify
            ops.copy_tensor(scratch, history)
            history = scratch
            retained = retained[:batch.num_tokens]
        sliding = pages.sliding
        width = sliding.size(3) if sliding is not None else 0
        position, seq, mapping, local_mapping, local_table = batch_metadata(batch, width, self.block_size)
        plan = (seq, position, batch.cu_seqlens, batch.state_indices)
        table = converted(batch.block_tables, torch.int32)
        def visit_inputs(i):
            if sliding is None:
                k, v = pages.full[i].unbind()
                table_i, mapping_i = table, mapping
            elif model.layers[model.visits[i]].local:
                k, v = sliding[self.page_indices[i]].flatten(1, 2).unbind()
                table_i, mapping_i = local_table, local_mapping
            else:
                k, v = pages.full[self.page_indices[i]].unbind()
                table_i, mapping_i = table, mapping
            state = SimpleNamespace(k=k, v=v, history=history[:, i].split(self.channels, dim=-1))
            if retained is not None:
                state.retain = retained[:,i].split(self.channels, dim=-1)
            return state, table_i, mapping_i
        hidden = model.hidden(batch.input_ids, position, seq, visit_inputs, plan, batch.is_prefill,
                              taps=taps[:batch.num_tokens].view(batch.num_tokens,self.num_taps,-1) if taps is not None else None,
                              tap_at=self.tap_at)
        if batch.is_prefill:
            hidden = ops.gather_rows(hidden, batch.last_row)
        return model.logits(hidden) if batch.need_logits else None

    def _forward_eager(self, batch):
        self.n_eager_forwards += 1
        self.last_hidden = None
        return self._run(self.model, batch, self.history, self.pages, taps=getattr(self,'taps',None),
                         verify=(self.verify_history,self.verify_inputs) if self.roll_forward else ())

    def linear_advance(self, state_indices, num_accepted, B, T, cu=None):
        from ..ops.inkling import commit_history
        commit_history(self.verify_inputs, self.history, state_indices, num_accepted, T, cu)

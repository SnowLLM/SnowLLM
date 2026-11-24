# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .. import ops
from ..errors import SnowLLMError
from .cache import tensor
from .cache_resources import QwenResources
from .inputs import synthetic_batch
from .prefix_cache import Residue
from .qwen_runner import QwenRunner
from .resources import ResourceSet


@dataclass
class InklingPages:
    full: torch.Tensor
    sliding: torch.Tensor | None = None


class InklingResidue(Residue):
    mid_walk = False

    def __init__(self, history, checkpoints, sliding, windows):
        self.history, self.checkpoints = history, checkpoints
        self.sliding, self.windows = sliding, windows
        self.positions = {}

    def _window(self, i, slot, position, restore):
        if self.windows is None:
            return
        source = self.sliding[:, :, slot].flatten(2, 3)
        snapshot = self.windows[i]
        count = min(position, snapshot.size(2))
        start = (position - count) % source.size(2)
        first = min(count, source.size(2) - start)
        for offset, ring, rows in ((0, start, first), (first, 0, count - first)):
            if rows:
                live = source.narrow(2, ring, rows)
                saved = snapshot.narrow(2, offset, rows)
                if restore:
                    live.copy_(saved)
                else:
                    saved.copy_(live)

    def bytes_each(self):
        return self.history[0].nbytes + (self.windows[0].nbytes if self.windows is not None else 0)

    def save(self, i, slot, m, r):
        self.checkpoints[i].copy_(self.history[slot])
        self._window(i, slot, m, False)
        self.positions[i] = m
        return True

    def load(self, i, slot, raw_blocks=None, held=None):
        self.history[slot].copy_(self.checkpoints[i])
        self._window(i, slot, self.positions[i], True)

    def drop(self, i):
        self.positions.pop(i, None)


class InklingResources(ResourceSet):
    def __init__(self, runner):
        super().__init__()
        self.r = r = runner
        self.messages = self.add(self.allocate_messages, [(r, ('d_ids', 'd_slot', 'd_seq', 'd_bt',
            'd_sidx', 'd_nacc', 'd_pos', 'd_cu', '_graph_inputs'))])
        self.states = self.add(self.allocate_history, [(r, ('history',))])
        self.cache = self.add(self.allocate_cache, [(r, ('pages',))])
        self.checkpoints = self.add(self.allocate_checkpoints, [(r, ('saved', 'saved_windows'))])
        self.support = (self.states, self.messages)

    allocate_messages = QwenResources.allocate_messages

    def allocate_history(self, slabs):
        self.r.history = self.history_tensor(slabs, self.r.max_num_seqs)

    def allocate_checkpoints(self, slabs, rows):
        r = self.r
        r.saved = self.history_tensor(slabs, rows) if rows else None
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
        pass

    def probe_units(self):
        return 1


class InklingRunner(QwenRunner):
    text_only = True
    tunable_rope = False
    plan_query_tiles = False

    def _check_chunk(self, chunk):
        return chunk

    def _configure(self):
        if self.kv_int8 or self.num_spec or self.draft_geo is not None:
            raise SnowLLMError("Inkling text inference requires BF16 KV and regular decoding")
        super()._configure()

    def _setup_resources(self):
        self.arena = None
        self.decode_rows = self.max_num_seqs
        cfg = self.model.config
        width = cfg['num_key_value_heads'] * cfg['head_dim']
        self.channels = (width, width, self.geo.hidden, self.geo.hidden)
        self.history_rows = cfg['sconv_kernel_size'] - 1
        self.window = cfg['sliding_window']
        self.kv_shape = (cfg['num_key_value_heads'], cfg['head_dim'])
        local = [self.model.layers[i].local for i in self.model.visits]
        self.local_count = sum(local)
        # Storing a prefill chunk must leave the first query's visible window intact.
        ring = ops.kv_blocks_for(self.window - 1 + self.candidates()[0], self.block_size)
        self.ring_pages = ring if ring * self.max_num_seqs < self._cache_blocks() else 0
        self.page_indices = []
        counts = [0, 0]
        for sliding in local:
            self.page_indices.append(counts[sliding])
            counts[sliding] += 1
        self.resources = InklingResources(self)

    def prefix_residue(self):
        return InklingResidue(self.history, self.saved, self.pages.sliding, self.saved_windows)

    def fixed_bytes(self, chunk=0):
        return (self.resources.states.bytes_for() + self.resources.messages.bytes_for()
                + self._cache_bytes(self._cache_blocks())
                + (2 * self.max_num_seqs * self._checkpoint_bytes() if self.prefix_memory_ratio else 0))

    def _pool_plan(self, graph_bytes):
        return super()._pool_plan(graph_bytes + self.act_bytes_for(self.max_prefill_tokens))

    def act_bytes_for(self, rows):
        if rows not in self._act:
            model = getattr(self.model, 'planning_model', self.model)
            peaks = []
            with ops.dry_load():
                history = self.resources.history_tensor(None, self.max_num_seqs)
                with self.resources.cache.temporary(self.ring_pages or 1):
                    pages = self.pages
                for tokens, prefill in ((rows, True), (self.max_num_seqs, False)):
                    b = min(tokens, self.max_num_seqs)
                    lengths = [tokens // b + (i < tokens % b) for i in range(b)]
                    last = [sum(lengths[:i+1])-1 for i in range(b)]
                    batch = synthetic_batch(lengths, self.max_blocks_per_seq, last_row=last,
                                            is_prefill=prefill)
                    with ops.dry_load() as measurement:
                        for t in (*model.weight_tensors(), history, pages.full, pages.sliding, *vars(batch).values()):
                            if isinstance(t, torch.Tensor):
                                measurement.note(t)
                        resident = measurement.device
                        self._run(model, batch, history, pages)
                        peaks.append(measurement.peak_device - resident)
            self._act[rows] = max(peaks)
        return self._act[rows]

    def _run(self, model, batch, history, pages):
        b, n = batch.batch_size, batch.num_tokens
        device = batch.input_ids.device
        if batch.is_prefill:
            ends = batch.cu_seqlens[1:].long()
            seq = torch.searchsorted(ends, torch.arange(n, device=device), right=True).int()
        else:
            seq = torch.arange(b, device=device, dtype=torch.int32)
        position = batch.positions[0].int()
        plan = (seq, position, batch.cu_seqlens, batch.state_indices)
        table = batch.block_tables.contiguous()
        mapping = batch.slot_mapping.long().clamp_min_(0)
        sliding = pages.sliding
        if sliding is not None:
            width = sliding.size(3)
            offset = batch.state_indices.long() * width
            local_table = (offset[:, None] + torch.arange(width, device=device)).int()
            ring_tokens = width * self.block_size
            local_mapping = offset.index_select(0, seq.long()) * self.block_size + position.long() % ring_tokens
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
            return state, table_i, mapping_i
        hidden = model.hidden(batch.input_ids, position, seq, visit_inputs, plan, batch.is_prefill)
        if batch.is_prefill:
            hidden = hidden.index_select(0, batch.last_row)
        return model.logits(hidden) if batch.need_logits else None

    def _forward_eager(self, batch):
        self.n_eager_forwards += 1
        self.last_hidden = None
        return self._run(self.model, batch, self.history, self.pages)

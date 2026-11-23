# SPDX-License-Identifier: Apache-2.0
import os
import weakref

import torch

from .. import ops
from .forward_context import Batch
from .graphs import GraphCase, GraphInputs
from .inputs import i32
from .spec_decode import SpecDecoder


class Qwen4ExpSpecDecoder(SpecDecoder):
    def __init__(self, runner, sampler, slots, tables, stage, **kwargs):
        super().__init__(runner, sampler, slots, tables, stage, **kwargs)
        self.verify_hist = [0] * (self.num_spec + 1)
        self.indexer = getattr(runner.model.mtp.layer.attn, 'indexer', None)
        first_scratch = runner.dummy_slot + 1
        self.scratch_states = i32(list(range(first_scratch, first_scratch + runner.max_num_seqs)))
        self.selected_states = torch.empty_like(self.scratch_states)
        self.use_graphs = os.environ.get('SNOWLLM_MTP_DRAFT_GRAPH', '1') != '0' and not self.window
        if self.use_graphs:
            batch, rows = runner.max_num_seqs, runner.max_num_seqs * (self.num_spec + 1)
            self.graph_pos = torch.zeros(3 * rows, dtype=torch.int64, device='cuda')
            self.graph_seq = torch.ones(batch, dtype=torch.int32, device='cuda')
            self.graph_table = torch.zeros(batch, runner.max_blocks_per_seq, dtype=torch.int32, device='cuda')
            self.graph_cu = torch.zeros(batch + 1, dtype=torch.int32, device='cuda')
            self.graph_states = torch.arange(rows, dtype=torch.int32, device='cuda')
        if ops.drying() is None:
            runner.mtp_proposer = weakref.proxy(self)

    @staticmethod
    def graph_key(batch, width):
        return ('mtp-proposals', batch, width)

    def graph_case(self, shape):
        if not self.use_graphs or shape.lengths or shape.T <= 1:
            return None
        batch, width = shape.B, shape.T
        rows = batch * width
        self.d_tokens.zero_()
        self.d_accept.fill_(1)
        self.graph_seq.fill_(width)
        self.graph_cu[:batch + 1].copy_(torch.arange(batch + 1, dtype=torch.int32, device='cuda') * width)
        pos = self.graph_pos[:3 * rows].view(3, rows)
        seq, table, cu = self.graph_seq[:batch], self.graph_table[:batch], self.graph_cu[:batch + 1]
        states = self.graph_states[:rows]
        inputs = GraphInputs(dict(positions=pos, seq_lens=seq, block_tables=table, cu_seqlens=cu,
                                  state_indices=states))
        total = ops.prefill_q_plan([width] * batch)[0]
        return GraphCase(self.graph_key(batch, width), f'MTP proposals graph B={batch} T={width}',
                         lambda: self._run_proposals(batch, width, width - 1, pos, cu, total,
                                                     seq, table, states),
                         inputs, draft=True)

    def _run_proposals(self, batch, width, draft_k, pos, cu, total, seq, table, states):
        rows = batch * width
        self._bt = table
        tokens, accepted = self.d_tokens[:rows].view(batch, width), self.d_accept[:batch]
        ids, mapping, selected = self._first_metadata(tokens, accepted)
        bt, seq = self._window(table, seq)
        mb = Batch(input_ids=ids, positions=pos, slot_mapping=mapping, block_tables=bt,
                   seq_lens=seq, is_prefill=False, num_tokens=rows,
                   state_indices=states, cu_seqlens=cu,
                   total_q_blocks=total, chunk_decode=True)
        hidden = self._first_proposal(mb, self.runner._mtp_input(rows), selected)
        if draft_k > 1:
            live = ops.gather_rows(states, selected, out=self.selected_states[:batch])
            self._scratch = self._copy_carry(live, batch)
            self._propose_rest_dev(batch, hidden, draft_k, accepted)

    def _copy_carry(self, states, batch):
        if self.indexer is None:
            return self.slots.dummy_indices(batch)
        end = self.runner.dummy_slot + 1
        for field in (self.indexer.carry, self.indexer.carry_pos):
            ops.copy_rows(field[end:end + batch].flatten(1), field[:end].flatten(1),
                          src_index=states)
        return self.scratch_states[:batch]

    def _draft_states(self, batch):
        return self._copy_carry(i32([r.state_head for r in batch]), len(batch))

    def _first_proposal(self, batch, source, selected):
        batch.last_row = selected
        logits, hidden = self.runner.mtp_draft(source, batch.input_ids, batch)
        B = batch.batch_size
        ops.copy_rows(self.d_drafts[:B, 0], self._draft_tokens(logits))
        ops.copy_rows(self.runner.mtp_h[:B], hidden)
        return self.runner.mtp_h[:B]

    def _propose_verified(self, batch, pos, cu, total, k, tv, nacc, seq, states, draft_limit,
                          *, terminal=False):
        self.verify_hist[k] += len(batch)
        if terminal:
            return 0, nacc.tolist()
        self.propose_first(batch, pos, cu, total, k, tv, nacc, seq, states, k)
        return k, nacc.tolist()

    def propose_first(self, batch, pos, cu, total_q_blocks, k, tokens, accepted, seq,
                      states, draft_k):
        B, width = len(batch), k + 1
        graph = (self.runner.graphs.get(self.graph_key(B, width))
                 if self.use_graphs and draft_k == k else None)
        with self.runner._input_scope():
            rows = tokens.numel()
            if tokens.data_ptr() != self.d_tokens.data_ptr():
                ops.copy_rows(self.d_tokens[:rows], tokens.reshape(-1))
            if accepted.data_ptr() != self.d_accept.data_ptr():
                ops.copy_rows(self.d_accept[:B], accepted)
            table = self.tables([r.blocks for r in batch])
            if graph is None:
                return self._run_proposals(B, width, draft_k, pos, cu,
                                           total_q_blocks or ops.prefill_q_plan([width] * B)[0],
                                           seq, table, states)
            graph.replay(dict(positions=pos, seq_lens=seq, block_tables=table, cu_seqlens=cu,
                              state_indices=states))
        self.runner.n_draft_replays += 1

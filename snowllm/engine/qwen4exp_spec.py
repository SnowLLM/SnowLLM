# SPDX-License-Identifier: Apache-2.0
import os
import weakref

import torch

from .. import ops
from .forward_context import Batch
from .graphs import GraphCase, GraphInputs
from .spec_decode import SpecDecoder


class Qwen4ExpSpecDecoder(SpecDecoder):
    def __init__(self, runner, sampler, slots, tables, stage, **kwargs):
        super().__init__(runner, sampler, slots, tables, stage, **kwargs)
        self.use_graphs = os.environ.get('SNOWLLM_MTP_DRAFT_GRAPH', '1') != '0' and not self.window
        if self.use_graphs:
            batch, rows = runner.max_num_seqs, runner.max_num_seqs * (self.num_spec + 1)
            self.graph_pos = torch.zeros(3 * rows, dtype=torch.int64, device='cuda')
            self.graph_seq = torch.ones(batch, dtype=torch.int32, device='cuda')
            self.graph_table = torch.zeros(batch, runner.max_blocks_per_seq, dtype=torch.int32, device='cuda')
            self.graph_cu = torch.zeros(batch + 1, dtype=torch.int32, device='cuda')
            self.graph_states = torch.full((rows,), runner.dummy_slot, dtype=torch.int32, device='cuda')
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
        inputs = GraphInputs(dict(positions=pos, seq_lens=seq, block_tables=table, cu_seqlens=cu))
        total = ops.prefill_q_plan([width] * batch)[0]
        return GraphCase(self.graph_key(batch, width), f'MTP proposals graph B={batch} T={width}',
                         lambda: self._run_proposals(batch, width, pos, cu, total, seq, table),
                         inputs, draft=True)

    def _run_proposals(self, batch, width, pos, cu, total, seq, table):
        rows = batch * width
        self._bt = table
        tokens, accepted = self.d_tokens[:rows].view(batch, width), self.d_accept[:batch]
        ids, mapping, selected = self._first_metadata(tokens, accepted)
        mb = Batch(input_ids=ids, positions=pos, slot_mapping=mapping, block_tables=table,
                   seq_lens=seq, is_prefill=False, num_tokens=rows,
                   state_indices=self.graph_states[:rows], cu_seqlens=cu,
                   total_q_blocks=total, chunk_decode=True)
        logits, hidden = self.runner._draft_eager(self.runner._mtp_input(rows), ids, mb)
        ops.gather_rows(self._draft_tokens(logits), selected, out=self.d_drafts[:batch, 0])
        if width > 2:
            self._scratch = self.graph_states[:batch]
            hidden = ops.gather_rows(hidden, selected, out=self.runner.mtp_h[:batch])
            self._propose_rest_dev(batch, hidden, width - 1, accepted)

    def propose_first(self, batch, pos, cu, total_q_blocks, k, tokens, accepted, seq):
        graph = self.runner.graphs.get(self.graph_key(len(batch), k + 1)) if self.use_graphs else None
        if graph is None:
            return super().propose_first(batch, pos, cu, total_q_blocks, k, tokens, accepted, seq)
        with self.runner._input_scope():
            rows = tokens.numel()
            if tokens.data_ptr() != self.d_tokens.data_ptr():
                ops.copy_rows(self.d_tokens[:rows], tokens.reshape(-1))
            if accepted.data_ptr() != self.d_accept.data_ptr():
                ops.copy_rows(self.d_accept[:len(batch)], accepted)
            table = self.tables([r.blocks for r in batch])
            graph.replay(dict(positions=pos, seq_lens=seq, block_tables=table, cu_seqlens=cu))
        self.runner.n_draft_replays += 1

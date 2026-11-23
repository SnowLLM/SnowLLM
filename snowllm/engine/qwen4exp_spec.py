# SPDX-License-Identifier: Apache-2.0
import os
import weakref
from collections import deque

import torch

from .. import ops
from .forward_context import Batch
from .graphs import GraphCase, GraphInputs
from .inputs import i32
from .mtp_lookup import MtpLookup
from .spec_decode import SpecDecoder


class _ToolDraft:
    def __init__(self, fmt):
        self.parser = fmt.new_state()
        self.scanned = 0
        self.suffix = None
        self.native_depth = 2
        self.rounds = self.proposed = self.verified = self.accepted = 0

    def propose(self, request, limit):
        self.parser.advance(request.out[self.scanned:])
        self.scanned = len(request.out)
        draft = self.parser.propose(limit)
        native = request.drafts
        if len(draft) < 2 or len(draft) <= len(native):
            self.suffix = None
            return False
        if len(native) >= 2:
            self.native_depth = len(native)
        self.suffix = request.drafts = draft
        self.proposed += len(draft)
        return True


class MtpDepth:
    def __init__(self):
        self.samples = weakref.WeakKeyDictionary()

    def observe(self, batch, k, accepted):
        if k:
            for request, count in zip(batch, accepted):
                history = self.samples.get(request)
                if history is None:
                    self.samples[request] = history = deque(maxlen=8)
                history.append((k, count - 1))

    @staticmethod
    def extend(k, kept, history, gap, cap):
        if kept < 2 or not gap >= 3.:
            return 2
        if gap >= 6. and ((k >= 4 and kept == k) or
                          any(width >= 4 and accepted == width for width, accepted in history[-4:])):
            return min(7, cap)
        return min(4, cap)

    def select(self, batch, k, cap):
        if len(batch) == 1 and batch[0].params.temperature > 0:
            request = batch[0]
            remaining = request.params.max_new_tokens - len(request.out)
            depth = self._sampled(self.samples.get(request, ()), k, cap, remaining)
            if k == 2 and depth > 2 and not self._repeated_prefix(request, depth):
                return 2
            return depth
        if k < 1:
            return min(2, cap)
        k = min(k, cap)
        limit = min(k + 1, cap)
        tried, accepted = [0] * (limit + 1), [0] * (limit + 1)
        for request in batch:
            for width, kept in self.samples.get(request, ()):
                for depth in range(1, min(width, limit) + 1):
                    tried[depth] += 1
                    accepted[depth] += kept >= depth
        if not k or tried[k] < (5 if k == 2 else 3) * len(batch):
            return k
        survival = [1.] + [hit / n if n else 0. for hit, n in zip(accepted[1:], tried[1:])]
        # The UD3.1 decode sweep measured about four extra-draft steps per base step.
        target_cost = 4
        mean = 1.
        scores = []
        for depth in range(1, k + 1):
            mean += survival[depth]
            scores.append(mean / (target_cost + depth))
        best = max(range(k), key=scores.__getitem__) + 1
        if scores[best - 1] > 1.05 * scores[k - 1]:
            return best
        required = (5 if k == 2 or survival[k] < 1. else 3) * len(batch)
        if k == limit or tried[k] < required or not survival[k - 1]:
            return k
        # Use a recent observation of the next position when one is available.
        predicted = survival[k] * min(1., survival[k] / survival[k - 1])
        if tried[k + 1] >= 3 * len(batch):
            predicted = min(predicted, survival[k + 1])
        if (mean + predicted) / (target_cost + k + 1) > 1.025 * scores[k - 1]:
            return k + 1
        return k

    @staticmethod
    def _repeated_prefix(request, depth):
        matched = 3 * depth
        if len(request.out) < matched or len(request.drafts) < 2:
            return False
        prefix = request.out[-matched:] + request.drafts[:2]
        return (MtpDepth._matches_passage(request.prompt, len(request.prompt), 4096, prefix, depth - 2)
                or MtpDepth._matches_passage(request.out, len(request.out) - matched, 1024, prefix, depth - 2))

    @staticmethod
    def _matches_passage(tokens, end, window, prefix, tail):
        start = max(0, end - window) + len(prefix) - 1
        end -= tail
        while start < end:
            try:
                last = tokens.index(prefix[-1], start, end)
            except ValueError:
                break
            if tokens[last - len(prefix) + 1:last + 1] == prefix:
                return True
            start = last + 1
        return False

    @staticmethod
    def _sampled(history, k, cap, remaining):
        if cap <= 2 or k < 2:
            return min(2, cap)
        if k > cap:
            return cap
        recent = []
        for width, kept in reversed(history):
            if width != k:
                break
            recent.append(kept)
        wide = min(7, cap)
        if k == 2:
            if (remaining >= 2 * (wide + 1) and len(recent) >= 6
                    and recent[:2] == [2, 2] and recent[:6].count(2) >= 5):
                return wide
            return 2
        # History trails the current verification by one round so stable widths can replay their graph.
        if recent and recent[0] < 2:
            return 2
        if len(recent) < 2:
            return k
        recent = recent[:4]
        base = sum(1 + min(2, kept) for kept in recent)
        extra = sum(max(0, kept - 2) for kept in recent)
        # Compare extra accepted tokens to the marginal cost above a K=2 round (four base + two draft units).
        if 6 * extra <= (k - 2) * base:
            return 2
        if recent[:3] == [k, k, k] and remaining >= k + 1 + 2 * (wide + 1):
            return wide
        return k


class Qwen4ExpSpecDecoder(SpecDecoder):
    def __init__(self, runner, sampler, slots, tables, stage, *, adaptive=False, prompt_lookup=False,
                 **kwargs):
        super().__init__(runner, sampler, slots, tables, stage, **kwargs)
        self.adaptive = MtpDepth() if adaptive or prompt_lookup else None
        self.lookups = weakref.WeakKeyDictionary() if prompt_lookup else None
        self.tool_states = weakref.WeakKeyDictionary()
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

    def step_rows(self, n):
        if self.adaptive is not None:
            return tuple(range(1, max(1, self.verify_rows(n)) + 1))
        return super().step_rows(n)

    def release_lookup(self, request):
        if self.lookups is not None:
            self.lookups.pop(request, None)

    def tool_stats(self, request):
        state = self.tool_states.get(request)
        return {name: getattr(state, name, 0) for name in ('rounds', 'proposed', 'verified', 'accepted')}

    def _offer_tool_draft(self, request, cap):
        limit = min(7, cap, request.params.max_new_tokens - len(request.out) - 1)
        if limit < 2:
            return False
        state = self.tool_states.get(request)
        if state is None:
            state = self.tool_states[request] = _ToolDraft(request.tool_draft)
        return state.propose(request, limit)

    def propose_rest(self, batch, h, k):
        if (self.adaptive is not None and len(batch) == 1 and batch[0].tool_draft is not None
                and self._offer_tool_draft(batch[0], self.num_spec)):
            return
        super().propose_rest(batch, h, k)

    def verify_step(self, batch, T, draft_limit=None):
        request = batch[0]
        tool = len(batch) == 1 and request.tool_draft is not None
        if draft_limit is None or (not tool and self.lookups is None):
            return super().verify_step(batch, T, draft_limit)
        lookup = self.lookups.get(request) if self.lookups is not None else None
        used_suffix = T > 1 and lookup is not None and lookup.suffix is request.drafts
        state = self.tool_states.get(request)
        used_tool = T > 1 and state is not None and state.suffix is request.drafts
        super().verify_step(batch, T, draft_limit)
        if len(batch) > 1 or request.done:
            for row in batch:
                self.release_lookup(row)
            return
        limit = min(draft_limit, request.params.max_new_tokens - len(request.out) - 1)
        if tool and self._offer_tool_draft(request, limit):
            return
        if self.lookups is not None:
            if lookup is None and limit >= 3:
                lookup = self.lookups[request] = MtpLookup(request, self.runner.geo.vocab_size)
            if lookup is not None:
                lookup.propose(request, limit, used_suffix)
        if used_tool and limit >= 2 and len(request.drafts) == 1:
            # The committed MTP head already produced the first native token and its hidden state.
            super().propose_rest(batch, self.runner.mtp_h[:1], min(state.native_depth, limit))

    def _propose_verified(self, batch, pos, cu, total, k, tv, nacc, seq, states, draft_limit,
                          *, terminal=False):
        self.verify_hist[k] += len(batch)
        if draft_limit is None:
            if terminal:
                return 0, nacc.tolist()
            self.propose_first(batch, pos, cu, total, k, tv, nacc, seq, states, k)
            return k, nacc.tolist()
        greedy_single = len(batch) == 1 and batch[0].params.temperature <= 0
        sources = {}
        if k and (self.tool_states or self.lookups):
            for request in batch:
                state = self.tool_states.get(request)
                if state is not None and state.suffix is request.drafts:
                    sources[request] = state
                    continue
                source = self.lookups.get(request) if self.lookups else None
                if source and (source.suffix is request.drafts or source.guard is request.drafts):
                    sources[request] = source
                    if greedy_single:
                        self.adaptive.samples.pop(request, None)
        if terminal:
            draft_k, acc = 0, nacc.tolist()
        elif greedy_single and draft_limit > 2 and not sources:
            draft_k, acc = self._guarded_proposals(batch, pos, cu, total, k, tv, nacc, seq, states,
                                                   draft_limit)
        else:
            if sources and len(batch) == 1 and sources[batch[0]].suffix is batch[0].drafts:
                draft_k = min(1, draft_limit)
            elif greedy_single and sources:
                # Resume at K=2 after the single-token guard left by the final lookup.
                draft_k = min(2, draft_limit)
            else:
                previous = (min(sources[r].native_depth if r in sources else k for r in batch)
                            if sources else k)
                draft_k = self.adaptive.select(batch, previous, draft_limit)
            self.propose_first(batch, pos, cu, total, k, tv, nacc, seq, states, draft_k)
            acc = nacc.tolist()
        if sources:
            for request, count in zip(batch, acc):
                source = sources.get(request)
                if isinstance(source, _ToolDraft):
                    source.rounds += 1
                    source.verified += k
                    source.accepted += count - 1
                elif source is None:
                    self.adaptive.observe([request], k, [count])
        else:
            self.adaptive.observe(batch, k, acc)
        return draft_k, acc

    @staticmethod
    def _greedy_prompt_match(request, cap):
        if len(request.out) < 4 or len(request.drafts) < 2:
            return False
        prefix = request.out[-4:] + request.drafts[:2]
        return MtpDepth._matches_passage(request.prompt, len(request.prompt), 4096, prefix, cap - 2)

    def _guarded_proposals(self, batch, pos, cu, total, k, tv, nacc, seq, states, cap):
        history = tuple(self.adaptive.samples.get(batch[0], ()))
        stable = k == 7 and cap >= 7 and history[-2:] == ((7, 7), (7, 7))
        depth = 7 if stable else 2
        self.propose_first(batch, pos, cu, total, k, tv, nacc, seq, states, depth)
        if stable or k < 2 or not history or history[-1][1] < 2:
            return depth, nacc.tolist()
        copied = False
        if k == 2:
            remaining = batch[0].params.max_new_tokens - len(batch[0].out)
            if remaining < 2 * (min(7, cap) + 1):
                return depth, nacc.tolist()
            if len(history) < 6 or any(kept < 2 for _, kept in history[-6:]):
                copied = self._greedy_prompt_match(batch[0], cap)
                if not copied:
                    return depth, nacc.tolist()
        count, gap = self._second_draft_margin(nacc)
        if copied and count == 3 and gap >= 6.:
            depth = min(7, cap)
        else:
            depth = self.adaptive.extend(k, count - 1, history, gap, cap)
        if depth > 2:
            with self.runner._input_scope():
                # A replay updated the device buffers but left the Python views from graph capture.
                self._bt = self.tables([batch[0].blocks])
                self._scratch = self.scratch_states[:1] if self.indexer is not None else self.slots.dummy_indices(1)
                self._propose_rest_dev(1, self.runner.mtp_streams[:1], depth, self.d_accept[:1], start=2)
        return depth, [count]

    def _second_draft_margin(self, nacc):
        logits = self.runner.draft_logits[0]
        token = self.d_drafts[0, 1:2]
        # B=1 uses the spare logits row for the return message and the native argmax for second place.
        result = self.runner.draft_logits[1, :3]
        torch.gather(logits, 0, token, out=result[1:2])
        logits.index_fill_(0, token, -float('inf'))
        try:
            second = ops.argmax(logits.view(1, -1), out=self.d_argmax[:1])
            torch.gather(logits, 0, second, out=result[2:])
        finally:
            logits.index_copy_(0, token, result[1:2])
        result[:1].copy_(nacc)
        count, first, second = result.tolist()
        return int(count), first - second

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

# SPDX-License-Identifier: Apache-2.0
"""Build scheduler batches independently of their input storage."""
import torch

from .. import ops
from .block_manager import block_tables
from .forward_context import Batch, i32, i64


def prefill_batch(spans, slots, block_size, inputs=None, prev_context=0):
    ids, pos, mapping, states, lens, resuming = [], [], [], [], [], []
    cu = [0]
    embed_rows, embeds = [], []
    visual = any(r.embeds is not None or r.mrope is not None for r, _, _ in spans)
    for r, lo, hi in spans:
        base = len(ids)
        ids.extend(r.prefill_src[lo:hi])
        pos.extend(range(lo, hi))
        mapping.extend(r.blocks[p // block_size] * block_size + p % block_size
                       for p in range(lo, hi))
        cu.append(len(ids))
        lens.append(hi)
        states.append(slots.name(r, 1)[0])
        resuming.append(int(lo > 0))
        selected = [k for k, row in enumerate(r.embed_rows) if lo <= row < hi]
        if selected:
            embed_rows.extend(base + r.embed_rows[k] - lo for k in selected)
            embeds.append(r.embeds[i64(selected)])

    last = [end - 1 for end in cu[1:]]
    lengths = [hi - lo for _, lo, hi in spans]
    if inputs is not None and not visual:
        d_ids, d_pos, d_last, d_map, d_lens, d_states, d_cu, d_resume = inputs(
            (ids, pos * 3, last), (mapping, lens, states, cu, resuming))
        positions = d_pos.view(3, -1)
        tables = inputs.tables([r.blocks for r, _, _ in spans])
        total, qmap = inputs.q_plan("prefill_qmap", lengths)
    else:
        # Draft/verify paths can retain these tensors across asynchronous calls;
        # give each prefill its own device storage rather than reuse staging.
        d_ids, d_last = i64(ids), i64(last)
        if visual:
            positions = torch.empty(3, len(ids), dtype=torch.int64, device="cuda")
            for at, (r, lo, hi) in zip(cu, spans):
                positions[:, at:at + hi - lo] = r.prefill_positions(lo, hi)
        else:
            positions = i64(pos * 3).view(3, -1)
        d_map, d_lens, d_states, d_cu = map(i32, (mapping, lens, states, cu))
        d_resume = i32(resuming) if any(resuming) else None
        tables = block_tables([r for r, _, _ in spans])
        total, qmap = ops.prefill_q_plan(lengths)

    return Batch(
        input_ids=d_ids, positions=positions, slot_mapping=d_map, block_tables=tables,
        seq_lens=d_lens, is_prefill=True, num_tokens=len(ids), state_indices=d_states,
        last_row=d_last, cu_seqlens=d_cu, has_state=d_resume if any(resuming) else None,
        total_q_blocks=total, q_block_map=qmap,
        host_ids=ids, host_seq_lens=lens, host_cu_seqlens=cu,
        need_logits=all(hi == r.prefill_len and not r.replay for r, _, hi in spans),
        embeds=torch.cat(embeds) if embeds else None,
        embed_rows=i64(embed_rows) if embed_rows else None,
        prev_ids=([list(r.prefill_src[max(0, lo - prev_context):lo]) for r, lo, _ in spans]
                  if prev_context else None))


def synthetic_batch(lengths, table_width, *, tensor=None, **fields):
    lengths = list(lengths)
    cu = [0]
    for n in lengths:
        cu.append(cu[-1]+n)
    B, M = len(lengths), cu[-1]
    if tensor is None:
        tensor = lambda name, values, dtype: torch.tensor(values, dtype=dtype, device='cuda')
    defaults = {
        'input_ids': ([0]*M, torch.int64),
        'positions': ([0]*(3*M), torch.int64),
        'slot_mapping': ([0]*M, torch.int32),
        'block_tables': ([0]*(B*table_width), torch.int32),
        'seq_lens': ([max(1, n) for n in lengths], torch.int32),
        'state_indices': ([0]*B, torch.int32),
        'cu_seqlens': (cu, torch.int32),
    }
    for name, (values, dtype) in defaults.items():
        value = fields.get(name, values)
        if value is not None and not isinstance(value, torch.Tensor):
            value = tensor(name, value, dtype)
        fields[name] = value
    for name, dtype in (('num_accepted', torch.int32), ('last_row', torch.int64),
                        ('has_state', torch.int32)):
        value = fields.get(name)
        if value is not None and not isinstance(value, torch.Tensor):
            fields[name] = tensor(name, value, dtype)
    fields['positions'] = fields['positions'].view(3, M)
    fields['block_tables'] = fields['block_tables'].view(B, table_width)
    fields.setdefault('is_prefill', False)
    fields.setdefault('host_seq_lens', [max(1, n) for n in lengths])
    fields.setdefault('host_cu_seqlens', cu)
    return Batch(num_tokens=M, **fields)


def history_batch(tokens, positions, firsts, lengths, slots, table, mapping, *,
                  min_table_width=0, messages=None, last_row=None, is_prefill=True):
    from .block_manager import i32_row
    cu = [0]
    for n in lengths:
        cu.append(cu[-1]+n)
    ends = [f+n for f, n in zip(firsts, lengths)]
    if messages is not None:
        messages.begin()
        seqs = [i for i, n in enumerate(lengths) for _ in range(n)]
        seqs.extend([len(lengths)-1]*(tokens.numel()-len(seqs)))
        seq = messages.tensor('row_sequences', seqs, torch.int32)
        cumulative = messages.tensor('cumulative_lengths', cu, torch.int32)
        lens = messages.tensor('lengths', ends, torch.int32)
        states = messages.tensor('state_slots', slots, torch.int32)
        total, qmap = messages.q_plan('query_blocks', lengths)
    else:
        ranges = [(i, 0, n) for i, n in enumerate(lengths)]
        ranges.append((len(lengths)-1, 0, tokens.numel()-sum(lengths)))
        seq = ops.metadata_ranges([(torch.int32, ranges)])[0]
        cumulative, lens = i32(cu), i32(ends)
        states = i32_row(slots).to('cuda')
        total, qmap = ops.prefill_q_plan(lengths)
    if table.shape[1] < min_table_width:
        wide = torch.empty(table.shape[0], min_table_width, dtype=table.dtype, device=table.device)
        ops.copy_rows(wide, table)
        table = wide
    return Batch(input_ids=tokens, positions=positions, seq_of_row=seq,
                 cu_seqlens=cumulative, seq_lens=lens, state_indices=states,
                 block_tables=table, slot_mapping=mapping, total_q_blocks=total,
                 q_block_map=qmap, last_row=last_row, is_prefill=is_prefill,
                 num_tokens=tokens.numel(), host_cu_seqlens=cu, host_seq_lens=ends)

# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import torch

from ...engine.cache import tensor
from ... import ops
from ..geometry import V41_MOE_CHUNK


class EncoderTail:
    def __init__(self, geo, slots, slabs):
        self.window = geo.sliding_window
        self.streams = tensor(slabs, (slots, self.window, geo.hc_mult, geo.hidden), torch.bfloat16)
        self.pre = tensor(slabs, (slots, self.window, geo.hc_mult), torch.float32)
        self.ids = tensor(slabs, (slots, self.window), torch.int64)
        self.positions = tensor(slabs, (slots, self.window), torch.int64)
        self.mapping = tensor(slabs, (slots, self.window), torch.int32)
        self.sequences = tensor(slabs, (self.window,), torch.int32)
        self.cumulative = tensor(slabs, (2,), torch.int32)
        self.last = tensor(slabs, (1,), torch.int32)

    def append(self, ctx, streams):
        b, a = ctx.batch, ctx.arena
        first = b.host_seq_lens[0] - ctx.M
        take = min(ctx.M, self.window)
        keep = b.ced_keep
        slot = 0
        trace = ops.tracing()
        launch = trace is None or trace.launch
        for dest, source in ((self.streams[slot], streams), (self.pre[slot], ctx.pre_mix),
                             (self.ids[slot], b.input_ids), (self.positions[slot], b.positions),
                             (self.mapping[slot], b.slot_mapping)):
            if keep:
                # The retained prefix overlaps its destination; stage it inside the planned arena.
                with a.frame():
                    saved = a.new(keep, *dest.shape[1:], dtype=dest.dtype)
                    if launch:
                        ops.copy_tensor(saved, dest[min(first, self.window) - keep:min(first, self.window)])
                        ops.copy_tensor(dest[:keep], saved)
            if launch:
                ops.copy_tensor(dest[keep:keep + take], source[-take:])

    def retained(self, batch):
        rows = batch.input_ids.numel()
        return min(batch.host_seq_lens[0] - rows, max(0, self.window - rows))

    def batch(self, source):
        end = source.host_seq_lens[0]
        rows = min(end, self.window)
        ops.fill_ranges_(((self.sequences, 0, 0), (self.cumulative, 0, rows), (self.last, rows - 1, 0)))
        return replace(source, input_ids=self.ids[0, :rows], positions=self.positions[0, :rows],
                       slot_mapping=self.mapping[0, :rows], seq_of_row=self.sequences[:rows],
                       cu_seqlens=self.cumulative, host_cu_seqlens=[0, rows], host_ids=None,
                       num_tokens=rows, last_row=self.last, total_q_blocks=0, q_block_map=None,
                       inplace=False, split=False, ced_stage="replay", swa_start=end - rows,
                       v41_microbatches=-(-rows // V41_MOE_CHUNK))

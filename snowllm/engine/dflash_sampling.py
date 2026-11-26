# SPDX-License-Identifier: Apache-2.0
import torch

from .. import ops
from .host_io import HostArena


class DFlashSampling:
    def __init__(self, sampler, geo, max_seqs, block, vocab):
        self.sampler = sampler
        self.block = block
        self.length = block - 1
        self.inputs = HostArena()
        self.proposals = [None] * max_seqs
        self.temperature = torch.zeros(max_seqs, dtype=torch.float32, device="cuda")
        self.slots = torch.arange(max_seqs, dtype=torch.int32, device="cuda")
        self.uniform = torch.zeros(max_seqs, self.length, dtype=torch.float32, device="cuda")
        width = min(16, vocab) if geo.inkling else geo.selector_top_k
        self.candidates = torch.empty(max_seqs, self.length, width,
                                      dtype=torch.int64, device="cuda")
        self.q = torch.empty_like(self.candidates, dtype=torch.float32)
        self.pools = {self.length: (self.candidates, self.q)}
        rows = max_seqs * block
        width = min(vocab, ops.SAMPLING_MAX_K)
        self.p = torch.empty(rows * width, dtype=torch.float32, device="cuda")
        self.p_ids = torch.empty(rows * width, dtype=torch.int64, device="cuda")
        self.tokens = torch.empty(rows, dtype=torch.int64, device="cuda")
        self.accepted = torch.empty(max_seqs, dtype=torch.int64, device="cuda")

    def reserve_width(self, length):
        shape = (self.q.size(0), length, self.q.size(2))
        self.pools[length] = (self.candidates.new_empty(shape), self.q.new_empty(shape))

    def stage(self, batch):
        count = len(batch)
        temperatures = [r.params.temperature for r in batch]
        with self.inputs.reading(reuse=True):
            temp = self.inputs.tensor("draft_temperature", temperatures, torch.float32)
            slots = self.inputs.tensor("draft_slots", [r.slot for r in batch], torch.int32)
            ops.copy_rows(self.temperature[:count], temp)
            ops.copy_rows(self.slots[:count], slots)
        self.draw_uniforms(batch)

    def draw_uniforms(self, batch):
        if any(r.params.temperature > 0 for r in batch):
            self.uniform[:len(batch)].uniform_(generator=self.sampler.rng)

    def propose(self, draft, hidden, logits, anchor, path):
        count, length = path.shape
        candidates, q = self.pools[length]
        draft.propose(hidden, logits, anchor, path, temperature=self.temperature[:count],
                      uniform=self.uniform.view(-1)[:count*length].view(count, length), slots=self.slots[:count],
                      candidate_ids=candidates, probabilities=q)

    def remember(self, batch, drafts):
        for request, tokens in zip(batch, drafts):
            self.proposals[request.slot] = tuple(tokens)

    def release(self, request):
        self.proposals[request.slot] = None

    def verify(self, logits, batch, drafts):
        count, width = drafts.shape[0], drafts.shape[1] + 1
        rows = [r for r in batch for _ in range(width)]
        tokens = self.tokens[:count * width].view(count, width)
        accepted = self.accepted[:count]
        if all(r.params.temperature <= 0 for r in batch):
            self.sampler.sample(logits, rows, out=tokens.view(-1))
            return tokens, ops.spec_accept(tokens, drafts, out=accepted)
        p, ids = self.sampler.probabilities(logits, rows, self.p, self.p_ids)
        # Proposal probabilities belong to stable request slots, not transient batch rows.
        valid = [int(tuple(r.drafts) == self.proposals[r.slot]) for r in batch]
        uniform = self.sampler._uniform(count * width).view(count, width)
        with self.inputs.reading(reuse=True):
            length = width - 1
            lengths = [len(self.proposals[r.slot]) if ok else self.length for r, ok in zip(batch, valid)]
            packed = length in self.pools and all(n == length or not ok for n, ok in zip(lengths, valid))
            source_slots = [r.slot for r in batch] if packed else [0]
            slots = self.inputs.tensor("verify_slots", source_slots, torch.int32)
            current = self.inputs.tensor("verify_valid", valid, torch.int32)
            if packed:
                candidates, q = self.pools[length]
                ops.dflash.reject(p, ids, drafts, candidates, q, slots, current, uniform,
                                  tokens, accepted)
            else:
                # A rebatch can mix proposal strides. Each slot prefix is contiguous on its own.
                for i, request in enumerate(batch):
                    row = slice(i, i + 1)
                    target = slice(i * width, (i + 1) * width)
                    prefix = (slice(request.slot, request.slot + 1), slice(0, width - 1))
                    candidates, q = self.pools[lengths[i]]
                    ops.dflash.reject(p[target], ids[target], drafts[row], candidates[prefix],
                                      q[prefix], slots, current[row], uniform[row], tokens[row],
                                      accepted[row])
        return tokens, accepted

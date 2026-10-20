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
        self.candidates = torch.empty(max_seqs, self.length, geo.selector_top_k,
                                      dtype=torch.int64, device="cuda")
        self.q = torch.empty_like(self.candidates, dtype=torch.float32)
        rows = max_seqs * block
        width = min(vocab, ops.SAMPLING_MAX_K)
        self.p = torch.empty(rows * width, dtype=torch.float32, device="cuda")
        self.p_ids = torch.empty(rows * width, dtype=torch.int64, device="cuda")
        self.tokens = torch.empty(rows, dtype=torch.int64, device="cuda")
        self.accepted = torch.empty(max_seqs, dtype=torch.int64, device="cuda")

    def stage(self, batch):
        count = len(batch)
        temperatures = [r.params.temperature for r in batch]
        with self.inputs.reading(reuse=True):
            temp = self.inputs.tensor("draft_temperature", temperatures, torch.float32)
            slots = self.inputs.tensor("draft_slots", [r.slot for r in batch], torch.int32)
            ops.copy_rows(self.temperature[:count], temp)
            ops.copy_rows(self.slots[:count], slots)
        if any(t > 0 for t in temperatures):
            self.uniform[:count].uniform_(generator=self.sampler.rng)

    def propose(self, draft, hidden, logits, anchor, path):
        count = path.shape[0]
        draft.propose(hidden, logits, anchor, path, temperature=self.temperature[:count],
                      uniform=self.uniform[:count], slots=self.slots[:count],
                      candidate_ids=self.candidates, probabilities=self.q)

    def remember(self, batch, drafts):
        for request, tokens in zip(batch, drafts):
            self.proposals[request.slot] = tuple(tokens)

    def release(self, request):
        self.proposals[request.slot] = None

    def verify(self, logits, batch, drafts):
        count, width = len(batch), self.block
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
            slots = self.inputs.tensor("verify_slots", [r.slot for r in batch], torch.int32)
            current = self.inputs.tensor("verify_valid", valid, torch.int32)
            ops.dflash.reject(p, ids, drafts, self.candidates, self.q, slots, current, uniform,
                               tokens, accepted)
        return tokens, accepted

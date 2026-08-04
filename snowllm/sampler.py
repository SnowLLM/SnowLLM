# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from . import ops
from .request import Request
from .trace import span


class Sampler:
    def __init__(self, stop_token_ids: frozenset[int], seed: int | None = None):
        self.stop_token_ids = stop_token_ids
        self.rng = torch.Generator(device="cuda")
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample(self, logits: torch.Tensor, rows: list[Request]) -> torch.Tensor:
        with span("sample"):
            if any(r.params.penalized() for r in rows):
                self._penalize(logits, rows)
            if all(r.params.temperature <= 0.0 for r in rows):
                return ops.argmax(logits)
            ks = [r.params.kernel_top_k() for r in rows]
            top_p = torch.tensor([r.params.top_p for r in rows], dtype=torch.float32,
                                 device="cuda")
            temp = torch.tensor([r.params.kernel_temperature() for r in rows],
                                dtype=torch.float32, device="cuda")
            k = torch.tensor(ks, dtype=torch.int32, device="cuda")
            return ops.sample(logits, k, top_p, temp, max(ks), generator=self.rng)

    def _histogram(self, r: Request, V: int) -> tuple[torch.Tensor, torch.Tensor]:
        if r.out_counts is None:
            r.out_counts = torch.zeros(V, dtype=torch.int32, device="cuda")
            r.prompt_seen = torch.zeros(V, dtype=torch.bool, device="cuda")
            if r.prompt:
                r.prompt_seen[torch.tensor(r.prompt, dtype=torch.int64, device="cuda")] = True
            if r.out:
                ids = torch.tensor(r.out, dtype=torch.int64, device="cuda")
                r.out_counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        return r.out_counts, r.prompt_seen

    def _penalize(self, logits: torch.Tensor, rows: list[Request]) -> None:
        depth: dict[int, int] = {}
        for i, r in enumerate(rows):
            t = depth.get(id(r), 0)
            depth[id(r)] = t + 1
            p = r.params
            if not p.penalized():
                continue
            counts, in_prompt = self._histogram(r, logits.shape[1])
            row, drafted = logits[i], r.drafts[:t]
            idx = torch.tensor(drafted, dtype=torch.int64, device="cuda") if drafted else None

            if p.repetition_penalty != 1.0:
                hit = in_prompt | (counts > 0)
                if idx is not None:
                    hit = hit.clone()
                    hit[idx] = True
                row.copy_(torch.where(hit, torch.where(row > 0, row / p.repetition_penalty,
                                                       row * p.repetition_penalty), row))
            if p.frequency_penalty:
                row.sub_(counts.to(row.dtype), alpha=p.frequency_penalty)
                if idx is not None:
                    row.index_add_(0, idx, torch.full((len(drafted),), -p.frequency_penalty,
                                                      dtype=row.dtype, device="cuda"))
            if p.presence_penalty:
                row.sub_((counts > 0).to(row.dtype), alpha=p.presence_penalty)
                if idx is not None:
                    uniq = torch.tensor(list(dict.fromkeys(drafted)), dtype=torch.int64,
                                        device="cuda")
                    fresh = (counts.index_select(0, uniq) == 0).to(row.dtype)
                    row.index_add_(0, uniq, fresh.mul_(-p.presence_penalty))

    def emit(self, logits: torch.Tensor, batch: list[Request]) -> None:
        for r, tok in zip(batch, self.sample(logits, batch).tolist()):
            self.append(r, tok)

    def append(self, r: Request, tok: int) -> None:
        r.out.append(tok)
        if r.out_counts is not None:
            r.out_counts[tok] += 1
        if tok in self.stop_token_ids or tok in r.params.stop_token_ids:
            r.done, r.finish_reason = True, "stop"
        elif len(r.out) >= r.params.max_new_tokens:
            r.done, r.finish_reason = True, "length"

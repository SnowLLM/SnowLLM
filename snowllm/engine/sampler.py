# SPDX-License-Identifier: Apache-2.0
from collections import Counter
import ctypes

import torch

from .. import ops
from ..trace import span
from ..ops.sampling import penalties, probabilities, sample_workspace_bytes
from .host_io import HostArena, MappedBuffer
from .request import Request


class Sampler:
    """Native sampling reads host parameters and publishes tokens directly to the CPU."""
    def __init__(self, stop_token_ids, seed=None, *, native_rng=False):
        self.stop_token_ids = stop_token_ids
        self.native_rng = native_rng
        self._init_rng(seed)
        self.inputs = HostArena()
        self.output = MappedBuffer(torch.int64)

    def _init_rng(self, seed):
        self.rng = torch.Generator(device="cuda")
        if seed is not None:
            self.rng.manual_seed(seed)

    def _uniform(self, n):
        if self.native_rng:
            from ..ops.sampling import uniform_
            out = self.uniforms[:n] if hasattr(self, "uniforms") else torch.empty(n, dtype=torch.float32, device="cuda")
            return uniform_(out, self.rng)
        if hasattr(self, "uniforms"):
            return torch.rand(n, generator=self.rng, out=self.uniforms[:n])
        return torch.rand(n,dtype=torch.float32,device="cuda",generator=self.rng)

    def reserve(self, rows, vocab):
        self.reserve_device(rows, vocab)
        self.output.reserve(rows)

    def reserve_device(self, rows, vocab):
        self.uniforms = torch.empty(rows,dtype=torch.float32,device="cuda")
        self.workspace = torch.empty(
            sample_workspace_bytes(rows,vocab,min(vocab,ops.SAMPLING_MAX_K)),
            dtype=torch.uint8,device="cuda")

    def _parameters(self, rows, vocab):
        ks = [r.params.kernel_top_k() for r in rows]
        top_k = self.inputs.tensor("top_k", ks, torch.int32)
        top_p = self.inputs.tensor("top_p", [r.params.top_p for r in rows], torch.float32)
        temp = self.inputs.tensor("temperature", [r.params.kernel_temperature() for r in rows],
                                  torch.float32)
        return top_k, top_p, temp, min(max(ks), vocab)

    def sample(self, logits, rows, *, out=None):
        with span("sample"):
            self.inputs.begin()
            try:
                if any(r.params.penalized() for r in rows):
                    self._penalize(logits,rows,prepared=True)
                if all(r.params.temperature<=0 for r in rows):
                    return ops.argmax(logits,out=out)
                k, top_p, temp, width = self._parameters(rows, logits.shape[1])
                uniform = self._uniform(len(rows))
                return ops.sample(logits,k,top_p,temp,width,uniform=uniform,out=out,
                                  workspace=getattr(self,"workspace",None))
            finally:
                self.inputs.used()

    def probabilities(self, logits, rows, out, indices):
        with self.inputs.reading(reuse=True):
            if any(r.params.penalized() for r in rows):
                self._penalize(logits, rows, prepared=True)
            top_k, top_p, temp, width = self._parameters(rows, logits.shape[1])
            n = len(rows) * width
            values, ids = out[:n].view(-1, width), indices[:n].view(-1, width)
            probabilities(logits, top_k, top_p, temp, width, values, ids, self.workspace)
        return values, ids

    def _read_tokens(self, logits, rows=None):
        n = logits.shape[0]
        if not n:
            return []
        self.inputs.begin()
        self.output.reserve(n)
        out = self.output.device[:n]
        try:
            if rows is None:
                try:
                    ops.argmax(logits,out=out)
                finally:
                    self.inputs.used()
            else:
                self.sample(logits,rows,out=out)
        finally:
            # The sampling fence protects parameters and publishes the result;
            # no second event or device-to-host copy is necessary.
            self.inputs.begin()
        return list((ctypes.c_int64*n).from_address(self.output.host.data_ptr()))

    def greedy_list(self, logits):
        return self._read_tokens(logits)

    def sample_list(self, logits, rows):
        return self._read_tokens(logits,rows)

    def _penalize(self, logits, rows, *, prepared=False):
        if not prepared:
            self.inputs.begin()
        try:
            depth = {}
            for i,r in enumerate(rows):
                t=depth.get(id(r),0);depth[id(r)]=t+1
                p=r.params
                if not p.penalized():
                    continue
                counts=Counter(r.out);counts.update(r.drafts[:t])
                if not hasattr(r,"_sample_prompt_tokens"):
                    r._sample_prompt_tokens=set(r.prompt)
                tokens=sorted(r._sample_prompt_tokens | counts.keys())
                ids=self.inputs.tensor(("penalty_ids",i),tokens)
                cnt=self.inputs.tensor(("penalty_counts",i),[counts.get(v,0) for v in tokens],torch.int32)
                penalties(logits[i],ids,cnt,p.presence_penalty,p.frequency_penalty,p.repetition_penalty)
        finally:
            if not prepared:
                self.inputs.used()

    def __del__(self):
        if hasattr(self, "inputs"):
            self.inputs.begin()

    def emit(self, logits: torch.Tensor, batch: list[Request]) -> None:
        for r, tok in zip(batch, self.sample_list(logits, batch)):
            self.append(r, tok)

    def append(self, r: Request, tok: int) -> None:
        r.out.append(tok)
        if tok in self.stop_token_ids or tok in r.params.stop_token_ids:
            r.done, r.finish_reason = True, "stop"
        elif len(r.out) >= r.params.max_new_tokens:
            r.done, r.finish_reason = True, "length"

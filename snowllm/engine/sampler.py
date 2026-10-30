# SPDX-License-Identifier: Apache-2.0
from collections import Counter
import array
import ctypes
import contextlib

import torch

from .. import ops
from ..trace import span
from ..errors import SnowLLMError
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

    def reserve(self, rows, vocab, *, max_tokens=None, max_requests=None, arena=None):
        self.reserve_device(rows, vocab,arena=arena)
        self.output.reserve(rows)
        for name, dtype in (("top_k", torch.int32), ("top_p", torch.float32),
                            ("temperature", torch.float32)):
            self.inputs.reserve(name, rows, dtype)
        limit = vocab if max_tokens is None else min(vocab, max_tokens)
        requests = rows if max_requests is None else max_requests
        if not 0 < requests <= rows or limit <= 0:
            raise SnowLLMError('invalid penalty buffer dimensions')
        # IDs are shared by verification rows of the same request. Counts
        # remain distinct until all asynchronous penalty kernels finish.
        self._penalty_rows, self._penalty_requests, self._penalty_limit = rows,requests,limit
        self._penalty_id_bytes = requests*limit*8
        self._penalty_slab = MappedBuffer(torch.uint8)
        self._penalty_slab.reserve(self._penalty_id_bytes+rows*limit*4,exact=True)
        self._penalty_ids = self._penalty_slab.device[:self._penalty_id_bytes].view(torch.int64).view(requests,limit)
        self._penalty_counts = self._penalty_slab.device[self._penalty_id_bytes:].view(torch.int32).view(rows,limit)

    def reserve_device(self, rows, vocab, *, arena=None):
        self.uniforms = torch.empty(rows,dtype=torch.float32,device="cuda")
        self.arena = arena
        self._workspace_bytes = sample_workspace_bytes(rows,vocab,min(vocab,ops.SAMPLING_MAX_K))
        self.workspace = (None if arena is not None else
                          torch.empty(self._workspace_bytes,dtype=torch.uint8,device='cuda'))

    @contextlib.contextmanager
    def _workspace_scope(self):
        arena = getattr(self,'arena',None)
        if arena is None:
            yield getattr(self,'workspace',None)
        else:
            with arena.frame():
                arena.at = max(arena.at,arena.tail_high)
                yield arena.workspace(self._workspace_bytes,'workspace:sampling')

    def _parameters(self, rows, vocab):
        ks = [r.params.kernel_top_k() for r in rows]
        top_k = self.inputs.tensor("top_k", ks, torch.int32)
        top_p = self.inputs.tensor("top_p", [r.params.top_p for r in rows], torch.float32)
        temp = self.inputs.tensor("temperature", [r.params.kernel_temperature() for r in rows],
                                  torch.float32)
        return top_k, top_p, temp, min(max(ks), vocab)

    def sample(self, logits, rows, *, out=None):
        if out is None and self.output.device is not None:
            out = self.output.device[:len(rows)]
        with span("sample"):
            self.inputs.begin()
            try:
                if any(r.params.penalized() for r in rows):
                    self._penalize(logits,rows,prepared=True)
                if all(r.params.temperature<=0 for r in rows):
                    return ops.argmax(logits,out=out)
                k, top_p, temp, width = self._parameters(rows, logits.shape[1])
                uniform = self._uniform(len(rows))
                with self._workspace_scope() as workspace:
                    return ops.sample(logits,k,top_p,temp,width,uniform=uniform,out=out,
                                      workspace=workspace)
            finally:
                self.inputs.used()

    def probabilities(self, logits, rows, out, indices):
        with self.inputs.reading(reuse=True):
            if any(r.params.penalized() for r in rows):
                self._penalize(logits, rows, prepared=True)
            top_k, top_p, temp, width = self._parameters(rows, logits.shape[1])
            n = len(rows) * width
            values, ids = out[:n].view(-1, width), indices[:n].view(-1, width)
            with self._workspace_scope() as workspace:
                probabilities(logits, top_k, top_p, temp, width, values, ids, workspace)
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
            self._penalize_rows(logits,rows)
        finally:
            if not prepared:
                self.inputs.used()

    def _penalty_write(self, values, code, offset):
        if len(values)>self._penalty_limit:
            raise SnowLLMError('penalty token capacity exceeded')
        data = array.array(code,values)
        if data:
            ctypes.memmove(self._penalty_slab.host.data_ptr()+offset,
                           data.buffer_info()[0],len(data)*data.itemsize)

    def _penalize_rows(self, logits, rows):
        if len(rows)>self._penalty_rows:
            raise SnowLLMError('penalty row capacity exceeded')
        groups = {}
        for i,r in enumerate(rows):
            if r.params.penalized():
                groups.setdefault(id(r),(r,[]))[1].append(i)
        if len(groups)>self._penalty_requests:
            raise SnowLLMError('penalty request capacity exceeded')
        for slot,(r,indices) in enumerate(groups.values()):
            if not hasattr(r,'_sample_prompt_tokens'):
                r._sample_prompt_tokens=set(r.prompt)
            counts=Counter(r.out)
            tokens=sorted(r._sample_prompt_tokens | counts.keys())
            seen=set(tokens)
            ends=[len(tokens)]
            # New draft IDs are appended in first-occurrence order. Each row's
            # prefix view excludes tokens appearing only in future draft rows.
            for token in r.drafts[:len(indices)-1]:
                if token not in seen:
                    tokens.append(token)
                    seen.add(token)
                ends.append(len(tokens))
            self._penalty_write(tokens,'q',slot*self._penalty_limit*8)
            p=r.params
            for depth,i in enumerate(indices):
                if depth and depth<=len(r.drafts):
                    counts[r.drafts[depth-1]]+=1
                n=ends[min(depth,len(ends)-1)]
                values=[counts.get(t,0) for t in tokens[:n]]
                self._penalty_write(values,'i',self._penalty_id_bytes+i*self._penalty_limit*4)
                penalties(logits[i],self._penalty_ids[slot,:n],self._penalty_counts[i,:n],
                          p.presence_penalty,p.frequency_penalty,p.repetition_penalty)

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

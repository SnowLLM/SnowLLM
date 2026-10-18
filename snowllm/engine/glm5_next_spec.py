# SPDX-License-Identifier: Apache-2.0
"""GLM NextN drafting with shared scheduling and state slots, and native GPU math."""
from dataclasses import replace

import torch

from .. import ops
from .._capi import SnowLLMError
from .forward_context import Batch
from .spec_decode import SpecDecoder


class Glm5NextSpecDecoder(SpecDecoder):
    def __init__(self,runner,sampler,slots,tables,stage,*,num_spec,window=0,sinks=64):
        if window:
            raise SnowLLMError("GLM MTP uses its learned sparse selection; mtp_window must be 0")
        super().__init__(runner,sampler,slots,tables,stage,num_spec=num_spec,window=window,sinks=sinks)
        self.accept_hist = []

    def _states(self, requests):
        canonical = self.messages.tensor("canonical",[r.slot for r in requests],torch.int32)
        scratch = self.messages.tensor("scratch",[r.slot+self.runner.max_num_seqs for r in requests],torch.int32)
        return canonical,scratch

    def propose_first(self,requests,pos,cu,total_q_blocks,k,tokens,accepted,seq):
        b,t = tokens.shape
        with self.messages.reading(reuse=True):
            canonical,self._scratch = self._states(requests)
            self._bt = self.tables([r.blocks for r in requests])
            ids,mapping,rows = self._first_metadata(tokens,accepted)
            mb = Batch(input_ids=ids,positions=pos,slot_mapping=mapping,block_tables=self._bt,
                       seq_lens=seq,state_indices=canonical,cu_seqlens=cu,is_prefill=False,
                       num_tokens=b*t,num_accepted=accepted,last_row=rows)
            logits,hidden = self.runner.mtp_draft(self.runner.last_hidden,ids,mb)
            ops.copy_rows(self.d_drafts[:b,0],self._draft_tokens(logits))
            if k>1:
                self._propose_rest_dev(b,hidden,k,accepted)

    def _draft_states(self, requests):
        return self._states(requests)[1]

    def _draft_source_offset(self, step):
        return -self.runner.max_num_seqs if step == 1 else 0

    def propose_after_prefill(self,r,b,lo,hi,M,sampled=None):
        # The checkpoint trains h[t] together with embedding(token[t+1]), with
        # the MTP cache still at position t. Intermediate chunks have the next
        # prompt token available; the final chunk uses the target's sampled token.
        nxt = r.tokens[lo+1:hi]+([sampled] if sampled is not None else [r.tokens[hi]])
        ids = self.runner.host_tensor("mtp_ids",nxt)
        mb = replace(b,input_ids=ids,state_indices=self.runner.host_tensor("mtp_slot",[r.slot],torch.int32),
                     chunk_decode=False,last_row=self.runner.host_tensor("mtp_last",[M-1]),
                     need_logits=sampled is not None)
        return self.runner.mtp_draft(self.runner.last_hidden,ids,mb)

    def after_decode(self,requests,starts):
        live = [(i,r) for i,r in enumerate(requests) if not r.done]
        if live:
            h = self.runner.gather_hidden(self.runner.last_hidden,[i for i,_ in live])
            n = len(live)
            with self.messages.reading(reuse=True):
                canonical,_ = self._states([r for _,r in live])
                self._bt = self.tables([r.blocks for _,r in live])
                ops.copy_rows(self.d_c[:n],self.messages.tensor("cached",[starts[i]-1 for i,_ in live]))
                ops.copy_rows(self.d_pd[:n],self.messages.tensor("delta",[r.pos_delta for _,r in live]))
                accepted = self.messages.tensor("accepted",[1]*n)
                pos,mapping,table,lens = self._step_metadata(n,accepted,1)
                ids = self.messages.tensor("last",[r.last for _,r in live])
                b = Batch(input_ids=ids,positions=pos,slot_mapping=mapping,block_tables=table,
                          seq_lens=lens,state_indices=canonical,is_prefill=False,num_tokens=n,need_logits=False)
                self.runner.mtp_draft(h,ids,b)

# SPDX-License-Identifier: Apache-2.0
import torch

from ... import ops
from ...checkpoint.gguf.dflash import DRAFT_FORMATS
from ...ops.allocation import empty
from ...ops.inkling import (candidate_groups, candidate_tiles, candidate_token, draft_gates, draft_mix, draft_scores, normalize_rows,
                           pack_search, packed_logits, sample_draft_step, search_logits, selected_logits)
from ..projection import pack_bf16, pack_kquant
from ..qwen3_5.dflash_draft import DFlashDraft


class InklingDraft(DFlashDraft):
    def context_feature(self, taps):
        return taps

    def recurrent_feature(self, taps):
        return draft_mix(taps, self.w["rnn_mix"], self.w["rnn_norm"], self.eps)

    def write_context(self, taps, positions, slots, *, local_slots=None):
        g, d = self.geo, ops.dflash
        rows = taps.size(0)
        cos = self.arena.new(rows, g.head_size//2, dtype=torch.float32)
        sin = self.arena.new(rows, g.head_size//2, dtype=torch.float32)
        d.rope_cos_sin(positions, self.inv_freq, cos, sin)
        with self.arena.frame():
            kv = self.arena.new(rows, 2*g.kv_dim)
            need = max((ops.gemm_kquant_a_ws_bytes if isinstance(w["ctx_kv"],ops.KQuantProjWeight)
                        else ops.gemm_bf16_a_ws_bytes)(rows,g.hidden) for w in self.w["layers"])
            scratch = self.arena.workspace(need)
            for layer,(w,(kc,vc)) in enumerate(zip(self.w["layers"], self.pools)):
                with self.arena.frame():
                    feat = draft_mix(taps, w["mix"], w["ctx_norm"], self.eps)
                    weight = w["ctx_kv"]
                    if isinstance(weight,ops.KQuantProjWeight):
                        size = ops.gemm_kquant_a_ws_bytes(rows,g.hidden)
                        ops.gemm_kquant_a(weight.fmt,feat,weight.quant,weight.meta,kv,
                                          rows,2*g.kv_dim,g.hidden,scratch[:size])
                    else:
                        size = ops.gemm_bf16_a_ws_bytes(rows,g.hidden)
                        ops.gemm_bf16_a(feat,weight,kv,rows,2*g.kv_dim,g.hidden,scratch[:size])
                    d.norm_rope_store(kv, None, w["ctx_k_norm"], cos, sin, None, kc, vc,
                                      local_slots if g.sliding(layer) and local_slots is not None else slots,
                                      self.eps, self.block_size, layer=0)

    def embed_block(self, model, anchor, noise):
        count, hidden = anchor.numel(), self.geo.hidden
        width = self.geo.block_size
        view = noise.view(count, width, hidden)
        embedded = ops.gather_rows(model.embedding, anchor)
        ops.copy_rows(view[:,0], normalize_rows(embedded, self.eps, self.w["embed_norm"]))
        for b in range(count):
            ops.copy_rows(view[b,1:], self.w["mask"])

    def propose_recurrent(self, model, hidden, anchor, context, out, initial_zeros, sampling=None, confidence=None):
        g = self.geo
        batch, n = out.shape
        hidden = hidden.view(batch, g.block_size, g.hidden)
        w = self.w
        vocab = w["fold_bias"].numel()
        search = (candidate_tiles(batch, n, model.output_columns, g.hidden, g.markov_rank)
                  if sampling is None and getattr(model, "search_head", None) is not None and "search_fold" in w else 0)
        if search:
            head = hidden[0, 1:n+1]
            rough = search_logits(head, model.search_head)
        else:
            head = hidden if n + 1 == g.block_size else hidden[:, 1:n+1]
            base = packed_logits(head.reshape(-1, g.hidden), model.output, model.output_columns)
            base = base.view(batch, head.size(1), -1)[:, -n:]
        state = empty(batch, g.markov_rank, dtype=torch.bfloat16, device=hidden.device)
        ops.copy_rows(state, initial_zeros.flatten()[:state.numel()].view_as(state))
        joint = empty(batch, 2*g.markov_rank + 2*g.hidden, device=hidden.device, dtype=torch.bfloat16)
        state_in, token_in, hidden_in, context_in = joint.split((g.markov_rank,g.markov_rank,g.hidden,g.hidden),dim=1)
        ops.copy_rows(context_in, context)
        prev = anchor
        bias = w["fold_bias"]
        for step in range(n):
            ops.copy_rows(token_in, w["w1"], src_index=prev)
            ops.copy_batch((state_in, hidden_in), (state, hidden[:,step+1]))
            proj = packed_logits(joint, *w["joint"])
            latent = draft_gates(proj, w["joint_bias"], state)
            if search:
                if step == 0:
                    tiles, ids, bias = candidate_groups(rough, search_logits(latent, w["search_fold"]),
                        w["search_bias"], w["tile_offsets"], bias, search)
                    base = selected_logits(head, model.output, model.output_columns, tiles, vocab).unsqueeze(0)
                fold = selected_logits(latent, *w["fold"], tiles, vocab)
            else:
                fold = packed_logits(latent, *w["fold"])
            scores = draft_scores(base[:,step], fold, bias, model.config["final_logit_softcapping"])
            if sampling is None:
                if search:
                    prev = candidate_token(scores, ids, confidence[:,step] if confidence is not None else None)
                else:
                    prev = ops.argmax(scores)
            else:
                prev = sample_draft_step(scores, sampling.temperature[:batch], sampling.uniform[:batch, step],
                                   sampling.slots[:batch], sampling.candidates[:, step], sampling.q[:, step])
            ops.copy_rows(out[:,step], prev)


def load_inkling(src, w, pools, geo, arena, block_size, candidate_search=False):
    rd = src.rd
    def weight(name, dtype=torch.bfloat16):
        return rd.tensor(name, dtype).cuda()
    def packed(name):
        shuffled, columns, _, _ = pack_bf16(weight(name))
        return shuffled, columns
    w.update(rnn_mix=weight("markov_ctx_mix.weight", torch.float32),
             rnn_norm=weight("markov_ctx_norm.weight", torch.float32),
             mask=weight("mask_embd.weight"), embed_norm=weight("token_embd_norm.weight", torch.float32),
             joint=packed("joint_proj.weight"), joint_bias=weight("joint_proj.bias", torch.float32),
             fold_bias=weight("markov_w2.bias",torch.float32),
             w1=weight("markov_w1.weight"))
    fold = weight("markov_w2.weight")
    shuffled, columns, _, _ = pack_bf16(fold)
    w["fold"] = shuffled, columns
    if candidate_search and candidate_tiles(1, 1, columns, geo.hidden, geo.markov_rank):
        w["search_fold"] = pack_search(fold, columns)
        w["search_bias"] = torch.nn.functional.pad(w["fold_bias"], (0, columns - w["fold_bias"].numel()), value=-float("inf"))
        w["tile_offsets"] = torch.arange(32, device=fold.device, dtype=torch.int64)
    del fold
    for i, layer in enumerate(w["layers"]):
        prefix=f"blk.{i}."
        names=[prefix+"attn_"+v+"_ctx.weight" for v in ("k","v")]
        formats = {DRAFT_FORMATS.get(rd.gguf[name].quant.name) for name in names}
        if len(formats) == 1 and None not in formats:
            ctx = pack_kquant(torch.cat([rd.raw(n) for n in names]), formats.pop(), 2*geo.kv_dim, geo.hidden)
        else:
            ctx = pack_bf16(torch.cat([weight(n) for n in names]))[0]
        layer.update(mix=weight(prefix+"ctx_mix.weight",torch.float32),
                     ctx_norm=weight(prefix+"ctx_norm.weight",torch.float32),
                     ctx_k_norm=weight(prefix+"attn_k_ctx_norm.weight"),
                     ctx_kv=ctx)
    return InklingDraft(w,pools,geo,arena,block_size)

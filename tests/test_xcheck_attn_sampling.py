# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Paged attention and sampling against torch.

The reference is `scaled_dot_product_attention`, `topk` and `cumsum` -- not a reference written
here. Attention masking is subtle enough that a reference and a kernel written from the same
reading of it would agree on a misreading.

So the causal mask is derived from the DEFINITION: query token i of a request sits at absolute
position `seq_len - S_b + i` in its sequence (the cache already holds `seq_len - S_b` earlier
tokens), and causal means it attends to every key at position <= its own.
"""

import sys

import torch
import torch.nn.functional as F

from snowllm import ops
from snowllm._capi import build_geometry, check as launched, lib

import _harness

CFG = build_geometry()
BS = CFG.block_size
Hq, Hk, D = CFG.num_heads, CFG.num_kv_heads, CFG.head_size
check = _harness.Checks(46)


def build_cache(seq_lens, max_blocks):
    """A paged pool with a SHUFFLED block table -- a kernel that ignored the indirection and read
    the pool linearly would still pass with an identity table."""
    B = len(seq_lens)
    num_blocks = B * max_blocks
    g = torch.Generator(device="cuda").manual_seed(4)
    k_cache = torch.randn(num_blocks, BS, Hk, D, generator=g, device="cuda").to(torch.bfloat16)
    v_cache = torch.randn(num_blocks, BS, Hk, D, generator=g, device="cuda").to(torch.bfloat16)
    phys = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    block_tables = phys.view(B, max_blocks).contiguous()
    # The pool is NOT stored in this natural [block, slot, head, d] order, and how it IS stored is
    # the library's business -- so the same values go in through the production writer, one whole
    # block per call, and the REFERENCE keeps the natural-layout tensors. Agreement then means the
    # kernel read both the block-table indirection and its own layout correctly, rather than that
    # both sides were handed the same rearrangement.
    k_kernel, v_kernel = (ops.empty_bytes(n) for n in ops.kv_pool_bytes(num_blocks, False))
    row = Hk * D
    ident = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(num_blocks, 1)
    seq = torch.arange(num_blocks, dtype=torch.int32, device="cuda").repeat_interleave(BS)
    pos = torch.arange(BS, dtype=torch.int32, device="cuda").repeat(num_blocks)
    ops.reshape_and_cache(k_cache.reshape(-1, row), v_cache.reshape(-1, row), k_kernel, v_kernel,
                          ops.resolve_slots(ident, seq, pos), row, row)
    return k_cache, k_kernel, v_cache, v_kernel, block_tables


def gather_kv(k_cache, v_cache, block_tables, b, seq_len):
    """The request's keys/values in logical order, straight out of the block table."""
    ks, vs = [], []
    for j in range((seq_len + BS - 1) // BS):
        p = int(block_tables[b, j])
        ks.append(k_cache[p])
        vs.append(v_cache[p])
    return torch.cat(ks)[:seq_len], torch.cat(vs)[:seq_len]  # [T, Hk, D]


def sdpa_ref(q, k, v, q_len):
    """torch's SDPA with the mask written from the definition of causal attention.
    q [S, Hq, D], k/v [T, Hk, D] -> [S, Hq, D] fp32."""
    T = k.shape[0]
    rep = Hq // Hk
    kk = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)  # [Hq, T, D]
    vv = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    qq = q.float().permute(1, 0, 2)  # [Hq, S, D]

    pos = torch.arange(T - q_len, T, device=q.device).unsqueeze(1)  # each query's absolute position
    key = torch.arange(T, device=q.device).unsqueeze(0)
    mask = key <= pos  # causal: attend to every key at or before your own position
    out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=D ** -0.5)
    return out.permute(1, 0, 2)  # [S, Hq, D]


def test_prefill():
    # Mixed: a fresh request, one continuing from a long context (S << seq_len -- the case where a
    # bottom-right-aligned mask differs from a plain lower-triangular one), and a ragged tail.
    cases = [(48, 48), (7, 500), (129, 129), (1, 33), (64, 200)]  # (S_b, seq_len_b)
    S = [c[0] for c in cases]
    seq = [c[1] for c in cases]
    B = len(cases)
    max_blocks = max((t + BS - 1) // BS for t in seq)
    print(f"=== paged prefill vs torch SDPA, B={B}, S={S}, seq_lens={seq} ===")

    k_cache, k_kernel, v_cache, v_kernel, bt = build_cache(seq, max_blocks)
    M = sum(S)
    torch.manual_seed(9)
    q = (torch.randn(M, Hq, D, device="cuda") * 0.5).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(S).cumsum(0)), dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq, dtype=torch.int32, device="cuda")

    total_q_blocks, qmap = ops.prefill_q_plan(list(S))
    out = torch.empty(M, Hq, D, dtype=torch.bfloat16, device="cuda")
    ops.paged_attn_prefill(q, k_kernel, v_kernel, out, cu, bt, seq_lens, total_q_blocks, D ** -0.5,
                           qmap)
    ops.synchronize()

    off = 0
    for b in range(B):
        k, v = gather_kv(k_cache, v_cache, bt, b, seq[b])
        want = sdpa_ref(q[off:off + S[b]], k, v, S[b])
        check.close(f"req {b}: S={S[b]:<4} seq_len={seq[b]:<4} "
                    f"(context={seq[b] - S[b]})", out[off:off + S[b]], want, 0.02)
        off += S[b]


def test_decode():
    seq = [1, 16, 17, 300, 4096]
    B = len(seq)
    max_blocks = max((t + BS - 1) // BS for t in seq)
    print(f"\n=== paged decode vs torch SDPA, B={B}, seq_lens={seq} ===")
    k_cache, k_kernel, v_cache, v_kernel, bt = build_cache(seq, max_blocks)
    torch.manual_seed(10)
    q = (torch.randn(B, Hq, D, device="cuda") * 0.5).to(torch.bfloat16)
    seq_lens = torch.tensor(seq, dtype=torch.int32, device="cuda")

    nslots = ops.paged_decode_num_slots(B)
    ws = ops.empty_bytes(ops.paged_decode_workspace_size(nslots))
    plan = torch.zeros(ops.paged_decode_plan_elems(B, nslots), dtype=torch.int32, device="cuda")
    out = torch.empty(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    ops.paged_attn_decode_plan(seq_lens, plan, nslots)
    ops.paged_attn_decode(q, seq_lens, k_kernel, v_kernel, out, bt, plan, ws, B, nslots, D ** -0.5)
    ops.synchronize()

    for b in range(B):
        k, v = gather_kv(k_cache, v_cache, bt, b, seq[b])
        want = sdpa_ref(q[b:b + 1], k, v, 1)[0]  # decode: one query, sees everything
        # Max abs err rather than rel L2: one decode row's norm is small enough that a rel score
        # is dominated by whichever element happened to be near zero.
        e = (out[b].float() - want).abs().max().item()
        check(f"req {b}: seq_len={seq[b]}", e < 0.05, f"max abs err {e:.5f}")


def test_decode_verify():
    """T query rows per request -- the speculative verify step. Each row must see the earlier rows
    of its OWN step (they are already in the cache) and nothing after itself, so an off-by-one in
    the tail mask shows up here and nowhere else. seq_lens counts all T rows."""
    seq = [17, 64, 300, 4096]
    B = len(seq)
    max_blocks = max((t + BS - 1) // BS for t in seq)
    for T in range(2, ops.PAGED_DECODE_MAX_Q_TOKENS + 1):
        print(f"\n=== paged decode verify T={T} vs torch SDPA, B={B}, seq_lens={seq} ===")
        k_cache, k_kernel, v_cache, v_kernel, bt = build_cache(seq, max_blocks)
        torch.manual_seed(11 + T)
        q = (torch.randn(B * T, Hq, D, device="cuda") * 0.5).to(torch.bfloat16)
        seq_lens = torch.tensor(seq, dtype=torch.int32, device="cuda")

        nslots = ops.paged_decode_num_slots(B)
        ws = ops.empty_bytes(ops.paged_decode_workspace_size(nslots, T))
        plan = torch.zeros(ops.paged_decode_plan_elems(B, nslots), dtype=torch.int32, device="cuda")
        out = torch.empty(B * T, Hq, D, dtype=torch.bfloat16, device="cuda")
        ops.paged_attn_decode_plan(seq_lens, plan, nslots)
        ops.paged_attn_decode(q, seq_lens, k_kernel, v_kernel, out, bt, plan, ws, B, nslots,
                              D ** -0.5, T)
        ops.synchronize()

        for b in range(B):
            k, v = gather_kv(k_cache, v_cache, bt, b, seq[b])
            want = sdpa_ref(q[b * T:(b + 1) * T], k, v, T)
            e = (out[b * T:(b + 1) * T].float() - want).abs().max().item()
            check(f"req {b}: seq_len={seq[b]}", e < 0.05, f"max abs err {e:.5f}")


def torch_filter(probs_row, k, top_p):
    """torch's own top-k, then the filter from its stated semantics: renormalize the k survivors,
    zero everything past cumulative mass top_p, renormalize again. -> (probs[k], ids[k])."""
    w_p, w_i = torch.topk(probs_row, k, dim=-1)
    w_p = w_p / w_p.sum(-1, keepdim=True)
    cum = w_p.cumsum(-1)
    keep = torch.ones_like(w_p, dtype=torch.bool)
    keep[..., 1:] = cum[..., :-1] < top_p  # slot 0 always survives, whatever top_p is
    w_p = torch.where(keep, w_p, torch.zeros_like(w_p))
    return w_p / w_p.sum(-1, keepdim=True), w_i


def test_sampling():
    print("\n=== sampling vs torch (uniform batch) ===")
    B, V, k = 8, CFG.vocab_size, 64
    torch.manual_seed(12)
    logits = (torch.randn(B, V, device="cuda") * 3.0).float()
    temp, top_p = 0.8, 0.9

    s = torch.cuda.current_stream().cuda_stream
    d_temp = torch.full((B,), temp, dtype=torch.float32, device="cuda")
    d_topk = torch.full((B,), k, dtype=torch.int32, device="cuda")
    d_topp = torch.full((B,), top_p, dtype=torch.float32, device="cuda")

    probs = torch.empty(B, V, dtype=torch.float32, device="cuda")
    launched(lib.snowllm_sampling_softmax(logits.data_ptr(), probs.data_ptr(), B, V,
                                          d_temp.data_ptr(), s), "softmax")
    ops.synchronize()
    want_probs = torch.softmax(logits / temp, dim=-1)
    check.close("softmax(logits/T)", probs, want_probs, 1e-5)

    top_probs = torch.empty(B, k, dtype=torch.float32, device="cuda")
    top_idx = torch.empty(B, k, dtype=torch.int64, device="cuda")
    launched(lib.snowllm_sampling_topk_topp(probs.data_ptr(), top_probs.data_ptr(),
                                            top_idx.data_ptr(), B, V, k, d_topk.data_ptr(),
                                            d_topp.data_ptr(), s), "topk_topp")
    ops.synchronize()
    w_p, w_i = torch_filter(want_probs, k, top_p)
    check.exact("top-k indices == torch.topk", top_idx, w_i)
    check.close("top-p filtered probs", top_probs, w_p, 1e-4)

    # multinomial: same uniforms into both, so the draw is a function, not a distribution.
    uni = torch.rand(B, dtype=torch.float32, device="cuda")
    tok = torch.empty(B, dtype=torch.int64, device="cuda")
    launched(lib.snowllm_sampling_multinomial(top_probs.data_ptr(), top_idx.data_ptr(),
                                              uni.data_ptr(), tok.data_ptr(), B, k, s),
             "multinomial")
    ops.synchronize()
    slot = torch.searchsorted(w_p.cumsum(-1).contiguous(), uni.unsqueeze(1)).clamp_(max=k - 1)
    want_tok = w_i.gather(1, slot).squeeze(1)
    check("multinomial (inverse CDF)", torch.equal(tok, want_tok),
          f"{(tok != want_tok).sum().item()}/{B} rows differ")

    launched(lib.snowllm_sampling_argmax(logits.data_ptr(), tok.data_ptr(), B, V, s), "argmax")
    ops.synchronize()
    check.exact("argmax == torch.argmax", tok, logits.argmax(-1))


def test_sampling_per_row():
    """The whole point of the per-row parameters: a batch where every row wants something DIFFERENT.
    A kernel that quietly used row 0's parameters for everyone, or the batch-wide width k for every
    row's cut, passes every uniform-batch test above and fails here.

    Row 0 is greedy expressed as top_k=1, which must come out EXACTLY equal to torch.argmax."""
    print("\n=== sampling vs torch (per-row parameters) ===")
    s = torch.cuda.current_stream().cuda_stream

    rows = [  # (temperature, top_k, top_p)
        (1.0, 1, 1.0),     # greedy: top_k=1 IS argmax
        (0.7, 20, 0.95),   # the checkpoint's generation_config
        (1.3, 64, 0.8),
        (0.5, 5, 1.0),     # no top-p cut
        (1.0, 50, 0.3),    # aggressive nucleus
    ]
    B, V = len(rows), CFG.vocab_size
    k = max(r[1] for r in rows)  # the output WIDTH; every row cuts to its own top_k inside it
    torch.manual_seed(21)
    logits = (torch.randn(B, V, device="cuda") * 3.0).float()

    d_temp = torch.tensor([r[0] for r in rows], dtype=torch.float32, device="cuda")
    d_topk = torch.tensor([r[1] for r in rows], dtype=torch.int32, device="cuda")
    d_topp = torch.tensor([r[2] for r in rows], dtype=torch.float32, device="cuda")

    probs = torch.empty(B, V, dtype=torch.float32, device="cuda")
    top_probs = torch.empty(B, k, dtype=torch.float32, device="cuda")
    top_idx = torch.empty(B, k, dtype=torch.int64, device="cuda")
    tok = torch.empty(B, dtype=torch.int64, device="cuda")
    uni = torch.rand(B, dtype=torch.float32, device="cuda")
    launched(lib.snowllm_sampling_softmax(logits.data_ptr(), probs.data_ptr(), B, V,
                                          d_temp.data_ptr(), s), "softmax")
    launched(lib.snowllm_sampling_topk_topp(probs.data_ptr(), top_probs.data_ptr(),
                                            top_idx.data_ptr(), B, V, k, d_topk.data_ptr(),
                                            d_topp.data_ptr(), s), "topk_topp")
    launched(lib.snowllm_sampling_multinomial(top_probs.data_ptr(), top_idx.data_ptr(),
                                              uni.data_ptr(), tok.data_ptr(), B, k, s),
             "multinomial")
    ops.synchronize()

    for b, (t, kk, pp) in enumerate(rows):
        want_probs = torch.softmax(logits[b:b + 1] / t, dim=-1)
        w_p, w_i = torch_filter(want_probs, kk, pp)  # torch cuts at THIS row's own k

        e_probs = _harness.rel(top_probs[b, :kk], w_p[0])
        same_idx = torch.equal(top_idx[b, :kk], w_i[0])
        pads_zero = bool((top_probs[b, kk:] == 0).all())  # slots past this row's k must be dead

        slot = torch.searchsorted(w_p.cumsum(-1).contiguous(), uni[b].view(1, 1)).clamp_(max=kk - 1)
        same_tok = int(tok[b]) == int(w_i.gather(1, slot).squeeze())

        check(f"T={t:<4} k={kk:<3} p={pp}", same_idx and pads_zero and e_probs < 1e-4 and same_tok,
              f"idx {'EXACT' if same_idx else 'DIFFER'}  probs {e_probs:.1e}  "
              f"pads {'0' if pads_zero else 'DIRTY'}  tok {'EXACT' if same_tok else 'DIFFER'}")

    check("row 0 (top_k=1) == torch.argmax", int(tok[0]) == int(logits[0].argmax()))


def main():
    test_prefill()
    test_decode()
    test_decode_verify()
    test_sampling()
    test_sampling_per_row()
    return check.done()


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0

import functools

import torch

from ._common import _chk, _p, _stream, check, lib, SnowLLMError


def hc_init(mix, pre=None):
    _chk(mix, "V4.1 HC initial mix", torch.float32, mix.shape[0], 24)
    if pre is not None:
        from ._common import _chk_rows
        _chk_rows(pre, "V4.1 saved HC mix", torch.float32)
        if pre.shape != (mix.shape[0], 4):
            raise SnowLLMError("saved HC mix must have four columns per token")
    check(lib.snowllm_dsv41_hc_init(_p(mix), _p(pre), mix.shape[0],
          0 if pre is None else pre.stride(0), _stream()), "dsv41_hc_init")


@functools.lru_cache(None)
def attention_scratch_bytes(rows, compressed):
    query = getattr(lib, "snowllm_dsv41_attention_scratch_bytes", None)
    return query(rows, int(compressed)) if query is not None else 0


def attention(q, raw, table, positions, seq, sinks, out, window, *, compressed=None,
              comp_table=None, selection=None, counts=None, start=0, reference=False,
              scratch=None):
    rows = q.shape[0]
    _chk(q, "V4.1 attention Q", torch.bfloat16, rows, 64, 512)
    _chk(raw, "V4.1 raw KV", torch.bfloat16)
    _chk(out, "V4.1 attention output", torch.bfloat16, rows, 64, 512)
    _chk(table, "V4.1 raw table", torch.int32)
    _chk(positions, "V4.1 attention positions", torch.int64, rows)
    _chk(seq, "V4.1 attention requests", torch.int32, rows)
    _chk(sinks, "V4.1 attention sinks", torch.float32, 64)
    if not 0 < window <= 128 or start < 0:
        raise SnowLLMError("V4.1 attention requires window 1..128 and a nonnegative replay start")
    if compressed is not None:
        _chk(compressed, "V4.1 compressed KV", torch.uint8, compressed.shape[0], 288)
        _chk(comp_table, "V4.1 compressed table", torch.int32)
        _chk(selection, "V4.1 attention selection", torch.int32, rows, selection.shape[1])
        _chk(counts, "V4.1 attention counts", torch.int32, rows)
        if selection.shape[1] > 512:
            raise SnowLLMError("V4.1 attention supports at most 512 selected compressed tokens")
    name = "snowllm_dsv41_attention_reference" if reference else "snowllm_dsv41_attention"
    function = getattr(lib, name, None)
    if function is None:
        raise SnowLLMError(f"the kernel library has no {name}; rebuild it")
    extra = []
    if scratch is not None and not reference:
        _chk(scratch, "V4.1 attention scratch", torch.uint8)
        need = attention_scratch_bytes(rows, compressed is not None)
        if scratch.nbytes < need:
            raise SnowLLMError(f"V4.1 attention needs {need} scratch bytes")
        function = lib.snowllm_dsv41_attention_split
        extra = [_p(scratch), scratch.nbytes]
    check(function(*[_p(t) for t in
        (q, raw, compressed, table, comp_table, positions, seq, selection, counts, sinks, out)],
        rows, table.shape[1], 0 if comp_table is None else comp_table.shape[1],
        0 if selection is None else selection.shape[1], window, start, *extra, _stream()), "dsv41_attention")


def hc_mixes(x, weight, out, eps):
    rows, hc, hidden = x.shape
    _chk(x, "V4.1 HC residual", torch.bfloat16, rows, 4, hidden)
    _chk(weight, "V4.1 HC projection", torch.float32, 24, hc * hidden)
    _chk(out, "V4.1 HC mixes", torch.float32, rows, 24)
    check(lib.snowllm_dsv41_hc_mixes(_p(x), _p(weight), _p(out), rows, hc * hidden,
        eps, _stream()), "dsv41_hc_mixes")


def fp8_quant(x):
    _chk(x, "V4.1 FP8 activation", torch.bfloat16)
    if x.numel() % 32:
        raise SnowLLMError("V4.1 FP8 activations require groups of 32")
    check(lib.snowllm_dsv41_fp8_quant(_p(x), x.numel(), _stream()), "dsv41_fp8_quant")


def index_scores(q, key, table, weights, positions, seq, lengths, out, ratio,
                 candidates=None, candidate_block=8):
    rows = q.shape[0]
    _chk(q, "V4.1 index Q", torch.bfloat16, rows, 32, 128)
    _chk(key, "V4.1 index K", torch.uint8, key.shape[0], 68)
    _chk(weights, "V4.1 index weights", torch.float32, rows, 32)
    _chk(table, "V4.1 index table", torch.int32)
    _chk(positions, "V4.1 positions", torch.int64, rows)
    _chk(seq, "V4.1 request index", torch.int32, rows)
    _chk(lengths, "V4.1 compressed lengths", torch.int32)
    _chk(out, "V4.1 index scores", torch.float32, rows, out.shape[1])
    if ratio <= 0 or candidate_block <= 0:
        raise SnowLLMError("V4.1 index ratios and candidate blocks must be positive")
    if candidates is not None:
        _chk(candidates, "V4.1 candidate mask", torch.int8, rows, candidates.shape[1])
        if candidates.shape[1] * candidate_block < out.shape[1]:
            raise SnowLLMError("V4.1 candidate mask is shorter than the score axis")
    check(lib.snowllm_dsv41_index_scores(*[_p(t) for t in
        (q, key, table, weights, positions, seq, lengths, candidates, out)],
        rows, out.shape[1], table.shape[1], ratio,
        0 if candidates is None else candidates.shape[1], candidate_block, _stream()), "dsv41_index_scores")


def candidate_scores(scores, out, positions, lengths, seq, block_positions, block_lengths, block):
    _chk(scores, "V4.1 candidate input", torch.float32)
    _chk(out, "V4.1 candidate scores", torch.float32)
    _chk(positions, "V4.1 candidate positions", torch.int64, scores.shape[0])
    _chk(seq, "V4.1 candidate requests", torch.int32, scores.shape[0])
    _chk(lengths, "V4.1 candidate lengths", torch.int32)
    _chk(block_positions, "V4.1 block positions", torch.int64, scores.shape[0])
    _chk(block_lengths, "V4.1 block lengths", torch.int32, *lengths.shape)
    if block <= 0 or out.shape[0] != scores.shape[0]:
        raise SnowLLMError("invalid V4.1 candidate dimensions")
    check(lib.snowllm_dsv41_candidate_scores(*[_p(t) for t in
        (scores, out, positions, lengths, seq, block_positions, block_lengths)],
        scores.shape[0], scores.shape[1], out.shape[1], block, _stream()), "dsv41_candidate_scores")


def engram_unpack(raw, out):
    _chk(raw, "Engram row bytes", torch.uint8)
    _chk(out, "Engram embeddings", torch.bfloat16)
    if raw.numel() % 264 or out.numel() != raw.numel() // 264 * 256:
        raise SnowLLMError("Engram rows require 256 E4M3 weights and eight E8M0 scales")
    check(lib.snowllm_dsv41_engram_unpack(_p(raw), _p(out), raw.numel() // 264, _stream()), "dsv41_engram_unpack")


def engram_gate(x, kv, weight, out, eps):
    rows, hc, hidden = x.shape
    _chk(x, "Engram residual", torch.bfloat16, rows, 4, hidden)
    _chk(kv, "Engram projection", torch.bfloat16, rows, 5 * hidden)
    _chk(weight, "Engram gate weights", torch.float32, 4, hidden)
    _chk(out, "Engram output", torch.bfloat16, rows, hc, hidden)
    check(lib.snowllm_dsv41_engram_gate(_p(x), _p(kv), _p(weight), _p(out),
        rows, hidden, eps, _stream()), "dsv41_engram_gate")

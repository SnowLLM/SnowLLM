# SPDX-License-Identifier: Apache-2.0

import sys

import torch
import torch.nn.functional as F

import _harness
from snowllm import ops
from snowllm._capi import SnowLLMError


def norm(x, gamma, beta, eps=1e-6):
    return F.layer_norm(x.float(), (x.shape[-1],), gamma.float(), beta.float(), eps).bfloat16()


def window_reference(qkv):
    q, k, v = qkv.view(-1, 4, 3, 16, 72).float().unbind(2)
    score = torch.einsum("bqhd,bkhd->bhqk", q, k) * 72 ** -.5
    return torch.einsum("bhqk,bkhd->bqhd", score.softmax(-1), v).flatten(0, 1).flatten(1).bfloat16()


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(4729)
    check = _harness.Checks(68)
    device, dtype = "cuda", torch.bfloat16
    for rows in (1, 7, 65):
        gamma = torch.randn(4608, device=device, dtype=dtype)
        beta = torch.randn_like(gamma)
        for offset in (0., 1024.):
            x = (torch.randn(rows, 4608, device=device) * 16 + offset).bfloat16()
            actual = torch.empty_like(x)
            expected = norm(x, gamma, beta)
            ops.vision_layernorm(x, gamma, beta, actual, 1e-6)
            check.close(f"4608 LayerNorm rows={rows} offset={offset}", actual, expected, .001)
            ops.vision_layernorm(x, gamma, beta, x, 1e-6)
            check.exact("4608 LayerNorm supports in-place output", x, actual)
    for rows in (1, 17, 129):
        x = torch.randn(rows*4, 1152, device=device, dtype=dtype)
        gamma = torch.randn(4608, device=device, dtype=dtype)
        beta = torch.randn_like(gamma)
        index = torch.randperm(rows*4, device=device)
        out = torch.empty(rows, 4608, device=device, dtype=dtype)
        mean = torch.empty(rows, 1152, device=device, dtype=dtype)
        for gather in (False, True):
            ordered = (x[index] if gather else x).view(rows, 4, 1152)
            expected = norm(ordered.flatten(1), gamma, beta)
            expected_mean = ordered.float().mean(1).bfloat16()
            ops.vision_merge_norm(x, gamma, beta, out, 1e-6,
                                  indices=index if gather else None, mean=mean)
            check.close(f"fused merge norm rows={rows} gather={gather}", out, expected, .001)
            check.exact("fused four-row mean", mean, expected_mean)
            ops.vision_merge_norm(x, gamma, beta, out, 1e-6,
                                  indices=index if gather else None)
            check.close("merge normalization without mean output", out, expected, .001)
    table = torch.randn(4900, 1152, device=device, dtype=dtype)
    ids = torch.randint(0, 4900, (37,), device=device)
    hidden = torch.randn(37, 1152, device=device, dtype=dtype)
    expected = hidden + table[ids]
    ops.vision_pos_embed_add(hidden, table, ids.int())
    check.exact("position lookup + addition rounds once to BF16", hidden, expected)
    for windows in (1, 3, 256):
        qkv = torch.randn(windows*4, 3456, device=device, dtype=dtype)
        actual = torch.empty(windows*4, 1152, device=device, dtype=dtype)
        ops.vision_attention(qkv, None, None, None, actual, 16, 72)
        check.close(f"four-token attention windows={windows}", actual, window_reference(qkv), .001)
        alone = torch.empty(4, 1152, device=device, dtype=dtype)
        ops.vision_attention(qkv[:4], None, None, None, alone, 16, 72)
        check.exact("attention isolates each window", actual[:4], alone)
    lengths = (3, 19, 7)
    m = sum(lengths)
    qkv = torch.randn(m, 3456, device=device, dtype=dtype)
    original = qkv.clone()
    actual = torch.empty(m, 1152, device=device, dtype=dtype)
    cu = torch.tensor([0, 3, 22, 29], device=device, dtype=torch.int32)
    ops.vision_attention(qkv, None, None, cu, actual, 16, 72)
    expected, start = [], 0
    for n in lengths:
        q, k, v = qkv[start:start+n].view(n, 3, 16, 72).float().unbind(1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * 72 ** -.5
        expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).flatten(1))
        start += n
    check.close("ragged attention without RoPE", actual, torch.cat(expected), .006)
    check.exact("no-RoPE attention leaves QKV unchanged", qkv, original)
    ops.vision_merge_norm(x, gamma, beta, out, 1e-6, indices=index, mean=mean)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.vision_merge_norm(x, gamma, beta, out, 1e-6, indices=index, mean=mean)
    out.zero_()
    graph.replay()
    check.exact("fused merge captures and replays in HIP graph", out, eager)
    empty = torch.empty(0, 1152, device=device, dtype=dtype)
    ops.vision_merge_norm(empty, gamma, beta, empty.view(0, 4608), 1e-6)
    ops.vision_attention(empty.view(0, 3456), None, None, None, empty, 16, 72)
    ops.vision_pos_embed_add(empty, table, ids[:0].int())
    ops.vision_attention(empty.view(0, 3456), None, None, cu[:1], empty, 16, 72)
    check("empty inputs require no scratch or kernel launch", True)
    for work in (
        lambda: ops.vision_attention(qkv, None, None, None, actual, 16, 72),
        lambda: ops.vision_merge_norm(qkv, gamma, beta, out, 1e-6),
        lambda: ops.vision_attention(qkv, None, actual, cu, actual, 16, 72),
    ):
        try:
            work()
        except SnowLLMError:
            check("malformed shapes fail before launching a kernel", True)
        else:
            check("malformed shapes fail before launching a kernel", False)
    return check.done()


if __name__ == "__main__":
    sys.exit(main())

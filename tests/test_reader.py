# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import torch

import _harness

CKPT = _harness.checkpoint()

from safetensors import safe_open  # noqa: E402

from snowllm.checkpoint.reader import Reader  # noqa: E402


def main() -> int:
    r = Reader(CKPT)
    keys = list(r.keys())
    ok = True

    shard0 = r._shard(keys[0])
    in_shard = [k for k in keys if r.weight_map[k] == pathlib.Path(shard0.path).name]
    by_off = sorted(in_shard, key=lambda k: shard0.spec(k)[2])
    unaligned = [k for k in by_off if shard0.spec(k)[2] % 4096 != 0]
    picks = list(dict.fromkeys(
        [by_off[0], by_off[-1]] + unaligned[:3]
        + ["model.language_model.layers.0.mlp.experts.gate_up_proj",
           "model.language_model.layers.3.self_attn.q_proj.weight",
           "lm_head.weight"]))

    print("=== bit-exactness vs safetensors ===")
    for k in picks:
        s = r._shard(k)
        _, shape, offset, nbytes = s.spec(k)
        got = r.to_device(k)
        torch.cuda.synchronize()
        with safe_open(s.path, framework="pt") as h:
            want = h.get_tensor(k).cuda()
        same = torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        ok &= same
        print(f"  {k[-52:]:<52} {nbytes / 1e6:8.1f} MB  off%4096={offset % 4096:<5} "
              f"{'EXACT' if same else 'MISMATCH'}")
        del got, want

    print("\n=== slot recycling: tiny chunks, minimum slot headroom ===")
    big = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    with safe_open(r._shard(big).path, framework="pt") as h:
        want = h.get_tensor(big).cuda()
    for chunk_kib, depth, nbuf in ((64, 8, 9), (4, 4, 5), (1024, 16, 17)):
        rr = Reader(CKPT, queue_depth=depth, chunk_bytes=chunk_kib << 10, num_buffers=nbuf)
        got = rr.to_device(big)
        torch.cuda.synchronize()
        nch = (r._shard(big).spec(big)[3] + (chunk_kib << 10) - 1) // (chunk_kib << 10)
        same = torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        ok &= same
        print(f"  chunk={chunk_kib:>4} KiB  depth={depth:>2}  buffers={nbuf:>2}  "
              f"{nch:>6} chunks -> {'EXACT' if same else 'CORRUPT'}")
        rr.close()
        del got
    del want

    print("\n=== read_shard() views ===")
    name = pathlib.Path(shard0.path).name
    buf, views = r.read_shard(name)
    torch.cuda.synchronize()
    bad = 0
    for k in in_shard:
        one = r.to_device(k)
        torch.cuda.synchronize()
        if not torch.equal(views[k].reshape(-1).view(torch.uint8),
                           one.reshape(-1).view(torch.uint8)):
            bad += 1
        del one
    ok &= bad == 0
    print(f"  {len(in_shard)} views vs per-tensor reads: {len(in_shard) - bad} exact, {bad} wrong")
    del buf, views

    r.close()
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

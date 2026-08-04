# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The io_uring reader against safetensors, bit-exactly, then its throughput.

The reader reads the 4096-ALIGNED SUPERSET of a tensor's byte range (O_DIRECT has no choice) and
copies only the middle out. That arithmetic is where an off-by-one would live, and it would not
crash -- it would shift the bytes. So every tensor checked here is compared bit-for-bit against
safetensors' own read, and the set is chosen to hit the edges: the first tensor in a shard, the
last, and ones whose offsets are not multiples of 4096.

Needs the real checkpoint; skips (exit 0) without it.
"""

import pathlib
import sys
import time

import torch

import _harness

CKPT = _harness.checkpoint()

from safetensors import safe_open  # noqa: E402

from snowllm.reader import Reader  # noqa: E402


def main():
    r = Reader(CKPT)
    keys = list(r.keys())
    ok = True

    # Pick tensors that stress the alignment math: unaligned offsets, plus a shard's first and last.
    shard0 = r._shard(keys[0])
    in_shard = [k for k in keys if r.weight_map[k] == pathlib.Path(shard0.path).name]
    by_off = sorted(in_shard, key=lambda k: shard0.spec(k)[2])
    unaligned = [k for k in by_off if shard0.spec(k)[2] % 4096 != 0]
    picks = list(dict.fromkeys(
        [by_off[0], by_off[-1]] + unaligned[:3]
        + ["model.language_model.layers.0.mlp.experts.gate_up_proj",  # 1.07 GB, many chunks
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

    # Slot recycling under pressure. A staging slot is busy from submit until its read is reaped AND
    # its H2D drains; deriving the slot from a chunk COUNTER instead is unsound, because io_uring
    # completions are unordered and one stalled chunk lets the submit index run a whole num_buffers
    # ahead of it. A 64 KiB chunk over a 1 GB tensor is ~16k chunks recycled through 9 slots --
    # the configuration that would expose it.
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

    # read_shard()'s zero-copy views must be the same bytes as a per-tensor read.
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

    # Cold read of the WHOLE checkpoint to GPU. O_DIRECT never populates the page cache, so this is
    # cold by construction -- no fadvise needed, and none of it can be a warm re-read.
    print("\n=== whole checkpoint, SSD -> GPU (no shuffling) ===")
    torch.cuda.synchronize()
    t0 = time.time()
    total = 0
    for name in r.shards():
        buf, _ = r.read_shard(name)
        total += buf.numel()
        del buf  # the shuffling pass is a later stage; here we only prove the bytes land
    torch.cuda.synchronize()
    dt = time.time() - t0
    gbps = total / 1e9 / dt
    print(f"  {total / 1e9:.1f} GB in {dt:.1f} s  ->  {gbps:.2f} GB/s")

    r.close()
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

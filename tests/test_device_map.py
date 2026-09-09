# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import gc
import os
import pathlib
import sys
from collections.abc import Callable

import torch

from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.models import placement as P

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))
GENERIC_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_GENERIC_DIR",
                   pathlib.Path.home() / "models/Qwen3.6-35B-A3B-UD-Q4_K_XL"))
LAYERS = 2
GIB = 1 << 30
MIB = 1 << 20


def shmem() -> int:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("Shmem:"):
            return int(line.split()[1]) << 10
    return 0


def parts(w: object) -> list[torch.Tensor]:
    return [t for t in (getattr(w, n, None) for n in ("p0", "p1", "p2", "quant", "meta"))
            if t is not None]


def routed(model: object) -> list[torch.Tensor]:
    return [t for lyr in model.layers for t in parts(lyr.moe.gate_up) + parts(lyr.moe.down)]


def resident(model: object) -> list[torch.Tensor]:
    out = [t for lyr in model.layers
           for t in parts(lyr.moe.shared_gate_up) + parts(lyr.moe.shared_down)]
    return out + [lyr.attn.fanout.gamma for lyr in model.layers]


def only_loaded(real: Callable[[object], dict]) -> Callable[[object], dict]:
    return lambda g: {k: {i: v for i, v in by.items() if i < LAYERS} for k, by in real(g).items()}


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        return 0
    ck = _harness.Checks(64)

    torch.zeros(1, device="cuda")
    free0 = torch.cuda.mem_get_info()[0]
    plain = loader.load(MODEL_DIR, layers=range(LAYERS))
    took_plain = free0 - torch.cuda.mem_get_info()[0]
    want = [t.cpu() for t in routed(plain)]
    on_device = not any(ops.host_mapped(t) for t in routed(plain))
    del plain
    gc.collect()
    torch.cuda.empty_cache()

    P.layer_bytes = only_loaded(P.layer_bytes)
    s0 = shmem()
    free0 = torch.cuda.mem_get_info()[0]
    mapped = loader.load(MODEL_DIR, layers=range(LAYERS), device_map=f"experts:{LAYERS}")
    took_mapped = free0 - torch.cuda.mem_get_info()[0]
    grew = shmem() - s0
    kept = took_plain - took_mapped
    got = routed(mapped)
    asked = sum(t.nbytes for t in got)

    print("\n=== the named group is the group that moved ===")
    ck("without a map every routed expert buffer is in the carve-out", on_device)
    ck(f"with experts:{LAYERS} every one of them is in host pages",
       all(ops.host_mapped(t) for t in got), f"{len(got)} buffers, {asked / GIB:.2f} GiB")
    ck("and nothing else went with them -- the shared expert and the norms stay put",
       not any(ops.host_mapped(t) for t in resident(mapped)))
    ck("the same load costs the carve-out that much less", kept > asked * 0.95,
       f"{took_plain / GIB:.2f} -> {took_mapped / GIB:.2f} GiB, {kept / GIB:.2f} saved for "
       f"{asked / GIB:.2f} moved")

    print("\n=== placement does not change a byte of the weights ===")
    same = [i for i, (a, b) in enumerate(zip(got, want)) if not a.cpu().equal(b)]
    ck("every moved buffer is bit-identical to the one the carve-out got", not same,
       f"{len(got) - len(same)}/{len(got)} match")
    probe = got[0]
    echo = torch.empty(probe.numel(), dtype=torch.uint8, device="cuda")
    echo.copy_(probe)
    torch.cuda.synchronize()
    ck("and the device reads it through the registered mapping, which is what the kernels do",
       echo.cpu().equal(probe.cpu()), f"{probe.nbytes / MIB:.1f} MiB round trip")

    print("\n=== the host pays exactly what was asked for, with no allocator rounding ===")
    ck("what pinned is what the map said would pin", ops.host_pinned_bytes() == asked,
       f"{ops.host_pinned_bytes() / GIB:.4f} vs {asked / GIB:.4f} GiB")
    ck("and Shmem, which is where ROCm bills pinned pages, grew by the same",
       asked <= grew < asked * 1.02, f"{grew / GIB:.3f} GiB billed for {asked / GIB:.3f} asked, "
       f"{grew / asked:.3f}x")

    generic(ck)
    return ck.done()


def generic_routed(model: object) -> list[torch.Tensor]:
    return [t for lyr in model.layers for t in parts(lyr.mlp.gate_up_w) + parts(lyr.mlp.down_w)]


def generic(ck: _harness.Checks) -> None:
    if not GENERIC_DIR.is_dir():
        print(f"\n== skipped the generic arm: {GENERIC_DIR} is not here")
        return
    def live() -> int:
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()

    a0 = live()
    plain = loader.load(GENERIC_DIR, layers=range(LAYERS), mtp=False)
    took_plain = live() - a0
    on_device = not any(ops.host_mapped(t) for t in generic_routed(plain))
    want = [t.cpu() for t in generic_routed(plain)]
    del plain

    before = ops.host_pinned_bytes()
    a0 = live()
    mapped = loader.load(GENERIC_DIR, layers=range(LAYERS), mtp=False,
                         device_map=f"experts:{LAYERS}")
    took_mapped = live() - a0
    got = generic_routed(mapped)
    asked = sum(t.nbytes for t in got)
    pinned = ops.host_pinned_bytes() - before

    print("\n=== and the same map on the geometry that used to be refused ===")
    ck("without a map the generic loader keeps its experts in the carve-out", on_device)
    ck("with one, every one of them is in host pages",
       all(ops.host_mapped(t) for t in got),
       f"{asked / GIB:.2f} GiB over {len(got)} buffers")
    ck("the shared expert stayed, so the group named is the group that moved",
       not any(ops.host_mapped(t) for t in
               [t for lyr in mapped.layers
                for t in parts(lyr.mlp.shared_gate_up_w) + parts(lyr.mlp.shared_down_w)]))
    ck("what pinned is what those buffers weigh", pinned == asked,
       f"{pinned / GIB:.4f} vs {asked / GIB:.4f} GiB")
    ck("the same load costs the carve-out that much less", took_plain - took_mapped > asked * 0.95,
       f"{took_plain / GIB:.2f} -> {took_mapped / GIB:.2f} GiB, "
       f"{(took_plain - took_mapped) / GIB:.2f} saved for {asked / GIB:.2f} moved")
    same = [i for i, (a, b) in enumerate(zip(got, want)) if not a.cpu().equal(b)]
    ck("and every moved buffer is bit-identical to the one the carve-out got", not same,
       f"{len(got) - len(same)}/{len(got)} match")


if __name__ == "__main__":
    sys.exit(main())

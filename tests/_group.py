#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
import gc
import importlib
import inspect
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _harness


def _share_loads() -> None:
    from snowllm.checkpoint import loader

    inner, held = loader.load, [None, None]

    def shared(path, **kw):
        key = (str(path),) + tuple(sorted((k, repr(v)) for k, v in kw.items()))
        if held[0] != key:
            held[0] = held[1] = None
            _reclaim()
            held[0], held[1] = key, inner(path, **kw)
        return held[1]

    loader.load = shared


def _reclaim() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _run(name: str) -> tuple[str, float]:
    t0 = time.monotonic()
    _harness.skipped = False
    sys.argv = [name]
    try:
        mod = importlib.import_module(name)
        fn = getattr(mod, "main", None)
        rc = 0
        if fn is not None:
            out = fn()
            if inspect.iscoroutine(out):
                out = asyncio.run(out)
            rc = int(out or 0)
    except SystemExit as e:
        rc = int(e.code or 0)
    except BaseException:
        traceback.print_exc()
        rc = 1
    sys.modules.pop(name, None)
    _reclaim()
    took = time.monotonic() - t0
    if rc:
        return "FAIL", took
    return ("SKIP" if _harness.skipped else "PASS"), took


def main() -> int:
    _share_loads()
    bad = 0
    for path in sys.argv[1:]:
        name = pathlib.Path(path).stem
        print(f"\n== {name}", flush=True)
        status, took = _run(name)
        bad += status == "FAIL"
        print(f">> {name} {status} {took:.0f}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

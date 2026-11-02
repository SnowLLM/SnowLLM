# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path


def main():
    import torch
    from snowllm.checkpoint import loader
    from snowllm.mineru.graphs import GraphCache

    torch.manual_seed(101)
    root = Path(
        os.environ.get("SNOWLLM_MINERU_DIR", "~/models/MinerU-4.1")
    ).expanduser()
    model = loader.load(
        root, mtp=False, kv=dict(ctx=2048, slots=1, util=0.9, chunk=128)
    )
    visual = model.visual
    cache = GraphCache(lambda: 12 << 30)
    visual._memory_admit = cache.admit
    saved = snapshot = None
    grids = (
        [[1, 16, 16]],
        [[1, 8, 32]],
        [[1, 16, 16]],
        [[1, 8, 32]],
        [[1, 8, 16], [1, 8, 16]],
        [[1, 8, 8], [1, 12, 16]],
        [[1, 8, 16], [1, 8, 16]],
        [[1, 8, 8], [1, 12, 16]],
    )
    with torch.inference_mode():
        for grid in grids:
            pixels = torch.randn(256, visual.geo.patch_in)
            grid = torch.tensor(grid, dtype=torch.int64)
            visual._native_graph_cache = None
            expected = visual.forward(pixels, grid)
            visual._native_graph_cache = cache
            actual = visual.forward(pixels, grid)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if saved is not None:
                torch.testing.assert_close(saved, snapshot, atol=0, rtol=0)
            saved, snapshot = actual, actual.clone()
        assert cache.native_captures == 2 and cache.native_replays == 6, cache.status()
        print(
            "PASS: native vision graphs refresh pixels, RoPE and image boundaries; outputs remain independent",
            cache.status(),
            flush=True,
        )
    cache.clear()


if __name__ == "__main__":
    main()

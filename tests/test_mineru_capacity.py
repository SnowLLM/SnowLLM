# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


def main():
    import torch

    from snowllm.mineru.graphs import GraphCache
    from snowllm.models.vision import VisionModel
    from snowllm.ops import image as op
    from snowllm.ops.allocation import active_arena

    class Model(VisionModel):
        def __init__(self):
            self._native_eager = True

        def forward(self, x, out, *, arena=None, admit=True):
            def walk():
                a = op.activation(op.add(x, 1), "relu")
                scratch = active_arena()
                with scratch.frame():
                    scratch.workspace(1024 if x.shape[-1] % 2 else 0).zero_()
                return op.copy(out, op.add(op.multiply(a, a), x))

            return self.execute_plan(
                ("capacity-test", tuple(x.shape)),
                x.shape[0],
                walk,
                arena=arena,
                admit=admit,
                family=("capacity-test",),
            )

    with torch.inference_mode():
        for dtype in (torch.float32, torch.float16):
            model = Model()
            shapes = ((8, 1200), (6, 613), (6, 618), (6, 630), (4, 1094))
            saved = []
            for shape in (*shapes, *reversed(shapes)):
                x = torch.randn(*shape, dtype=dtype, device="cuda")
                out = torch.empty_like(x)
                actual = model.forward(x, out)
                a = torch.relu(x + 1)
                torch.testing.assert_close(actual, a * a + x, atol=0, rtol=0)
                for previous, snapshot in saved:
                    torch.testing.assert_close(previous, snapshot, atol=0, rtol=0)
                saved.append((actual, actual.clone()))
            capacity = model._eager_arena
            assert capacity.probes == 1, capacity.probes
            assert capacity.rebuilds == 1, capacity.rebuilds
            assert capacity.scratch_bytes >= 1024
            for plan in capacity.capacities.values():
                assert not plan.overlaps(), plan.overlaps()[:3]
            print(
                "PASS: capacity reuse, non-monotone workspace, FP16/FP32 outputs",
                dtype,
                flush=True,
            )

        model = Model()
        model._native_graph_after = 3
        cache = GraphCache(lambda: 12 << 30)
        model._memory_admit = cache.admit
        x = torch.randn(4, 65, device="cuda")
        for _ in range(5):
            x.normal_()
            actual = cache.native(model.forward, (x,), torch.empty_like(x))
            a = torch.relu(x + 1)
            torch.testing.assert_close(actual, a * a + x, atol=0, rtol=0)
        assert cache.captures == 1 and cache.replays == 2, cache.status()
        assert cache.eager_forwards == 2, cache.status()
        cache.clear()
        assert not hasattr(model, "_eager_arena")
        print(
            "PASS: repeated shapes promote to graphs and refresh their inputs",
            flush=True,
        )

        model = Model()
        x = torch.randn(4, 64, device="cuda")
        out = torch.full_like(x, float("nan"))
        with patch.object(
            model,
            "_memory_admit",
            side_effect=MemoryError("capacity denied"),
            create=True,
        ):
            try:
                model.forward(x, out)
            except MemoryError:
                pass
            else:
                raise AssertionError(
                    "capacity allocation must respect the shared budget"
                )
        assert model._eager_arena.buf is None
        assert torch.isnan(out).all()
        print(
            "PASS: insufficient budget fails before allocation and GPU execution",
            flush=True,
        )

        import os
        from pathlib import Path

        from snowllm.checkpoint.loader import load_auxiliary

        root = Path(
            os.environ.get("SNOWLLM_MINERU_DIR", "~/models/MinerU-4.1")
        ).expanduser()
        folder = root / "small-models/OCR/paddleocr"
        for filename, options, shapes in (
            (
                "ch_PP-OCRv6_tiny_det_infer.safetensors",
                {"detection": True},
                (
                    (1, 3, 96, 128),
                    (2, 3, 128, 160),
                    (4, 3, 160, 704),
                    (2, 3, 128, 960),
                    (1, 3, 160, 672),
                    (3, 3, 192, 768),
                ),
            ),
            (
                "ch_PP-OCRv6_small_rec_infer.safetensors",
                {},
                (
                    (1, 3, 48, 64),
                    (2, 3, 48, 320),
                    (6, 3, 48, 613),
                    (6, 3, 48, 618),
                    (4, 3, 48, 1094),
                    (6, 3, 48, 1090),
                    (3, 3, 48, 324),
                    (1, 3, 48, 1072),
                ),
            ),
            (
                "seal_PP-OCRv4_det_infer.pth",
                {"seal": True},
                ((1, 3, 128, 160), (2, 3, 64, 96), (3, 3, 96, 128)),
            ),
        ):
            model = load_auxiliary(folder / filename, "PPOCRModel", **options)
            cache = GraphCache(lambda: 12 << 30)
            model._memory_admit = cache.admit
            model._native_graph_after = 3
            saved = snapshot = None
            output_key = "maps" if options else "ctc_logits"
            for shape in (*shapes, *reversed(shapes)):
                x = torch.randn(*shape, device="cuda", dtype=model.dtype)
                model._native_eager, model._native_graph_cache = False, None
                expected = model.forward(x)[output_key]
                model._native_eager, model._native_graph_cache = True, cache
                actual = model.forward(x)[output_key]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                if saved is not None:
                    torch.testing.assert_close(saved, snapshot, atol=0, rtol=0)
                saved, snapshot = actual, actual.clone()
            shape = shapes[1]
            x = torch.randn(
                *shape[:-1], shape[-1] * 2, device="cuda", dtype=model.dtype
            )[..., ::2]
            model._native_eager, model._native_graph_cache = False, None
            expected = model.forward(x)[output_key]
            model._native_eager, model._native_graph_cache = True, cache
            torch.testing.assert_close(
                model.forward(x)[output_key], expected, atol=0, rtol=0
            )
            print(
                "PASS: dynamic native OCR is bitwise identical to exact-shape execution",
                filename,
                cache.status(),
                flush=True,
            )
            cache.clear()

        from snowllm.checkpoint import loader

        main_model = loader.load(
            root, mtp=False, kv=dict(ctx=2048, slots=1, util=0.9, chunk=128)
        )
        visual = main_model.visual
        cache = GraphCache(lambda: 12 << 30)
        visual._memory_admit = cache.admit
        for grid in (
            [[1, 16, 16]],
            [[1, 8, 64]],
            [[1, 8, 16], [1, 8, 16]],
            [[1, 78, 58]],
            [[1, 16, 16]],
        ):
            rows = sum(t * h * w for t, h, w in grid)
            pixels = (
                torch.randn(
                    rows, visual.geo.patch_in, dtype=torch.bfloat16, device="cuda"
                )
                * 0.1
            )
            grid = torch.tensor(grid, dtype=torch.int64)
            visual._native_eager, visual._native_graph_cache = False, None
            expected = visual.forward(pixels, grid)
            visual._native_eager, visual._native_graph_cache = True, cache
            actual = visual.forward(pixels, grid)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert cache.captures == 0
        print(
            "PASS: capacity vision preserves ragged-image and long-image BF16 outputs",
            cache.status(),
            flush=True,
        )
        cache.clear()


if __name__ == "__main__":
    main()

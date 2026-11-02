# SPDX-License-Identifier: Apache-2.0
import base64
import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from snowllm.hub import recipes
from snowllm.mineru.api import configure
from snowllm.mineru.assets import configure_models
from snowllm._platform import DeviceMemory
from snowllm.mineru.transport import InternalAsyncTransport
from snowllm.ngram import banned_ngram_tokens
from snowllm.serve.state import _REQUEST_STATE
from snowllm.serve.routing import ModelRouting


class MinerURecipeTests(unittest.TestCase):
    def test_preimported_table_factories_are_patched_and_restored(self):
        original = lambda *args, **kwargs: "original"
        factory = lambda *args, **kwargs: "gpu"
        saved = SimpleNamespace()
        model = SimpleNamespace(model_copy=lambda **kwargs: saved)
        config = SimpleNamespace(model=model)
        paths = SimpleNamespace(ensure=original)
        hybrid = SimpleNamespace(get_device=original)
        onnx = SimpleNamespace(table_ort_session=original)
        names = (
            "cls.paddle_table_cls",
            "rec.slanet_plus.table_structure",
            "rec.slanet_plus.table_structure_utils",
            "rec.unet_table.table_structure_unet",
            "rec.unet_table.utils",
        )
        tables = {
            f"mineru.model.table.{name}": SimpleNamespace(table_ort_session=original)
            for name in names
        }
        modules = {
            "mineru.config": SimpleNamespace(get_config=lambda: config),
            "mineru.model": SimpleNamespace(download=SimpleNamespace(ModelPath=paths)),
            "mineru.model.runtime": SimpleNamespace(hybrid=hybrid, onnx=onnx),
            **tables,
        }
        with patch.dict("sys.modules", modules):
            with contextlib.ExitStack() as stack:
                configure_models(stack, "/models/MinerU", factory)
                for module in (onnx, *tables.values()):
                    self.assertIs(module.table_ort_session, factory)
                self.assertEqual(hybrid.get_device(), "cuda")
            for module in (onnx, *tables.values()):
                self.assertIs(module.table_ort_session, original)
            self.assertIs(config.model, saved)

    def test_pinned_complete_bundle(self):
        raw = json.loads(
            (Path(__file__).resolve().parents[1] / "recipes/mineru.json").read_text()
        )
        recipe = recipes.Recipe(raw)
        self.assertEqual(recipe.runtime, "mineru")
        self.assertEqual(len(recipe.sources), 2)
        self.assertTrue(all(len(source.revision) == 40 for source in recipe.sources))
        self.assertEqual(recipe.sources[1].subdir, "small-models")

    def test_only_mineru_downloads_when_started(self):
        for runtime in ("mineru", "llm"):
            recipe = recipes.Recipe(
                dict(id="document", repo="acme/models", runtime=runtime)
            )
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(recipes, "catalogue", return_value=[recipe]),
                patch.object(recipes, "models_root", return_value=Path(directory)),
                patch.object(recipes, "pull") as pull,
            ):
                if runtime == "mineru":
                    self.assertEqual(
                        recipes.locate("document"),
                        (recipe.path(Path(directory)), recipe),
                    )
                    pull.assert_called_once()
                else:
                    with self.assertRaisesRegex(Exception, "Run `snowllm pull"):
                        recipes.locate("document")
                    pull.assert_not_called()

    def test_manifest_detects_missing_files_and_source_changes(self):
        recipe = recipes.Recipe(
            dict(id="document", repo="acme/models", revision="a" * 40)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weight").write_bytes(b"model")
            blob = recipes.Blob("https://example.com/weight", root / "weight", 5)
            recipes.write_manifest(recipe, root, [blob])
            self.assertTrue(recipes.complete(recipe, root))
            recipe.sources[0].revision = "b" * 40
            self.assertFalse(recipes.complete(recipe, root))
            recipe.sources[0].revision = "a" * 40
            (root / "weight").write_bytes(b"partial")
            self.assertFalse(recipes.complete(recipe, root))

    def test_remaining_gpu_budget(self):
        gib = 1 << 30
        self.assertEqual(
            DeviceMemory(32 * gib, 8 * gib, 0, 32 * gib, 8 * gib).allowance(12 * gib),
            4 * gib,
        )
        self.assertEqual(
            DeviceMemory(32 * gib, 8 * gib, gib, 32 * gib, 8 * gib).allowance(12 * gib),
            5 * gib,
        )
        self.assertLess(
            DeviceMemory(32 * gib, 14 * gib, 0, 32 * gib, 14 * gib).allowance(12 * gib),
            0,
        )

    def test_ngram_masks_only_repeated_completions(self):
        self.assertEqual(banned_ngram_tokens([1, 2, 3, 1, 2], 3), [3])
        self.assertEqual(banned_ngram_tokens([1, 2, 3, 1, 4], 3), [])
        self.assertEqual(banned_ngram_tokens([3, 1, 3], 1), [1, 3])
        self.assertEqual(banned_ngram_tokens([1, 2], 0), [])
        self.assertEqual(banned_ngram_tokens([1, 2], 4), [])


class MinerUApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_calls_keep_document_binding_during_draining(self):
        app = FastAPI()
        app.add_middleware(ModelRouting)
        state = SimpleNamespace(model_name="documents")
        registry = SimpleNamespace(
            acquire=lambda name: self.fail("must reuse the document lease")
        )
        app.state.registry = registry

        @app.post("/v1/chat/completions")
        async def chat():
            self.assertIs(_REQUEST_STATE.get(), state)
            return {"choices": []}

        runtime = SimpleNamespace(
            loop=asyncio.get_running_loop(), state=state, check_cancelled=lambda: None
        )
        with patch.dict("sys.modules", {"snowllm.serve.api": SimpleNamespace(app=app)}):
            async with httpx.AsyncClient(
                transport=InternalAsyncTransport(runtime)
            ) as client:
                response = await client.post(
                    "http://snowllm.internal/v1/chat/completions", json={}
                )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(_REQUEST_STATE.get())

    async def test_vlm_requests_return_to_the_serving_loop(self):
        loop = asyncio.get_running_loop()
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            self.assertIs(asyncio.get_running_loop(), loop)
            return {"choices": []}

        runtime = SimpleNamespace(loop=loop, check_cancelled=lambda: None)

        def other_loop():
            async def send():
                async with httpx.AsyncClient(
                    transport=InternalAsyncTransport(runtime)
                ) as client:
                    return await client.post(
                        "http://snowllm.internal/v1/chat/completions", json={}
                    )

            return asyncio.run(send())

        with patch.dict("sys.modules", {"snowllm.serve.api": SimpleNamespace(app=app)}):
            response = await asyncio.wait_for(asyncio.to_thread(other_loop), timeout=5)
        self.assertEqual(response.status_code, 200)

    async def test_parse_routes_selected_model_and_cleans_upload(self):
        app = FastAPI()
        configure(app)
        paths = []

        async def parse(path, **kwargs):
            self.assertEqual(path.read_bytes(), b"example pdf")
            self.assertEqual(kwargs, dict(tier="standard", page_range="1-2"))
            paths.append(path)
            return dict(markdown="Result", document={})

        state = SimpleNamespace(
            engine=SimpleNamespace(mineru=SimpleNamespace(parse=parse)),
            model_name="documents",
        )
        token = _REQUEST_STATE.set(state)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/documents/parse",
                    json=dict(
                        filename="../../report.pdf",
                        file=base64.b64encode(b"example pdf").decode(),
                        model="documents",
                        page_range="1-2",
                    ),
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["model"], "documents")
                self.assertEqual(response.json()["filename"], "report.pdf")
            self.assertFalse(paths[0].exists())
        finally:
            _REQUEST_STATE.reset(token)

    async def test_invalid_document_rejected_before_execution(self):
        app = FastAPI()
        configure(app)
        token = _REQUEST_STATE.set(
            SimpleNamespace(engine=SimpleNamespace(mineru=object()))
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                for filename, content in [
                    ("file.pdf", "not base64"),
                    ("file.exe", "eA=="),
                ]:
                    response = await client.post(
                        "/v1/documents/parse",
                        json=dict(filename=filename, file=content),
                    )
                    self.assertEqual(response.status_code, 400)
        finally:
            _REQUEST_STATE.reset(token)


if __name__ == "__main__":
    unittest.main()

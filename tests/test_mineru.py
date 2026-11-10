# SPDX-License-Identifier: Apache-2.0
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
                patch.object(recipes, "models_roots", return_value=[Path(directory)]),
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


class MinerUInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_binary_inference_routes_selected_model_and_releases_lease(self):
        import numpy as np
        from snowllm.mineru.wire import pack, unpack

        calls = []

        async def infer(component, inputs):
            calls.append((component, inputs))
            return {'maps': inputs['input'] * 2}

        state = SimpleNamespace(engine=SimpleNamespace(mineru=SimpleNamespace(
            infer=infer, describe=lambda: {'protocol': 1})), model_name='documents')
        record = SimpleNamespace(state=state, active=0)

        def acquire(name):
            self.assertEqual(name, 'documents')
            record.active += 1
            return record

        app = FastAPI()
        app.state.registry = SimpleNamespace(acquire=acquire)
        app.add_middleware(ModelRouting)
        configure(app)
        values = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/inference/ocr_det?model=documents',
                                         content=pack({'input': values}))
            self.assertEqual(response.status_code, 200, response.text)
            np.testing.assert_array_equal(unpack(response.content)['maps'], values * 2)
            self.assertEqual(record.active, 0)
            self.assertEqual(calls[0][0], 'ocr_det')
            response = await client.get('/v1/inference?model=documents')
            self.assertEqual(response.json(), {'protocol': 1})
            self.assertEqual(record.active, 0)

    async def test_document_business_route_is_absent(self):
        app = FastAPI()
        configure(app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/documents/parse', json={})
            self.assertEqual(response.status_code, 404)

    async def test_invalid_tensor_does_not_reach_executor(self):
        called = False

        async def infer(*args):
            nonlocal called
            called = True

        token = _REQUEST_STATE.set(SimpleNamespace(engine=SimpleNamespace(
            mineru=SimpleNamespace(infer=infer))))
        app = FastAPI()
        configure(app)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post('/v1/inference/ocr_det', content=b'%PDF-1.7')
                self.assertEqual(response.status_code, 400)
            self.assertFalse(called)
        finally:
            _REQUEST_STATE.reset(token)


class MinerUTransportTests(unittest.TestCase):
    def test_forged_tensor_shape_is_rejected_before_allocation(self):
        import io
        import zipfile
        import numpy as np
        from snowllm.mineru.wire import unpack

        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(
            descr='<f4', fortran_order=False, shape=(1 << 40,)))
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, 'w') as archive:
            archive.writestr('input.npy', header.getvalue())
        with self.assertRaisesRegex(ValueError, 'shape'):
            unpack(payload.getvalue())

    def test_tensors_round_trip_without_pickle(self):
        import io
        import numpy as np
        from snowllm.mineru.wire import pack, unpack

        values = {'image': np.arange(24, dtype=np.float16).reshape(2, 3, 4),
                  'ids': np.array([1, 2], dtype=np.int64), 'flag': True}
        received = unpack(pack(values))
        for name, expected in values.items():
            np.testing.assert_array_equal(received[name], expected)
        bad = io.BytesIO()
        np.savez(bad, item=np.array([{}], dtype=object))
        with self.assertRaises(ValueError):
            unpack(bad.getvalue())

    def test_model_client_pins_resolved_model_and_sends_binary_arrays(self):
        import numpy as np
        from snowllm.mineru.client import ModelClient, RemoteSession
        from snowllm.mineru.wire import pack, unpack

        def handle(request):
            if request.method == 'GET':
                return httpx.Response(200, json={'protocol': 1, 'model': 'resolved', 'assets': {},
                    'components': {'table_cls': {'checkpoint': 'Table/cls.onnx',
                        'architecture': 'MinerUTableModel', 'options': {'kind': 'classifier'},
                        'kind': 'tensor', 'inputs': ['x'], 'outputs': ['scores'], 'metadata': {}}}})
            self.assertEqual(request.url.params['model'], 'resolved')
            values = unpack(request.content)
            return httpx.Response(200, content=pack({'scores': values['input'] + 1}))

        client = ModelClient('http://test/v1', transport=httpx.MockTransport(handle))
        try:
            session = RemoteSession(client, '/models/cls.onnx')
            actual = session.run(None, {'x': np.array([2], dtype=np.float32)})
            np.testing.assert_array_equal(actual[0], [3])
            with self.assertRaises(ValueError):
                session.run(None, {'wrong': np.array([2])})
        finally:
            client.close()


class MinerUEntryTests(unittest.TestCase):
    def test_webui_child_uses_injected_entry_and_preserves_native_arguments(self):
        import sys
        from snowllm.mineru.cli import patch_webui

        class Managed:
            def _command(self, port, directory):
                return [sys.executable, '-m', 'mineru.kit.main', 'api-server',
                        '--port', str(port), '--upload-dir', str(directory)]

        original = Managed._command
        with patch.dict('sys.modules', {'mineru.kit.gradio.client': SimpleNamespace(
                ManagedLocalApiServer=Managed)}):
            with contextlib.ExitStack() as stack:
                patch_webui(stack, 'http://inference:8000', 'documents')
                command = Managed()._command(1234, '/uploads')
                self.assertEqual(command[:3], [sys.executable, '-m', 'snowllm.mineru.cli'])
                self.assertIn('documents', command)
                self.assertEqual(command[-5:], ['api-server', '--port', '1234', '--upload-dir', '/uploads'])
            self.assertIs(Managed._command, original)

    def test_help_forwards_without_opening_model_connection(self):
        import sys
        from snowllm.mineru.cli import main

        seen = []
        with (
            patch('snowllm.mineru.injection.require_version'),
            patch.dict('sys.modules', {'mineru.kit.main': SimpleNamespace(
                main=lambda: seen.append(list(sys.argv)))}),
            patch('snowllm.mineru.client.ModelClient') as client,
        ):
            main(['--snowllm-url', 'http://test', 'api-server', '--help'])
        self.assertEqual(seen, [['snowllm-mineru', 'api-server', '--help']])
        client.assert_not_called()

    def test_adapter_refuses_late_injection(self):
        from snowllm.mineru.injection import install

        hybrid = SimpleNamespace(AtomModelSingleton=SimpleNamespace(_models={'loaded': object()}))
        with (
            patch('snowllm.mineru.injection.require_version'),
            patch.dict('sys.modules', {
                'mineru.config': SimpleNamespace(get_config=lambda: None),
                'mineru.model.runtime': SimpleNamespace(hybrid=hybrid)}),
            contextlib.ExitStack() as stack,
            self.assertRaisesRegex(RuntimeError, 'before loading'),
        ):
            install(stack, None)


if __name__ == '__main__':
    unittest.main()

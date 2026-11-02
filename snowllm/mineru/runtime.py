# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import gc
import inspect
import sys
import threading
from pathlib import Path

import httpx
import torch

from .assets import configure_models
from .onnx import TableSession
from .graphs import GraphCache
from .transport import InternalAsyncTransport, InternalTransport


class GPUModel:
    def __init__(self, model, runtime):
        self._model, self._runtime = model, runtime

    def __getattr__(self, name):
        value = getattr(self._model, name)
        if (
            name in ("text_detector", "text_recognizer", "text_classifier")
            and value is not None
        ):
            return GPUModel(value, self._runtime)
        if callable(value) and name in (
            "predict",
            "batch_predict",
            "batch_predict_iter",
            "predict_batch",
            "run",
            "ocr",
            "batch_det",
            "batch_rec",
        ):

            def invoke(*args, **kwargs):
                result = self._runtime.gpu_call(lambda: value(*args, **kwargs))
                if not inspect.isgenerator(result):
                    return result

                def iterate():
                    end = object()
                    try:
                        while True:
                            item = self._runtime.gpu_call(lambda: next(result, end))
                            if item is end:
                                return
                            yield item
                    finally:
                        self._runtime.gpu_call(result.close)

                return iterate()

            return invoke
        return value

    def __call__(self, *args, **kwargs):
        return self._runtime.gpu_call(lambda: self._model(*args, **kwargs))


class MinerURuntime:
    def __init__(self, root, limits, *, enforce_eager=False):
        if sys.version_info < (3, 11):
            raise RuntimeError("MinerU requires Python 3.11 or newer")
        try:
            from importlib.metadata import version

            if version("mineru") != "4.1.3":
                raise RuntimeError(
                    "MinerU recipe requires mineru==4.1.3; install snowllm[mineru]"
                )
        except ImportError as exc:
            raise RuntimeError(
                "install snowllm[mineru] with a ROCm PyTorch environment"
            ) from exc
        from mineru.model.runtime.ownership import RuntimeOwner

        self.root = Path(root).resolve()
        self.limits = limits
        self.state = self.engine = self.loop = self.predictor = None
        self.owner = RuntimeOwner()
        self.restores = contextlib.ExitStack()
        self.graphs = GraphCache(lambda: self.limit, enabled=not enforce_eager)
        self.native_models = {}
        self.cancelled = threading.Event()
        self.gpu_thread = threading.get_ident()
        self.closed = False
        self.disposed = False
        self.tasks = set()
        self._lock = asyncio.Lock()
        from mineru.model.runtime.ownership import runtime_owner

        token = runtime_owner.set(self.owner)
        try:
            self._configure()
            from mineru.model.runtime.hybrid import HybridLocalModelContextSingleton

            before = torch.cuda.memory_allocated()
            with torch.inference_mode():
                self.context = HybridLocalModelContextSingleton().get_model()
                for name in (
                    "mfr_model",
                    "wireless_table_model",
                    "wired_table_model",
                    "table_cls_model",
                    "table_orientation_cls_model",
                    "seal_model",
                ):
                    getattr(self.context, name)
            self.weight_bytes = max(0, torch.cuda.memory_allocated() - before)
        except BaseException:
            self.close()
            raise
        finally:
            runtime_owner.reset(token)

    @property
    def limit(self):
        return self.limits.gpu_limit(torch.cuda.get_device_properties(0).total_memory)

    def _patch(self, target, name, value):
        original = getattr(target, name)
        setattr(target, name, value)
        self.restores.callback(setattr, target, name, original)

    def _configure(self):
        from mineru.model.runtime import hybrid

        configure_models(
            self.restores,
            self.root,
            lambda path, **kwargs: TableSession(
                path, graphs=self.graphs, load_model=self._load_native
            ),
        )
        original_init = hybrid.atom_model_init

        def initialize(*args, **kwargs):
            name = kwargs.get("model_name", args[0] if args else None)
            if str(name) == "mfr":
                from .mfr import Formula

                return GPUModel(
                    Formula(
                        self.root / "small-models/MFR/pp_formulanet_plus_m",
                        self._load_native,
                    ),
                    self,
                )
            if str(name) == "layout":
                from .layout import Layout

                return GPUModel(
                    Layout(
                        self.root / "small-models/Layout/PP-DocLayoutV2",
                        self._load_native,
                    ),
                    self,
                )
            if str(name) == "ocr":
                from .ocr import OCR

                return GPUModel(
                    OCR(
                        self._load_ocr,
                        **{
                            key: value
                            for key, value in kwargs.items()
                            if key not in ("model_name", "small_backend")
                        },
                    ),
                    self,
                )
            model = original_init(*args, **kwargs)
            return GPUModel(model, self)

        self._patch(hybrid, "atom_model_init", initialize)
        from mineru.backend.analysis.pdf import pipeline

        def cleanup(device="cuda"):
            from .._platform import release_device_cache

            self.gpu_call(release_device_cache)
            gc.collect(0)

        self._patch(pipeline, "clean_memory", cleanup)

        self._patch(
            pipeline,
            "get_vlm_predictor",
            lambda settings=None: (self.predictor, "http-client"),
        )

    def _load_ocr(self, path, **options):
        return self._load_native(path, "PPOCRModel", **options)

    def _load_native(self, path, architecture, **options):
        from ..checkpoint.loader import load_auxiliary

        key = str(path), architecture
        if key not in self.native_models:
            model = load_auxiliary(
                path,
                architecture,
                admit=self.graphs.admit if self.graphs else None,
                **options,
            )
            model._native_graph_cache = self.graphs
            if architecture == 'PPOCRModel':
                model._native_eager = True
                model._native_graph_after = 3
            if self.graphs is not None:
                model._memory_admit = self.graphs.admit
            self.native_models[key] = model
        return self.native_models[key]

    def bind(self, state):
        self.state = state
        self.engine = state.engine
        self.name = state.model_name
        state.engine.mineru = self
        if self.graphs is not None:
            state.model.visual._memory_admit = self.graphs.admit
            state.model.visual._native_graph_cache = self.graphs
            state.model.visual._native_eager = True
            state.model.visual._native_graph_after = 3
            state.model.visual._native_graph_max_rows = 1024
        config = state.engine.engine.runtime_config
        config.update(
            runtime="mineru",
            auxiliary_weight_bytes=self.weight_bytes,
            document_parse_path="/v1/documents/parse",
        )
        state.engine.memory["auxiliary_weights"] = self.weight_bytes
        for name in ("weights", "total", "load_peak"):
            if name in state.engine.memory:
                state.engine.memory[name] += self.weight_bytes

    def start(self):
        self.loop = asyncio.get_running_loop()
        from mineru_vl_utils import MinerUClient

        self.predictor = MinerUClient(
            backend="http-client",
            model_name=self.name,
            server_url="http://snowllm.internal/v1",
            skip_model_name_checking=True,
            max_concurrency=1,
            use_tqdm=False,
            max_retries=0,
        )
        client = self.predictor.client
        client._client.close()
        client._client = httpx.Client(transport=InternalTransport(self))

        async def aio_client():
            return httpx.AsyncClient(transport=InternalAsyncTransport(self))

        client._new_aio_client = aio_client

    def check_cancelled(self):
        if self.closed or self.cancelled.is_set():
            raise RuntimeError("MinerU document parsing cancelled")

    def gpu_call(self, function):
        self.check_cancelled()

        def run():
            self.check_cancelled()
            self.gpu_thread = threading.get_ident()
            with torch.inference_mode():
                return function()

        if threading.get_ident() == self.gpu_thread:
            return run()
        return asyncio.run_coroutine_threadsafe(
            self.engine.call(run), self.loop
        ).result()

    async def parse(self, path, *, tier="standard", page_range=""):
        if self.loop is None:
            self.start()
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            async with self._lock:
                self.check_cancelled()

                def run():
                    from mineru.parser.mineru_parser import MinerUParser
                    from mineru.model.runtime.ownership import runtime_owner

                    token = runtime_owner.set(self.owner)
                    try:
                        with torch.inference_mode():
                            result = MinerUParser(tier=tier).parse(
                                path, page_range=page_range
                            )
                            return dict(
                                markdown=result.markdown(), document=result.to_dict()
                            )
                    finally:
                        runtime_owner.reset(token)

                work = asyncio.create_task(asyncio.to_thread(run))
                try:
                    return await asyncio.shield(work)
                except asyncio.CancelledError:
                    self.cancelled.set()
                    while not work.done():
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await asyncio.shield(work)
                    if not work.cancelled():
                        work.exception()
                    raise
                finally:
                    if not self.closed:
                        self.cancelled.clear()
        finally:
            self.tasks.discard(task)

    def cancel(self):
        self.cancelled.set()
        for task in tuple(self.tasks):
            task.cancel()

    async def aclose(self):
        self.closed = True
        self.cancel()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        if self.predictor is not None:
            await self.predictor.aclose()
            self.predictor = None

    def close(self):
        from mineru.model.runtime.ownership import release_runtime_owner

        if self.disposed:
            return
        self.disposed = True
        self.closed = True
        try:
            release_runtime_owner(self.owner)
        finally:
            self.state = None
            self.context = None
            if self.graphs is not None:
                self.graphs.clear()
            for model in self.native_models.values():
                close = getattr(model, "close", None)
                if close is not None:
                    close()
            self.native_models.clear()
            self.restores.close()

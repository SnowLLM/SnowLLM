# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections import deque
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class FakeEngine:
    def __init__(self):
        self.waiting = deque()
        self.running = []
        self.fail_next = False

    def add(self, prompt, params, **kwargs):
        return SimpleNamespace(
            prompt=prompt, params=params, out=[], done=False, finish_reason=None
        )

    def step(self):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("test step failure")
        while self.waiting:
            request = self.waiting.popleft()
            if not request.done:
                request.out.append(42)
                request.done, request.finish_reason = True, "stop"


def async_engine_class():
    package = ModuleType("snowllm.engine")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "snowllm" / "engine")]
    package.Engine = FakeEngine
    package.EngineStats = package.Request = package.SamplingParams = SimpleNamespace
    path = Path(package.__path__[0]) / "async_engine.py"
    spec = importlib.util.spec_from_file_location(
        "snowllm.engine._single_path_test", path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"snowllm.engine": package}):
        spec.loader.exec_module(module)
    return module


class SingleModelPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_standalone_uses_original_private_executor_without_model_switches(
        self,
    ):
        module = async_engine_class()
        with patch.object(
            module, "ModelExecutionContext", side_effect=AssertionError("model switch")
        ):
            first = module.AsyncEngine(FakeEngine())
            second = module.AsyncEngine(FakeEngine())
        try:
            self.assertIsNone(first.scheduler)
            self.assertIsNone(first.context)
            self.assertIsNot(first._worker, second._worker)
            workers = await asyncio.gather(
                first.call(threading.get_ident), second.call(threading.get_ident)
            )
            self.assertNotEqual(workers[0], workers[1])
            self.assertNotIn(threading.get_ident(), workers)
        finally:
            await first.close()
            await second.close()

    async def test_standalone_keeps_serving_after_a_failed_step(self):
        module = async_engine_class()
        native = FakeEngine()
        engine = module.AsyncEngine(native)
        engine.start()
        try:
            native.fail_next = True
            request = engine.submit([1], SimpleNamespace())
            with patch.object(module.traceback, "print_exc", Mock()):
                self.assertEqual([token async for token in engine.stream(request)], [])
            self.assertEqual(request.finish_reason, "error")
            self.assertIsNone(engine.error)
            request = engine.submit([2], SimpleNamespace())
            self.assertEqual([token async for token in engine.stream(request)], [42])
            self.assertEqual(request.finish_reason, "stop")
        finally:
            await engine.close()


if __name__ == "__main__":
    unittest.main()

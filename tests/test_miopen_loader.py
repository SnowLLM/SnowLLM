import ctypes
import ctypes.util
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "snowllm/ops/miopen.py"


class MIOpenLoaderTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            version=SimpleNamespace(hip="7.14.0"),
            __file__="/example/venv/site-packages/torch/__init__.py",
        )
        spec = importlib.util.spec_from_file_location(
            "snowllm.ops._miopen_test", SOURCE
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"torch": self.torch}):
            spec.loader.exec_module(self.module)

    def test_uses_torch_dependency_scope_and_caches_handle(self):
        lib = Mock()
        with (
            patch.object(self.module.sys, "platform", "linux"),
            patch.object(ctypes, "CDLL", return_value=lib) as load,
            patch.object(
                ctypes.util, "find_library", side_effect=AssertionError("system lookup")
            ),
        ):
            self.assertIs(self.module.library(), lib)
            self.assertIs(self.module.library(), lib)
        load.assert_called_once_with(
            "/example/venv/site-packages/torch/lib/libtorch_hip.so"
        )
        self.assertEqual(lib.miopenCreate.argtypes, [ctypes.POINTER(ctypes.c_void_p)])
        self.assertEqual(lib.miopenConvolutionForwardImmediate.restype, ctypes.c_int)

    def test_rejects_non_rocm_torch_before_loading(self):
        self.torch.version.hip = None
        with patch.object(ctypes, "CDLL") as load:
            with self.assertRaisesRegex(RuntimeError, "require ROCm PyTorch"):
                self.module.library()
        load.assert_not_called()

    def test_missing_torch_library_does_not_fall_back_to_system(self):
        with (
            patch.object(self.module.sys, "platform", "linux"),
            patch.object(ctypes, "CDLL", side_effect=OSError("missing")) as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "does not expose"):
                self.module.library()
        self.assertEqual(load.call_count, 1)

    def test_torch_without_miopen_symbols_fails_clearly(self):
        with (
            patch.object(self.module.sys, "platform", "linux"),
            patch.object(ctypes, "CDLL", return_value=SimpleNamespace()),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not expose"):
                self.module.library()

    def test_windows_reuses_loaded_dll(self):
        path = r"C:\venv\Lib\site-packages\torch\lib\MIOpen.dll"
        platform = SimpleNamespace(
            _win_modules=lambda: [path, r"C:\Windows\kernel32.dll"]
        )
        with (
            patch.object(self.module.sys, "platform", "win32"),
            patch.dict("sys.modules", {"snowllm._platform": platform}),
            patch.object(ctypes, "CDLL", return_value=Mock()) as load,
        ):
            self.module.library()
        load.assert_called_once_with(path)

    def test_windows_refuses_missing_or_multiple_dlls(self):
        for paths in ([], [r"C:\torch\MIOpen.dll", r"C:\system\MIOpen.dll"]):
            with (
                self.subTest(paths=paths),
                patch.object(self.module.sys, "platform", "win32"),
                patch.dict(
                    "sys.modules",
                    {"snowllm._platform": SimpleNamespace(_win_modules=lambda: paths)},
                ),
                patch.object(ctypes, "CDLL") as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "already loaded"):
                    self.module.library()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

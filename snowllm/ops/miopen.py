# SPDX-License-Identifier: Apache-2.0
import ctypes as C
import functools
import sys
from pathlib import Path

import torch


@functools.cache
def library():
    if not torch.version.hip:
        raise RuntimeError("MinerU GPU convolutions require ROCm PyTorch with MIOpen")
    if sys.platform == "win32":
        from .._platform import _win_modules
        import ntpath

        paths = {path for path in _win_modules() if ntpath.basename(path).lower().startswith("miopen")
                 and path.lower().endswith(".dll")}
        if len(paths) != 1:
            raise RuntimeError("MinerU requires one MIOpen DLL already loaded by ROCm PyTorch")
        path = paths.pop()
    else:
        path = str(Path(torch.__file__).resolve().parent / "lib" / "libtorch_hip.so")
    try:
        module = C.CDLL(path)
        module.miopenCreate
        return module
    except (OSError, AttributeError) as exc:
        raise RuntimeError("ROCm PyTorch does not expose the MIOpen convolution library") from exc


class Descriptor(C.Structure):
    _fields_ = ([(key, C.c_int * 4) for key in ("x", "w", "y")]
                + [(key, C.c_int * 2) for key in ("stride", "padding", "dilation")]
                + [(key, C.c_int) for key in ("groups", "dtype", "transpose", "act", "channels_last",
                    "dense_params", "fp32_bias", "affine", "config", "allow_pointwise", "allow_depthwise")]
                + [("alignment", C.c_int64)])


class Convolution:
    def __init__(self, desc):
        from .._capi import lib
        from ..errors import SnowLLMError

        self.lib = lib
        self.pointer = C.c_void_p()
        size, dense_input = C.c_int64(), C.c_int()
        try:
            self.module = library()
        except RuntimeError as exc:
            self.module = None
            load_error = exc
        error = lib.snowllm_convolution_plan(C.byref(desc), self.module._handle if self.module else None,
                                             C.byref(self.pointer), C.byref(size), C.byref(dense_input))
        if error:
            if self.module is None:
                raise load_error
            raise SnowLLMError(error.decode())
        self.workspace, self.dense_input = size.value, bool(dense_input.value)

    def __del__(self):
        if self.pointer:
            self.lib.snowllm_convolution_destroy(self.pointer)

    def __call__(self, x, weight, bias, scale, shift, out, workspace):
        from ..errors import SnowLLMError
        from ._common import _p, _stream

        with torch.cuda.device(x.device):
            error = self.lib.snowllm_convolution_execute(
                self.pointer, _p(x), _p(weight), _p(bias), _p(scale), _p(shift), _p(out), _p(workspace), _stream())
        if error:
            raise SnowLLMError(error.decode())


@functools.lru_cache(maxsize=1024)
def convolution(device, *geometry):
    with torch.cuda.device(device):
        return Convolution(Descriptor(*geometry))

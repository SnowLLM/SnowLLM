# SPDX-License-Identifier: Apache-2.0
import ctypes as C
import functools
from pathlib import Path
import sys

import torch


class Solution(C.Structure):
    _fields_ = [
        ("time", C.c_float),
        ("workspace_size", C.c_size_t),
        ("solution_id", C.c_uint64),
        ("algorithm", C.c_int),
    ]


@functools.cache
def library():
    if not torch.version.hip:
        raise RuntimeError("MinerU GPU convolutions require ROCm PyTorch with MIOpen")
    if sys.platform == "win32":
        from .._platform import _win_modules
        import ntpath

        paths = {path for path in _win_modules()
                 if ntpath.basename(path).lower().startswith("miopen")
                 and path.lower().endswith(".dll")}
        if len(paths) != 1:
            raise RuntimeError("MinerU requires one MIOpen DLL already loaded by ROCm PyTorch")
        path = paths.pop()
    else:
        # Resolve symbols through PyTorch's dependencies instead of searching the system ROCm.
        path = str(Path(torch.__file__).resolve().parent / "lib" / "libtorch_hip.so")
    try:
        lib = C.CDLL(path)
        lib.miopenCreate
    except (OSError, AttributeError) as exc:
        raise RuntimeError("ROCm PyTorch does not expose the MIOpen convolution library") from exc
    pointer, integer, size = C.c_void_p, C.c_int, C.c_size_t
    signatures = {
        "miopenCreate": [C.POINTER(pointer)],
        "miopenSetStream": [pointer, pointer],
        "miopenCreateTensorDescriptor": [C.POINTER(pointer)],
        "miopenSet4dTensorDescriptor": [
            pointer,
            integer,
            integer,
            integer,
            integer,
            integer,
        ],
        "miopenDestroyTensorDescriptor": [pointer],
        "miopenCreateConvolutionDescriptor": [C.POINTER(pointer)],
        "miopenDestroyConvolutionDescriptor": [pointer],
        "miopenInitConvolutionNdDescriptor": [
            pointer,
            integer,
            C.POINTER(integer),
            C.POINTER(integer),
            C.POINTER(integer),
            integer,
        ],
        "miopenSetConvolutionGroupCount": [pointer, integer],
    }
    for direction in ("Forward", "BackwardData"):
        prefix = "miopenConvolution" + direction
        signatures[prefix + "GetSolutionCount"] = [pointer] * 5 + [C.POINTER(size)]
        signatures[prefix + "GetSolution"] = [pointer] * 5 + [
            size,
            C.POINTER(size),
            C.POINTER(Solution),
        ]
        signatures[prefix + "CompileSolution"] = [pointer] * 5 + [C.c_uint64]
        signatures[prefix + "Immediate"] = [pointer] * 9 + [size, C.c_uint64]
    for name, arguments in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, integer
    return lib


def check(status, operation):
    if status:
        raise RuntimeError(f"MIOpen {operation} failed with status {status}")


@functools.cache
def handle(device):
    value = C.c_void_p()
    check(library().miopenCreate(C.byref(value)), "create handle")
    return value


class Convolution:
    def __init__(
        self,
        xshape,
        wshape,
        yshape,
        dtype,
        stride,
        padding,
        dilation,
        groups,
        transpose,
        device,
    ):
        self.lib = library()
        self.handle = handle(device)
        self.tensors = []
        self.conv = C.c_void_p()
        for shape in (xshape, wshape, yshape):
            desc = C.c_void_p()
            check(
                self.lib.miopenCreateTensorDescriptor(C.byref(desc)),
                "create tensor descriptor",
            )
            self.tensors.append(desc)
            check(
                self.lib.miopenSet4dTensorDescriptor(
                    desc, {torch.float32: 1, torch.float16: 0}[dtype], *shape
                ),
                "set tensor descriptor",
            )
        check(
            self.lib.miopenCreateConvolutionDescriptor(C.byref(self.conv)),
            "create convolution descriptor",
        )
        arrays = [(C.c_int * 2)(*value) for value in (padding, stride, dilation)]
        check(
            self.lib.miopenInitConvolutionNdDescriptor(self.conv, 2, *arrays, 0),
            "set convolution descriptor",
        )
        check(self.lib.miopenSetConvolutionGroupCount(self.conv, groups), "set groups")
        self.transpose = transpose
        x, weight, out = self.tensors
        self.descriptors = (
            (x, weight, self.conv, out) if transpose else (weight, x, self.conv, out)
        )
        prefix = "miopenConvolution" + ("BackwardData" if transpose else "Forward")
        count = C.c_size_t()
        check(
            getattr(self.lib, prefix + "GetSolutionCount")(
                self.handle, *self.descriptors, C.byref(count)
            ),
            "count solutions",
        )
        candidates = (Solution * count.value)()
        check(
            getattr(self.lib, prefix + "GetSolution")(
                self.handle, *self.descriptors, count.value, C.byref(count), candidates
            ),
            "query solutions",
        )
        if not count.value:
            raise RuntimeError("MIOpen returned no convolution solution")
        self.solution = candidates[0].solution_id
        self.workspace = candidates[0].workspace_size
        self.compile = getattr(self.lib, prefix + "CompileSolution")
        self.forward = getattr(self.lib, prefix + "Immediate")
        self.compiled = False

    def __del__(self):
        for tensor in self.tensors:
            self.lib.miopenDestroyTensorDescriptor(tensor)
        if self.conv:
            self.lib.miopenDestroyConvolutionDescriptor(self.conv)

    def __call__(self, x, weight, out, workspace):
        check(
            self.lib.miopenSetStream(
                self.handle, torch.cuda.current_stream().cuda_stream
            ),
            "set stream",
        )
        if not self.compiled:
            check(
                self.compile(self.handle, *self.descriptors, self.solution),
                "compile solution",
            )
            self.compiled = True
        inp, filt, output = self.tensors
        arguments = (
            (inp, x.data_ptr(), filt, weight.data_ptr())
            if self.transpose
            else (filt, weight.data_ptr(), inp, x.data_ptr())
        )
        check(
            self.forward(
                self.handle,
                *arguments,
                self.conv,
                output,
                out.data_ptr(),
                workspace.data_ptr(),
                self.workspace,
                self.solution,
            ),
            "execute convolution",
        )


@functools.lru_cache(maxsize=1024)
def convolution(
    xshape, wshape, yshape, dtype, stride, padding, dilation, groups, transpose, device
):
    return Convolution(
        xshape,
        wshape,
        yshape,
        dtype,
        stride,
        padding,
        dilation,
        groups,
        transpose,
        device,
    )

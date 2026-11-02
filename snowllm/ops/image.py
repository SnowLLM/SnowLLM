# SPDX-License-Identifier: Apache-2.0
import contextlib
import math

import torch

from torch.utils._pytree import tree_leaves

from .allocation import empty, active_arena
from .operation import operator
from .._capi import check, lib
from ._common import _p, _stream, drying


def pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


@operator
def evaluate(function, shape, args, kwargs=None, *, dtype=None):
    first = next(
        value for value in tree_leaves(args) if isinstance(value, torch.Tensor)
    )
    out = empty(shape, dtype=dtype or first.dtype, device=first.device)
    if out.device.type != "meta":
        function(*args, **(kwargs or {}), out=out)
    return out


def dense(value):
    if value.is_contiguous():
        return value
    out = empty(value.shape, dtype=value.dtype, device=value.device)
    return copy(out, value)


@operator
def copy(out, value):
    if drying() is None and out.device.type != "meta" and value.device.type != "meta":
        out.copy_(value)
    return out


def add(x, y):
    return evaluate(
        torch.add,
        torch.broadcast_shapes(x.shape, y.shape)
        if isinstance(y, torch.Tensor)
        else x.shape,
        (x, y),
    )


def multiply(x, y):
    return evaluate(
        torch.mul,
        torch.broadcast_shapes(x.shape, y.shape)
        if isinstance(y, torch.Tensor)
        else x.shape,
        (x, y),
    )


def concatenate(values, dim=1):
    shape = list(values[0].shape)
    shape[dim] = sum(value.shape[dim] for value in values)
    return evaluate(torch.cat, shape, (values,), {"dim": dim}, dtype=values[0].dtype)


@operator
def activation(x, kind, scale=None, shift=None):
    if kind is None and scale is None and shift is None:
        return x
    out = empty(x.shape, dtype=x.dtype, device=x.device)
    acts = {
        None: 0,
        "relu": 4,
        "silu": 5,
        "sigmoid": 9,
        "hardsigmoid": 11,
        "paddle_hsigmoid": 7,
        "paddle_hswish": 12,
        "hardswish": 6,
        "gelu": 2,
        "tanh": 10,
    }
    if (
        x.dtype == torch.float16
        and kind in ("relu", "silu", "gelu", "sigmoid", "hardsigmoid")
        and scale is None
    ):
        if out.device.type != "meta":
            getattr(
                torch.ops.aten,
                {
                    "relu": "relu",
                    "silu": "silu",
                    "gelu": "gelu",
                    "sigmoid": "sigmoid",
                    "hardsigmoid": "hardsigmoid",
                }[kind],
            ).out(x, out=out)
        return out
    x = dense(x)
    batch, channels, inner = (
        (x.shape[0], x.shape[1], math.prod(x.shape[2:]))
        if scale is not None
        else (1, 1, x.numel())
    )
    check(
        lib.snowllm_auxiliary_affine_activation(
            _p(x),
            _p(scale),
            _p(shift),
            _p(out),
            batch,
            channels,
            inner,
            {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[x.dtype],
            (kind if isinstance(kind, int) else acts[kind]),
            _stream(),
        ),
        "image_activation",
    )
    return out


@operator
def convolution(
    x,
    weight,
    bias=None,
    *,
    stride=1,
    padding=0,
    groups=1,
    transpose=False,
    dilation=1,
    output_padding=0,
    scale=None,
    shift=None,
    act=None,
):
    stride, padding, dilation, output_padding = (
        pair(stride),
        pair(padding),
        pair(dilation),
        pair(output_padding),
    )
    kernel = weight.shape[2:]
    spatial = tuple(
        (size - 1) * step - 2 * pad + k
        if transpose
        else (size + 2 * pad - k) // step + 1
        for size, step, pad, k in zip(
            x.shape[2:],
            stride,
            padding,
            tuple((k - 1) * d + 1 for k, d in zip(kernel, dilation)),
        )
    )
    if transpose:
        spatial = tuple(n + p for n, p in zip(spatial, output_padding))
    channels = weight.shape[1] * groups if transpose else weight.shape[0]
    acts = {None: 0, "relu": 4, "silu": 5, "gelu": 2, "sigmoid": 9}
    if (
        x.dtype == torch.float32
        and dilation == (1, 1)
        and not transpose
        and act in acts
    ):
        if groups == 1 and kernel == (1, 1) and stride == (1, 1) and padding == (0, 0):
            return pointwise_f32(x, weight, bias, scale, shift, acts[act])
        if (
            groups == x.shape[1] == channels
            and kernel[0] == kernel[1]
            and kernel[0] in (3, 5)
            and stride[0] == stride[1]
            and stride[0] in (1, 2)
            and padding == (kernel[0] // 2,) * 2
        ):
            return depthwise_conv(x, weight, bias, scale, shift, stride[0], acts[act])
    out = empty((x.shape[0], channels, *spatial), dtype=x.dtype, device=x.device)
    from .miopen import convolution as plan_convolution

    x = dense(x)
    plan = plan_convolution(
        tuple(x.shape),
        tuple(weight.shape),
        tuple(out.shape),
        x.dtype,
        stride,
        padding,
        dilation,
        groups,
        transpose,
        torch.cuda.current_device(),
    )
    arena = active_arena()
    with arena.frame() if arena is not None else contextlib.nullcontext():
        workspace = empty((plan.workspace,), dtype=torch.uint8, device=x.device)
        if out.device.type != "meta":
            plan(x, weight, out, workspace)
    if bias is not None:
        out = add(out, bias[None, :, None, None])
    return activation(out, act, scale, shift)


@operator
def batch_norm(x, weight, bias, mean, variance, eps=1e-5):
    out = empty(x.shape, dtype=x.dtype, device=x.device)
    arena = active_arena()
    with arena.frame() if arena is not None else contextlib.nullcontext():
        scratch = [
            empty((x.shape[1],), dtype=torch.float32, device=x.device) for _ in range(2)
        ]
        if out.device.type != "meta":
            torch.ops.aten._native_batch_norm_legit_no_training.out(
                x,
                weight,
                bias,
                mean,
                variance,
                0.1,
                eps,
                out0=out,
                out1=scratch[0],
                out2=scratch[1],
            )
    return out


@operator
def layer_norm(x, weight, bias, eps=1e-6):
    out = empty(x.shape, dtype=x.dtype, device=x.device)
    shape = (*x.shape[:-1], 1)
    arena = active_arena()
    with arena.frame() if arena is not None else contextlib.nullcontext():
        stats = [empty(shape, dtype=torch.float32, device=x.device) for _ in range(2)]
        if out.device.type != "meta":
            torch.ops.aten.native_layer_norm.out(
                x,
                (x.shape[-1],),
                weight,
                bias,
                eps,
                out0=out,
                out1=stats[0],
                out2=stats[1],
            )
    return out


@operator
def linear(x, weight, bias=None):
    x = dense(x)
    out = empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)
    if out.device.type != "meta":
        if bias is None:
            torch.mm(
                x.view(-1, x.shape[-1]), weight.T, out=out.view(-1, weight.shape[0])
            )
        else:
            torch.addmm(
                bias,
                x.view(-1, x.shape[-1]),
                weight.T,
                out=out.view(-1, weight.shape[0]),
            )
    return out


def pad(x, pads, value=0):
    shape = list(x.shape)
    for i in range(len(pads) // 2):
        shape[-1 - i] += pads[2 * i] + pads[2 * i + 1]
    return evaluate(torch.ops.aten.constant_pad_nd.out, shape, (x, pads, value))


def interpolate(x, *, scale=None, size=None):
    size = (
        tuple(size)
        if size is not None
        else tuple(int(value * scale) for value in x.shape[2:])
    )
    return evaluate(
        torch.ops.aten.upsample_nearest2d.out, (*x.shape[:2], *size), (x, size)
    )


@operator
def pool(x, kernel, stride=None, *, padding=0, ceil=False, average=False):
    kernel, stride, padding = pair(kernel), pair(stride or kernel), pair(padding)
    sizes = []
    for size, k, s, p in zip(x.shape[2:], kernel, stride, padding):
        n = (size + 2 * p - k + (s - 1 if ceil else 0)) // s + 1
        if ceil and (n - 1) * s >= size + p:
            n -= 1
        sizes.append(n)
    out = empty((*x.shape[:2], *sizes), dtype=x.dtype, device=x.device)
    if average:
        if out.device.type != "meta":
            torch.ops.aten.avg_pool2d.out(
                x, kernel, stride, padding, ceil, True, None, out=out
            )
    else:
        indices = empty(out.shape, dtype=torch.int64, device=x.device)
        if out.device.type != "meta":
            torch.ops.aten.max_pool2d_with_indices.out(
                x, kernel, stride, padding, (1, 1), ceil, out=out, indices=indices
            )
    return out


def adaptive_pool(x, size=(1, 1)):
    if pair(size) == (1, 1):
        return evaluate(
            torch.mean, (*x.shape[:2], 1, 1), (x,), {"dim": (-2, -1), "keepdim": True}
        )
    return evaluate(
        torch.ops.aten._adaptive_avg_pool2d.out,
        (*x.shape[:2], *pair(size)),
        (x, pair(size)),
    )


@operator
def matmul(x, y):
    batch = tuple(torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]))
    shape = (*batch, x.shape[-2], y.shape[-1])
    out = empty(shape, dtype=x.dtype, device=x.device)
    if not batch:
        if out.device.type != "meta":
            torch.mm(x, y, out=out)
    elif y.ndim == 2:
        value = dense(x)
        if out.device.type != "meta":
            torch.mm(value.view(-1, x.shape[-1]), y, out=out.view(-1, y.shape[-1]))
    else:

        def flatten(value):
            value = value.expand(*batch, *value.shape[-2:])
            target = (math.prod(batch), *value.shape[-2:])
            try:
                return value.view(target)
            except RuntimeError:
                return dense(value).view(target)

        first, second = flatten(x), flatten(y)
        if out.device.type != "meta":
            torch.bmm(first, second, out=out.view(math.prod(batch), *shape[-2:]))
    return out


def softmax(x, dim=-1):
    return evaluate(torch.ops.aten._softmax.out, x.shape, (x, dim, False))


def subtract(x, y):
    return evaluate(
        torch.sub,
        torch.broadcast_shapes(x.shape, y.shape)
        if isinstance(y, torch.Tensor)
        else x.shape,
        (x, y),
    )


def divide(x, y):
    return evaluate(
        torch.div,
        torch.broadcast_shapes(x.shape, y.shape)
        if isinstance(y, torch.Tensor)
        else x.shape,
        (x, y),
    )


def unary(function, x, **kwargs):
    return evaluate(function, x.shape, (x,), kwargs)


def clamp(x, minimum=None, maximum=None):
    return unary(torch.clamp, x, min=minimum, max=maximum)


def stack(values, dim=0):
    return concatenate([x.unsqueeze(dim) for x in values], dim)


def arange(size, *, device, dtype=torch.int64):
    out = empty((size,), dtype=dtype, device=device)
    if out.device.type != "meta":
        torch.arange(size, out=out)
    return out


def cast(x, dtype):
    return copy(empty(x.shape, dtype=dtype, device=x.device), x)


def full(shape, value, *, device, dtype=torch.float32):
    out = empty(shape, dtype=dtype, device=device)
    if out.device.type != "meta":
        out.fill_(value)
    return out


def embedding(weight, indices):
    indices = dense(indices)
    out = evaluate(
        torch.index_select,
        (indices.numel(), weight.shape[1]),
        (weight, 0, indices.view(-1)),
    )
    return out.view(*indices.shape, weight.shape[1])


def gather(x, indices, dim=1):
    return evaluate(torch.gather, indices.shape, (x, dim, indices), dtype=x.dtype)


def where(condition, first, second):
    shape = torch.broadcast_shapes(
        condition.shape,
        first.shape if isinstance(first, torch.Tensor) else (),
        second.shape if isinstance(second, torch.Tensor) else (),
    )
    dtype = first.dtype if isinstance(first, torch.Tensor) else second.dtype
    return evaluate(torch.where, shape, (condition, first, second), dtype=dtype)


def compare(function, x, y):
    shape = (
        torch.broadcast_shapes(x.shape, y.shape)
        if isinstance(y, torch.Tensor)
        else x.shape
    )
    return evaluate(function, shape, (x, y), dtype=torch.bool)


def reduce(function, x, dim, *, keepdim=False, dtype=None):
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    dims = tuple(d % x.ndim for d in dims)
    shape = (
        tuple(1 if i in dims else n for i, n in enumerate(x.shape))
        if keepdim
        else tuple(n for i, n in enumerate(x.shape) if i not in dims)
    )
    return evaluate(
        function, shape, (x,), {"dim": dim, "keepdim": keepdim}, dtype=dtype
    )


@operator
def topk(x, count):
    shape = (*x.shape[:-1], count)
    values = empty(shape, dtype=x.dtype, device=x.device)
    indices = empty(shape, dtype=torch.int64, device=x.device)
    if values.device.type != "meta":
        torch.topk(x, count, dim=-1, out=(values, indices))
    return values, indices


@operator
def sort(x, *, descending=False):
    values = empty(x.shape, dtype=x.dtype, device=x.device)
    indices = empty(x.shape, dtype=torch.int64, device=x.device)
    if values.device.type != "meta":
        torch.sort(x, dim=-1, descending=descending, out=(values, indices))
    return values, indices


def grid_sample(x, grid):
    return evaluate(
        torch.ops.aten.grid_sampler_2d.out,
        (x.shape[0], x.shape[1], grid.shape[1], grid.shape[2]),
        (x, grid, 0, 0, False),
    )


@operator
def linear_half(x, weight, bias=None):
    x = dense(x)
    rows = x.numel() // x.shape[-1]
    columns, contraction = weight.shape
    out = empty((*x.shape[:-1], columns), dtype=x.dtype, device=x.device)
    arena = active_arena()
    with arena.frame() if arena is not None else contextlib.nullcontext():
        workspace = empty(
            (16 * rows * columns if columns <= 1024 else 0,),
            dtype=torch.uint8,
            device=x.device,
        )
        check(
            lib.snowllm_auxiliary_gemm_half(
                _p(x),
                _p(weight),
                _p(bias),
                _p(out),
                _p(workspace),
                rows,
                columns,
                contraction,
                _stream(),
            ),
            "image_linear_half",
        )
    return out


@operator
def index_copy(out, dim, indices, value):
    if (
        out.device.type != "meta"
        and value.device.type != "meta"
        and indices.device.type != "meta"
    ):
        out.index_copy_(dim, indices, value)
    return out


def attention_half(q, k, v, mask=None):
    q, k, v = [cast(value, torch.float32) for value in (q, k, v)]
    scores = matmul(q, k.transpose(-1, -2))
    if mask is not None:
        scores = add(scores, cast(mask, torch.float32))
    return cast(matmul(softmax(scores), v), torch.float16)


def bilinear(x, size):
    return evaluate(
        torch.ops.aten.upsample_bilinear2d.out,
        (*x.shape[:2], *size),
        (x, tuple(size), False),
    )


def prepare_input(x, dtype, admit=None):
    if not x.is_cuda or x.dtype != dtype:
        size = x.numel() * torch.empty((), dtype=dtype, device="meta").element_size()
        if admit:
            admit((size + 511) // 512 * 512)
        x = x.to(device="cuda", dtype=dtype)
    return x


@operator
def pointwise_f32(x, weight, bias=None, scale=None, shift=None, act=0, config=0):
    x = dense(x)
    out = empty(
        (x.shape[0], weight.shape[0], *x.shape[2:]), dtype=x.dtype, device=x.device
    )
    check(
        lib.snowllm_auxiliary_pointwise_f32(
            _p(x),
            _p(weight),
            _p(bias),
            _p(scale),
            _p(shift),
            _p(out),
            x.shape[0],
            x.shape[1],
            math.prod(x.shape[2:]),
            weight.shape[0],
            act,
            config,
            _stream(),
        ),
        "image_pointwise",
    )
    return out


@operator
def depthwise_conv(x, weight, bias=None, scale=None, shift=None, stride=1, act=0):
    x = dense(x)
    height, width = ((n - 1) // stride + 1 for n in x.shape[2:])
    out = empty((x.shape[0], x.shape[1], height, width), dtype=x.dtype, device=x.device)
    check(
        lib.snowllm_auxiliary_depthwise(
            _p(x),
            _p(weight),
            _p(bias),
            _p(scale),
            _p(shift),
            _p(out),
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            weight.shape[-1],
            stride,
            act,
            {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[x.dtype],
            _stream(),
        ),
        "image_depthwise",
    )
    return out


def allocate(shape, *, dtype=torch.float32, device="cuda", admit=None, fill=None):
    size = math.prod(shape) * torch.empty((), dtype=dtype, device="meta").element_size()
    if admit:
        admit((size + 511) // 512 * 512)
    out = torch.empty(shape, dtype=dtype, device=device)
    if fill is not None:
        out.fill_(fill)
    return out

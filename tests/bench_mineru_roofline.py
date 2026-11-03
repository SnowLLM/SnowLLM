# SPDX-License-Identifier: Apache-2.0
import argparse
from collections import Counter, defaultdict
import contextlib
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time
from types import SimpleNamespace
from unittest.mock import patch


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, SimpleNamespace):
        yield from tensors(vars(value))


def volume(value):
    found = set()
    total = 0
    for item in tensors(value):
        key = item.data_ptr(), tuple(item.shape), tuple(item.stride()), item.dtype
        if key not in found:
            found.add(key)
            total += item.numel() * item.element_size()
    return total


def weights(model):
    import torch
    seen, stores = set(), {}

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                storage = value.untyped_storage()
                stores[storage.data_ptr()] = str(value.dtype), storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, '__dict__'):
            for key, item in vars(value).items():
                if not key.startswith('_') and key not in ('runners', 'source'):
                    visit(item)
    visit(model)
    result = Counter()
    for dtype, size in stores.values():
        result[dtype] += size
    return dict(result)


class Audit:
    def __init__(self):
        self.depth = 0
        self.rows = defaultdict(Counter)

    @contextlib.contextmanager
    def install(self):
        from snowllm.ops import image as op
        from snowllm.models.vision import VisionModel

        def instrument(name, original):
            def call(*args, **kwargs):
                outer = self.depth == 0
                self.depth += 1
                try:
                    result = original(*args, **kwargs)
                finally:
                    self.depth -= 1
                if not outer:
                    return result
                first = next(tensors(args), None)
                if first is None:
                    return result
                if name == 'activation' and args[1] is None and all(v is None for v in args[2:]):
                    return result
                flops = 0
                out = next(tensors(result), None)
                if name in ('convolution', 'pointwise_f32', 'depthwise_conv'):
                    x, weight = args[:2]
                    transpose = kwargs.get('transpose', False)
                    flops = (2 * x.numel() * weight.shape[1] * math.prod(weight.shape[2:])
                             if transpose else 2 * out.numel() * weight.shape[1] * math.prod(weight.shape[2:]))
                elif name in ('linear', 'linear_half', 'matmul'):
                    flops = 2 * out.numel() * args[0].shape[-1]
                traffic = volume((args, kwargs)) + volume(result)
                if name == 'index_copy':
                    traffic = 2 * volume(args[3]) + volume(args[2])
                if name == 'copy':
                    traffic = volume(args[0]) + volume(args[1])
                key = name + '/' + str(first.dtype)
                row = self.rows[key]
                row.update(calls=1, flops=flops, logical_bytes=traffic)
                return result
            return call

        names = ('convolution', 'pointwise_f32', 'depthwise_conv', 'linear', 'linear_half',
                 'matmul', 'evaluate', 'activation', 'copy', 'batch_norm', 'layer_norm',
                 'pool', 'topk', 'sort', 'index_copy')
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(VisionModel, 'execute_plan',
                lambda self, key, rows, walk, **kw: walk()))
            for name in names:
                stack.enter_context(patch.object(op, name, instrument(name, getattr(op, name))))
            yield


def measure(function, repeats=9, warmups=5):
    import torch
    for _ in range(warmups):
        value = function()
    torch.cuda.synchronize()
    wall, device = [], []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        before = time.perf_counter()
        start.record()
        value = function()
        stop.record()
        stop.synchronize()
        wall.append((time.perf_counter() - before) * 1000)
        device.append(start.elapsed_time(stop))
    return dict(wall_ms=statistics.median(wall), device_ms=statistics.median(device),
                wall_samples_ms=wall, device_samples_ms=device), value


def checkpoint(root):
    totals = defaultdict(Counter)
    with (root / 'model.safetensors').open('rb') as file:
        header = json.loads(file.read(struct.unpack('<Q', file.read(8))[0]))
    for name, info in header.items():
        if name == '__metadata__':
            continue
        group = 'vision' if 'visual.' in name else 'language'
        totals[group][info['dtype']] += info['data_offsets'][1] - info['data_offsets'][0]
    return {key: dict(value) for key, value in totals.items()}


def profile(args):
    import ctypes
    import torch
    from snowllm import ops
    from snowllm._capi import GEO_MINERU_12B
    from bench_mineru_ablation import build_case

    torch.set_num_threads(2)
    torch.manual_seed(127)
    ops.select_geometry(GEO_MINERU_12B)
    roctx = ctypes.CDLL('/opt/rocm/lib/librocprofiler-sdk-roctx.so')
    roctx.roctxProfilerPause(0)
    for name in ('layout_b1', 'layout_b8', 'ocr_det', 'ocr_rec', 'seal_det',
                 'formula_b1', 'formula_b8', 'table_cls', 'table_unet', 'table_slanet'):
        gc.collect()
        torch.cuda.empty_cache()
        with contextlib.ExitStack() as stack, torch.inference_mode():
            function, cache, models = build_case(name, 'baseline', args, stack)
            if name.startswith('formula'):
                model = models[0]
                captured = []
                original = type(model).__call__
                def capture(instance, x):
                    captured.append(x.detach().to(device='cuda', dtype=torch.float16))
                    return original(instance, x)
                with patch.object(type(model), '__call__', capture):
                    function()
                x = captured[0]
                function = lambda: model(x)
            for _ in range(5):
                value = function()
            torch.cuda.synchronize()
            roctx.roctxProfilerResume(0)
            roctx.roctxRangePushA(name.encode())
            value = function()
            torch.cuda.synchronize()
            roctx.roctxRangePop()
            roctx.roctxProfilerPause(0)
            print('PROFILE', name, flush=True)
        del function, cache, models, value
        if name.startswith('formula'):
            del model, x, captured


def main(args):
    import torch
    from snowllm import ops
    from snowllm._capi import GEO_MINERU_12B
    from bench_mineru_ablation import build_case

    torch.set_num_threads(2)
    torch.manual_seed(127)
    torch.backends.cuda.matmul.allow_tf32 = False
    ops.select_geometry(GEO_MINERU_12B)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        protocol=dict(fma_flops=2, warmups=5, repeats=9, input_resident=True,
            flops='convolution and matrix products only; excludes elementwise/SFU',
            bytes='sum of logical operator inputs/outputs, excludes internal scratch and cache effects',
            timing='fresh wall and device event medians; CPU preprocessing, RPC and loading excluded'),
        checkpoint=checkpoint(args.model), ceilings={}, models=[])
    import os
    report['library_sha256'] = hashlib.sha256(Path(os.environ['SNOWLLM_LIB']).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    for size in (64 << 20, 256 << 20):
        x = torch.ones(size // 4, device='cuda')
        y = torch.empty_like(x)
        timing, _ = measure(lambda: y.copy_(x), repeats=15)
        timing.update(bytes=2 * size, gbps=2 * size / (timing['device_ms'] * 1e6))
        report['ceilings'][f'copy_{size}'] = timing
        print('COPY', size, timing['gbps'], flush=True)
        del x, y
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for size in (2048, 4096):
            a = torch.randn(size, size, dtype=dtype, device='cuda')
            b = torch.randn_like(a)
            c = torch.empty_like(a)
            timing, _ = measure(lambda: torch.mm(a, b, out=c))
            timing.update(tflops=2 * size ** 3 / (timing['device_ms'] * 1e9))
            report['ceilings'][f'gemm_{dtype}_{size}'] = timing
            print('GEMM', dtype, size, timing['tflops'], flush=True)
            del a, b, c
    save()

    cases = ['layout_b1', 'layout_b8', 'ocr_det', 'ocr_rec', 'seal_det',
             'formula_b1', 'formula_b8', 'table_cls', 'table_unet', 'table_slanet']
    for name in cases:
        gc.collect()
        torch.cuda.empty_cache()
        with contextlib.ExitStack() as stack, torch.inference_mode():
            function, cache, models = build_case(name, 'baseline', args, stack)
            persistent = weights(models[0])
            if name.startswith('formula'):
                model = models[0]
                captured = []
                original = type(model).__call__
                def capture(instance, x):
                    captured.append(x.detach().to(device='cuda', dtype=torch.float16))
                    return original(instance, x)
                with patch.object(type(model), '__call__', capture):
                    function()
                x = captured[0]
                function = lambda: model(x)
            timing, value = measure(function)
            shapes = [list(item.shape) for item in tensors(value)]
            graph_status = cache.status()
            cache.enabled = False
            audit = Audit()
            with audit.install():
                audited = function()
            torch.cuda.synchronize()
            if name.startswith('formula'):
                assert torch.equal(value, audited)
            else:
                for expected, actual in zip(tensors(value), tensors(audited)):
                    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)
            flops = sum(row['flops'] for row in audit.rows.values())
            traffic = sum(row['logical_bytes'] for row in audit.rows.values())
            row = dict(case=name, **timing, persistent_bytes_by_dtype=persistent,
                output_shapes=shapes, flops=flops, logical_bytes=traffic,
                intensity=flops / traffic, tflops=flops / (timing['wall_ms'] * 1e9),
                logical_gbps=traffic / (timing['wall_ms'] * 1e6),
                operators={k: dict(v) for k, v in audit.rows.items()}, graphs=graph_status)
            report['models'].append(row)
            save()
            print('MODEL', name, {k: round(row[k], 3) for k in
                  ('wall_ms', 'tflops', 'logical_gbps', 'intensity')}, persistent, flush=True)
        del function, cache, models, value, audited, audit
        if name.startswith('formula'):
            del model, x, captured
    report['completed'] = True
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    (profile if args.profile else main)(args)

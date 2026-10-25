# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
from unittest.mock import patch

import torch

import _harness
from snowllm import host_alloc, ops
from snowllm._platform import gpu_mem_info, hip_runtimes
from snowllm.engine.graphs import GraphBudget


class PointerAttributes(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('device', ctypes.c_int),
                ('devicePointer', ctypes.c_void_p), ('hostPointer', ctypes.c_void_p),
                ('isManaged', ctypes.c_int), ('allocationFlags', ctypes.c_uint)]


def main():
    ck = _harness.Checks()
    runtimes = hip_runtimes()
    if len(runtimes) != 1:
        raise AssertionError('pointer audit requires one HIP runtime')
    hip = ctypes.CDLL(runtimes.pop())
    hip.hipPointerGetAttributes.argtypes = [ctypes.POINTER(PointerAttributes), ctypes.c_void_p]
    hip.hipPointerGetAttributes.restype = ctypes.c_int
    def memory_type(tensor):
        attributes = PointerAttributes()
        status = hip.hipPointerGetAttributes(ctypes.byref(attributes), tensor.data_ptr())
        if status:
            raise AssertionError(f'hipPointerGetAttributes failed: {status}')
        return attributes.type
    control = torch.empty(1024, device='cuda')
    ck('HIP identifies the default PyTorch allocation as device memory', memory_type(control) == 2)
    del control
    torch.cuda.empty_cache()
    host_alloc.enable()
    from snowllm import capacity
    from snowllm._capi import lib
    baseline = host_alloc.allocated_bytes()
    original_budget = capacity.budget
    capacity.configure(capacity.MemoryBudget(max_host_mem_gib=1))
    rejected = lib.snowllm_host_alloc(2 << 30, 0, None)
    ck('native host allocator rejects segments beyond the host ceiling', not rejected)
    ck('rejected segments do not consume the host budget', host_alloc.allocated_bytes() == baseline)
    try:
        oversized = torch.empty(2 << 30, dtype=torch.uint8, device='cuda')
    except (torch.OutOfMemoryError, RuntimeError):
        ck('PyTorch rejects an oversized host tensor without device fallback', True)
    else:
        ck('PyTorch rejects an oversized host tensor without device fallback', False)
        del oversized
    lease_before = capacity._host_reserved
    capacity.reserve_host(1 << 30)
    try:
        ck('pinned staging and host tensor segments share one budget',
           not lib.snowllm_host_alloc(1, 0, None))
    finally:
        capacity.release_host(1 << 30)
        capacity.configure(original_budget)
    ck('staging reservations are released', capacity._host_reserved == lease_before)
    x = torch.arange(1024, dtype=torch.float32, device='cuda')
    ck('torch factories allocate mapped host storage', x.is_cuda and ops.host_mapped(x))
    ck('HIP independently identifies PyTorch storage as host memory', memory_type(x) == 1)
    constructed = [torch.tensor([1, 2], device='cuda'),
                   torch.as_tensor([1., 2.], device='cuda'),
                   torch.tensor(3., device='cuda')]
    ck('Python tensor constructors allocate host memory', all(memory_type(t) == 1 for t in constructed))
    converted = torch.arange(1024, dtype=torch.float32).to('cuda')
    cloned = converted.to(torch.float64).clone()
    ck('CPU-to-GPU transfer and dtype conversion allocate host memory',
       memory_type(converted) == memory_type(cloned) == 1)
    before = host_alloc.allocated_bytes()
    large = torch.empty(64 << 20, dtype=torch.uint8, device='cuda')
    ck('large torch.empty requests reach the hipHostMalloc callback',
       memory_type(large) == 1 and host_alloc.allocated_bytes() - before >= 64 << 20)
    del large
    gc.collect()
    before = host_alloc.allocated_bytes()
    large = torch.empty(64 << 20, dtype=torch.uint8, device='cuda')
    ck('PyTorch reuses cached host segments',
       memory_type(large) == 1 and host_alloc.allocated_bytes() == before)
    ck('views retain mapped host identity', ops.host_mapped(x[13:]))
    raw = ops.empty_bytes(8 << 20)
    ck('SnowLLM buffers use the same host allocator', ops.host_mapped(raw))
    expected = torch.arange(1024, dtype=torch.float32) * 2 + 1
    with ThreadPoolExecutor(max_workers=1, initializer=host_alloc.enter_thread) as worker:
        copied = worker.submit(lambda: x * 2 + 1).result()
        ck('worker threads use mapped host allocation', ops.host_mapped(copied))
        ck('HIP identifies worker results as host memory', memory_type(copied) == 1)
        ck.close('worker GPU computation is correct', copied.cpu(), expected, 1e-6)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    pool = host_alloc.graph_pool()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        with torch.cuda.graph(graph, pool=pool.id):
            output = x * 2 + 1
    torch.cuda.current_stream().wait_stream(side)
    ck('graph outputs use mapped host storage', ops.host_mapped(output))
    ck('HIP identifies graph allocations as host memory', memory_type(output) == 1)
    noise = [torch.full_like(x, 123) for _ in range(8)]
    x.add_(1)
    graph.replay()
    ck.close('graph replay survives ordinary pool reuse', output.cpu(), expected + 2, 1e-6)
    with patch('snowllm._platform.host_total', return_value=256 << 30), \
         patch('snowllm._platform.host_available', return_value=128 << 30), \
         patch.object(torch.cuda, 'mem_get_info', side_effect=AssertionError('VRAM used as RAM budget')):
        free, total = gpu_mem_info()
    ck('capacity uses RAM instead of the VRAM carve-out', total == int(.9 * (256 << 30)) and free >= 128 << 30)
    ck('host allocation accounting includes native segments', host_alloc.allocated_bytes() >= raw.nbytes)
    budget = GraphBudget()
    with patch('snowllm.engine.graphs.device_mem_info', return_value=(64 << 20, 128 << 20)):
        ck('host RAM and graph-driver VRAM have separate budgets', budget.room(1., 0))
    with patch('snowllm.engine.graphs.device_mem_info', return_value=(48 << 20, 128 << 20)):
        budget.charge()
    with patch('snowllm.engine.graphs.device_mem_info', return_value=(8 << 20, 128 << 20)):
        ck('measured driver VRAM limits additional graph capture',
           budget.driver_measured == 16 << 20 and not budget.room(1., 1))
    del copied, output, graph, pool, noise, raw, x, converted, cloned, large, constructed
    gc.collect()
    torch.cuda.synchronize()
    return ck.done()


if __name__ == '__main__':
    raise SystemExit(main())

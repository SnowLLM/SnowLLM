# SPDX-License-Identifier: Apache-2.0
import contextlib
import ctypes
import dataclasses
import functools
import os
import struct
import tempfile
import threading
import weakref
from contextvars import ContextVar
from pathlib import Path

import torch

from .. import _hip_memory
from .._capi import SnowLLMError, check, lib
from ..checkpoint.expert_cache import expert_cache_directories
from ..capacity import release_host, reserve_host
from ._common import _p, _stream, drying, empty_bytes, host_memory, zero_bytes
from .inference import _weights
from .moe import KQuantExpertWeight, LowBitExpertWeight
from .operation import operator, workspace_capacity

_build = ContextVar('disk_expert_build', default=None)


@contextlib.contextmanager
def building(store):
    token = _build.set(store)
    try:
        yield
    finally:
        _build.reset(token)


@contextlib.contextmanager
def planning_copy():
    store = _build.get()
    with building(Store(*store.options) if isinstance(store, Store) else store):
        yield


def align(n, boundary=512):
    return -(-n // boundary) * boundary


@dataclasses.dataclass(frozen=True)
class Plane:
    width: int
    groups: int = 1
    padding: int = 0
    tail: int = 0

    def size(self, count):
        return self.groups * self.pitch(count) + self.tail

    def pitch(self, count):
        return self.width * count + self.padding


@dataclasses.dataclass(frozen=True)
class Layout:
    experts: int
    gate: tuple
    down: tuple
    prefix: int = 0
    vq: bool = False
    original: int = 0
    batch_rows: int = 0

    def planes(self, mode):
        return self.down if mode == 'down' else self.gate + self.down

    def size(self, mode, count):
        return self.prefix + sum(p.size(count) for p in self.planes(mode))

    def record(self, mode):
        return sum(align(p.width * p.groups) for p in self.planes(mode))


def weight_plane(tensor, count, kind, index, fmt=0, columns=0, stride=0):
    values = (ctypes.c_int64 * 4)()
    if lib.snowllm_disk_weight_plane(kind, fmt, index, tensor.nbytes, count, columns, stride, values):
        raise SnowLLMError('invalid disk expert weight layout')
    return Plane(*values)


def standard_planes(weight, scale, count, columns):
    buffers = tensors(weight, scale)
    if isinstance(weight, LowBitExpertWeight):
        return tuple(weight_plane(t, count, 1, i, weight.fmt, columns, weight.stride)
                     for i, t in enumerate(buffers))
    kind = 2 if isinstance(weight, KQuantExpertWeight) else 0
    return tuple(weight_plane(t, count, kind, i) for i, t in enumerate(buffers))


def tensors(weight, scale):
    if isinstance(weight, LowBitExpertWeight):
        return [t for t in (weight.p0, weight.p1, weight.p2) if t is not None]
    if isinstance(weight, KQuantExpertWeight):
        return [weight.quant, weight.meta]
    return [t for t in (weight, scale) if t is not None]


def compact_weight(original, buffers, capacity, role):
    if isinstance(original, LowBitExpertWeight):
        stride = getattr(lib, 'snowllm_moe_lowbit_' + role + '_stride')(capacity)
        return LowBitExpertWeight(*(buffers + [None])[:3], stride, original.fmt)
    if isinstance(original, KQuantExpertWeight):
        return KQuantExpertWeight(*buffers, original.fmt)
    return buffers[0].view(original.dtype)


def cache_layout(specs, mode, count):
    offsets, size = {}, 0
    for layer, layout in specs.items():
        sections = {}
        sizes = ({'blob': [layout.size(mode, count)]} if layout.vq else
                 dict(down=[p.size(count) for p in layout.down]))
        if not layout.vq and mode == 'all':
            sizes['gate_up'] = [p.size(count) for p in layout.gate]
        for role, lengths in sizes.items():
            sections[role] = []
            for length in lengths:
                at = align(size)
                sections[role].append((at, length))
                size = at + length
        offsets[layer] = sections
    return offsets, size


class Catalog:
    def __init__(self):
        self.layers = {}
        self.load_sizes = {}

    def add(self, layer, layout):
        self.layers[layer] = layout


def load_buffer(layer, role, key, size):
    build = _build.get()
    if isinstance(build, Catalog):
        build.load_sizes.setdefault(layer, {})[role, key] = size
    elif (isinstance(build, Store) and build.active and layer in build.unpacked
          and build.dry == (drying() is not None)):
        region = build.load_offsets.get(layer, {}).get((role, key))
        if region is not None:
            at, planned = region
            if size != planned:
                raise SnowLLMError('disk expert packing buffer changed after planning')
            if build.load_slab is None:
                with host_memory(False):
                    build.load_slab = empty_bytes(build.load_bytes)
            return build.load_slab.narrow(0, at, size)
    return None


def gguf_load_buffer(name, size):
    if _build.get() is None:
        return None
    parts = name.split('.')
    if len(parts) == 4 and parts[0] == 'blk' and parts[1].isdigit() and parts[3] == 'weight':
        role = {'ffn_gate_exps': 'gate_up', 'ffn_up_exps': 'gate_up',
                'ffn_gate_up_exps': 'gate_up', 'ffn_down_exps': 'down'}.get(parts[2])
        from ..models.placement import current_placement
        layer = int(parts[1])
        if role and current_placement() in (('experts', layer), ('experts.' + role, layer)):
            return load_buffer(layer, role, name, size)
    return None


def packed_buffer(role, key, count, size):
    if _build.get() is not None:
        from ..models.placement import current_placement
        scope = current_placement()
        if scope is not None and scope[0] in ('experts', 'experts.' + role):
            from .._capi import build_geometry
            if count == build_geometry().moe_num_experts:
                buf = load_buffer(scope[1], role, key, size)
                if buf is not None:
                    return buf
    return empty_bytes(size)


@functools.cache
def _copy2d():
    fn = _hip_memory._runtime().hipMemcpy2DAsync
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


class Records:
    def __init__(self, layout, mode, output, count):
        self.planes = layout.planes(mode)
        self.size = layout.record(mode)
        self.output, self.count = output, count

    def pack(self, stage, slot, stream, source, count):
        host = stage.data_ptr()
        for p, t in zip(self.planes, source, strict=True):
            device = t.data_ptr() + slot * p.width
            _hip_memory._check(_copy2d()(host, p.width, device, p.pitch(count), p.width, p.groups,
                                       2, stream), 'pack disk expert')
            host += align(p.width * p.groups)

    def ranges(self, expert):
        yield self.file, expert * self.size, 0, self.size

    def destinations(self):
        at = 0
        for p, t in zip(self.planes, self.output, strict=True):
            yield t.data_ptr(), at, p.width, p.pitch(self.count), p.groups
            at += align(p.width * p.groups)

    patches = ()
    patch_value = 0


def file_handle(file):
    if os.name == 'nt':
        import msvcrt
        return msvcrt.get_osfhandle(file.fileno())
    return file.fileno()


def array(dtype, values):
    return (dtype * len(values))(*values)


class Store:
    def __init__(self, catalog, dmap, topk, mode, count, target, batch_limit=None):
        self.host_layers = [i for i in catalog.layers if dmap and any(
            dmap.on_host(g, i) for g in ('experts.gate_up', 'experts.down'))]
        self.specs = {i: layout for i, layout in catalog.layers.items()
                      if i not in self.host_layers and (dmap is None or dmap.on_disk(i))}
        self.active = bool(self.specs)
        self.host_bytes = dmap.host_bytes if dmap else 0
        self.offload_target = target
        self.disk_cache = None
        self.options = (catalog, dmap, topk, mode, count, target, batch_limit)
        self.topk = topk
        if not catalog.layers:
            raise SnowLLMError('this model has no routed experts to offload')
        self.experts = next(iter(catalog.layers.values())).experts
        if not 1 <= topk <= self.experts or mode not in ('all', 'down', 'auto'):
            raise SnowLLMError('invalid disk expert routing or cache mode')
        if count > self.experts:
            raise SnowLLMError(f'expert cache capacity exceeds {self.experts}')
        if count and count < topk:
            raise SnowLLMError(f'this disk expert path needs at least top-k ({topk}) cache slots')
        if any(s.vq for s in self.specs.values()) and mode == 'down':
            raise SnowLLMError('VQ disk offload requires all or auto mode')
        from ..defaults import DEFAULT_PREFILL_CHUNK
        if batch_limit is None:
            batch_limit = DEFAULT_PREFILL_CHUNK
        def overflow(m, c):
            sections, size = {}, 0
            if m == 'all' and hasattr(lib, 'snowllm_disk_proxy_overflow'):
                for i, layout in self.specs.items():
                    if layout.batch_rows and c < layout.experts:
                        slots = layout.experts - c
                        offsets, length = cache_layout({i: layout}, m, slots)
                        sections[i] = (slots, offsets[i])
                        size = max(size, length)
            return sections, size
        def savings(m, c):
            if c == self.experts:
                return 0
            _, allocated = cache_layout(self.specs, m, c)
            return sum(s.original or s.size(m, self.experts) for s in self.specs.values()) - allocated - overflow(m, c)[1]
        def capacity(m, amount):
            lo, hi = topk - 1, self.experts + 1
            while hi - lo > 1:
                n = (lo + hi) // 2
                if savings(m, n) >= amount:
                    lo = n
                else:
                    hi = n
            return lo
        if mode == 'auto':
            amount = target if target else savings('all', count)
            mode = ('down' if not any(s.vq for s in self.specs.values()) and
                    savings('down', topk) >= amount else 'all')
            count = capacity(mode, amount)
        if target:
            count = capacity(mode, target)
            if count < topk:
                raise SnowLLMError(f'requested {target} bytes of disk offload, maximum {savings(mode, topk)}')
        self.mode, self.capacity = mode, count
        self.overflow_offsets, self.overflow_bytes = overflow(mode, count)
        self.disk_saved_bytes = savings(mode, count) if count < self.experts else 0
        self.disk_weight_bytes = self.disk_saved_bytes + self.overflow_bytes
        self.saved_bytes = self.disk_saved_bytes + self.host_bytes
        self.active = self.active and count < self.experts
        self.rows = max(1, count // topk)
        self.ids_per_row = topk
        self.record_bytes = max((s.record(mode) for s in self.specs.values()), default=0)
        self.graph_native = hasattr(lib, 'snowllm_disk_proxy_graph_buffers')
        self.host_staging_bytes = 0
        if self.active:
            self.load_offsets, self.load_bytes, self.load_slab = {}, 0, None
            self.unpacked = set(self.specs)
            for layer in self.specs:
                regions, end = {}, 0
                for key, length in catalog.load_sizes.get(layer, {}).items():
                    if mode == 'all' or key[0] == 'down':
                        at = align(end)
                        regions[key] = (at, length)
                        end = at + length
                self.load_offsets[layer] = regions
                self.load_bytes = max(self.load_bytes, end)
            self.offsets, size = cache_layout(self.specs, mode, count)
            with host_memory(False):
                self.slab = zero_bytes(size)
                self.overflow_slab = zero_bytes(self.overflow_bytes) if self.overflow_bytes else None
            from .._capi import build_geometry
            self.ids_per_row = max(topk, build_geometry().moe_topk_all)
            self.graph_width = min(self.ids_per_row, count) if self.graph_native else 0
            self.batch_rows = max(s.batch_rows for s in self.specs.values()) if mode == 'all' else 0
            self.batch_rows = min(self.batch_rows, batch_limit)
            self.route_rows = max(self.rows, self.batch_rows)
            self.stage_count = max(2, self.graph_width)
            self.host_staging_bytes = self.stage_count * self.record_bytes + 4 * (
                self.route_rows * self.ids_per_row + (self.experts + 1) * (1 + bool(self.batch_rows) + bool(self.overflow_bytes)) +
                (self.graph_width + 1 if self.graph_native else 0))
        self.dry = drying() is not None
        self.lock = threading.Lock()
        self.resources = contextlib.ExitStack()
        self._close = weakref.finalize(self, self.resources.close)
        if self.active and not self.dry:
            self.directories = expert_cache_directories() or (Path.home() / '.cache' / 'snowllm' / 'experts',)
            try:
                self._open()
            except BaseException:
                self._close()
                raise

    def buffers(self, layer, role):
        return [self.slab.narrow(0, at, length) for at, length in self.offsets[layer][role]]

    def _open(self):
        if not hasattr(lib, 'snowllm_disk_proxy_create'):
            raise SnowLLMError('disk offload requires SnowLLM-Kernels with the native disk proxy')
        reserve_host(self.host_staging_bytes)
        self.resources.callback(release_host, self.host_staging_bytes)
        self.ids = torch.empty(self.route_rows * self.ids_per_row, dtype=torch.int32, pin_memory=True)
        self.mapping = torch.empty(self.experts + 1, dtype=torch.int32, pin_memory=True)
        self.hit_map = torch.empty_like(self.mapping, pin_memory=True) if self.batch_rows else None
        self.cold_map = torch.empty_like(self.mapping, pin_memory=True) if self.overflow_bytes else None
        self.records = torch.empty((self.stage_count, self.record_bytes), dtype=torch.uint8, pin_memory=True)
        self.stages = self.records[:2].unbind()
        self.graph_slots = (torch.empty(self.graph_width + 1, dtype=torch.int32, pin_memory=True)
                            if self.graph_native else None)
        self.io = torch.cuda.Stream()
        # Keep host pointers and the transfer stream alive until the worker has joined.
        self.resources.callback(lambda *keepalive: None, self.ids, self.mapping, self.hit_map, self.cold_map, self.records, self.graph_slots, self.io)
        self.proxy = lib.snowllm_disk_proxy_create(self.experts, self.capacity, _p(self.ids),
            _p(self.mapping), self.ids.numel(), *map(_p, self.stages), self.record_bytes, self.io.cuda_stream)
        if not self.proxy:
            raise SnowLLMError('cannot start the disk expert proxy thread')
        self.files = {}
        self.read_fds = set()
        self.resources.callback(lib.snowllm_disk_proxy_destroy, self.proxy)
        if self.graph_native:
            check(lib.snowllm_disk_proxy_graph_buffers(self.proxy, _p(self.records), self.graph_width,
                                                       _p(self.graph_slots)), 'graph expert staging')
        self.consumed = torch.cuda.Event()
        self.resources.callback(self.consumed.synchronize)

    def close(self):
        with self.lock:
            self._close()
            self.slab = self.load_slab = self.stages = self.records = self.graph_slots = None
            self.ids = self.mapping = self.hit_map = self.cold_map = self.overflow_slab = self.io = None

    def check_open(self):
        if not self._close.alive:
            raise SnowLLMError('disk expert store is closed')

    @contextlib.contextmanager
    def execution(self):
        from .trace import tracing
        from ..engine.graphs import expert_capture
        if self.dry or expert_capture(self) or (tracing() is not None and not tracing().launch):
            yield
            return
        with self.lock:
            self.check_open()
            stream = torch.cuda.current_stream()
            stream.wait_event(self.consumed)
            try:
                yield
            finally:
                self.consumed.record(stream)

    def layer(self, i, layout, backend=None):
        if layout != self.specs[i]:
            raise SnowLLMError(f'layer {i} expert layout changed after planning')
        return (backend or OffloadMoE)(self, i)

    def spool(self, layer, records, source, count):
        try:
            self._spool(layer, records, source, count)
        finally:
            self.unpacked.discard(layer)
            if not self.unpacked:
                self.load_slab = None

    def _spool(self, layer, records, source, count):
        if self.dry:
            return
        directory = self.directories[layer % len(self.directories)]
        directory.mkdir(parents=True, exist_ok=True)
        records.file = self.resources.enter_context(tempfile.TemporaryFile(dir=directory, buffering=0))
        stream = torch.cuda.current_stream()
        views = [memoryview(stage.numpy())[:records.size] for stage in self.stages]
        ready = [torch.cuda.Event(), torch.cuda.Event()]
        try:
            for step in range(self.experts + 1):
                if step < self.experts:
                    slot = step % 2
                    records.pack(self.stages[slot], step, stream.cuda_stream, source, count)
                    ready[slot].record(stream)
                if step:
                    slot = (step - 1) % 2
                    ready[slot].synchronize()
                    view = views[slot]
                    while view:
                        wrote = records.file.write(view)
                        if not wrote:
                            raise OSError('short write of disk expert')
                        view = view[wrote:]
            if hasattr(os, 'POSIX_FADV_DONTNEED') and hasattr(os, 'fdatasync'):
                # Release newly packed pages before writing the next layer.
                fd = records.file.fileno()
                os.fdatasync(fd)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
        except BaseException:
            stream.synchronize()
            raise

    def register(self, records):
        if self.dry:
            return None
        starts, spans = [0], []
        for expert in range(self.experts):
            for file, offset, at, size in records.ranges(expert):
                fd = file.fileno()
                if hasattr(os, 'posix_fadvise') and fd not in self.read_fds:
                    # Router misses are random. Sequential readahead fetches adjacent unused experts.
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                    except OSError:
                        pass
                    self.read_fds.add(fd)
                spans.extend((file_handle(file), offset, at, size))
            starts.append(len(spans) // 4)
        planes = [v for plane in records.destinations() for v in plane]
        native = lib.snowllm_disk_proxy_layer(self.proxy, array(ctypes.c_int32, starts),
            array(ctypes.c_int64, spans), len(spans) // 4, array(ctypes.c_int64, planes),
            len(planes) // 5, array(ctypes.c_int64, records.patches), len(records.patches), records.patch_value)
        if native < 0:
            raise SnowLLMError('invalid disk expert source spans or transfer layout')
        return native

    def check_graph(self):
        check(lib.snowllm_disk_proxy_graph_status(self.proxy), 'graph disk expert read')

    def acquire(self, layer, count, remap):
        from .trace import tracing
        if self.dry or (tracing() is not None and not tracing().launch):
            return
        from ..engine.graphs import expert_capture
        graph = expert_capture(self)
        if graph is not None and not self.graph_native:
            graph.boundary(functools.partial(self.acquire, layer, count, remap))
            return
        check(lib.snowllm_disk_proxy_submit(self.proxy, layer.native, count, remap, _stream()),
              f'layer {layer.index} disk expert read')


class OffloadMoE:
    def __init__(self, store, layer):
        self.store, self.index = store, layer
        self.batch_rows = min(store.specs[layer].batch_rows, store.batch_rows)
        self.rows = self.batch_rows or store.rows

    def stats(self):
        hits, misses = ctypes.c_int64(), ctypes.c_int64()
        with self.store.lock:
            self.store.check_open()
            if self.store.graph_native:
                self.store.check_graph()
            check(lib.snowllm_disk_proxy_stats(self.store.proxy, self.native, ctypes.byref(hits),
                                               ctypes.byref(misses)), 'disk expert cache statistics')
        return hits.value, misses.value

    def workspace_bytes(self, rows):
        return workspace_capacity(('disk_moe',), min(rows, self.rows),
                                  lib.snowllm_moe_cache_workspace_bytes) if rows else 0

    def remap(self, ids):
        from .trace import tracing
        if self.store.dry or ids.is_meta or (tracing() is not None and not tracing().launch):
            return
        host = self.store.ids[:ids.numel()].view_as(ids)
        host.copy_(ids, non_blocking=True)
        self.store.acquire(self, ids.numel(), True)
        ids.copy_(host, non_blocking=True)

    @operator
    def forward(self, hidden, router, shared_gate, shared_down, out, workspace, token_ids=None, tid2eid=None):
        with self.store.execution():
            self._forward(hidden, router, shared_gate, shared_down, out, workspace, token_ids, tid2eid)

    def forward_norm(self, x, residual, gamma, hidden_row, eps, router, shared_gate, shared_down,
                     out, workspace):
        from . import rmsnorm_residual
        rmsnorm_residual(x, residual, gamma, hidden_row, eps)
        self.forward(hidden_row, router, shared_gate, shared_down, out, workspace)

    def _forward(self, hidden, router, shared_gate, shared_down, out, workspace, token_ids, tid2eid):
        store = self.store
        if hidden.shape != out.shape or hidden.dtype != torch.bfloat16 or out.dtype != hidden.dtype:
            raise SnowLLMError('disk MoE expects matching BF16 input and output')
        if workspace.nbytes < self.workspace_bytes(len(hidden)):
            raise SnowLLMError('insufficient disk MoE workspace')
        weights = _weights((self.gate, self.gate_scale), (self.down, self.down_scale),
                           (shared_gate, self.shared_gate_scale), (shared_down, self.shared_down_scale))
        for at in range(0, len(hidden), self.rows):
            x = hidden[at:at + self.rows]
            tokens = token_ids[at:at + len(x)] if token_ids is not None else None
            ids = getattr(store, 'ids', None)
            if self.batch_rows:
                check(lib.snowllm_moe_disk_batch(getattr(store, 'proxy', None), getattr(self, 'native', -1),
                    weights, _p(x), _p(router), _p(tokens), _p(tid2eid), _p(ids),
                    _p(getattr(store, 'hit_map', None)), _p(out[at:]), _p(workspace), len(x),
                    store.capacity, _stream()), 'schedule disk experts')
                continue
            check(lib.snowllm_moe_cache_route(_p(x), _p(router), _p(tokens), _p(tid2eid),
                  _p(ids), _p(workspace), len(x), _stream()), 'route disk experts')
            store.acquire(self, len(x) * store.ids_per_row, False)
            check(lib.snowllm_moe_disk_execute(weights, _p(x), _p(getattr(store, 'mapping', None)),
                  _p(out[at:]), _p(workspace), len(x), store.capacity, store.mode == 'all',
                  _stream()), 'execute disk experts')


def binding(layer):
    store = _build.get()
    if isinstance(store, Catalog):
        return store
    if store is not None and store.active and layer in store.specs and store.dry == (drying() is not None):
        return store
    return None


def bind_standard(module, layer):
    build = binding(layer)
    if build is None or layer is None:
        return module
    from .._capi import build_geometry
    geo = build_geometry()
    e = geo.moe_num_experts
    gate, down = module.gate_up, module.down
    gs, ds = getattr(module.backend, 'gate_scale', None), getattr(module.backend, 'down_scale', None)
    embedded = isinstance(gate, torch.Tensor) and module.shared_gate_up is None
    count = geo.moe_num_slabs if embedded else e
    batch_rows = 0
    if hasattr(lib, 'snowllm_moe_disk_batch_rows'):
        weights = _weights((gate, gs), (down, ds), (module.shared_gate_up, None),
                           (module.shared_down, None))
        batch_rows = lib.snowllm_moe_disk_batch_rows(weights)
    layout = Layout(e, standard_planes(gate, gs, count, 2 * geo.moe_inter),
                    standard_planes(down, ds, count, geo.hidden), batch_rows=batch_rows)
    if isinstance(build, Catalog):
        build.add(layer, layout)
        return module
    if not hasattr(lib, 'snowllm_moe_disk_execute'):
        raise SnowLLMError('disk experts require SnowLLM-Kernels with snowllm_moe_disk_execute')
    cache = build.layer(layer, layout)
    source, output = [], []
    for role, original, scale, shared, scale_attr in (
            ('gate_up', gate, gs, 'shared_gate_up', 'gate_scale'),
            ('down', down, ds, 'shared_down', 'down_scale')):
        src = tensors(original, scale)
        shared_scale = None
        if embedded and count > e:
            # Separate the always-active shared slab before compacting routed IDs.
            part = [t.reshape(count, -1)[e:] for t in src]
            if build.mode == 'all' or role == 'down':
                part = [t.clone() for t in part]
            setattr(module, shared, part[0])
            shared_scale = part[1] if scale is not None else None
        setattr(cache, 'shared_' + scale_attr, shared_scale)
        cached = build.mode == 'all' or role == 'down'
        if cached:
            dst = build.buffers(layer, role)
            packed = compact_weight(original, dst, build.capacity, role)
            scale = dst[-1].view(scale.dtype) if scale is not None else None
            source.extend(src)
            output.extend(dst)
        else:
            packed = original
        setattr(cache, 'gate' if role == 'gate_up' else 'down', packed)
        setattr(cache, scale_attr, scale)
    cache.records = Records(layout, build.mode, output, build.capacity)
    build.spool(layer, cache.records, source, count)
    cache.native = build.register(cache.records)
    if layer in build.overflow_offsets and not build.dry:
        width, regions = build.overflow_offsets[layer]
        buffers = [build.overflow_slab.narrow(0, at, size) for role in ('gate_up', 'down')
                   for at, size in regions[role]]
        planes = [v for plane in Records(layout, 'all', buffers, width).destinations() for v in plane]
        check(lib.snowllm_disk_proxy_overflow(build.proxy, cache.native, array(ctypes.c_int64, planes),
              len(planes) // 5, width, _p(build.cold_map)), 'register prefill expert workspace')
    module.backend = cache
    return module


def inkling_buffers(proj):
    return [t for t in (proj.weight, proj.scale, proj.meta, *(proj.compact or ())) if t is not None]


class OffloadInklingMoE(OffloadMoE):
    def project(self, module, x, ids, prefill):
        with self.store.execution():
            if len(x) <= self.store.rows:
                return self._project(module, x, ids, prefill)
            from .allocation import empty
            states = empty(len(x), module.topk, x.shape[-1], device=x.device, dtype=x.dtype)
            for at in range(0, len(x), self.store.rows):
                chunk = slice(at, at + self.store.rows)
                states[chunk].copy_(self._project(module, x[chunk], ids[chunk], prefill))
            return states

    def _project(self, module, x, ids, prefill):
        if self.store.mode == 'all':
            self.remap(ids)
        activated, plan = module.activate(x, ids, prefill)
        if self.store.mode == 'down':
            self.remap(ids)
            plan = module.plan(ids, prefill, module.down.scale is not None)
        return module.down(activated, ids, prefill, plan=plan)


def bind_inkling(module, layer):
    build = binding(layer)
    if build is None:
        return
    e = module.router.shape[0]
    def planes(proj):
        return tuple(weight_plane(t, e, 2 if proj.meta is not None else 0, i)
                     for i, t in enumerate(inkling_buffers(proj)))
    layout = Layout(e, planes(module.gate_up), planes(module.down))
    if isinstance(build, Catalog):
        build.add(layer, layout)
        return
    source, output = [], []
    for role in ('gate_up', 'down'):
        if role == 'gate_up' and build.mode == 'down':
            continue
        proj = getattr(module, role)
        src = inkling_buffers(proj)
        dst = build.buffers(layer, role)
        packed = [b.view(t.dtype) for b, t in zip(dst, src)]
        if proj.meta is None:
            packed[0] = packed[0].reshape(build.capacity, *proj.weight.shape[1:])
        replacement = dataclasses.replace(proj, weight=packed[0],
            scale=packed[1] if proj.scale is not None else None,
            meta=packed[1] if proj.meta is not None else None,
            compact=tuple(packed[-2:]) if proj.compact is not None else None,
            slots=build.capacity)
        setattr(module, role, replacement)
        source.extend(src)
        output.extend(dst)
    cache = build.layer(layer, layout, OffloadInklingMoE)
    cache.records = Records(layout, build.mode, output, build.capacity)
    build.spool(layer, cache.records, source, e)
    cache.native = build.register(cache.records)
    return cache


class OffloadVQMoE(OffloadMoE):
    def chunk_rows(self, rows):
        return min(rows, self.store.rows)

    def launch(self, config, args, book):
        store = self.store
        if config.rows > store.rows or config.topk != store.topk:
            raise SnowLLMError('VQ route batch exceeds the planned disk expert slots')
        args = (ctypes.byref(config), *args, _p(book), _p(store.ids))
        with store.execution():
            check(lib.snowllm_vq_moe_disk(*args, 0, _stream()), 'route VQ disk experts')
            store.acquire(self, config.rows * config.topk, True)
            check(lib.snowllm_vq_moe_disk(*args, 1, _stream()), 'execute VQ disk experts')


class VQRecords:
    def __init__(self, store, gguf, name, layout):
        self.gguf, self.name, self.layout, self.store = gguf, name, layout, store
        e = layout.experts
        self.book = align(16 + e * 3 * 8)
        self.prefix = align(self.book + layout.matrices[0].codewords * 8)
        self.offsets, size = [], 0
        for m in layout.matrices[:3]:
            self.offsets.append(size)
            size += align(m.end - m.offset, 8)
        self.size = align(size)

        self.patches = tuple(at + 24 for at in self.offsets)
        self.patch_value = self.book

    def ranges(self, expert):
        for at, matrix in zip(self.offsets, self.layout.matrices[expert * 3:expert * 3 + 3]):
            for path, offset, length in self.gguf.ranges(self.name, matrix.offset, matrix.end - matrix.offset):
                files = self.store.files
                if path not in files:
                    files[path] = self.store.resources.enter_context(path.open('rb', buffering=0))
                yield files[path], offset, at, length
                at += length

    def destinations(self):
        yield self.blob.data_ptr() + self.prefix, 0, self.size, self.size, 1


def vq_blob(rd, name, layout, layer):
    store = binding(layer)
    if store is not None:
        records = VQRecords(store, rd.gguf, name, layout)
        spec = Layout(layout.experts, (), (Plane(records.size),), records.prefix, vq=True, original=layout.nbytes)
        if isinstance(store, Catalog):
            store.add(layer, spec)
        else:
            if not hasattr(lib, 'snowllm_vq_moe_disk'):
                raise SnowLLMError('VQ disk offload requires SnowLLM-Kernels with snowllm_vq_moe_disk')
            cache = store.layer(layer, spec, OffloadVQMoE)
            cache.records = records
            size = records.prefix + store.capacity * records.size
            blob, = store.buffers(layer, 'blob')
            records.blob = blob
            if not store.dry:
                header = bytearray(records.prefix)
                header[:16] = rd.gguf.read(name, 0, 16)
                matrix = layout.matrices[0]
                codebook = rd.gguf.read(name, matrix.codebook, matrix.codewords * 8)
                header[records.book:records.book + len(codebook)] = codebook
                for expert in range(layout.experts):
                    base = records.prefix + (expert % store.capacity) * records.size
                    struct.pack_into('<3Q', header, 16 + expert * 24, *(base + n for n in records.offsets))
                blob[:len(header)].copy_(torch.frombuffer(header, dtype=torch.uint8))
            cache.native = store.register(records)
            return blob, dataclasses.replace(layout, nbytes=size), cache
    blob = empty_bytes(layout.nbytes)
    rd.raw(name, out=blob)
    return blob, layout, None

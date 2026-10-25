# SPDX-License-Identifier: Apache-2.0
import ctypes
from pathlib import Path

import torch

from .._capi import SnowLLMError, check, lib
from ..expert_lru import ExpertLRU, cache_config, offload_config
from ..checkpoint.expert_cache import ExpertDiskCache, align
from . import moe
from ._common import _p, _stream, drying, empty_bytes, host_memory, zero_bytes
from .inference import _weight, _weights
from .trace import tracing


def _sizes(fmt, count, role):
    kind, code = fmt
    names = ('quant', 'meta') if kind == 'kquant' else ('p0', 'p1', 'p2')
    return tuple(int(getattr(lib, f'snowllm_moe_{kind}_{role}_{p}_bytes')(code, count))
                 for p in names)


def _weight_from(fmt, role, planes, count):
    kind, code = fmt
    if kind == 'kquant':
        return moe.KQuantExpertWeight(*planes, code)
    stride = getattr(lib, f'snowllm_moe_lowbit_{role}_stride')(count)
    return moe.LowBitExpertWeight(*planes, stride, code)


def _layout(fmt, count, role):
    segments, at = [], 0
    for size in _sizes(fmt, count, role):
        at = -(-at // 512) * 512
        segments.append((at, size))
        at += size
    return segments, at


def allocate_weight(fmt, count, role, buffer=None):
    segments, size = _layout(fmt, count, role)
    buffer = zero_bytes(size) if buffer is None else buffer
    if buffer.numel() < size:
        raise SnowLLMError('expert shuffle buffer is smaller than its planned layout')
    planes = [buffer[at:at + size] if size else None for at, size in segments]
    return _weight_from(fmt, role, planes, count)


def _specs(rd, geo):
    specs = {}
    experts = geo.moe_num_experts
    for i in range(geo.num_layers):
        if geo.is_hashed(i) or f'blk.{i}.ffn_gate_tid2eid.weight' in rd.gguf:
            continue
        if f'blk.{i}.ffn_exps_vq.blob' in rd.gguf:
            raise SnowLLMError('VQ MoE does not support the expert LRU cache yet')
        names = tuple(f'blk.{i}.ffn_{part}_exps.weight' for part in ('gate', 'up', 'down'))
        tensors = [rd.gguf[n] for n in names]
        if any(t.shape[0] != experts or t.nbytes % experts for t in tensors):
            raise SnowLLMError(f'blk.{i}: invalid routed expert tensor shape')
        fmts = []
        for tensor in tensors:
            name = tensor.quant.name
            if name in moe.LOWBIT_FORMATS:
                fmts.append(('lowbit', moe.LOWBIT_FORMATS[name]))
            elif name in moe.KQUANT_FORMATS:
                fmts.append(('kquant', moe.KQUANT_FORMATS[name]))
            else:
                raise SnowLLMError(f'{tensor.name}: MoE LRU does not support {name}')
        if fmts[0] != fmts[1] or fmts[0][0] != fmts[2][0]:
            raise SnowLLMError(f'blk.{i}: MoE LRU needs matching gate/up and one routed storage family')
        spans = tuple((str(rd.gguf.file_of(n)).encode(), rd.gguf.file_offset(n), t.nbytes // experts)
                      for n, t in zip(names, tensors))
        specs[i] = (fmts[0], fmts[2], spans)
    return specs


def _buffers(specs, mode, packed):
    raw = scratch = 0
    for gu, dn, spans in specs.values():
        roles = (('gate_up', gu), ('down', dn)) if mode == 'all' else (('down', dn),)
        sizes = [_layout(fmt, 1, role)[1] for role, fmt in roles]
        raw = max(raw, sum(s[2] for s in (spans if mode == 'all' else spans[2:])))
        scratch = max(scratch, *sizes)
        if packed:
            raw = max(raw, sum(map(align, sizes)))
    return raw, scratch


class ExpertStore:
    def __init__(self, rd, geo, capacity=0, mode='all', offload=None, device_map=None):
        if offload is None and not 1 <= capacity <= geo.moe_num_experts:
            raise SnowLLMError(f'moe_cache_experts must be 1..{geo.moe_num_experts}')
        if not all(hasattr(lib, 'snowllm_moe_cache_' + name)
                   for name in ('route', 'copy', 'execute', 'workspace_bytes')):
            raise SnowLLMError('MoE LRU needs SnowLLM-Kernels with expert-cache entry points')
        if not hasattr(lib, 'snowllm_file_reader_readv'):
            raise SnowLLMError('MoE LRU needs SnowLLM-Kernels with batched file reading')
        if mode not in ('all', 'down', 'auto'):
            raise SnowLLMError('moe_cache_mode must be all, down or auto')
        specs = _specs(rd, geo)
        self.resident_layers = tuple(i for i in range(geo.num_layers) if i not in specs)
        def on_host(role, layer):
            return device_map is not None and device_map.on_host('experts.' + role, layer)
        self.host_layers = tuple(i for i in specs if on_host('gate_up', i) or on_host('down', i))
        self.host_bytes = sum(sum(_sizes(fmt, geo.moe_num_experts, role))
                              for i, (gu, dn, _) in specs.items()
                              for role, fmt in (('gate_up', gu), ('down', dn)) if on_host(role, i))
        self.fixed_bytes = sum(sum(_sizes(fmt, geo.moe_num_experts, role))
                               for i in self.host_layers for role, fmt in
                               (('gate_up', specs[i][0]), ('down', specs[i][1])) if not on_host(role, i))
        self.specs = {i: spec for i, spec in specs.items() if i not in self.host_layers}
        self.experts, self.topk = geo.moe_num_experts, geo.moe_topk
        self.disk_cache = ExpertDiskCache.from_env()
        self.buffers = {m: _buffers(self.specs, m, self.disk_cache is not None)
                        for m in (('all', 'down') if mode == 'auto' else (mode,))}
        self.resident_bytes = sum(sum(_sizes(gu, self.experts, 'gate_up')) +
                                  sum(_sizes(dn, self.experts, 'down'))
                                  for gu, dn, _ in specs.values())
        self.reader = self._file_reader = self.stream = self.io_stream = None
        self.offload_target = offload
        self.active = bool(self.specs)
        if not self.active:
            if offload:
                raise SnowLLMError(f'disk cache planning: requested {offload} bytes, maximum 0 bytes; '
                                   'hash and shared experts stay resident')
            self.mode, self.capacity = 'all', self.experts
            self.saved_bytes, self.disk_saved_bytes, self.host_staging_bytes = self.host_bytes, 0, 0
            self.disk_weight_bytes = 0
            return
        if offload is None:
            mode, capacity = cache_config(mode, capacity, self.experts, self.allocation_bytes)
        else:
            try:
                mode, capacity = offload_config(mode, self.experts, offload,
                                                self.resident_bytes - self.host_bytes,
                                                self.allocation_bytes)
            except ValueError as exc:
                raise SnowLLMError(f'disk cache planning: {exc}') from exc
        self.saved_bytes = self.resident_bytes - self.allocation_bytes(mode, capacity)
        self.disk_saved_bytes = self.saved_bytes - self.host_bytes
        raw, scratch = self.buffers[mode]
        self.disk_weight_bytes = self.disk_saved_bytes + raw * min(capacity, self.topk) + scratch
        if mode == 'all' and not hasattr(lib, 'snowllm_moe_cache_execute_phase'):
            raise SnowLLMError('MoE LRU needs SnowLLM-Kernels with shared expert phases')
        if mode == 'down' and not all(hasattr(lib, 'snowllm_moe_cache_' + name)
                                      for name in ('gate_up', 'down')):
            raise SnowLLMError('down LRU needs SnowLLM-Kernels with split expert execution')
        self.mode, self.overlap = mode, True
        self.capacity, self.experts = capacity, geo.moe_num_experts
        self.topk, self.all = geo.moe_topk, geo.moe_topk + geo.moe_num_shared
        self.rows = int(lib.snowllm_paged_prefill_max_tokens())
        self.packed = {}
        identities = {}
        raw, scratch = self.buffers[mode]
        host_packed = 0
        for i, (gu_fmt, dn_fmt, spans) in self.specs.items():
            if self.disk_cache is not None:
                roles = (('gate_up', gu_fmt, spans[:2]), ('down', dn_fmt, spans[2:]))
                packed, offset = [], 0
                for role, fmt, sources in roles:
                    if mode == 'down' and role == 'gate_up':
                        continue
                    segments, size = _layout(fmt, 1, role)
                    source_info = []
                    for path, start, length in sources:
                        if path not in identities:
                            file = Path(path.decode()).resolve()
                            stat = file.stat()
                            identities[path] = (str(file), stat.st_size, stat.st_mtime_ns,
                                                stat.st_dev, stat.st_ino)
                        source_info.append((identities[path], start, length))
                    identity = dict(abi=lib.snowllm_abi_version(), fmt=fmt, layout=segments,
                                    geometry=(geo.hidden, geo.moe_inter),
                                    source=source_info)
                    record = self.disk_cache.plan(i, role, size, self.experts, identity)
                    packed.append((role, fmt, offset, size, record))
                    offset += record
                    host_packed = max(host_packed, record)
                self.packed[i] = tuple(packed)
        self.io_experts = min(capacity, self.topk)
        self.raw_stride = raw
        with host_memory(False):
            self.raw = empty_bytes(raw * self.io_experts)
            self.scratch = empty_bytes(scratch)
        self.parts = 3 if mode == 'all' else 1
        count = self.parts * self.io_experts
        if self.disk_cache is not None:
            count = max(count, len(self.disk_cache.directories) * len(next(iter(self.packed.values()))) * self.io_experts)
        self.io_paths = (ctypes.c_char_p * count)()
        self.io_offsets = (ctypes.c_int64 * count)()
        self.io_sizes = (ctypes.c_int64 * count)()
        self.io_destinations = (ctypes.c_void_p * count)()
        striped = self.disk_cache is not None and len(self.disk_cache.directories) > 1
        reader_config = (16, 512 << 10, 24) if striped else (8, 4 << 20, 12)
        self.host_staging_bytes = (host_packed + (self.rows * self.all + self.experts + 1) * 4 +
                                   reader_config[1] * reader_config[2])
        if drying() is None:
            import weakref
            from ..capacity import check_host, reserve_host, release_host
            from ._common import host_pinned_bytes
            check_host(self.host_staging_bytes, host_pinned_bytes())
            nbytes = self.host_staging_bytes - reader_config[1] * reader_config[2]
            reserve_host(nbytes)
            self._host_reservation = weakref.finalize(self, release_host, nbytes)
        self.host_ids = torch.empty(self.rows, self.all, dtype=torch.int32,
                                    pin_memory=drying() is None)
        self.host_map = torch.empty(self.experts + 1, dtype=torch.int32,
                                    pin_memory=drying() is None)
        self.host_ids_array = (ctypes.c_int32 * self.host_ids.numel()).from_address(self.host_ids.data_ptr())
        self.host_map_array = (ctypes.c_int32 * self.host_map.numel()).from_address(self.host_map.data_ptr())
        self.host_packed = None
        if host_packed:
            dry = drying()
            self.host_packed = (dry.meta((host_packed,), torch.uint8, host=True) if dry is not None
                                else torch.empty(host_packed, dtype=torch.uint8, pin_memory=True))
        if drying() is None:
            self.io_stream = torch.cuda.Stream()
            self.route_ready = torch.cuda.Event()
            self.io_ready = torch.cuda.Event()
            self.hits_done = torch.cuda.Event()
            from ..checkpoint.reader import FileReader
            self._file_reader = FileReader(*reader_config)
            self.reader = self._file_reader._r

    def allocation_bytes(self, mode, count):
        weights = sum((sum(_sizes(gu, self.experts, 'gate_up')) if mode == 'down'
                       else _layout(gu, count, 'gate_up')[1]) + _layout(dn, count, 'down')[1]
                      for gu, dn, _ in self.specs.values())
        raw, scratch = self.buffers[mode]
        return self.fixed_bytes + weights + raw * min(count, self.topk) + scratch

    def load_many(self, layer, assignments, gate_up, down):
        if self.disk_cache is not None:
            ready, missing = [], []
            for expert, slot in assignments:
                records = [self.disk_cache.lookup(layer, role, expert)
                           for role, *_ in self.packed[layer]]
                if all(record is not None for record in records):
                    ready.append((expert, slot, records))
                else:
                    missing.append((expert, slot))
            self._load_packed(layer, ready, gate_up, down)
            assignments = missing
        _, _, spans = self.specs[layer]
        if self.mode == 'down':
            spans = spans[2:]
        for start in range(0, len(assignments), self.io_experts):
            batch = assignments[start:start + self.io_experts]
            tensors = []
            for i, (expert, _) in enumerate(batch):
                raw, at = [], i * self.raw_stride
                for part, (path, offset, size) in enumerate(spans):
                    tensor = self.raw[at:at + size]
                    j = self.parts * i + part
                    self.io_paths[j] = path
                    self.io_offsets[j] = offset + expert * size
                    self.io_sizes[j] = size
                    self.io_destinations[j] = _p(tensor)
                    raw.append(tensor)
                    at += size
                tensors.append(raw)
            check(lib.snowllm_file_reader_readv(self.reader, self.io_paths, self.io_offsets,
                self.io_sizes, self.io_destinations, self.parts * len(batch), _stream()), 'read cached experts')
            for (expert, slot), raw in zip(batch, tensors):
                self._insert(layer, raw, slot, gate_up, down, expert=expert)

    def _load_packed(self, layer, assignments, gate_up, down):
        for start in range(0, len(assignments), self.io_experts):
            batch = assignments[start:start + self.io_experts]
            count = 0
            for i, (_, _, records) in enumerate(batch):
                for (_, _, offset, _, _), record in zip(self.packed[layer], records):
                    for path, file_offset, size, destination in record:
                        self.io_paths[count] = path
                        self.io_offsets[count] = file_offset
                        self.io_sizes[count] = size
                        self.io_destinations[count] = self.raw.data_ptr() + i * self.raw_stride + offset + destination
                        count += 1
                        self.disk_cache.read_bytes += size
            check(lib.snowllm_file_reader_readv(self.reader, self.io_paths, self.io_offsets,
                self.io_sizes, self.io_destinations, count, _stream()), 'read prepacked experts')
            for i, (_, slot, _) in enumerate(batch):
                for role, fmt, offset, size, _ in self.packed[layer]:
                    begin = i * self.raw_stride + offset
                    temp = allocate_weight(fmt, 1, role, self.raw[begin:begin + size])
                    self._copy(temp, gate_up if role == 'gate_up' else down, role, slot)

    def _insert(self, layer, raw, slot, gate_up, down, expert=None):
        gu_fmt, dn_fmt, _ = self.specs[layer]
        for role, fmt, target in (('gate_up', gu_fmt, gate_up), ('down', dn_fmt, down)):
            if self.mode == 'down' and role == 'gate_up':
                continue
            temp = allocate_weight(fmt, 1, role, self.scratch)
            if self.disk_cache is not None:
                self.scratch[:_layout(fmt, 1, role)[1]].zero_()
            data = (temp.quant, temp.meta) if fmt[0] == 'kquant' else (temp.p0, temp.p1, temp.p2)
            inputs = raw if self.mode == 'down' else raw[:2] if role == 'gate_up' else raw[2:]
            check(getattr(lib, f'snowllm_moe_{fmt[0]}_shuffle_{role}')(
                fmt[1], *map(_p, inputs), *map(_p, data), 1, _stream()), 'shuffle cached expert')
            self._copy(temp, target, role, slot)
            if self.disk_cache is not None and expert is not None:
                size = _layout(fmt, 1, role)[1]
                record = align(size)
                self.host_packed[:record].zero_()
                self.host_packed[:size].copy_(self.scratch[:size], non_blocking=True)
                torch.cuda.current_stream().synchronize()
                self.disk_cache.publish(layer, role, expert,
                    memoryview(self.host_packed.numpy())[:record])

    def _copy(self, source, target, role, slot):
        source_desc, target_desc = _weight(source), _weight(target)
        check(lib.snowllm_moe_cache_copy(ctypes.byref(source_desc), ctypes.byref(target_desc),
            int(role == 'down'), self.capacity, slot, _stream()), 'insert cached expert')

    def close(self):
        if self.reader is not None:
            if self.stream is not None:
                self.stream.synchronize()
            if self.io_stream is not None:
                self.io_stream.synchronize()
            self._file_reader.close()
            self.reader = None

    def __del__(self):
        if getattr(self, 'reader', None) is not None:
            self.close()


class CachedExperts:
    def __init__(self, store, layer, gate_up=None):
        self.store, self.layer = store, layer
        gu_fmt, dn_fmt, _ = store.specs[layer]
        with host_memory(False):
            if store.mode == 'down' and gate_up is None:
                raise SnowLLMError('down LRU requires resident gate_up weights')
            self.gate_up = gate_up if gate_up is not None else allocate_weight(gu_fmt, store.capacity, 'gate_up')
            self.down = allocate_weight(dn_fmt, store.capacity, 'down')
        self.lru = ExpertLRU(store.capacity, store.experts)
        self.weights = None
        self.weights_key = None

    def workspace_bytes(self, rows):
        return lib.snowllm_moe_cache_workspace_bytes(rows)

    def forward(self, hidden, router, shared_gate, shared_down, out, workspace,
                token_ids=None, tid2eid=None):
        store = self.store
        rows = hidden.shape[0]
        if not rows:
            return
        if rows > store.rows:
            raise SnowLLMError('MoE batch exceeds its startup routing capacity')
        if workspace.nbytes < self.workspace_bytes(rows):
            raise SnowLLMError('MoE cache workspace is smaller than its planned layout')
        key = (self.gate_up, self.down, shared_gate, shared_down)
        if self.weights_key != key:
            self.weights = _weights((self.gate_up, None), (self.down, None),
                                    (shared_gate, None), (shared_down, None))
            self.weights_key = key
        weights = self.weights
        trace = tracing() is not None
        if not trace:
            if store.reader is None:
                raise SnowLLMError('the MoE expert store is closed')
            stream = torch.cuda.current_stream()
            if store.stream is not None and stream != store.stream:
                store.stream.synchronize()
            store.stream = stream
        check(lib.snowllm_moe_cache_route(_p(hidden), _p(router), _p(token_ids), _p(tid2eid),
            _p(store.host_ids), _p(workspace), rows, _stream()), 'route cached experts')
        down_only = store.mode == 'down'
        if down_only:
            if not trace:
                store.route_ready.record(store.stream)
            check(lib.snowllm_moe_cache_gate_up(weights, _p(hidden), _p(workspace), rows,
                _stream()), 'execute resident gate_up')
        if trace:
            groups = [()]
        else:
            if down_only and store.overlap:
                store.route_ready.synchronize()
            else:
                store.stream.synchronize()
            needed = list(dict.fromkeys(e for e in reversed(
                store.host_ids_array[:rows * store.all]) if e < store.experts))
            needed.reverse()
            groups = [needed[at:at + store.capacity] for at in range(0, len(needed), store.capacity)]
        for index, group in enumerate(groups):
            computed = set()
            shared_computed = False
            if not trace:
                if index:
                    store.stream.synchronize()
                def load(pending):
                    nonlocal shared_computed
                    if not down_only and store.overlap and len(groups) == 1:
                        missing = {e for e, _ in pending}
                        computed.update(e for e in group if e not in missing)
                        ctypes.memset(store.host_map_array, 255, ctypes.sizeof(store.host_map_array))
                        store.host_map_array[store.experts] = store.experts
                        for expert in computed:
                            store.host_map_array[expert] = self.lru.resident[expert]
                        check(lib.snowllm_moe_cache_execute_phase(weights, _p(hidden), _p(store.host_map),
                            _p(out), _p(workspace), rows, store.capacity, 0, _stream()),
                            'execute cache hits and shared expert')
                        shared_computed = True
                        store.hits_done.record(store.stream)
                        try:
                            with torch.cuda.stream(store.io_stream):
                                store.load_many(self.layer, pending, self.gate_up, self.down)
                                store.io_ready.record(store.io_stream)
                        except BaseException:
                            store.io_stream.synchronize()
                            store.hits_done.synchronize()
                            raise
                        # The miss pass reuses the pinned map submitted by the hit pass.
                        store.hits_done.synchronize()
                        store.stream.wait_event(store.io_ready)
                        return
                    if down_only and store.overlap:
                        if index:
                            store.io_stream.wait_stream(store.stream)
                        else:
                            store.io_stream.wait_event(store.route_ready)
                        with torch.cuda.stream(store.io_stream):
                            store.load_many(self.layer, pending, self.gate_up, self.down)
                            store.io_ready.record(store.io_stream)
                        store.stream.wait_event(store.io_ready)
                    else:
                        store.load_many(self.layer, pending, self.gate_up, self.down)
                mapping = self.lru.acquire(group, load, batched=True)
                ctypes.memset(store.host_map_array, 255, ctypes.sizeof(store.host_map_array))
                store.host_map_array[store.experts] = store.experts
                for expert, slot in mapping.items():
                    if expert not in computed:
                        store.host_map_array[expert] = slot
            args = (weights,) if down_only else (weights, _p(hidden))
            execute = lib.snowllm_moe_cache_down if down_only else lib.snowllm_moe_cache_execute
            if shared_computed:
                execute = lib.snowllm_moe_cache_execute_phase
            check(execute(*args, _p(store.host_map), _p(out), _p(workspace), rows,
                store.capacity, int(index == len(groups) - 1), _stream()),
                'execute cached experts')

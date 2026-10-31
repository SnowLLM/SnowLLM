# SPDX-License-Identifier: Apache-2.0
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps


_current = contextvars.ContextVar('memory_report', default=None)


def current():
    return _current.get()


@contextmanager
def suspended():
    token = _current.set(None)
    try:
        yield
    finally:
        _current.reset(token)


def size(n):
    if n is None:
        return 'n/a'
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024 or unit == 'TiB':
            return f'{n:.3f} {unit}'
        n /= 1024


@dataclass
class MemoryReport:
    values: dict = field(default_factory=dict)
    suggestion: int | None = None
    max_host_offload: int | None = None
    weight_lower_bound: tuple[int, int, int] | None = None

    def available(self, gpu, host):
        print(f'[planner] available GPU={size(gpu)}, host={size(host)}', flush=True)

    def ready(self, *, weights, host_weights, workspace, snapshot_count, device_used):
        v = self.values
        if not v:
            return
        workspace += v.get('fixed_workspace', 0)
        each = v['snapshot_each']
        snapshots = each * snapshot_count
        cache = v.get('cache_snapshot_each', 0) * (snapshot_count-v['snapshot_count'])
        v['cache'] += cache
        v['total'] += weights - v['weights'] + workspace - v['workspace'] + snapshots - v['snapshots'] + cache
        v.update(weights=weights, host_weights=host_weights, workspace=workspace,
                 snapshots=snapshots, snapshot_count=snapshot_count, device_used=device_used)

    def shortfall(self, nbytes):
        if 'total' in self.values:
            self.values['room'] = self.values['total'] - nbytes
            self.values['measured_room'] = True

    def lines(self, status):
        v = self.values
        if not v:
            if self.weight_lower_bound is not None:
                weights, gpu, host = self.weight_lower_bound
                yield (f'weights need at least {size(weights)}; available: '
                       f'GPU {size(gpu)} + host up to {size(host)}')
                yield (f'{status}: at least {size(max(0, weights-gpu-host))} more memory needed '
                       '(KV cache and workspace not included)')
                return
            yield f'{status} memory=n/a'
            return
        target = v.get('allocation', 'GPU')
        gpu = 0 if target == 'host' else v['weights']
        host = v['host_weights'] + (v['weights'] if target == 'host' else 0)
        weights = f'weights {size(gpu+host)}'
        if self.max_host_offload is not None and target != 'host':
            weights += f', max host offload={size(self.max_host_offload)}'
        yield weights
        yield f'weights GPU={size(gpu)}, host={size(host)}'
        context = f'ctx={v["context"]}'
        if v.get('per_request', v['context']) != v['context']:
            context += f', max/request={v["per_request"]}'
        yield f'{context}, KV={size(v["cache"])}, {size(v["cache_100k"])} per 100K tok'
        if v['snapshot_each']:
            cached = v['snapshot_count'] / v['requests']
            average = '~' if v['snapshot_count'] % v['requests'] else ''
            copies = f'{average}{cached:g} cached'
            if v['linear']:
                copies = f'{v["slots"]} live + {copies}'
            states = v['linear'] + v['snapshots']
            line = (f'reqs={v["requests"]}, state snapshots={size(states)}, '
                    f'{size(v["snapshot_each"])} per snapshot, {copies} per req')
            yield line
        yield (f'workspace={size(v["workspace"])}, '
               f'extra/request~{size(v["other"] / v["requests"])}')
        total = f'{status} {target} tensors={size(v["total"])}'
        if 'device_used' in v:
            total += f' (device={size(v["device_used"])})'
        if 'host_allocated' in v:
            total += f', host-pool={size(v["host_allocated"])}'
        if status == 'failed':
            missing = max(0, v['total'] - v['room'])
            loading = max(0, v['load_peak'] - v.get('load_room', v['room']))
            if missing or loading or self.suggestion:
                total += f', short={size(max(missing, loading))}'
                if self.suggestion:
                    total += f', --max-model-len {self.suggestion} (est)'
        yield total

    def emit(self, status):
        for line in self.lines(status):
            print(f'[planner] {line}', flush=True)


def startup_report(build):
    @wraps(build)
    def wrapped(*args, **kwargs):
        report = MemoryReport()
        token = _current.set(report)
        try:
            result = build(*args, **kwargs)
        except Exception:
            report.emit('failed')
            raise
        else:
            report.emit('ready')
            return result
        finally:
            _current.reset(token)
    return wrapped

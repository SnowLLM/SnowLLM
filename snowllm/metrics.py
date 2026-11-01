# SPDX-License-Identifier: Apache-2.0
"""CPU-only request accounting, hardware sensor reads, and Prometheus formatting."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RequestTiming:
    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    queue_seconds: float = 0.0
    scheduler_wait_seconds: float = 0.0

    def add(self, kind, tokens, seconds):
        if tokens > 0:
            setattr(self, kind + '_tokens', getattr(self, kind + '_tokens') + tokens)
            setattr(self, kind + '_seconds', getattr(self, kind + '_seconds') + seconds)

    def snapshot(self):
        result = dict(vars(self))
        for kind in ('prefill', 'decode'):
            seconds = result[kind + '_seconds']
            result[kind + '_tok_s'] = result[kind + '_tokens'] / seconds if seconds > 0 else None
        return result


def gpu_sensors(root=Path('/sys/class/drm')):
    """Read Linux DRM/hwmon sensors without spawning GPU tools.

    Missing sensors are omitted, rather than reporting zero utilization/power.
    Power values are Linux hwmon microwatts; temperatures are millidegrees C.
    """
    devices = []
    for card in sorted(root.glob('card[0-9]*')):
        if not card.name[4:].isdigit():
            continue
        device = card / 'device'
        if not device.exists():
            continue
        row = {'device': device.resolve().name, 'card': card.name}

        def read(path, key, scale=1):
            try:
                value = float(path.read_text().strip()) / scale
                if value >= 0:
                    row[key] = value
            except (OSError, ValueError):
                pass

        read(device / 'gpu_busy_percent', 'utilization_percent')
        read(device / 'mem_info_vram_used', 'memory_used_bytes')
        read(device / 'mem_info_vram_total', 'memory_total_bytes')
        read(device / 'mem_info_gtt_used', 'shared_memory_used_bytes')
        read(device / 'mem_info_gtt_total', 'shared_memory_total_bytes')
        for hwmon in sorted((device / 'hwmon').glob('hwmon*')):
            read(hwmon / 'power1_average', 'power_watts', 1e6)
            if 'power_watts' not in row:
                read(hwmon / 'power1_input', 'power_watts', 1e6)
            read(hwmon / 'temp1_input', 'temperature_celsius', 1e3)
        if len(row) > 2:
            devices.append(row)
    return devices


def prometheus(snapshot):
    lines = []
    counters = {'preemptions', 'cache_hits', 'cache_misses', 'prefill_tokens_saved',
                'graph_replays', 'eager_forwards', 'draft_replays', 'draft_eager'}

    def emit(name, value, kind='gauge', labels=''):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        lines.extend([f'# HELP {name} SnowLLM {name.removeprefix("snowllm_").replace("_", " ")}.',
                      f'# TYPE {name} {kind}', f'{name}{labels} {value}'])

    for key, value in snapshot.get('engine', {}).items():
        if key == 'accounting':
            for metric, number in value.items():
                counter = metric.endswith(('_tokens', '_seconds', '_steps', '_rows'))
                emit('snowllm_' + metric + ('_total' if counter else ''), number,
                     'counter' if counter else 'gauge')
        else:
            emit('snowllm_' + key + ('_total' if key in counters else ''), value,
                 'counter' if key in counters else 'gauge')
    model_rows = {}
    for row in snapshot.get('models', []):
        label = str(row['id']).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        engine = row.get('engine') or {}
        values = {**engine, **engine.get('accounting', {}), **(row.get('scheduling') or {})}
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                model_rows.setdefault(key, []).append((label, value))
    for key, values in model_rows.items():
        counter = key in counters or key.endswith(('_tokens', '_seconds', '_steps', '_rows')) or key == 'steps'
        name = 'snowllm_model_' + key + ('_total' if counter else '')
        lines.extend([f'# HELP {name} Per-model {key.replace("_", " ")}.',
                      f'# TYPE {name} {"counter" if counter else "gauge"}'])
        lines.extend(f'{name}{{model="{label}"}} {value}' for label, value in values)
    # Each metric's HELP/TYPE lines must appear only once for multiple GPUs.
    sensor_rows = {}
    for gpu in snapshot.get('gpus', []):
        label = str(gpu['device']).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        for key, value in gpu.items():
            if isinstance(value, (int, float)):
                sensor_rows.setdefault(key, []).append((label, value))
    for key, values in sensor_rows.items():
        name = 'snowllm_gpu_' + key
        lines.extend([f'# HELP {name} GPU {key.replace("_", " ")}.', f'# TYPE {name} gauge'])
        lines.extend(f'{name}{{device="{label}"}} {value}' for label, value in values)
    return '\n'.join(lines) + '\n'

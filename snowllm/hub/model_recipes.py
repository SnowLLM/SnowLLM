# SPDX-License-Identifier: Apache-2.0
import re
import sys
from pathlib import Path

from ..errors import SnowLLMError


def system_memory_gib():
    from .._platform import WINDOWS, host_total
    if WINDOWS:
        import ctypes
        kib = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kib)):
            return kib.value / (1 << 20)
    else:
        try:
            data = Path('/run/udev/data/+dmi:id').read_text()
            sizes = re.findall(r'^E:MEMORY_DEVICE_\d+_SIZE=(\d+)$', data, re.M)
            total = sum(map(int, sizes))
            if total:
                return total / (1 << 30)
        except OSError:
            pass
    return round(host_total() / (1 << 30))


def validate_parameter(name, value, rule):
    if 'values' in rule:
        valid = any(type(value) is type(v) and value == v for v in rule['values'])
    else:
        valid = (rule['type'] == 'integer' and type(value) is int and rule['min'] <= value <= rule['max'])
    if not valid:
        raise SnowLLMError(f"{name}={value!r} is outside the recipe range {rule}")


def draft_specs(model):
    specs = model['specs'].copy()
    for name, spec in specs.items():
        if 'checkpoint' in spec:
            checkpoint = spec['checkpoint']
            subdir = model['sources'][checkpoint].get('subdir')
            sources = list(dict.fromkeys([*spec.get('sources', []), checkpoint]))
            defaults = {**spec.get('defaults', {}), 'dflash': f'@model/{subdir}' if subdir else '@model'}
            specs[name] = {**spec, 'sources': sources, 'defaults': defaults}
    return specs


def variant_specs(specs, variant):
    shared = [name for name, spec in specs.items()
              if name == 'none' or (spec.get('sources') and not spec.get('include')
                                    and spec.get('defaults', {}).get('dflash'))]
    local = variant.get('specs', [])
    excluded = variant.get('exclude_specs', [])
    unknown = (set(local) | set(excluded)) - specs.keys()
    if unknown:
        raise ValueError(f"unknown draft methods: {sorted(unknown)}")
    return [name for name in dict.fromkeys([*local, *shared]) if name not in excluded]


def expand(model):
    presets, variants, specs = model['presets'], model['variants'], draft_specs(model)
    supported = {name: variant_specs(specs, variant) for name, variant in variants.items()}
    rule = model['recommended']
    if rule['mode'] == 'fixed':
        candidates, tiers = [rule['preset']], {}
    elif rule['mode'] == 'by_memory':
        candidates, tiers = rule['candidates'], rule['min_system_gib']
        if set(candidates) != set(tiers):
            raise ValueError('each memory candidate needs a system RAM tier')
    else:
        raise ValueError(f"unknown recommendation mode {rule['mode']}")
    if not candidates or not set(candidates) <= presets.keys() or len(set(candidates)) != len(candidates):
        raise ValueError('invalid recommended presets')

    def resolve(key, variant, draft, preset, alias):
        v, s = variants[variant], specs[draft]
        if draft not in supported[variant]:
            raise ValueError(f"{variant} does not support {draft}")
        raw = {k: model[k] for k in ('model', 'type', 'family', 'serve_as', 'runtime') if k in model}
        if 'local_files' in v:
            raw.update(repo='TBD', include=v['local_files'])
        else:
            raw.update(model['sources'][v['source']], revision=v['revision'],
                       include=v['include'] + s.get('include', []), exclude=v.get('exclude', []))
        raw['sources'] = [model['sources'][n] for n in v.get('sources', []) + s.get('sources', [])]
        base = model.get('defaults', {}) | v.get('defaults', {}) | s.get('defaults', {})
        defaults = base | preset.get('draft_params', {}) | preset.get('inference', {}) | alias.get('defaults', {})
        if not alias:
            raw['base_defaults'] = {k: value for k, value in base.items() if value is not None}
            if 'spec_label' in s:
                raw['spec_label'] = s['spec_label']
        raw.update(id=key, precision=v['precision'], dir=alias.get('dir', v['dir']),
                   modalities=v['modalities'], spec=alias.get('spec_label', draft), bytes=v.get('download_bytes', {}).get(draft, 0),
                   defaults={k: value for k, value in defaults.items() if value is not None})
        raw.update({k: preset[k] for k in ('summary', 'note') if k in preset})
        raw.update({k: alias[k] for k in ('priority', 'status', 'requires') if k in alias})
        heads = model.get('vision_heads', {}).get(v.get('source'))
        if heads:
            for key, head in heads.items():
                if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', key) or key.casefold() == 'none'
                        or not re.fullmatch(r'(?:mmproj-[\w.-]+|[\w.-]+\.vision)\.gguf', head['file']) or head['bytes'] <= 0):
                    raise ValueError(f"{variant} has an invalid vision head {key!r}")
            selected = [key for key, head in heads.items() if head['file'] in raw['include']]
            if 'default_vision_head' in v:
                default = v['default_vision_head']
                if default != 'none' and default not in selected:
                    raise ValueError(f"{variant} must download its default vision head {default!r}")
                selected = [] if default == 'none' else [default]
            if len(selected) != int('image' in v['modalities']):
                raise ValueError(f"{variant} must include its declared default vision head")
            raw['vision_head'] = selected[0] if selected else 'none'
            raw['vision_file'] = heads[selected[0]]['file'] if selected else 'none'
            raw['vision_heads'] = heads
        return raw

    for name, preset in presets.items():
        draft = preset['draft']
        parameters = specs[draft].get('parameters', {})
        for k, v in preset['draft_params'].items():
            validate_parameter(k, v, parameters[k])
        row = resolve(model['id'] + ':' + name, preset['variant'], draft, preset, {})
        row.update(model_id=model['id'], variant=preset['variant'], preset=name, parameters=parameters,
                   inference=preset['inference'], draft_params=preset['draft_params'])
        if name in candidates:
            tier = tiers.get(name, 0)
            row.update(recommend_order=candidates.index(name), min_system_gib=tier)
        yield row
    covered = {(p['variant'], p['draft']) for p in presets.values()}
    for variant, methods in supported.items():
        for draft in methods:
            if (variant, draft) in covered:
                continue
            key = f"{model['id']}:{variant}:{draft}"
            row = resolve(key, variant, draft, {}, {})
            row.update(model_id=model['id'], variant=variant, parameters=specs[draft].get('parameters', {}))
            yield row
    for key, alias in model.get('legacy_ids', {}).items():
        preset = presets[alias['preset']] if 'preset' in alias else {}
        yield resolve(key, preset.get('variant', alias.get('variant')), preset.get('draft', alias.get('spec')),
                      preset, alias)


def recommended(name, candidates, memory_gib=None):
    ordered = sorted((r for r in candidates if 'recommend_order' in r.raw),
                     key=lambda r: r.raw['recommend_order'])
    for r in ordered:
        tier = r.raw['min_system_gib']
        if isinstance(tier, dict):
            tier = tier.get(sys.platform, tier['default'])
        if not tier:
            return r
        if memory_gib is None:
            memory_gib = system_memory_gib()
        if memory_gib >= tier:
            return r
    raise SnowLLMError(f"No recommended {name} preset for {memory_gib:g} GiB system RAM. "
                       f"Select a preset explicitly using {name}:PRESET.")

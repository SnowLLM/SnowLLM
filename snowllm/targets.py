# SPDX-License-Identifier: Apache-2.0
from typing import Literal


Target = Literal['gpu', 'npu']
TARGETS = ('gpu', 'npu')
NPU_UNAVAILABLE = 'NPU backend is not configured; NPU requires a dedicated runtime and device-specific weights, not generic GGUF checkpoints'


def normalize_target(value):
    if not isinstance(value, str) or value.lower() not in TARGETS:
        raise ValueError('target must be gpu or npu')
    return value.lower()


def resolve_target(options, target=None):
    selected = None
    for index, option in enumerate(options):
        if option == '--target':
            if index + 1 >= len(options):
                raise ValueError('--target requires gpu or npu')
            value = options[index + 1]
        elif option.startswith('--target='):
            value = option.partition('=')[2]
        else:
            continue
        value = normalize_target(value)
        if selected is not None and selected != value:
            raise ValueError('conflicting target options')
        selected = value
    if target is not None:
        target = normalize_target(target)
        if selected is not None and target != selected:
            raise ValueError('target conflicts with --target in model options')
    return target or selected or 'gpu'

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
"""ROCm wheel discovery that does not import Torch or initialize the GPU."""
from importlib.machinery import PathFinder
import os
from pathlib import Path
import sys


def configure_kpack_search() -> None:
    # ROCm's loader accepts absolute Torch archives via PATH_PREFIX, preserving
    # built-in paths. Mixing other libraries can shadow kernels on older loaders.
    if sys.platform != 'win32' or any(name in os.environ for name in
            ('ROCM_KPACK_PATH', 'ROCM_KPACK_PATH_PREFIX')):
        return
    try:
        spec = PathFinder.find_spec('torch')
    except (ImportError, ValueError, OSError):
        # Missing or mocked Torch must not break management commands.
        return
    if spec is None:
        return
    archives = set()
    for location in spec.submodule_search_locations or ():
        root = Path(location)
        for directory in (root/'.kpack', root/'lib/.kpack', root/'bin/.kpack'):
            try:
                archives.update(str(path.resolve()) for path in directory.glob('torch_*.kpack')
                                if path.is_file())
            except OSError:
                continue
    if archives:
        os.environ.setdefault('ROCM_KPACK_PATH_PREFIX', os.pathsep.join(sorted(archives)))

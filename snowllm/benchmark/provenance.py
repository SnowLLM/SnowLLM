# SPDX-License-Identifier: Apache-2.0
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def native_identity():
    from .._capi import _lib_path, ABI_VERSION
    try:
        release = version('snowllm-kernels')
    except PackageNotFoundError:
        release = None
    return dict(version=release, abi=ABI_VERSION, sha256=digest(_lib_path))


def source_identity():
    root = Path(__file__).resolve().parents[2]
    result = dict(client_sha256=digest(Path(__file__).with_name('context.py')))
    if (root / '.git').exists():
        for key, arguments in [('commit', ['rev-parse', 'HEAD']),
                               ('source_dirty', ['status', '--porcelain', '--untracked-files=normal'])]:
            value = subprocess.run(['git', '-C', str(root), *arguments],
                                   capture_output=True, text=True, timeout=10)
            if value.returncode == 0:
                result[key] = bool(value.stdout.strip()) if key == 'source_dirty' else value.stdout.strip()
    return result


def model_identity(root):
    root = Path(root).expanduser()
    result = {}
    manifest = root / 'snowllm-recipe.json'
    if manifest.is_file():
        result['recipe'] = json.loads(manifest.read_text())
        result['recipe_sha256'] = digest(manifest)
    files = [root] if root.is_file() else sorted(root.rglob('*'))
    result['files'] = [dict(path=path.name if root.is_file() else str(path.relative_to(root)), size=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns)
                       for path in files if path.is_file()]
    result['metadata_sha256'] = {str(path.relative_to(root)): digest(path)
                                 for path in sorted(root.glob('*.json'))
                                 if path.stat().st_size < 32 * 1024**2}
    return result

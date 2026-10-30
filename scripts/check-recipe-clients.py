#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIENTS = ('0.3.13',)
PROBE = '''
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from snowllm._version import __version__
from snowllm.hub import recipes
assert __version__ == sys.argv[2], __version__
book = recipes.catalogue(sys.argv[3], refresh=True)
current = json.loads(pathlib.Path(sys.argv[4]).read_text())['recipes']
expected = {r['id']: r for r in current
            if recipes.version(str(r.get('requires') or '0')) <= recipes.version(__version__)}
assert expected, 'no recipes compatible with this client'
assert {r.id for r in book} == set(expected), 'catalogue does not match client version requirements'
for raw in expected.values():
    target = recipes.find(raw['id'], book)
    defaults = dict(raw.get('defaults') or {})
    if __version__ == '0.3.13' and 'max_gpu_mem_util' in defaults:
        defaults['gpu_memory_utilization'] = defaults.pop('max_gpu_mem_util')
    assert target.defaults == defaults, (target.id, target.defaults, defaults)
    assert set(target.defaults) <= set(recipes.SETTABLE), target.id
    assert target.servable == (raw.get('status', 'supported') == 'supported'), target.id
print(json.dumps(dict(version=__version__, recipes=len(book))))
'''


def check(wheels, web):
    current_catalogue = web/'recipe-v2.json'
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        env = dict(os.environ, SNOWLLM_CONFIG=str(base/'config.json'),
                   XDG_CACHE_HOME=str(base/'cache'), SNOWLLM_MODELS=str(base/'models'),
                   PYTHONDONTWRITEBYTECODE='1')
        for client in CLIENTS:
            candidates = list(wheels.glob(f'snowllm-{client}-py3-none-any.whl'))
            if len(candidates) != 1:
                raise ValueError(f'expected one published SnowLLM {client} wheel')
            source = base/client
            with zipfile.ZipFile(candidates[0]) as archive:
                if any(Path(name).is_absolute() or '..' in Path(name).parts for name in archive.namelist()):
                    raise ValueError('invalid published wheel path')
                archive.extractall(source)
            catalogue = web/'recipes'/f'{client}.json'
            result = subprocess.run([sys.executable, '-c', PROBE, str(source), client,
                                     str(catalogue.resolve()), str(current_catalogue.resolve())], env=env, cwd=base,
                                    capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f'published client {client} failed:\n{result.stderr}')
            print(result.stdout.strip())
            if client == '0.3.13':
                result = subprocess.run([sys.executable, '-c', PROBE, str(source), client,
                                         str((web/'recipe.json').resolve()),
                                         str(current_catalogue.resolve())], env=env, cwd=base,
                                        capture_output=True, text=True)
                if result.returncode:
                    raise ValueError(f'the unversioned catalogue broke 0.3.13:\n{result.stderr}')
        from snowllm.hub import recipes
        book = recipes.catalogue(str(current_catalogue))
        print(f'current client: {len(book)} recipes, schema {recipes.SCHEMA} PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wheels', type=Path)
    parser.add_argument('--web', type=Path, default=ROOT/'web')
    args = parser.parse_args()
    try:
        check(args.wheels.resolve(), args.web.resolve())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f'check-recipe-clients.py: {error}\n')

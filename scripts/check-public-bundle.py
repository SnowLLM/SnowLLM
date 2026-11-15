import argparse
from pathlib import Path
import re
import subprocess


VERSION_TAG = re.compile(r'refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(?:\.dev[0-9]+)?')


def check(bundle, expected_main=None):
    output = subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], text=True)
    refs = {}
    for row in output.splitlines():
        oid, name = row.split()
        if name in refs or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', oid):
            raise ValueError('duplicate ref or invalid object ID in public bundle')
        if name not in ('HEAD', 'refs/heads/main') and not VERSION_TAG.fullmatch(name):
            raise ValueError('non-release ref in public bundle')
        refs[name] = oid
    main = refs.get('refs/heads/main')
    if main is None:
        raise ValueError('public bundle must contain main')
    if refs.get('HEAD', main) != main or expected_main is not None and main != expected_main:
        raise ValueError('public bundle main does not match the expected snapshot')
    print(f'public bundle: {bundle.name}, main and {sum(bool(VERSION_TAG.fullmatch(n)) for n in refs)} version tags PASS')
    return refs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--expected-main')
    args = parser.parse_args()
    try:
        check(args.bundle, args.expected_main)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'public bundle: {error}\n')

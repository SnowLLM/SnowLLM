# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import sys
import time

import httpx

from ..auth import bearer_headers, read_key
from ..daemon.cli import management_client_arguments
from .config import BenchmarkConfig, sizes
from .jobs import TERMINAL


def parser():
    p = argparse.ArgumentParser(prog='snowllm bench', description='Benchmark loaded models using the standard PG-19 or File Edit HTTP workload')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('context', add_help=False, help='run the standard HTTP client against an existing server')
    for name in ('run', 'list', 'show', 'cancel', 'export'):
        cmd = sub.add_parser(name)
        management_client_arguments(cmd)
        cmd.add_argument('--admin-port', type=int, default=8001)
        if name in ('show', 'cancel', 'export'):
            cmd.add_argument('run_id')
        if name == 'run':
            cmd.add_argument('--model-id', required=True)
            cmd.add_argument('--workload', choices=('pg19', 'file-edit'), default='pg19')
            cmd.add_argument('--isl', type=sizes, default=[1024, 8192, 32768, 131072, 225280])
            cmd.add_argument('--osl', type=int, default=argparse.SUPPRESS, help='output cap (PG-19: 256, File Edit: 1024)')
            cmd.add_argument('--conc', type=sizes, default=[1])
            cmd.add_argument('--reps', type=int, default=4)
            cmd.add_argument('--warm', type=int, default=1)
            cmd.add_argument('--thinking', choices=('auto', 'on', 'off'), default='auto')
            cmd.add_argument('--label', default='')
            cmd.add_argument('--detach', action='store_true', help='return immediately with the run ID')
            cmd.add_argument('--out', type=Path, help='export standard result rows after completion')
        if name == 'export':
            cmd.add_argument('--out', type=Path, required=True)
            cmd.add_argument('--results-only', action='store_true')
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['context']:
        from .context import main as context_main
        context_main(argv[1:], prog='snowllm bench context')
        return 0
    p = parser()
    a = p.parse_args(argv)
    if not 1 <= a.admin_port <= 65535:
        p.error('admin port must be between 1 and 65535')
    config = None
    if a.action == 'run':
        try:
            config = BenchmarkConfig(**{key: vars(a)[key] for key in BenchmarkConfig.model_fields if key in vars(a)})
        except ValueError as exc:
            p.error(str(exc))
        if a.detach and a.out:
            p.error('--out requires waiting for completion; remove --detach')
    from ..daemon.server import local_url
    root = (a.admin_url or local_url(a.admin_host, a.admin_port)) + '/daemon/benchmarks'
    headers = bearer_headers(read_key(p, a.admin_api_key_file, admin=True))
    run_id = None
    try:
        with httpx.Client(timeout=30, trust_env=False, headers=headers) as client:
            def request(method, url, **kwargs):
                response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()

            if a.action == 'run':
                result = request('POST', root, json=config.model_dump())
                run_id = result['id']
                print(f"Benchmark {run_id}", file=sys.stderr, flush=True)
                if not a.detach:
                    previous = None
                    while result['state'] not in TERMINAL:
                        result = request('GET', root + '/' + run_id)
                        progress = result.get('progress') or {}
                        message = (result['state'], progress.get('completed'), progress.get('total'))
                        if message != previous:
                            print(f"{message[0]} {message[1] or 0}/{message[2] or '?'}", file=sys.stderr, flush=True)
                            previous = message
                        if result['state'] not in TERMINAL:
                            time.sleep(.5)
                    if a.out:
                        a.out.parent.mkdir(parents=True, exist_ok=True)
                        a.out.write_text(json.dumps(result['rows'], indent=2) + '\n', encoding='utf-8')
            elif a.action == 'list':
                result = request('GET', root)
            elif a.action == 'cancel':
                result = request('POST', root + '/' + a.run_id + '/cancel')
            elif a.action == 'export':
                result = request('GET', root + '/' + a.run_id + '/export',
                                 params={'results_only': str(a.results_only).lower()})
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                print(str(a.out))
                return 0
            else:
                result = request('GET', root + '/' + a.run_id)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if isinstance(result, dict) and result.get('state') in ('failed', 'cancelled', 'interrupted') else 0
    except KeyboardInterrupt:
        if run_id:
            try:
                with httpx.Client(timeout=10, trust_env=False, headers=headers) as client:
                    response = client.post(root + '/' + run_id + '/cancel')
                    response.raise_for_status()
                print(f'Cancellation requested for {run_id}', file=sys.stderr)
            except httpx.HTTPError as exc:
                print(f'Could not cancel {run_id}: {exc}', file=sys.stderr)
        return 130
    except httpx.HTTPStatusError as exc:
        print(f'snowllm bench: {exc.response.text}', file=sys.stderr)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        print(f'snowllm bench: {exc}', file=sys.stderr)
    return 1

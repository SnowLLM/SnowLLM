# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
from pathlib import Path
import sys
import webbrowser

import httpx
from ..config import default_config_path


def parser():
    p = argparse.ArgumentParser(prog="snowllm", description="Local model management and Web UI")
    commands = p.add_subparsers(dest="command", required=True)
    daemon = commands.add_parser("webui",
                                 help="open the browser interface for models, chat, and settings")
    daemon.add_argument("--host", default="127.0.0.1", help="inference listener address")
    daemon.add_argument("--port", type=int, default=8000, help="inference listener port")
    daemon.add_argument("--admin-host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1",
                        help="local management listener address")
    daemon.add_argument("--admin-port", type=int, default=8001, help="Web UI and management port")
    daemon.add_argument("--config", type=Path,
                        default=default_config_path(), help="JSON configuration file")
    daemon.add_argument("--models-dir")
    daemon.add_argument("--load-timeout", type=float, default=600)
    daemon.add_argument("--no-browser", action="store_true",
                        help="do not open the Web UI in a browser")
    sub = daemon.add_subparsers(dest="action")
    for name in ("status", "unload", "default", "logs", "config", "pull", "load"):
        action = sub.add_parser(name)
        action.add_argument("--admin-port", "--port", dest="admin_port", type=int, default=argparse.SUPPRESS,
                            help="management port (--port is a compatibility alias)")
        action.add_argument("--admin-host", choices=("127.0.0.1", "localhost", "::1"), default=argparse.SUPPRESS)
        arguments(action, name)
    for name in ("status", "load", "unload", "default"):
        action = commands.add_parser(name)
        action.add_argument("--admin-port", "--port", dest="admin_port", type=int, default=8001,
                            help="management port (--port is a compatibility alias)")
        action.add_argument("--admin-host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
        arguments(action, name)
    return p


def arguments(p, name):
    if name == "load":
        p.add_argument("--force", action="store_true",
                       help="replace the sole loaded model and interrupt its requests")
        p.add_argument("--id", metavar='MODEL_ID', help="assign a management ID when loading one model")
        p.add_argument("model", nargs='+', metavar='MODEL')
        p.epilog = "Inference options after the model list apply to every model; use -- to separate them explicitly."
    elif name == "unload":
        selection = p.add_mutually_exclusive_group()
        selection.add_argument("id", nargs='?', metavar='MODEL_ID',
                               help="model ID from snowllm status; may be omitted when only one model is loaded")
        selection.add_argument("--all", action="store_true", help="unload all models in the service")
        p.add_argument("--force", action="store_true", help="interrupt requests for the selected model(s)")
    elif name == "default":
        p.add_argument("id", metavar='MODEL_ID', help="use this loaded model for requests without a model")
    elif name == "logs":
        p.add_argument("--download", action="store_true")
    elif name == "config":
        p.add_argument("--models-dir")
        p.add_argument("--recipes-url", help="catalogue URL or local JSON file; empty resets the default")
        p.add_argument("--library-path", help="development kernel library file; empty uses bundled kernels")
    elif name == "pull":
        p.add_argument("recipe")


def main(argv):
    p = parser()
    trailing = []
    if '--' in argv and (argv[:1] == ['load'] or argv[:1] == ['webui'] and 'load' in argv):
        index = argv.index('--')
        argv, trailing = argv[:index], argv[index + 1:]
    a, options = p.parse_known_args(argv)
    action = a.action if a.command == "webui" else a.command
    if options and action != "load":
        p.error("unrecognized arguments: " + " ".join(options))
    if not 1 <= a.admin_port <= 65535:
        p.error("admin port must be between 1 and 65535")
    from .server import local_url
    admin_url = local_url(a.admin_host, a.admin_port)
    if action is None:
        if not 1 <= a.port <= 65535:
            p.error("inference port must be between 1 and 65535")
        if a.port == a.admin_port:
            p.error("inference and management ports must be different")
        if a.load_timeout <= 0:
            p.error("load timeout must be positive")
        # Probe before creating resources: a second daemon must not disturb the first.
        with httpx.Client(timeout=1, trust_env=False) as client:
            try:
                response = client.get(admin_url + "/daemon/status")
                if response.is_success:
                    print(f"SnowLLM Web UI already running at {admin_url}")
                    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
                    if not a.no_browser:
                        open_browser(admin_url)
                    return 0
            except httpx.HTTPError:
                pass
        from .server import serve
        from .manager import DaemonManager
        try:
            manager = DaemonManager(a.config, a.load_timeout, a.models_dir)
        except (OSError, ValueError) as exc:
            print(f"snowllm webui: {exc}", file=sys.stderr)
            return 1
        print(f"SnowLLM Web UI / management: {admin_url}", flush=True)
        print(f"SnowLLM inference API: {local_url(a.host, a.port)}/v1", flush=True)
        print("Use snowllm load MODEL, snowllm status, snowllm unload MODEL_ID or --all; Ctrl+C stops all owned tasks.", flush=True)
        try:
            asyncio.run(serve(manager, host=a.host, port=a.port,
                              admin_host=a.admin_host, admin_port=a.admin_port,
                              browser=None if a.no_browser else open_browser))
        except (OSError, RuntimeError) as exc:
            print(f"snowllm webui: {exc}", file=sys.stderr)
            return 1
        return 0
    method, endpoint, payload, params = "GET", action, None, None
    if action == "load":
        method = "POST"
        options = [*options, *trailing]
        if a.id is not None:
            from ..serve.state import validate_model_id
            try:
                validate_model_id(a.id)
            except ValueError as exc:
                p.error(str(exc))
            if len(a.model) != 1 or a.force:
                p.error('--id requires one model and cannot be combined with --force')
        from ..targets import resolve_target
        entries = []
        for source in a.model:
            prefix, separator, path = source.partition(':')
            target = prefix if separator and prefix in ('gpu', 'npu') else None
            try:
                selected = resolve_target(options, target)
            except ValueError as exc:
                p.error(str(exc))
            entries.append(dict(model=path if target else source, options=list(options), target=selected))
        if a.id is not None:
            entries[0]['id'] = a.id
        if a.force:
            if len(entries) != 1:
                p.error('--force supports switching to one model; omit it for a batch load')
            item = entries[0]
            payload = dict(model=item['model'], options=item['options'], target=item['target'], force=True)
        else:
            endpoint, payload = 'models/load', dict(models=entries)
    elif action == "unload":
        method = "POST"
        if a.id is not None:
            endpoint, payload = 'models/unload', dict(id=a.id, force=a.force)
        else:
            params = {"force": str(a.force).lower()}
            if a.all:
                params['all_models'] = 'true'
    elif action == "default":
        method, endpoint, payload = "POST", 'models/default', dict(id=a.id)
    elif action == "config":
        method, payload = "PUT", {key: getattr(a, key) for key in ("models_dir", "recipes_url", "library_path")
                                  if getattr(a, key) is not None}
        if not payload:
            p.error("config requires --models-dir, --recipes-url, or --library-path")
    elif action == "pull":
        method, payload = "POST", dict(recipe=a.recipe)
    elif action == "logs":
        params = {"download": str(a.download).lower()}
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.request(method, f"{admin_url}/daemon/{endpoint}",
                                      json=payload, params=params)
            response.raise_for_status()
            result = response.json()
        notice = result.pop('load_advice_new', None)
        result.pop('load_advice', None)
        if notice is not None:
            print(f"snowllm: {notice['message']} {notice['url']} "
                  '(hide with SNOWLLM_NO_LOAD_ADVICE=1)', file=sys.stderr)
        print(result["text"] if action == "logs" else json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except httpx.HTTPStatusError as exc:
        print(f"snowllm {action}: {exc.response.text}", file=sys.stderr)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"snowllm {action}: cannot reach the Web UI service; start `snowllm webui`: {exc}", file=sys.stderr)
    return 1


def open_browser(url):
    try:
        if webbrowser.open(url):
            return
    except Exception as exc:
        print(f"snowllm webui: browser could not be opened: {exc}", file=sys.stderr)
    print(f"Open the Web UI manually: {url}", flush=True)

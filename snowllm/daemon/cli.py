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
    for name in ("status", "unload", "logs", "config", "pull", "load"):
        action = sub.add_parser(name)
        action.add_argument("--admin-port", "--port", dest="admin_port", type=int, default=argparse.SUPPRESS,
                            help="management port (--port is a compatibility alias)")
        action.add_argument("--admin-host", choices=("127.0.0.1", "localhost", "::1"), default=argparse.SUPPRESS)
        arguments(action, name)
    for name in ("status", "load", "unload"):
        action = commands.add_parser(name)
        action.add_argument("--admin-port", "--port", dest="admin_port", type=int, default=8001,
                            help="management port (--port is a compatibility alias)")
        action.add_argument("--admin-host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
        arguments(action, name)
    return p


def arguments(p, name):
    if name == "load":
        p.add_argument("--force", action="store_true", help="interrupt active requests")
        p.add_argument("model")
        p.add_argument("options", nargs=argparse.REMAINDER,
                       help="inference options after MODEL, optionally separated by --")
    elif name == "unload":
        p.add_argument("--force", action="store_true")
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
    a = p.parse_args(argv)
    action = a.action if a.command == "webui" else a.command
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
        print("Use snowllm load MODEL, snowllm status, snowllm unload; Ctrl+C stops all owned tasks.", flush=True)
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
        options = a.options[1:] if a.options[:1] == ["--"] else a.options
        payload = dict(model=a.model, options=options, force=a.force)
    elif action == "unload":
        method, params = "POST", {"force": str(a.force).lower()}
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

# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
from pathlib import Path
import sys
import webbrowser

import httpx
from ..auth import WEBUI_BASIC_USERNAME, auth_enabled, bearer_headers, read_key, startup_key
from ..config import default_config_path, read_config


class EnableAuth(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.enable_api_auth = True
        namespace.enable_webui_auth = True


def parser(*, startup_defaults=None):
    p = argparse.ArgumentParser(prog="snowllm", description="Local model management and Web UI")
    commands = p.add_subparsers(dest="command", required=True)
    daemon = commands.add_parser("webui",
                                 help="open the browser interface for models, chat, and settings")
    daemon.add_argument("--host", default="127.0.0.1", help="inference listener address")
    daemon.add_argument("--port", type=int, default=8000, help="inference listener port")
    daemon.add_argument("--admin-host", default="127.0.0.1", help="management listener address")
    daemon.add_argument("--admin-port", type=int, default=8001, help="Web UI and management port")
    daemon.add_argument("--api-key-file", type=Path,
                        help="inference key file, defaults to keys/inference.key beside the config when enabled")
    daemon.add_argument("--admin-api-key-file", type=Path,
                        help="management key file, defaults to keys/admin.key beside the config when enabled")
    daemon.add_argument("--enable-auth", nargs=0, action=EnableAuth, default=argparse.SUPPRESS,
                        help="enable authentication on both listeners")
    daemon.add_argument("--generate-keys", nargs=0, action=EnableAuth, default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    daemon.add_argument("--enable-api-auth", action=argparse.BooleanOptionalAction,
                        help="enable inference authentication and initialize its key file if needed")
    daemon.add_argument("--enable-webui-auth", action=argparse.BooleanOptionalAction,
                        help="enable Web UI and management authentication and initialize its key file if needed")
    daemon.add_argument("--show-keys", action=argparse.BooleanOptionalAction, default=True,
                        help="print active API keys on startup (default: enabled)")
    daemon.add_argument("--admin-url", type=parse_admin_url, help="public Web UI origin for a reverse proxy")
    daemon.add_argument("--config", type=Path,
                        default=default_config_path(), help="JSON configuration file")
    storage = daemon.add_mutually_exclusive_group()
    storage.add_argument("--models-dir")
    storage.add_argument('--models-dirs', nargs='+', metavar='DIR',
                         help='ordered model roots; downloads use the first')
    daemon.add_argument("--load-timeout", type=float, default=600)
    browser = daemon.add_mutually_exclusive_group()
    browser.add_argument("--no-browser", action="store_true", help="do not open the Web UI in a browser")
    browser.add_argument("--browser", dest="no_browser", action="store_false", default=argparse.SUPPRESS,
                         help="open the Web UI in a browser")
    if startup_defaults:
        daemon.set_defaults(**startup_defaults)
    sub = daemon.add_subparsers(dest="action")
    for name in ("status", "unload", "default", "logs", "config", "pull", "load"):
        action = sub.add_parser(name)
        action.add_argument("--admin-port", "--port", dest="admin_port", type=int, default=argparse.SUPPRESS,
                            help="management port (--port is a compatibility alias)")
        management_client_arguments(action, nested=True)
        arguments(action, name)
    for name in ("status", "load", "unload", "default"):
        action = commands.add_parser(name)
        action.add_argument("--admin-port", "--port", dest="admin_port", type=int, default=8001,
                            help="management port (--port is a compatibility alias)")
        management_client_arguments(action)
        arguments(action, name)
    return p


def webui_defaults(parser, config):
    fields = {
        "host": (str,), "port": (int,), "admin_host": (str,), "admin_port": (int,),
        "admin_url": (str, type(None)), "load_timeout": (int, float), "no_browser": (bool,),
    }
    defaults = {}
    for name, allowed in fields.items():
        if name in config:
            value = config[name]
            if type(value) not in allowed or isinstance(value, str) and not value.strip():
                parser.error(f"invalid {name} in the configuration")
            defaults[name] = value
    return defaults


def parse_admin_url(value):
    try:
        url = httpx.URL(value)
        if (url.scheme in ("http", "https") and url.host and not url.userinfo
                and url.raw_path == b"/" and not url.fragment):
            return str(url).rstrip("/")
    except httpx.InvalidURL:
        pass
    raise argparse.ArgumentTypeError("expected an HTTP(S) origin, such as https://snowllm.example.com")


def management_client_arguments(parser, *, nested=False):
    default = argparse.SUPPRESS if nested else None
    parser.add_argument("--admin-host", default=argparse.SUPPRESS if nested else "127.0.0.1")
    parser.add_argument("--admin-url", type=parse_admin_url, default=default, help="management HTTP(S) origin")
    parser.add_argument("--admin-api-key-file", type=Path, default=default,
                        help="management key file (or set SNOWLLM_ADMIN_API_KEY)")


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
        storage = p.add_mutually_exclusive_group()
        storage.add_argument("--models-dir")
        storage.add_argument('--models-dirs', nargs='+', metavar='DIR')
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
    if action is None:
        try:
            config_path = a.config.expanduser().resolve()
            config = read_config(config_path)
        except (OSError, ValueError, RuntimeError) as exc:
            p.error(str(exc))
        defaults = webui_defaults(p, config)
        if defaults:
            p = parser(startup_defaults=defaults)
            a = p.parse_args(argv)
    if not 1 <= a.admin_port <= 65535:
        p.error("admin port must be between 1 and 65535")
    from .server import local_url, validate_listener_keys
    local_admin_url = local_url(a.admin_host, a.admin_port)
    if action is None:
        if not 1 <= a.port <= 65535:
            p.error("inference port must be between 1 and 65535")
        if a.port == a.admin_port:
            p.error("inference and management ports must be different")
        if not 0 < a.load_timeout < float("inf"):
            p.error("load timeout must be positive and finite")
        api_auth = auth_enabled(p, a.enable_api_auth, config)
        admin_auth = auth_enabled(p, a.enable_webui_auth, config, admin=True)
        api_key, api_key_path = startup_key(p, a.api_key_file, config_path, config, enabled=api_auth)
        admin_key, admin_key_path = startup_key(p, a.admin_api_key_file, config_path, config,
                                               enabled=admin_auth, admin=True)
        try:
            validate_listener_keys(admin_key, api_key)
        except ValueError as exc:
            p.error(str(exc))
        # Probe before creating the manager so a second daemon cannot disturb the first.
        browser_url = a.admin_url or local_admin_url
        with httpx.Client(timeout=1, trust_env=False) as client:
            try:
                response = client.get(local_admin_url + "/daemon/status", headers=bearer_headers(admin_key))
                if response.is_success:
                    print(f"SnowLLM Web UI already running at {local_admin_url}")
                    if api_auth or admin_auth:
                        print("Restart the service for changes to the authentication settings to take effect.")
                    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
                    if not a.no_browser:
                        open_browser(browser_url)
                    return 0
            except httpx.HTTPError:
                pass
        from .server import serve
        from .manager import DaemonManager
        try:
            manager = DaemonManager(config_path, a.load_timeout, a.models_dirs or a.models_dir)
        except (OSError, ValueError) as exc:
            print(f"snowllm webui: {exc}", file=sys.stderr)
            return 1
        print(f"SnowLLM Web UI / management: {browser_url}", flush=True)
        print(f"SnowLLM inference API: {local_url(a.host, a.port)}/v1", flush=True)
        if api_key_path:
            print(f"Inference key file: {api_key_path}", flush=True)
        if a.show_keys and api_key:
            print(f"Inference API key: {api_key}", flush=True)
        if admin_key_path:
            print(f"Management key file: {admin_key_path}", flush=True)
        if admin_key:
            print(f"WebUI Basic Auth username: {WEBUI_BASIC_USERNAME}", flush=True)
            if a.show_keys:
                print(f"Management API key: {admin_key}", flush=True)
        if a.show_keys and (api_key or admin_key):
            print("Use --no-show-keys to hide API keys.", flush=True)
        print("Use snowllm load MODEL, snowllm status, snowllm unload MODEL_ID or --all; Ctrl+C stops all owned tasks.", flush=True)
        try:
            asyncio.run(serve(manager, host=a.host, port=a.port,
                              admin_host=a.admin_host, admin_port=a.admin_port,
                              api_key=api_key, admin_key=admin_key, admin_url=a.admin_url,
                              browser=None if a.no_browser else open_browser))
        except (OSError, RuntimeError) as exc:
            print(f"snowllm webui: {exc}", file=sys.stderr)
            return 1
        return 0
    if a.command == "webui" and (a.enable_api_auth is not None or a.enable_webui_auth is not None):
        p.error("authentication switches apply only to Web UI startup")
    admin_key = read_key(p, a.admin_api_key_file, admin=True)
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
        method, payload = "PUT", {key: getattr(a, key) for key in ("models_dir", "models_dirs", "recipes_url", "library_path")
                                  if getattr(a, key) is not None}
        if not payload:
            p.error("config requires --models-dir, --models-dirs, --recipes-url, or --library-path")
    elif action == "pull":
        method, payload = "POST", dict(recipe=a.recipe)
    elif action == "logs":
        params = {"download": str(a.download).lower()}
    try:
        with httpx.Client(timeout=10, trust_env=False, headers=bearer_headers(admin_key)) as client:
            response = client.request(method, f"{a.admin_url or local_admin_url}/daemon/{endpoint}",
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

# SPDX-License-Identifier: Apache-2.0
from base64 import b64decode
from binascii import Error
from hmac import compare_digest
from ipaddress import ip_address
import os
from pathlib import Path
import re
from secrets import token_urlsafe
from tempfile import TemporaryDirectory

from .checkpoint._cache_io import locked

API_KEY_ENV = "SNOWLLM_API_KEY"
ADMIN_KEY_ENV = "SNOWLLM_ADMIN_API_KEY"
WEBUI_BASIC_USERNAME = "snowllm"


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def read_key(parser, file, *, admin=False):
    option, env = ("--admin-api-key-file", ADMIN_KEY_ENV) if admin else ("--api-key-file", API_KEY_ENV)
    try:
        key = file.expanduser().read_text(encoding="utf-8").strip() if file else os.environ.get(env)
    except (OSError, UnicodeError):
        parser.error(f"could not read {option}")
    if key is not None and not re.fullmatch(r"[!-~]+", key):
        parser.error(f"{option} or {env} must contain one nonempty ASCII key without whitespace")
    return key


def generated_key(parser, path, *, admin=False):
    try:
        if not _read_key_file(path):
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock_path = path.with_name(f".{path.name}.lock")
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+b") as lock, locked(lock.fileno(), exclusive=True):
                if not _read_key_file(path):
                    with TemporaryDirectory(dir=path.parent) as directory:
                        pending = Path(directory) / path.name
                        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, "w", encoding="ascii") as stream:
                            stream.write(token_urlsafe(32) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        pending.replace(path)
    except (OSError, UnicodeError) as exc:
        parser.error(f"could not initialize key file {path}: {exc}")
    return read_key(parser, path, admin=admin)


def _read_key_file(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def auth_enabled(parser, cli_value, config, *, admin=False):
    if cli_value is not None:
        return cli_value
    name = "enable_webui_auth" if admin else "enable_api_auth"
    source = name if name in config else "enable_auth"
    value = config.get(source)
    if source in config and not isinstance(value, bool):
        parser.error(f"{source} in the configuration must be a boolean")
    return value


def startup_key(parser, file, config_path, config, *, enabled, admin=False):
    if enabled is False:
        return None, None
    field, env, basename = (("admin_api_key_file", ADMIN_KEY_ENV, "admin.key") if admin
                            else ("api_key_file", API_KEY_ENV, "inference.key"))
    if file is None:
        if env in os.environ:
            return read_key(parser, None, admin=admin), None
        if (configured := config.get(field)) is not None:
            if not isinstance(configured, str) or not configured.strip():
                parser.error(f"{field} in the configuration must be a nonempty path")
            file = config_path.parent / Path(configured).expanduser()
        elif enabled:
            file = config_path.parent / "keys" / basename
        else:
            return None, None
    try:
        path = file.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        parser.error(f"invalid {field} path: {exc}")
    if enabled:
        return generated_key(parser, path, admin=admin), path
    return read_key(parser, path, admin=admin), None


def bearer_headers(key):
    return {"Authorization": f"Bearer {key}"} if key else {}


def authorized(headers, key, *, admin=False, messages=False):
    if key is None:
        return True
    scheme, _, value = headers.get("authorization", "").partition(" ")
    secret = key.encode("ascii")
    if scheme.lower() == "bearer" and compare_digest(value.encode("latin-1"), secret):
        return True
    if messages:
        candidate = headers.get("x-api-key", "")
        if compare_digest(candidate.encode("latin-1"), secret):
            return True
    if admin and scheme.lower() == "basic":
        try:
            credentials = b64decode(value, validate=True)
        except (Error, ValueError):
            return False
        return compare_digest(credentials, WEBUI_BASIC_USERNAME.encode("ascii") + b":" + secret)
    return False


def unauthorized(*, admin=False):
    from starlette.responses import JSONResponse
    challenge = 'Basic realm="SnowLLM WebUI"' if admin else "Bearer"
    return JSONResponse({"error": {"message": "Missing or invalid API key", "type": "authentication_error"}},
                        status_code=401, headers={"WWW-Authenticate": challenge, "Cache-Control": "no-store"})


def inference_auth_failure(request, key, *, ready):
    if key is None:
        return None
    path = request.url.path
    if authorized(request.headers, key, messages=path in ("/v1/messages", "/v1/messages/count_tokens")):
        return None
    if path == "/health" and request.method == "GET" and "authorization" not in request.headers:
        from starlette.responses import JSONResponse
        return JSONResponse({"status": "ok" if ready else "unavailable"}, status_code=200 if ready else 503)
    return unauthorized()

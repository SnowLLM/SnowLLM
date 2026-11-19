# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from .manager import DaemonManager
from ..auth import authorized, inference_auth_failure, is_loopback, unauthorized
from ..targets import Target
from ..loading import LoadBatch


class Load(BaseModel):
    model: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    force: bool = False
    target: Target | None = None


class AddModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    id: str | None = Field(default=None, min_length=1, max_length=256)
    target: Target | None = None


class SelectModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    force: bool = False


class Directory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models_dir: str | None = Field(None, min_length=1)
    models_dirs: list[str] | None = Field(None, min_length=1)
    recipes_url: str | None = None
    library_path: str | None = None


class ModelPath(BaseModel):
    path: str = Field(min_length=1)


class Pull(BaseModel):
    recipe: str = Field(min_length=1, pattern=r"^(?:b/)?[A-Za-z0-9][A-Za-z0-9_.-]*$")


def storage_bytes(path):
    # Never follow directory or file symlinks while measuring owned storage.
    import os
    total = 0
    for current, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        for name in files:
            file = Path(current) / name
            if not file.is_symlink():
                stat = file.stat()
                total += stat.st_blocks * 512 if hasattr(stat, "st_blocks") else stat.st_size
    return total


def create_app(manager: DaemonManager, *, admin_host="127.0.0.1", admin_key=None, admin_url=None):
    # The dual-server runner owns the shared manager's lifetime.
    app = FastAPI(title="SnowLLM management")
    assets = Path(__file__).parent / "static"
    public_host = urlsplit(admin_url).netloc if admin_url else None
    local_only = admin_key is None and is_loopback(admin_host.lower())

    @app.middleware("http")
    async def management(request, call_next):
        authority = request.headers.get("host", "").lower()
        try:
            host = urlsplit("//" + authority).hostname
        except ValueError:
            host = None
        if not host or (local_only and authority != public_host and host != "testserver" and not is_loopback(host)):
            return JSONResponse({"detail": "invalid management host"}, status_code=403)
        origin = request.headers.get("origin")
        expected_origin = admin_url if authority == public_host else str(request.base_url).rstrip("/")
        if (origin and origin != expected_origin) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "cross-origin requests are not allowed"}, status_code=403)
        if not authorized(request.headers, admin_key, admin=True):
            return unauthorized(admin=True)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'")
        if request.url.path == "/" or request.url.path.startswith("/daemon"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(assets / "index.html")

    app.mount("/ui", StaticFiles(directory=assets), name="ui")

    from ..benchmark.config import BenchmarkConfig

    @app.get('/daemon/benchmarks')
    async def benchmark_list():
        return {'runs': manager.benchmarks.list()}

    @app.post('/daemon/benchmarks')
    async def benchmark_start(body: BenchmarkConfig):
        try:
            return await manager.benchmarks.start(body)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/daemon/benchmarks/{run_id}')
    async def benchmark_detail(run_id: str):
        try:
            return manager.benchmarks.detail(run_id)
        except (ValueError, LookupError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post('/daemon/benchmarks/{run_id}/cancel')
    async def benchmark_cancel(run_id: str):
        try:
            return await manager.benchmarks.cancel(run_id)
        except (ValueError, LookupError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/daemon/benchmarks/{run_id}/export')
    async def benchmark_export(run_id: str, results_only: bool = False):
        try:
            record = manager.benchmarks.detail(run_id)
        except (ValueError, LookupError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return JSONResponse(record['rows'] if results_only else record,
                            headers={'Content-Disposition': f'attachment; filename="snowllm-bench-{run_id}.json"'})

    @app.get("/daemon/recipes")
    async def catalogue(refresh: bool = False, bundle: bool = False):
        from ..hub import recipes
        from ..errors import SnowLLMError

        def inventory():
            roots = [Path(value) for value in getattr(manager, 'models_dirs', [manager.models_dir])]
            rows = []
            for recipe in recipes.catalogue(url=manager.status()["recipes_url"], refresh=refresh, bundle=bundle):
                path = recipes.installed(recipe, roots) or recipe.path(roots[0])
                rows.append({**recipe.raw, "id": recipe.ref, "path": str(path), "servable": recipe.servable,
                             "download_path": str(recipe.path(roots[0])),
                             "loadable": recipes.loadable(path, recipe),
                             "downloaded": recipes.complete(recipe, path),
                             "exists": path.is_dir()})
            return {"recipes": rows}
        try:
            return await asyncio.to_thread(inventory)
        except (SnowLLMError, OSError, ValueError) as exc:
            raise HTTPException(502, str(exc))

    @app.get("/daemon/local-models")
    async def local_models():
        from ..hub.recipes import loadable, MANIFEST

        def inventory():
            rows, seen = [], set()
            for value in getattr(manager, 'models_dirs', [manager.models_dir]):
                root = Path(value)
                if not root.exists():
                    continue
                for p in sorted(root.iterdir()):
                    if not p.is_dir():
                        continue
                    identity = p.resolve()
                    if identity in seen:
                        continue
                    if loadable(p) or (p / MANIFEST).is_file() or (
                            not p.is_symlink() and any(p.rglob('*.snowllm-part'))):
                        rows.append({"name": p.name, "path": str(p), "root": str(root),
                                     "bytes": storage_bytes(p) if not p.is_symlink() else None,
                                     "loadable": loadable(p), "deletable": not p.is_symlink()})
                        seen.add(identity)
            return {"models": rows}
        try:
            return await asyncio.to_thread(inventory)
        except OSError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/daemon/directories")
    async def directories(path: str | None = None):
        def browse():
            current = Path(path or manager.models_dir).expanduser().resolve()
            # A not-yet-created model root opens at its closest existing parent.
            if path is None:
                while not current.exists() and current.parent != current:
                    current = current.parent
            if not current.is_dir():
                raise ValueError("not a directory")
            children = []
            for child in current.iterdir():
                try:
                    if child.is_dir():
                        children.append({"name": child.name, "path": str(child)})
                except OSError:
                    continue
            usage = shutil.disk_usage(current)
            return {"path": str(current), "parent": str(current.parent),
                    "directories": sorted(children, key=lambda x: x["name"].casefold()),
                    "free_bytes": usage.free, "total_bytes": usage.total}
        try:
            return await asyncio.to_thread(browse)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/daemon/status")
    async def status():
        return await manager.refresh()

    @app.get('/daemon/tasks')
    async def tasks():
        await manager.refresh()
        return {'tasks': manager.task_snapshot()}

    @app.post('/daemon/tasks/{task_id}/cancel')
    async def cancel_task(task_id: str):
        try:
            return await manager.jobs.cancel(task_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, 'could not reach the worker to cancel the task') from exc

    @app.post('/daemon/models/load')
    async def add_model(body: AddModel | LoadBatch):
        try:
            if isinstance(body, LoadBatch):
                return await manager.add_models([item.model_dump() for item in body.models])
            return await manager.add_model(body.model, body.options, body.id, target=body.target)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post('/daemon/models/unload')
    async def unload_model(body: SelectModel):
        try:
            return await manager.unload_model(body.id, body.force)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post('/daemon/models/default')
    async def default_model(body: SelectModel):
        try:
            return await manager.set_default(body.id)
        except (ValueError, httpx.HTTPError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/daemon/load")
    async def load(body: Load):
        try:
            options = list(body.options)
            if body.target is not None:
                from ..targets import resolve_target
                target = resolve_target(options, body.target)
                options.extend(['--target', target])
            return await manager.switch(body.model, options, body.force)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/daemon/unload")
    async def unload(force: bool = False, all_models: bool = False):
        try:
            return await manager.switch(None, [], force, all_models=all_models)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/daemon/load/cancel")
    async def cancel_load():
        try:
            return await manager.cancel_load()
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/daemon/models/delete")
    async def delete_model(body: ModelPath):
        try:
            return await manager.delete_model(body.path)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc))

    @app.put("/daemon/config")
    async def config(body: Directory):
        try:
            changes = body.model_dump(exclude_unset=True)
            return manager.configure(changes.pop("models_dir", None), **changes)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/daemon/pull")
    async def pull(body: Pull):
        try:
            return manager.pull(body.recipe)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/daemon/download/cancel")
    async def cancel_download():
        return await manager.cancel_download()

    @app.get("/daemon/logs")
    async def logs(download: bool = False):
        return {"text": manager.read_log(download)}

    @app.get("/daemon/metrics")
    async def metrics(request: Request):
        return await forward(manager, "metrics", request, strip_credentials=admin_key is not None)

    @app.get("/daemon/inference/models")
    async def chat_models(request: Request):
        return await forward(manager, "v1/models", request, strip_credentials=admin_key is not None)

    @app.post("/daemon/inference/chat/completions")
    async def chat(request: Request):
        return await forward(manager, "v1/chat/completions", request, strip_credentials=admin_key is not None)

    @app.post("/daemon/profile/{action}")
    async def profile(action: str, request: Request):
        if action not in ("start", "stop"):
            raise HTTPException(404, "unknown profiling action")
        return await forward(manager, action + "_profile", request, strip_credentials=admin_key is not None)

    return app


def create_inference_app(manager: DaemonManager, *, api_key=None):
    app = FastAPI(title="SnowLLM inference", docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
    async def proxy(path: str, request: Request):
        if not (path.startswith("v1/") or path in ("health", "metrics")):
            raise HTTPException(404, "unknown inference endpoint")
        failure = inference_auth_failure(request, api_key, ready=manager.state == "ready")
        if failure is not None:
            return failure
        return await forward(manager, path, request, strip_credentials=api_key is not None)

    return app


async def forward(manager: DaemonManager, path: str, request: Request, *, strip_credentials=False):
    body = await request.body()
    if manager.benchmarks.busy and request.method != 'GET':
        raise HTTPException(503, 'a benchmark has exclusive use of inference')
    if manager.status()["state"] != "ready":
        raise HTTPException(503, f"model is {manager.state}")
    backend = manager.backend
    cancelled = asyncio.Event()
    task = None
    if request.method == 'POST' and path.startswith('v1/'):
        name = request.query_params.get('model')
        try:
            content = json.loads(body)
            if isinstance(content, dict) and isinstance(content.get('model'), str):
                name = content['model']
        except (ValueError, UnicodeError):
            pass
        record = next((row for row in manager.models if row['id'] == (name or manager.default_model)
                       or name in row.get('aliases', [])), None)

        async def cancel_request():
            cancelled.set()

        task = manager.jobs.begin_request((record or {}).get('id') or name or manager.default_model,
                                          (record or {}).get('target'), cancel_request)

    async def wait_for_response(awaitable):
        work = asyncio.ensure_future(awaitable)
        stop = asyncio.create_task(cancelled.wait())
        retrieved = False
        try:
            await asyncio.wait((work, stop), return_when=asyncio.FIRST_COMPLETED)
            if cancelled.is_set():
                raise asyncio.CancelledError
            result = await work
            retrieved = True
            return result
        finally:
            for pending in (work, stop):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(work, stop, return_exceptions=True)
            if not retrieved and not work.cancelled() and work.exception() is None:
                unused = work.result()
                if isinstance(unused, httpx.Response):
                    await unused.aclose()

    manager.active += 1
    response = None

    async def watch_disconnect():
        while True:
            if (await request.receive())['type'] == 'http.disconnect':
                cancelled.set()
                return
            await asyncio.sleep(0)

    watcher = asyncio.create_task(watch_disconnect()) if task else None
    try:
        excluded = {"host", "connection", "content-length", "transfer-encoding"}
        if strip_credentials:
            excluded.update(("authorization", "x-api-key"))
        headers = {k: v for k, v in request.headers.items()
                   if k not in excluded}
        upstream = manager.client.build_request(request.method, backend + "/" + path,
                    params=request.query_params, headers=headers, content=body)
        sending = manager.client.send(upstream, stream=True)
        response = await wait_for_response(sending) if task else await sending
    except BaseException as exc:
        manager.active -= 1
        if response:
            await response.aclose()
        if task:
            manager.jobs.finish_request(task['id'], 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed',
                                         None if isinstance(exc, asyncio.CancelledError) else str(exc))
        if cancelled.is_set():
            raise HTTPException(499, 'request cancelled') from exc
        if isinstance(exc, httpx.HTTPError):
            raise HTTPException(502, "inference connection failed") from exc
        raise
    finally:
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    completed = False

    async def chunks():
        nonlocal completed
        iterator = response.aiter_raw().__aiter__()
        try:
            while True:
                try:
                    chunk = await wait_for_response(anext(iterator)) if task else await anext(iterator)
                except StopAsyncIteration:
                    completed = True
                    return
                yield chunk
        except asyncio.CancelledError:
            if not cancelled.is_set():
                raise

    class ProxyResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            error = None
            try:
                await super().__call__(scope, receive, send)
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                # Also runs if the client disconnects before the iterator starts.
                try:
                    await response.aclose()
                finally:
                    manager.active -= 1
                    if task:
                        state = ('cancelled' if cancelled.is_set() or not completed and not error
                                 else 'failed' if error or response.status_code >= 400 else 'completed')
                        manager.jobs.finish_request(task['id'], state,
                            error or (f'HTTP {response.status_code}' if response.status_code >= 400 else None))

    headers = {k: v for k, v in response.headers.items()
               if k.lower() not in ("connection", "transfer-encoding", "keep-alive")}
    if task:
        headers['X-SnowLLM-Task-ID'] = task['id']
    return ProxyResponse(chunks(), status_code=response.status_code, headers=headers)

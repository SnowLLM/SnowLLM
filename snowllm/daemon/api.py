# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
import shutil
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from .manager import DaemonManager


class Load(BaseModel):
    model: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    force: bool = False


class Directory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models_dir: str | None = Field(None, min_length=1)
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


def create_app(manager: DaemonManager):
    # The dual-server runner owns the shared manager's lifetime.
    app = FastAPI(title="SnowLLM management")
    assets = Path(__file__).parent / "static"

    @app.middleware("http")
    async def local_management(request, call_next):
        from fastapi.responses import JSONResponse
        host = urlsplit("//" + request.headers.get("host", "")).hostname
        if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
            return JSONResponse({"detail": "invalid local host"}, status_code=403)
        origin = request.headers.get("origin")
        if (origin and origin != str(request.base_url).rstrip("/")) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "cross-origin requests are not allowed"}, status_code=403)
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

    @app.get("/daemon/recipes")
    async def catalogue(refresh: bool = False, bundle: bool = False):
        from ..hub import recipes
        from ..errors import SnowLLMError

        def inventory():
            root = Path(manager.models_dir)
            rows = []
            for recipe in recipes.catalogue(url=manager.status()["recipes_url"], refresh=refresh, bundle=bundle):
                path = recipe.path(root)
                rows.append({**recipe.raw, "id": recipe.ref, "path": str(path), "servable": recipe.servable,
                             "loadable": recipes.loadable(path),
                             "downloaded": (path / recipes.MANIFEST).is_file(),
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
            root = Path(manager.models_dir)
            if not root.exists():
                return {"models": []}
            return {"models": [{"name": p.name, "path": str(p), "bytes": storage_bytes(p) if not p.is_symlink() else None,
                                "loadable": loadable(p), "deletable": not p.is_symlink()}
                                for p in sorted(root.iterdir()) if p.is_dir() and
                                (loadable(p) or (p / MANIFEST).is_file() or (not p.is_symlink() and any(p.rglob("*.snowllm-part"))))]}
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
        return manager.status()

    @app.post("/daemon/load")
    async def load(body: Load):
        try:
            return await manager.switch(body.model, body.options, body.force)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/daemon/unload")
    async def unload(force: bool = False):
        try:
            return await manager.switch(None, [], force)
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
        return await forward(manager, "metrics", request)

    @app.get("/daemon/inference/models")
    async def chat_models(request: Request):
        return await forward(manager, "v1/models", request)

    @app.post("/daemon/inference/chat/completions")
    async def chat(request: Request):
        return await forward(manager, "v1/chat/completions", request)

    @app.post("/daemon/profile/{action}")
    async def profile(action: str, request: Request):
        if action not in ("start", "stop"):
            raise HTTPException(404, "unknown profiling action")
        return await forward(manager, action + "_profile", request)

    return app


def create_inference_app(manager: DaemonManager):
    app = FastAPI(title="SnowLLM inference", docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
    async def proxy(path: str, request: Request):
        if not (path.startswith("v1/") or path in ("health", "metrics")):
            raise HTTPException(404, "unknown inference endpoint")
        return await forward(manager, path, request)

    return app


async def forward(manager: DaemonManager, path: str, request: Request):
    if manager.status()["state"] != "ready":
        raise HTTPException(503, f"model is {manager.state}")
    backend = manager.backend
    manager.active += 1
    response = None
    try:
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "connection", "content-length", "transfer-encoding")}
        upstream = manager.client.build_request(request.method, backend + "/" + path,
                    params=request.query_params, headers=headers, content=await request.body())
        response = await manager.client.send(upstream, stream=True)
    except BaseException as exc:
        manager.active -= 1
        if response:
            await response.aclose()
        if isinstance(exc, httpx.HTTPError):
            raise HTTPException(502, "inference connection failed") from exc
        raise

    async def chunks():
        async for chunk in response.aiter_raw():
            yield chunk

    class ProxyResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                # Also runs if the client disconnects before the iterator starts.
                try:
                    await response.aclose()
                finally:
                    manager.active -= 1

    headers = {k: v for k, v in response.headers.items()
               if k.lower() not in ("connection", "transfer-encoding", "keep-alive")}
    return ProxyResponse(chunks(), status_code=response.status_code, headers=headers)

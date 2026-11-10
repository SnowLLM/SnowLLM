# SPDX-License-Identifier: Apache-2.0
import json
import os
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from ..targets import Target
from ..loading import LoadBatch


class LoadModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    id: str | None = Field(default=None, min_length=1, max_length=256)
    models_dir: str | None = None
    models_dirs: list[str] | None = Field(None, min_length=1)
    recipes_url: str | None = None
    target: Target | None = None


class SelectModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1)
    force: bool = False


class BatchLoad(LoadBatch):
    models_dir: str | None = None
    models_dirs: list[str] | None = Field(None, min_length=1)
    recipes_url: str | None = None


def model_sources(body):
    sources = {'SNOWLLM_MODELS': body.models_dir, 'SNOWLLM_RECIPES_URL': body.recipes_url}
    if body.models_dirs is not None:
        sources['SNOWLLM_MODEL_DIRS'] = json.dumps(body.models_dirs)
    elif body.models_dir is not None and os.environ.get('SNOWLLM_MODEL_DIRS'):
        sources['SNOWLLM_MODEL_DIRS'] = None
    return sources


def configure(app, registry):
    app.state.registry = registry

    @app.get('/worker/status')
    async def status():
        return registry.snapshot()

    @app.post('/worker/load')
    async def load(body: LoadModel):
        try:
            return await registry.load(
                body.model,
                body.options,
                body.id,
                target=body.target,
                sources=model_sources(body),
            )
        except (ValueError, RuntimeError, MemoryError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/worker/unload')
    async def unload(body: SelectModel):
        try:
            return await registry.unload(body.id, body.force)
        except (ValueError, RuntimeError, LookupError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/worker/load-many')
    async def load_many(body: BatchLoad):
        try:
            return await registry.load_many([item.model_dump() for item in body.models],
                                            sources=model_sources(body))
        except (ValueError, RuntimeError, MemoryError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/worker/default')
    async def default(body: SelectModel):
        try:
            return registry.set_default(body.id)
        except (ValueError, RuntimeError, LookupError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/worker/load/cancel')
    async def cancel():
        try:
            return registry.cancel_load()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc


def main(argv):
    import uvicorn
    from ..cli import parser
    from ..serve.api import app
    from ..serve.registry import ModelRegistry
    from .logging import access_log_config

    args = parser().parse_args(argv)
    source = getattr(args, 'model', None) or args.model_flag
    # Keep the recipe source so its defaults are resolved by the same builder as later loads.
    index = argv.index(source) if source else None
    options = argv[:index] + argv[index + 1 :] if source else []
    app.state.initial_model = (source, options) if source else None
    configure(app, ModelRegistry())
    uvicorn.run(app, host=args.host, port=args.port, log_level='info', log_config=access_log_config())

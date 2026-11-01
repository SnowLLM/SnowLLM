# SPDX-License-Identifier: Apache-2.0
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class LoadModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    id: str | None = Field(default=None, min_length=1, max_length=256)
    models_dir: str | None = None
    recipes_url: str | None = None


class SelectModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1)
    force: bool = False


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
                sources={
                    'SNOWLLM_MODELS': body.models_dir,
                    'SNOWLLM_RECIPES_URL': body.recipes_url,
                },
            )
        except (ValueError, RuntimeError, MemoryError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/worker/unload')
    async def unload(body: SelectModel):
        try:
            return await registry.unload(body.id, body.force)
        except (ValueError, RuntimeError, LookupError) as exc:
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

    args = parser().parse_args(argv)
    source = getattr(args, 'model', None) or args.model_flag
    if not source:
        raise SystemExit('a managed worker needs an initial model')
    # Keep the recipe source so its defaults are resolved by the same builder as later loads.
    index = argv.index(source)
    options = argv[:index] + argv[index + 1 :]
    app.state.initial_model = (source, options)
    configure(app, ModelRegistry())
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')

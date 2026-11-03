# SPDX-License-Identifier: Apache-2.0
import asyncio

from fastapi import HTTPException, Request
from fastapi.responses import Response

from .wire import pack, unpack


def configure(app):
    def runtime():
        from ..serve.state import serving

        value = getattr(serving().engine, 'mineru', None)
        if value is None:
            raise HTTPException(400, 'select a loaded auxiliary model bundle')
        return value

    @app.get('/v1/inference')
    async def describe_auxiliary():
        return runtime().describe()

    @app.post('/v1/inference/{component}')
    async def infer_auxiliary(component: str, request: Request):
        selected = runtime()
        try:
            inputs = await asyncio.to_thread(unpack, await request.body())
            result = await selected.infer(component, inputs)
            return Response(await asyncio.to_thread(pack, result), media_type='application/x-npz')
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except (MemoryError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

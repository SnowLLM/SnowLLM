# SPDX-License-Identifier: Apache-2.0
import asyncio

import httpx


async def send_internal(runtime, request, payload):
    from ..serve.api import app

    async def bound_app(scope, receive, send):
        state = getattr(runtime, "state", None)
        if state is not None:
            scope = {**scope, "snowllm.internal_state": state}
        await app(scope, receive, send)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(bound_app)) as client:
        response = await client.request(
            request.method, str(request.url), headers=request.headers, content=payload
        )
        await response.aread()
        return response


class InternalTransport(httpx.BaseTransport):
    def __init__(self, runtime):
        self.runtime = runtime

    def handle_request(self, request):
        self.runtime.check_cancelled()
        payload = request.read()
        response = asyncio.run_coroutine_threadsafe(
            send_internal(self.runtime, request, payload), self.runtime.loop
        ).result()
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )


class InternalAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, runtime):
        self.runtime = runtime

    async def handle_async_request(self, request):
        self.runtime.check_cancelled()
        payload = await request.aread()
        if asyncio.get_running_loop() is self.runtime.loop:
            response = await send_internal(self.runtime, request, payload)
        else:
            response = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    send_internal(self.runtime, request, payload), self.runtime.loop
                )
            )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )

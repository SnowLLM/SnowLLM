# SPDX-License-Identifier: Apache-2.0
import json

from starlette.responses import JSONResponse

from .state import _REQUEST_STATE


class ModelRouting:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if (state := scope.get('snowllm.internal_state')) is not None:
            # Internal model calls remain covered by the document request's lease while draining.
            token = _REQUEST_STATE.set(state)
            try:
                return await self.app(scope, receive, send)
            finally:
                _REQUEST_STATE.reset(token)
        registry = getattr(scope['app'].state, 'registry', None)
        path = scope['path']
        if registry is None or not path.startswith('/v1/') or path == '/v1/models':
            return await self.app(scope, receive, send)

        def error(status, exc):
            if path.startswith('/v1/messages'):
                kind = (
                    'api_error'
                    if status >= 500
                    else 'not_found_error'
                    if status == 404
                    else 'invalid_request_error'
                )
                body = {'type': 'error', 'error': {'type': kind, 'message': str(exc)}}
            else:
                body = {'detail': str(exc)}
            return JSONResponse(body, status_code=status)

        chunks = []
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunks.append(message)
            if not message.get('more_body'):
                break
        try:
            body = json.loads(b''.join(item.get('body', b'') for item in chunks) or b'{}')
            name = body.get('model') if isinstance(body, dict) else None
            record = registry.acquire(name)
        except (ValueError, TypeError) as exc:
            return await error(400, exc)(scope, receive, send)
        except LookupError as exc:
            return await error(404, exc)(scope, receive, send)
        except RuntimeError as exc:
            return await error(503, exc)(scope, receive, send)

        async def replay():
            return chunks.pop(0) if chunks else await receive()

        token = _REQUEST_STATE.set(record.state)
        try:
            await self.app(scope, replay, send)
        finally:
            _REQUEST_STATE.reset(token)
            record.active -= 1

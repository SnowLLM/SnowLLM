# SPDX-License-Identifier: Apache-2.0
"""Run the inference and management listeners with one shared supervisor."""
import asyncio
from contextlib import contextmanager
import signal
import threading

import uvicorn

from .api import create_app, create_inference_app
from ..updates import session as update_session


def local_url(host, port):
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


class Listener(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # Signals belong to the coordinator, never to an individual listener.
        yield

    def install_signal_handlers(self):
        # Compatibility with older uvicorn releases.
        pass


async def serve(manager, *, host, port, admin_host, admin_port, browser=None):
    manager.inference_url = local_url(host, port)
    servers = [
        Listener(uvicorn.Config(create_inference_app(manager), host=host, port=port,
                               log_level="info", timeout_graceful_shutdown=15)),
        Listener(uvicorn.Config(create_app(manager), host=admin_host, port=admin_port,
                               log_level="info", proxy_headers=False, timeout_graceful_shutdown=15)),
    ]
    sockets, tasks, old_handlers = [], [], {}

    def shutdown(signum, frame):
        for server in servers:
            if server.should_exit:
                server.force_exit = True
            server.should_exit = True

    try:
        # Reserve both ports before serving either endpoint. A bind failure must not
        # leave a partially started daemon or launch a browser.
        for server in servers:
            sockets.append(server.config.bind_socket())
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, shutdown)
        tasks = [asyncio.create_task(server.serve(sockets=[sock]))
                 for server, sock in zip(servers, sockets)]
        while not all(server.started for server in servers):
            if any(task.done() for task in tasks):
                raise RuntimeError("a daemon listener stopped during startup")
            await asyncio.sleep(.01)
        manager.start_telemetry()
        if browser:
            threading.Thread(target=browser, args=(local_url(admin_host, admin_port),), daemon=True).start()
        async with update_session():
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for server in servers:
            server.should_exit = True
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
        for sock in sockets:
            sock.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        await manager.close()
    for result in results:
        if isinstance(result, BaseException):
            raise result

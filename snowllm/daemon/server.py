# SPDX-License-Identifier: Apache-2.0
"""Run the inference and management listeners with one shared supervisor."""
import asyncio
from contextlib import contextmanager
import logging
import signal
import socket
import threading

import uvicorn

from .api import create_app, create_inference_app
from .logging import access_log_config
from ..auth import is_loopback
from ..updates import session as update_session


def local_url(host, port):
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def advertised_url(host, port):
    if host in ("0.0.0.0", "::"):
        family, destination = ((socket.AF_INET, "192.0.2.1") if host == "0.0.0.0"
                               else (socket.AF_INET6, "2001:db8::1"))
        try:
            # UDP connect selects a local source address without sending a packet.
            with socket.socket(family, socket.SOCK_DGRAM) as probe:
                probe.connect((destination, 9))
                host = probe.getsockname()[0]
        except OSError:
            pass
    return local_url(host, port)


def validate_listener_keys(admin_key, api_key):
    if api_key and api_key == admin_key:
        raise ValueError("inference and management API keys must be different")


class Listener(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # Signals belong to the coordinator, never to an individual listener.
        yield

    def install_signal_handlers(self):
        # Compatibility with older uvicorn releases.
        pass


async def serve(manager, *, host, port, admin_host, admin_port, api_key=None, admin_key=None,
                admin_url=None, browser=None):
    validate_listener_keys(admin_key, api_key)
    manager.inference_url = advertised_url(host, port)
    log_config = access_log_config()
    servers = [
        Listener(uvicorn.Config(create_inference_app(manager, api_key=api_key), host=host, port=port,
                               log_level="info", log_config=log_config, timeout_graceful_shutdown=15)),
        Listener(uvicorn.Config(create_app(manager, admin_host=admin_host, admin_key=admin_key, admin_url=admin_url),
                               host=admin_host, port=admin_port,
                               log_level="info", log_config=log_config, proxy_headers=False,
                               timeout_graceful_shutdown=15)),
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
        if admin_key is None and (not is_loopback(admin_host.lower()) or admin_url):
            logging.getLogger("uvicorn.error").warning(
                "Web UI and management API exposed without auth. "
                "Use --enable-auth (both listeners) or --enable-webui-auth (management only).")
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
        browser_url = admin_url or advertised_url(admin_host, admin_port)
        print(f"SnowLLM inference API: {manager.inference_url}/v1", flush=True)
        print(f"SnowLLM Web UI / management: {browser_url}", flush=True)
        if browser:
            threading.Thread(target=browser, args=(browser_url,), daemon=True).start()
        async with update_session(manager.update):
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

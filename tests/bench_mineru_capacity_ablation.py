# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch


@contextlib.asynccontextmanager
async def frontend(app):
    import uvicorn
    from mineru.model.runtime.ownership import RuntimeOwner, runtime_owner, release_runtime_owner
    from mineru.parser.mineru_parser import MinerUParser
    from snowllm.mineru.client import ModelClient
    from snowllm.mineru.injection import install

    address = socket.socket()
    address.bind(('127.0.0.1', 0))
    address.listen()
    url = f'http://127.0.0.1:{address.getsockname()[1]}'
    app.state.initial_model = None
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    serving = asyncio.create_task(server.serve(sockets=[address]))
    try:
        while not server.started:
            if serving.done():
                await serving
            await asyncio.sleep(0.01)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'MINERU_DEVICE_MODE': 'cpu'}))
            client = await asyncio.to_thread(ModelClient, url, 'documents')
            stack.callback(client.close)
            install(stack, client)
            owner = RuntimeOwner()
            token = runtime_owner.set(owner)
            stack.callback(runtime_owner.reset, token)
            stack.callback(release_runtime_owner, owner)

            async def request(path):
                def parse():
                    result = MinerUParser(tier='standard').parse(path)
                    return dict(markdown=result.markdown(), document=result.to_dict())

                return await asyncio.to_thread(parse)

            yield request
    finally:
        server.should_exit = True
        await serving


async def worker(args):
    import torch
    from snowllm.ops.arena import Arena, CapacityArena
    import snowllm.ops.arena as arena_module
    from snowllm.serve.api import app
    from snowllm.serve.registry import ModelRegistry

    torch.set_num_threads(2)
    registry = ModelRegistry()
    app.state.registry = registry
    os.environ["SNOWLLM_MODELS"] = str(args.model.parent)
    fixtures = args.fixtures
    report = dict(mode=args.mode, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  library_sha256=hashlib.sha256(Path(os.environ["SNOWLLM_LIB"]).read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256((fixtures / "long.pdf").read_bytes()).hexdigest(),
                  requests=[])
    active = None
    original_trace, original_pack = Arena.trace, arena_module.pack

    def trace(*a, **kw):
        start = time.perf_counter()
        try:
            return original_trace(*a, **kw)
        finally:
            if active is not None:
                active["trace_seconds"] += time.perf_counter() - start
                active["trace_calls"] += 1

    def pack(*a, **kw):
        start = time.perf_counter()
        try:
            return original_pack(*a, **kw)
        finally:
            if active is not None:
                active["pack_seconds"] += time.perf_counter() - start
                active["pack_calls"] += 1

    def exact(self, key, rows, fn, *, admit=None, **kw):
        plan = self.checked.get(key)
        if plan is None:
            self.capacities.clear()
            plan = self.prepare(key, rows, fn)
        else:
            self.reuses += 1
        return self.run(key, plan, fn, admit)

    try:
        with contextlib.ExitStack() as stack:
            if args.mode == "exact":
                stack.enter_context(patch.object(CapacityArena, "execute", exact))
            stack.enter_context(patch.object(Arena, "trace", trace))
            stack.enter_context(patch.object(arena_module, "pack", pack))
            await registry.load("b/mineru", ["--max-gpu-mem", "12GiB", "--stats-interval", "0"],
                                "documents")
            runtime = registry.records["documents"].state.engine.mineru
            async with frontend(app) as request:
                for _ in range(2):
                    await request(fixtures / "warmup.pdf")
                for index in range(args.requests):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    active = dict(index=index, trace_seconds=0.0, pack_seconds=0.0,
                                  trace_calls=0, pack_calls=0)
                    before = runtime.graphs.status()
                    start = time.perf_counter()
                    value = await request(fixtures / "long.pdf")
                    torch.cuda.synchronize()
                    active["seconds"] = time.perf_counter() - start
                    active["markdown_sha256"] = hashlib.sha256(value["markdown"].encode()).hexdigest()
                    active["pages"] = len(value["document"]["pages"])
                    assert active["pages"] == 80
                    after = runtime.graphs.status()
                    active["graph_delta"] = {k: after[k] - before[k] for k in
                                             ("captures", "replays", "fallbacks", "capacity_rebuilds",
                                              "capacity_probes", "capacity_reuses")}
                    active["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    report["requests"].append(active)
                    print(json.dumps(active), flush=True)
                    active = None
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["completed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        await registry.close()


def main(args):
    if args.worker:
        asyncio.run(worker(args))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.pairs):
        for mode in (("capacity", "exact") if index % 2 == 0 else ("exact", "capacity")):
            output = args.output / f"{mode}-{index + 1}.json"
            command = [str(args.withgpu), "--timeout", "600", sys.executable,
                       str(Path(__file__).resolve()), "--worker", "--mode", mode,
                       "--output", str(output), "--model", str(args.model),
                       "--fixtures", str(args.fixtures), "--requests", str(args.requests)]
            print("START", mode, index + 1, flush=True)
            with output.with_suffix(".log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    returncode = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            if returncode:
                print(output.with_suffix(".log").read_text()[-6000:], flush=True)
                raise SystemExit(returncode)
            print("DONE", mode, index + 1, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("capacity", "exact"), default="capacity")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--withgpu", type=Path, default=Path.home() / "withgpu/withgpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--requests", type=int, default=4)
    main(parser.parse_args())

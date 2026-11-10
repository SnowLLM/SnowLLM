# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import asdict
import json
import shutil
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import uuid

import httpx
from ..updates import Update
from ..errors import is_capacity_error
from ..telemetry import Telemetry, enabled as telemetry_enabled, announce_opt_out
from ..config import (models_directory, read_config, write_config, recipes_source,
                      library_path, bundled_library_path, DEFAULT_RECIPES_URL)


class DaemonManager:
    def __init__(self, config: Path, load_timeout: float = 600, models_dir: str | None = None):
        self.config_path = config.expanduser().resolve()
        self.config = read_config(self.config_path)
        if not self.config_path.exists():
            # Persist the built-in default, not a temporary environment/CLI override.
            self.config = {"models_dir": str(Path("~/models").expanduser().resolve()),
                           "recipes_url": DEFAULT_RECIPES_URL, "library_path": None}
            write_config(self.config_path, self.config)
        self.load_timeout = load_timeout
        self.inference_url = "http://127.0.0.1:8000"
        self.models_override = str(Path(models_dir).expanduser().resolve()) if models_dir else None
        self.process = None
        self.download = None
        self.download_recipe = None
        self.download_root = None
        self.last_load = None
        self.runtime_config = None
        self.update = Update()
        self.progress_path = self.config_path.parent / "download-progress.json"
        self.task = None
        self.transition_stage = None
        self.state = "idle"
        self.model = None
        self.models = []
        self.default_model = None
        self.operation = None
        self.shared_worker = False
        self.cancel_requested = False
        self.process_config = None
        self.memory = {}
        self.queues = {}
        self.target_errors = {}
        self.load_advice = None
        self._load_advice_shown = False
        self.instance = uuid.uuid4().hex
        self.error = None
        self.backend = None
        self.active = 0
        self.lock = asyncio.Lock()
        self.closing = False
        self.storage_busy = False
        self.telemetry = None
        self.log_path = self.config_path.parent / "inference.log"
        self.download_log = self.config_path.parent / "download.log"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5), trust_env=False)

    def start_telemetry(self):
        if not telemetry_enabled(self.config_path):
            announce_opt_out()
        elif self.telemetry is None:
            self.telemetry = Telemetry(self.telemetry_sample)
            self.telemetry.start()

    async def telemetry_sample(self):
        backend = self.backend
        if not backend:
            return None
        response = await self.client.get(backend + "/health", timeout=1)
        response.raise_for_status()
        if backend == self.backend:
            return response.json().get("telemetry")

    @property
    def models_dir(self):
        return str(models_directory(self.config_path, self.config, self.models_override)[0])

    def status(self):
        directory, source = models_directory(self.config_path, self.config, self.models_override)
        if self.process and self.process.poll() is not None and self.state == "ready":
            self.state = "failed"
            self.error = self.worker_error()
            self.models = [{**row, "state": "failed", "error": self.error} for row in self.models]
        recipe_url, recipe_source = recipes_source(self.config_path, self.config)
        library, lib_source = library_path(self.config_path, self.config)
        progress = None
        if self.download:
            try:
                progress = json.loads(self.progress_path.read_text())
            except (OSError, ValueError):
                pass
        return dict(load_advice=self.load_advice, process_config=self.process_config, memory=self.memory, queues=self.queues, target_errors=self.target_errors, cancel_requested=self.cancel_requested, models=self.models, default_model=self.default_model, operation=self.operation, last_load=self.last_load, runtime_config=self.runtime_config, update=asdict(self.update), state=self.state, model=self.model, error=self.error,
                    error_kind="capacity" if self.error and is_capacity_error(self.error) else None,
                    inference_url=self.inference_url,
                    active_requests=self.active, models_dir=str(directory),
                    config_file=str(self.config_path),
                    saved_models_dir=self.config.get("models_dir"),
                    models_dir_source=source,
                    recipes_url=recipe_url, recipes_url_source=recipe_source,
                    saved_recipes_url=self.config.get("recipes_url"),
                    library_path=library or bundled_library_path(), library_path_source=lib_source,
                    saved_library_path=self.config.get("library_path"),
                    pid=self.process.pid if self.process else None,
                    download=(dict(recipe=self.download_recipe, pid=self.download.pid,
                                   exit_code=self.download.poll(), progress=progress, log=str(self.download_log))
                              if self.download else None), log=str(self.log_path))

    def configure(self, directory=None, **settings):
        updated = dict(self.config)
        if directory is not None:
            if not directory.strip():
                raise ValueError("model directory must not be empty")
            path = Path(directory).expanduser()
            if not path.is_absolute():
                path = self.config_path.parent / path
            path = path.resolve()
            path.mkdir(parents=True, exist_ok=True)
            import tempfile
            with tempfile.TemporaryFile(dir=path):
                pass
            updated["models_dir"] = str(path)
        for key, value in settings.items():
            if key == "library_path" and self.backend and value != self.config.get(key):
                raise ValueError("unload all models before changing the kernel library")
            if key not in ("recipes_url", "library_path"):
                raise ValueError(f"unknown setting: {key}")
            if value is None or value == "":
                updated.pop(key, None)
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
            value = value.strip()
            # Validate the saved value independently of any environment override.
            if key == "library_path" or not value.startswith(("http://", "https://")):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = self.config_path.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise ValueError(f"{key} file does not exist: {path}")
                value = str(path)
            updated[key] = value
        write_config(self.config_path, updated)
        self.config = updated
        return self.status()

    def read_log(self, download=False):
        path = self.download_log if download else self.log_path
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 65536))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def worker_error(self):
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', self.read_log())
        for line in reversed(text.splitlines()):
            if re.match(r'^(?:snowllm(?:: error)?|[\w.]+(?:Error|Exception)):\s*', line):
                return line[:2048]
        return f"inference exited with code {self.process.returncode}; see {self.log_path}"

    def spawn(self, args, log):
        env = os.environ.copy()
        env["SNOWLLM_CONFIG"] = str(self.config_path)
        env["SNOWLLM_MODELS"] = self.models_dir
        env["SNOWLLM_RECIPES_URL"] = recipes_source(self.config_path, self.config)[0]
        # Only the supervisor sends telemetry; workers expose local counters.
        env["SNOWLLM_TELEMETRY_CHILD"] = "1"
        env["SNOWLLM_NO_UPDATE_CHECK"] = "1"
        configured_library, _ = library_path(self.config_path, self.config)
        if configured_library:
            env["SNOWLLM_LIB"] = configured_library
        if args and args[0] == "pull":
            env["SNOWLLM_DOWNLOAD_PROGRESS"] = str(self.progress_path)
        else:
            env.pop("SNOWLLM_DOWNLOAD_PROGRESS", None)
        command = [sys.executable, "-m", "snowllm.cli", *args]
        if args and args[0] != 'pull':
            command.append('--managed-worker')
            wrapper = Path.home() / 'withgpu' / 'withgpu'
            if wrapper.is_file():
                command = [str(wrapper), *command]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as output:
            return subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=os.name != "nt")

    async def stop_process(self, proc):
        if proc is None:
            return
        # Kill the entire group even if the leader exited.
        def terminate(sig):
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   capture_output=True)
                else:
                    os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
        terminate(signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.to_thread(proc.wait), 15)
        except asyncio.TimeoutError:
            terminate(getattr(signal, "SIGKILL", signal.SIGTERM))
            await asyncio.to_thread(proc.wait)
        if os.name != "nt":
            terminate(signal.SIGKILL)

    async def switch(self, model, options, force=False, *, all_models=False):
        from ..targets import NPU_UNAVAILABLE, resolve_target
        async with self.lock:
            if (self.process and self.process.poll() is None
                    and len([row for row in self.models if row.get("runtime_config")]) > 1
                    and not (model is None and all_models)):
                raise ValueError("multiple models are loaded; use `snowllm unload MODEL_ID` or "
                                 "`snowllm unload --all`; use `snowllm load MODEL` to add a model")
            if self.closing or (self.task and not self.task.done()):
                raise ValueError("a model transition is already in progress")
            # The supervisor owns the backend address; callers cannot override it.
            if any(not isinstance(x, str) for x in options):
                raise ValueError("options must be strings")
            if any(x.split("=", 1)[0] in ("--host", "--port", "--model", "--managed-worker") for x in options):
                raise ValueError("--host, --port and --model are managed by the daemon")
            if model is not None and resolve_target(options) == 'npu':
                raise ValueError(NPU_UNAVAILABLE)
            if model is not None:
                self._advise_batch_load()
            if model is not None:
                self.last_load = dict(model=model, options=list(options), force=force)
            self.state = "draining"
            self.transition_stage = "waiting"
            self.error = None
            self.task = asyncio.create_task(self.transition(model, options, force))
        return self.status()

    async def transition(self, model, options, force):
        try:
            while self.active and not force:
                await asyncio.sleep(.1)
            self.transition_stage = "stopping"
            if self.telemetry:
                await self.telemetry.unloading()
            await self.stop_process(self.process)
            self.process = None
            self.backend = None
            self.models = []
            self.default_model = None
            self.shared_worker = False
            self.process_config = None
            self.memory = {}
            self.queues = {}
            self.target_errors = {}
            self.model = model
            self.runtime_config = None
            if model is None:
                self.state = "idle"
                return
            self.state = "loading"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.backend = f"http://127.0.0.1:{port}"
            self.process = self.spawn([model, *options, "--host", "127.0.0.1", "--port", str(port)], self.log_path)
            deadline = asyncio.get_running_loop().time() + self.load_timeout
            while asyncio.get_running_loop().time() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(self.worker_error())
                try:
                    response = await self.client.get(self.backend + "/health", timeout=1)
                    if response.status_code == 200:
                        data = response.json()
                        self.runtime_config = data.get("runtime_config")
                        self.state = "ready"
                        if "models" in data:
                            self.shared_worker = True
                            self.accept_snapshot(data)
                        if self.telemetry:
                            self.telemetry.loaded(response.json().get("telemetry"))
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.25)
            raise RuntimeError(f"model loading timed out after {self.load_timeout:g}s; see {self.log_path}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc)
            self.state = "failed"
            await self.stop_process(self.process)
            self.process = None
            self.backend = None
        finally:
            self.transition_stage = None

    def accept_snapshot(self, snapshot):
        self.models = snapshot.get('models', [])
        self.process_config = snapshot.get('process_config')
        self.memory = snapshot.get('memory', {})
        self.queues = snapshot.get('queues', {})
        self.target_errors = snapshot.get('target_errors', {})
        self.default_model = snapshot.get('default_model')
        previous = self.operation
        self.operation = snapshot.get('operation')
        if self.operation is None and previous and previous.get('stage') == 'queued' and self.task and not self.task.done():
            self.operation = previous
        self.cancel_requested = snapshot.get('cancel_requested', False)
        default = next((row for row in self.models if row['id'] == self.default_model), None)
        self.model = self.default_model
        self.runtime_config = default.get('runtime_config') if default else None
        ready = any(row['state'] == 'ready' for row in self.models)
        if ready:
            self.state = 'ready'
        elif self.operation:
            self.state = 'draining' if self.operation.get('kind') == 'unload' else 'loading'
        else:
            self.state = 'idle'
        if snapshot.get('worker_error'):
            self.error = snapshot['worker_error']
            self.models = [{**row, 'state': 'failed', 'error': self.error} for row in self.models]
            self.state = 'failed'
        elif not ready and any(row['state'] == 'failed' for row in self.models):
            self.state = 'failed'

    async def refresh(self):
        if self.transition_stage is not None:
            return self.status()
        if self.shared_worker and self.backend and self.process and self.process.poll() is None:
            backend = self.backend
            try:
                response = await self.client.get(backend + '/worker/status', timeout=2)
                response.raise_for_status()
                if self.transition_stage is not None or backend != self.backend:
                    return self.status()
                snapshot = response.json()
                self.accept_snapshot(snapshot)
                if snapshot.get('worker_error'):
                    await self.stop_process(self.process)
                    self.process = self.backend = None
                    self.shared_worker = False
            except httpx.HTTPError:
                pass
        return self.status()

    async def add_model(self, model, options, model_id=None, *, target=None):
        from ..serve.state import validate_model_id
        from ..targets import resolve_target
        validate_model_id(model_id)
        if any(not isinstance(value, str) for value in options):
            raise ValueError('model options must be strings')
        if any(value.split('=', 1)[0] in ('--host', '--port', '--model', '--managed-worker') for value in options):
            raise ValueError('--host, --port and --model are managed by the daemon')
        target = resolve_target(options, target)
        options = list(options)
        if not any(value.split('=', 1)[0] == '--target' for value in options):
            options.extend(['--target', target])
        if not self.backend or not self.process or self.process.poll() is not None:
            if model_id:
                options = ['--served-model-name', model_id, *options]
            return await self.switch(model, options)
        self.last_load = dict(model=model, options=list(options), id=model_id, target=target)
        return await self.model_operation('load', dict(model=model, options=options, id=model_id,
                                                     target=target,
                                                     models_dir=self.models_dir, recipes_url=recipes_source(self.config_path, self.config)[0]))

    async def add_models(self, models):
        from ..loading import LoadBatch
        from ..targets import resolve_target
        entries = [item.model_dump() for item in LoadBatch(models=models).models]
        for item in entries:
            item['target'] = resolve_target(item['options'], item['target'])
            if any(value.split('=', 1)[0] in ('--host', '--port', '--model', '--managed-worker')
                   for value in item['options']):
                raise ValueError('--host, --port and --model are managed by the daemon')
        if len(entries) == 1:
            item = entries[0]
            return await self.add_model(item['model'], item['options'], item['id'], target=item['target'])
        self.last_load = dict(models=entries)
        return await self.model_operation('load', dict(models=entries, models_dir=self.models_dir,
            recipes_url=recipes_source(self.config_path, self.config)[0]))

    def _advise_batch_load(self):
        from ..loading import MODEL_LOADING_GUIDE
        if self._load_advice_shown or not any(row['state'] == 'ready' for row in self.models):
            return
        self._load_advice_shown = True
        if os.environ.get('SNOWLLM_NO_LOAD_ADVICE', '').lower() in ('1', 'true', 'yes', 'on'):
            return
        message = ('Load all models together, or load the model with the largest workspace first. '
                   'Growing a shared workspace replans and recaptures models on that target. '
                   'GPU and NPU use separate workspaces.')
        self.load_advice = dict(id=self.instance, message=message, url=MODEL_LOADING_GUIDE)
        print(f'[snowllm] {message} {MODEL_LOADING_GUIDE} '
              '(hide with SNOWLLM_NO_LOAD_ADVICE=1)', flush=True)
        return self.load_advice

    async def _start_shared_worker(self):
        self.state = 'loading'
        self.shared_worker = False
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.backend = f'http://127.0.0.1:{port}'
        self.process = self.spawn(['--host', '127.0.0.1', '--port', str(port)], self.log_path)
        try:
            deadline = asyncio.get_running_loop().time() + self.load_timeout
            while asyncio.get_running_loop().time() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(self.worker_error())
                try:
                    response = await self.client.get(self.backend + '/health', timeout=1)
                    if response.is_success:
                        self.shared_worker = True
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.1)
            raise RuntimeError(f'worker startup timed out after {self.load_timeout:g}s')
        except BaseException:
            await self.stop_process(self.process)
            self.process = self.backend = None
            raise

    async def unload_model(self, model_id, force=False):
        if not self.backend or not self.process or self.process.poll() is not None:
            async with self.lock:
                if self.task and not self.task.done():
                    raise ValueError('wait for the model operation to finish')
                if not any(row['id'] == model_id for row in self.models):
                    raise ValueError(f'unknown model: {model_id}')
                await self.stop_process(self.process)
                self.process = self.backend = None
                self.shared_worker = False
                self.models = [row for row in self.models if row['id'] != model_id]
                if self.default_model == model_id:
                    self.model = self.default_model = self.runtime_config = None
                self.state = 'failed' if self.models else 'idle'
                if not self.models:
                    self.error = None
                    self.memory = {}
                    self.process_config = None
                return self.status()
        return await self.model_operation('unload', dict(id=model_id, force=force))

    async def set_default(self, model_id):
        if not self.backend:
            raise ValueError('no GPU worker is running')
        async with self.lock:
            if self.task and not self.task.done():
                raise ValueError('wait for the model operation to finish')
            response = await self.client.post(self.backend + '/worker/default', json=dict(id=model_id))
            if not response.is_success:
                raise ValueError(response.json().get('detail', 'could not set default model'))
            self.accept_snapshot(response.json())
            return self.status()

    async def model_operation(self, kind, payload):
        notice = None
        async with self.lock:
            if self.closing or self.storage_busy or (self.task and not self.task.done()):
                raise ValueError('a model operation is already in progress')
            self.error = None
            if kind == 'load':
                notice = self._advise_batch_load()
            row = next((row for row in self.models if row['id'] == payload.get('id')), None)
            first = payload.get('models', [{}])[0]
            self.operation = dict(kind=kind, model=payload.get('id') or payload.get('model') or first.get('id') or first.get('model'),
                                  target=payload.get('target') or (row or {}).get('target', 'gpu'), stage='queued')
            self.task = asyncio.create_task(self._model_operation(kind, payload))
        result = self.status()
        if notice is not None:
            result['load_advice_new'] = notice
        return result

    async def _model_operation(self, kind, payload):
        try:
            if not self.backend or not self.process or self.process.poll() is not None:
                await self._start_shared_worker()
            endpoint = 'load-many' if kind == 'load' and 'models' in payload else kind
            pending = asyncio.create_task(self.client.post(self.backend + '/worker/' + endpoint, json=payload))
            try:
                response = await asyncio.wait_for(asyncio.shield(pending), self.load_timeout) if kind == 'load' else await pending
            except asyncio.TimeoutError:
                self.error = f'model loading exceeded {self.load_timeout:g}s; cancelling at a safe boundary'
                await self.client.post(self.backend + '/worker/load/cancel')
                response = await pending
            except asyncio.CancelledError:
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass
                raise
            if not response.is_success:
                raise ValueError(response.json().get('detail', 'model operation failed'))
            self.accept_snapshot(response.json())
            if not any(row.get('runtime_config') for row in self.models):
                await self.stop_process(self.process)
                self.process = self.backend = None
                self.shared_worker = False
                self.process_config = None
                self.memory = {}
                self.state = 'idle'
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc)
            if self.process is None or self.process.poll() is not None:
                self.state = 'failed'
                self.shared_worker = False
            await self.refresh()
        finally:
            self.operation = None

    def pull(self, recipe):
        if self.closing or self.storage_busy or (self.download and self.download.poll() is None):
            raise ValueError("a download or storage operation is already running")
        self.progress_path.unlink(missing_ok=True)
        self.download_root = self.models_dir
        self.download = self.spawn(["pull", recipe], self.download_log)
        self.download_recipe = recipe
        return self.status()

    async def cancel_download(self):
        # Keep the process reference until it has stopped, preventing a second pull.
        await self.stop_process(self.download)
        return self.status()

    async def cancel_load(self):
        async with self.lock:
            if not self.task or self.task.done():
                raise ValueError("no model transition is in progress")
            if self.shared_worker and self.backend and self.operation and self.operation.get('kind') == 'load':
                response = await self.client.post(self.backend + '/worker/load/cancel')
                if not response.is_success:
                    raise ValueError(response.json().get('detail', 'could not cancel model loading'))
                self.accept_snapshot(response.json())
                return self.status()
            if self.shared_worker and self.operation:
                raise ValueError('only loading can be cancelled; an unload is already in progress')
            waiting = self.transition_stage == "waiting"
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            # Cancellation during draining preserves the old ready worker.
            if self.process and self.process.poll() is None and self.state == "draining" and waiting:
                self.state = "ready"
            else:
                await self.stop_process(self.process)
                self.process = None
                self.backend = None
                self.model = None
                self.runtime_config = None
                self.state = "idle"
            self.error = None
            return self.status()

    async def delete_model(self, path):
        async with self.lock:
            if self.closing or (self.task and not self.task.done()):
                raise ValueError("wait for the model transition to finish")
            if self.download and self.download.poll() is None:
                raise ValueError("stop the download before deleting models")
            root = Path(self.models_dir).resolve()
            target = Path(path).expanduser().absolute()
            if target.is_symlink() or target.resolve().parent != root or target.resolve() == root:
                raise ValueError("only model directories directly inside model storage can be deleted; symlinks are excluded")
            target = target.resolve()
            if self.config_path == target or target in self.config_path.parents:
                raise ValueError("the daemon configuration directory cannot be deleted")
            if not target.is_dir():
                raise ValueError("model directory does not exist")
            for row in self.models:
                runtime = row.get('runtime_config') or {}
                for used in (runtime.get('model_path'), runtime.get('dflash_path')):
                    if used and (Path(used).resolve() == target or target in Path(used).resolve().parents):
                        raise ValueError('unload the model before deleting files used by inference')
            # Resolve recipes as well as bare paths before checking the active worker.
            active = self.runtime_config and self.runtime_config.get("model_path")
            if self.model:
                if not active:
                    active = self.model if Path(self.model).is_absolute() else str(root / self.model)
                    from ..hub import recipes
                    try:
                        match = next((r for r in recipes.catalogue(url=self.status()["recipes_url"],
                            bundle=self.model.startswith("b/")) if r.ref == self.model), None)
                        if match:
                            active = str(match.path(root))
                    except Exception:
                        raise ValueError("cannot verify the active model path; unload the model first")
                active_path = Path(active).resolve()
                draft = self.runtime_config and self.runtime_config.get("dflash_path")
                if active_path == target or target in active_path.parents or (draft and (Path(draft).resolve() == target or target in Path(draft).resolve().parents)):
                    raise ValueError("unload the model before deleting files used by inference")
            from ..hub.recipes import loadable, MANIFEST
            if not loadable(target) and not (target / MANIFEST).is_file() and not any(target.rglob("*.snowllm-part")):
                raise ValueError("directory does not contain a recognized model or partial download")
            self.storage_busy = True
            deletion = asyncio.create_task(asyncio.to_thread(shutil.rmtree, target))
            try:
                await asyncio.shield(deletion)
            except asyncio.CancelledError:
                await deletion
                raise
            finally:
                self.storage_busy = False
            return {"deleted": str(target)}

    async def close(self):
        self.closing = True
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.telemetry:
            await self.telemetry.unloading()
        await self.stop_process(self.process)
        await self.stop_process(self.download)
        self.process = self.download = None
        if self.telemetry:
            await self.telemetry.close()
        await self.client.aclose()

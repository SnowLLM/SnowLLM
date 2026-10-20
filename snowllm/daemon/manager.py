# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import shutil
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import httpx
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
        self.progress_path = self.config_path.parent / "download-progress.json"
        self.task = None
        self.transition_stage = None
        self.state = "idle"
        self.model = None
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
            self.error = f"inference exited with code {self.process.returncode}; see {self.log_path}"
        recipe_url, recipe_source = recipes_source(self.config_path, self.config)
        library, lib_source = library_path(self.config_path, self.config)
        progress = None
        if self.download:
            try:
                progress = json.loads(self.progress_path.read_text())
            except (OSError, ValueError):
                pass
        return dict(last_load=self.last_load, runtime_config=self.runtime_config, state=self.state, model=self.model, error=self.error,
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

    async def switch(self, model, options, force=False):
        async with self.lock:
            if self.closing or (self.task and not self.task.done()):
                raise ValueError("a model transition is already in progress")
            # The supervisor owns the backend address; callers cannot override it.
            if any(not isinstance(x, str) for x in options):
                raise ValueError("options must be strings")
            if any(x.split("=", 1)[0] in ("--host", "--port", "--model") for x in options):
                raise ValueError("--host, --port and --model are managed by the daemon")
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
                    raise RuntimeError(f"inference exited with code {self.process.returncode}; see {self.log_path}")
                try:
                    response = await self.client.get(self.backend + "/health", timeout=1)
                    if response.status_code == 200:
                        self.runtime_config = response.json().get("runtime_config")
                        self.state = "ready"
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
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            # Cancellation during draining preserves the old ready worker.
            if self.process and self.process.poll() is None and self.state == "draining" and self.transition_stage == "waiting":
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
            # Resolve recipes as well as bare paths before checking the active worker.
            active = self.runtime_config and self.runtime_config.get("model_path")
            if self.model:
                if not active:
                    active = self.model if Path(self.model).is_absolute() else str(root / self.model)
                    from ..hub import recipes
                    try:
                        match = next((r for r in recipes.catalogue(url=self.status()["recipes_url"]) if r.id == self.model), None)
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

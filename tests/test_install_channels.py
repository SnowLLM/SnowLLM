import csv
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "install.sh").read_text()
SOURCE = INSTALL.replace("\ncheck_machine\n", "\ninstall_snowllm\nreturn 0\n", 1)
assert SOURCE != INSTALL


@unittest.skipIf(os.name == "nt", "The installer runs in a POSIX shell")
class InstallerChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        cls.mirror = cls.base / "mirror"
        cls.home = cls.base / "install"
        cls.python = cls.home / "venv/bin/python"
        subprocess.run([sys.executable, "-m", "venv", str(cls.home / "venv")],
                       check=True, capture_output=True, text=True, timeout=60)
        cls.wheels = {}
        cls.make_wheel("snowllm-kernels", "1.0.0")
        for version in ("0.3.18", "0.3.19.dev2", "0.4.0.dev1"):
            cls.make_wheel("snowllm", version, "snowllm-kernels==1.0.0")
        cls.write_index("snowllm-kernels", ("1.0.0",))
        cls.write_index("snowllm", ("0.3.18", "0.3.19.dev2", "0.4.0.dev1"))
        mirror = cls.mirror

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(mirror), **kwargs)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.addClassCleanup(cls.close_server)
        cls.env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
        cls.env.pop("PYTHONPATH", None)
        cls.env.update(PIP_CONFIG_FILE=os.devnull,
                       PIP_INDEX_URL=f"http://127.0.0.1:{cls.server.server_port}/simple",
                       PIP_NO_CACHE_DIR="1", NO_PROXY="127.0.0.1", SNOWLLM_HOME=str(cls.home))

    @classmethod
    def close_server(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    @classmethod
    def make_wheel(cls, project, version, dependency=None):
        name = project.replace("-", "_")
        dist = f"{name}-{version}.dist-info"
        metadata = f"Metadata-Version: 2.1\nName: {project}\nVersion: {version}\n"
        if dependency:
            metadata += f"Requires-Dist: {dependency}\n"
        entries = {
            f"{name}/__init__.py": "",
            f"{dist}/METADATA": metadata,
            f"{dist}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        }
        record = io.StringIO()
        writer = csv.writer(record)
        for filename in (*entries, f"{dist}/RECORD"):
            writer.writerow((filename, "", ""))
        entries[f"{dist}/RECORD"] = record.getvalue()
        wheel = cls.mirror / "packages" / f"{name}-{version}-py3-none-any.whl"
        wheel.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel, "w") as archive:
            for filename, value in entries.items():
                archive.writestr(filename, value)
        cls.wheels[project, version] = wheel

    @classmethod
    def write_index(cls, project, versions):
        links = []
        for version in versions:
            wheel = cls.wheels[project, version]
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            yanked = ' data-yanked="broken"' if version == "0.4.0.dev1" else ""
            links.append(f'<a href="/packages/{wheel.name}#sha256={digest}"{yanked}>{wheel.name}</a>')
        index = cls.mirror / "simple" / project / "index.html"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text("\n".join(links))

    def install(self, *args, **environment):
        return subprocess.run(["sh", "-s", "--", *args], input=SOURCE, text=True,
                              env={**self.env, **environment}, cwd=self.base,
                              capture_output=True, timeout=30)

    def version(self, project="snowllm"):
        result = subprocess.run([str(self.python), "-c",
                                "import importlib.metadata, sys; print(importlib.metadata.version(sys.argv[1]))",
                                project], env=self.env, cwd=self.base, text=True,
                                capture_output=True, check=True, timeout=5)
        return result.stdout.strip()

    def test_default_install_upgrade_and_switch_back_from_dev(self):
        for args, expected in (((), "0.3.18"), (("--channel", "dev"), "0.3.19.dev2"),
                               (("--channel", "dev"), "0.3.19.dev2"),
                               (("--channel", "stable"), "0.3.18")):
            with self.subTest(args=args):
                result = self.install(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.version(), expected)
                self.assertEqual(self.version("snowllm-kernels"), "1.0.0")

    def test_stable_filters_globally_enabled_prereleases(self):
        self.assertEqual(self.install("--channel", "dev").returncode, 0)
        result = self.install("--channel", "stable", PIP_PRE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.version(), "0.3.18")

    def test_failed_lookup_does_not_replace_the_existing_dev_install(self):
        self.assertEqual(self.install("--channel", "dev").returncode, 0)
        result = self.install("--channel", "stable", PIP_NO_INDEX="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not resolve the latest stable", result.stderr)
        self.assertEqual(self.version(), "0.3.19.dev2")

    def test_missing_stable_release_does_not_fall_back_to_dev(self):
        self.assertEqual(self.install("--channel", "dev").returncode, 0)
        self.write_index("snowllm", ("0.3.19.dev2",))
        try:
            result = self.install("--channel", "stable")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(self.version(), "0.3.19.dev2")
        finally:
            self.write_index("snowllm", ("0.3.18", "0.3.19.dev2", "0.4.0.dev1"))


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""CPU-only process ownership checks; no Torch or GPU required."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'scripts/run-job.py'
WORKER = '''import json, os, pathlib, signal, subprocess, sys, time
if os.environ.get('STUB_STUBBORN'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
time.sleep(.1)
pathlib.Path(os.environ['STUB_PIDS']).write_text(json.dumps([os.getpid(), child.pid]))
if not os.environ.get('STUB_ORPHAN'):
    time.sleep(60)
'''


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


class RunJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='snowllm-job-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.env = os.environ.copy()
        self.env.pop('SNOWLLM_TIMEOUT', None)
        self.env.pop('SNOWLLM_NO_SUITE', None)
        self.env['PYTHON'] = sys.executable
        self.env['STUB_PIDS'] = str(self.base / 'pids')
        self.script('withgpu', "raise SystemExit('unexpected GPU wrapper')")
        self.env['PATH'] = str(self.base) + os.pathsep + self.env['PATH']

    def script(self, name, code):
        p = self.base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'#!{sys.executable}\n' + code)
        p.chmod(0o755)
        return p

    def run_job(self, *args, env=None):
        return subprocess.run([sys.executable, str(JOB), *args], env=env or self.env,
                              capture_output=True, text=True, timeout=8)

    def start(self, command, env=None, cwd=None):
        p = subprocess.Popen(command, env=env or self.env, cwd=cwd,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def clean():
            if p.poll() is None:
                p.terminate()
                p.communicate(timeout=6)
            path = Path(self.env['STUB_PIDS'])
            if path.exists():
                for pid in json.loads(path.read_text()):
                    if alive(pid):
                        os.kill(pid, signal.SIGKILL)
        self.addCleanup(clean)
        deadline = time.monotonic() + 5
        while not Path(self.env['STUB_PIDS']).exists():
            if p.poll() is not None or time.monotonic() >= deadline:
                self.fail(f'worker not ready: {p.communicate(timeout=6)}')
            time.sleep(.02)
        return p

    def assert_collected(self):
        pids = json.loads(Path(self.env['STUB_PIDS']).read_text())
        self.assertFalse(any(alive(pid) for pid in pids), str(pids))

    def sandbox(self):
        for name in ('scripts/run-tests.sh', 'scripts/run-job.py', 'benchmarks/_jobs.sh',
                     'benchmarks/launch_snowllm.sh', 'benchmarks/launch_llama.cpp.sh',
                     'benchmarks/all_bench.sh', 'benchmarks/run.sh', 'benchmarks/demo/run.sh',
                     'benchmarks/run-aime.sh'):
            dst = self.base / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, dst)
        (self.base / 'tests').mkdir(exist_ok=True)

    def test_exit_codes_and_arguments(self):
        r = self.run_job('--', sys.executable, '-c',
                         "import sys; "
                         "assert sys.argv[1:] == ['a b', '$(literal)']; sys.exit(7)",
                         'a b', '$(literal)')
        self.assertEqual(r.returncode, 7, r.stderr)
        r = self.run_job('--', sys.executable, '-c',
                         'import os,signal; os.kill(os.getpid(), signal.SIGTERM)')
        self.assertEqual(r.returncode, 143)
        self.assertEqual(self.run_job('--', '/does/not/exist').returncode, 127)

    def test_no_implicit_gpu_wrapper(self):
        r = self.run_job('--', sys.executable, '-c', 'pass')
        self.assertEqual(r.returncode, 0)

    def test_invalid_timeout(self):
        for value in ('0', '-1', 'nan', 'inf'):
            self.assertEqual(self.run_job('--timeout', value, '--', 'true').returncode, 2)

    def test_timeout_cleans_stubborn_process_group(self):
        self.env['STUB_STUBBORN'] = '1'
        worker = self.script('worker', WORKER)
        p = self.start([sys.executable, str(JOB), '--timeout', '.5', '--', str(worker)])
        p.communicate(timeout=6)
        self.assertEqual(p.returncode, 124)
        self.assert_collected()

    def test_signal_cleans_process_group(self):
        worker = self.script('worker', WORKER)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            with self.subTest(signal=sig):
                Path(self.env['STUB_PIDS']).unlink(missing_ok=True)
                p = self.start([sys.executable, str(JOB), '--', str(worker)])
                p.send_signal(sig)
                p.communicate(timeout=6)
                self.assertEqual(p.returncode, 128 + sig)
                self.assert_collected()

    def test_normal_exit_collects_background_child(self):
        self.env['STUB_ORPHAN'] = '1'
        worker = self.script('worker', WORKER)
        p = self.start([sys.executable, str(JOB), '--', str(worker)])
        p.communicate(timeout=6)
        self.assertEqual(p.returncode, 0)
        self.assert_collected()




    def test_llama_launcher_cancel(self):
        self.sandbox()
        worker = self.script('llama-server', WORKER)
        gguf = self.base / 'stub.gguf'
        gguf.touch()
        self.env.update(LLAMA=str(worker),
                        GGUF=str(gguf), NAME='stub', STUB_ROOT=str(self.base))
        fake_py = self.script('metadata-python', '''import os, sys
if sys.argv[1] == '-c':
    if 'recipes.locate' in sys.argv[2]: print(os.environ['STUB_ROOT'], '-')
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        self.env['PYTHON'] = str(fake_py)
        p = self.start(['bash', 'benchmarks/launch_llama.cpp.sh', 'stub'], cwd=self.base)
        p.terminate()
        out, _ = p.communicate(timeout=6)
        self.assertEqual(p.returncode, 143, out)
        self.assert_collected()

    def test_demo_reuse_is_cpu_only(self):
        self.sandbox()
        self.env['NO_RECORD'] = '1'
        self.script('benchmarks/demo/gen_access_log.py', 'pass')
        fake_py = self.script('render-python', '''import os, sys
if sys.argv[1].endswith('render.py'):
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        self.env['PYTHON'] = str(fake_py)
        r = subprocess.run(['bash', 'benchmarks/demo/run.sh'], env=self.env, cwd=self.base,
                           capture_output=True, text=True, timeout=6)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def recording_stub(self):
        self.sandbox()
        worker = self.script('server', WORKER)
        self.env.update(STUB_SERVER=str(worker),
                        STUB_CLIENT=str(self.base/'client-pid'))
        self.env['PATH'] = str(self.base) + os.pathsep + self.env['PATH']
        self.script('curl', '''import os, pathlib, time
while not pathlib.Path(os.environ['STUB_PIDS']).exists(): time.sleep(.02)
''')
        self.script('llama', 'import os; os.execl(os.environ["STUB_SERVER"], "server")')
        fake_py = self.script('fake-python', '''import os, pathlib, sys, time
if sys.argv[1] == '-m':
    os.execl(os.environ['STUB_SERVER'], 'server')
elif sys.argv[1].endswith(('capture.py', 'eval_aime.py')):
    pathlib.Path(os.environ['STUB_CLIENT']).write_text(str(os.getpid()))
    time.sleep(60)
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        self.env['PYTHON'] = str(fake_py)

    def assert_recording_cancel(self, script):
        p = self.start(['bash', script], cwd=self.base)
        marker = Path(self.env['STUB_CLIENT'])
        deadline = time.monotonic() + 5
        while not marker.exists():
            if time.monotonic() >= deadline or p.poll() is not None:
                self.fail(f'client not started: {p.communicate(timeout=6)}')
            time.sleep(.02)
        client = int(marker.read_text())
        p.terminate()
        out, _ = p.communicate(timeout=6)
        self.assertEqual(p.returncode, 143, out)
        self.assert_collected()
        self.assertFalse(alive(client))

    def test_demo_record_cancel_cleans_server_and_client(self):
        self.recording_stub()
        self.script('benchmarks/demo/gen_access_log.py', 'pass')
        self.assert_recording_cancel('benchmarks/demo/run.sh')

    def test_aime_cancel_cleans_server_and_client(self):
        self.recording_stub()
        self.env['SNOWLLM_SERVER_LOG'] = str(self.base/'server.log')
        self.assert_recording_cancel('benchmarks/run-aime.sh')

    def benchmark_stub(self, client):
        self.sandbox()
        worker = self.script('server', WORKER)
        self.env['STUB_SERVER'] = str(worker)
        fake_py = self.script('metadata-python', '''import os, sys
if sys.argv[1] == '-c': sys.exit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        self.env['PYTHON'] = str(fake_py)
        self.script('snowllm', 'import os; os.execl(os.environ["STUB_SERVER"], "server")')
        self.env['PATH'] = str(self.base) + os.pathsep + self.env['PATH']
        self.script('benchmarks/perf_sweep.sh', client)

    def test_benchmark_failure_cleans_server(self):
        self.benchmark_stub('''import os, pathlib, time
while not pathlib.Path(os.environ['STUB_PIDS']).exists(): time.sleep(.02)
raise SystemExit(23)
''')
        p = self.start(['bash', 'benchmarks/all_bench.sh'], cwd=self.base)
        out, _ = p.communicate(timeout=6)
        self.assertEqual(p.returncode, 23, out)
        self.assert_collected()

    def test_benchmark_cancel_cleans_server_and_client(self):
        self.env['STUB_CLIENT'] = str(self.base / 'client-pid')
        self.benchmark_stub('''import os, pathlib, time
pathlib.Path(os.environ['STUB_CLIENT']).write_text(str(os.getpid()))
time.sleep(60)
''')
        self.assert_recording_cancel('benchmarks/all_bench.sh')

    def test_copied_benchmark_command_runs_independently(self):
        self.benchmark_stub('''import os, pathlib, sys, time
assert os.environ['ENGINE'] == 'snowllm'
assert os.environ['RECIPE'] == 'qwen3.6-35b-a3b-q4-k-xl'
assert os.environ['LABEL'] == 'qwen3_6_35b_a3b_q4_k_xl_mtp_snowllm'
assert os.environ['ISL'] == '1024,8192,32768,131072'
assert os.environ['COOL'] == '60'
assert sys.argv[1:] == ['127.0.0.1:8000']
while not pathlib.Path(os.environ['STUB_PIDS']).exists(): time.sleep(.02)
''')
        self.env['STUB_SERVER_ARGS'] = str(self.base / 'server-args')
        self.script('snowllm', '''import json, os, pathlib, sys
pathlib.Path(os.environ['STUB_SERVER_ARGS']).write_text(json.dumps(sys.argv[1:]))
os.execl(os.environ['STUB_SERVER'], 'server')
''')
        command = next(line for line in (ROOT / 'benchmarks/all_bench.sh').read_text().splitlines()
                       if line.endswith('--num-spec 2 & wait $!'))
        p = self.start(['bash', '-c', command], cwd=self.base)
        out, _ = p.communicate(timeout=6)
        self.assertEqual(p.returncode, 0, out)
        self.assert_collected()
        args = json.loads(Path(self.env['STUB_SERVER_ARGS']).read_text())
        self.assertEqual(args[0], 'qwen3.6-35b-a3b-q4-k-xl')
        self.assertEqual(args[-2:], ['--num-spec', '2'])


if __name__ == '__main__':
    unittest.main()

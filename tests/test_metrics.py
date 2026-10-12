# SPDX-License-Identifier: Apache-2.0
"""CPU-only accounting and sensor tests using synthetic hardware files."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from snowllm.metrics import RequestTiming, gpu_sensors


class MetricsTests(unittest.TestCase):
    def test_actual_scheduler_records_uncached_prefill_and_decode(self):
        # Execute the production scheduling method with a CPU runner and clock;
        # importing Engine itself would import the GPU kernel library.
        tree = ast.parse(Path('snowllm/engine/__init__.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_step')
        scope = {'time': SimpleNamespace(perf_counter=Mock(side_effect=[10., 10.5, 11., 11.1]))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<scheduler>', 'exec'), scope)
        req = SimpleNamespace(num_prefilled=80, prefilled=False, prompt=list(range(100)),
                              out=[], timing=RequestTiming())
        def prefill(r):
            r.num_prefilled = 100
            r.prefilled = True
            r.out.append(1)  # first output token belongs to prefill
        def decode(n):
            req.out.extend([2, 3, 4])  # speculative acceptance
            return 3, 1
        engine = SimpleNamespace(
            running=[req], max_num_seqs=1, _retire_finished=lambda: None,
            _pending_prefill=lambda: req if not req.prefilled else None,
            _prefill_turn=True, _prefill_chunk=prefill,
            _batchable_prefill_group=lambda: [], _decode_or_verify=decode)
        self.assertEqual(scope['_step'](engine), ('prefill', 20, 0))
        self.assertEqual(scope['_step'](engine), ('decode', 3, 1))
        self.assertEqual(req.timing.prefill_tokens, 20)
        self.assertEqual(req.timing.decode_tokens, 3)
        self.assertAlmostEqual(req.timing.snapshot()['prefill_tok_s'], 40)
        self.assertAlmostEqual(req.timing.snapshot()['decode_tok_s'], 30)

    def test_sensor_units_missing_values_and_multiple_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for card in ('card0', 'card1'):
                device = root / card / 'device'
                hwmon = device / 'hwmon' / 'hwmon0'
                hwmon.mkdir(parents=True)
                (device / 'gpu_busy_percent').write_text('73')
                (device / 'mem_info_vram_used').write_text('1073741824')
                (hwmon / 'power1_average').write_text('42000000')
                (hwmon / 'temp1_input').write_text('65000')
            (root / 'card0-DP-1').mkdir()
            (root / 'card1/device/gpu_busy_percent').write_text('unavailable')
            rows = gpu_sensors(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['power_watts'], 42)
            self.assertEqual(rows[0]['temperature_celsius'], 65)
            self.assertNotIn('utilization_percent', rows[1])
            self.assertEqual(gpu_sensors(root / 'missing'), [])

if __name__ == '__main__':
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import unittest

from snowllm import memory_report
from snowllm.capacity import memory_shortage


class MemoryReportTests(unittest.TestCase):
    def test_success_failure_and_early_failure_print_with_planner_prefix(self):
        @memory_report.startup_report
        def start(fail=False):
            self.assertIsNotNone(memory_report.current())
            if fail:
                raise MemoryError('host memory budget exhausted')
            return 42

        for fail in (False, True):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                if fail:
                    with self.assertRaisesRegex(MemoryError, 'host memory budget'):
                        start(True)
                else:
                    self.assertEqual(start(), 42)
            lines = output.getvalue().splitlines()
            self.assertTrue(all(line.startswith('[planner] ') for line in lines))
            self.assertIn('failed' if fail else 'ready', lines[0])
            self.assertIn('memory=n/a', lines[0])
            self.assertEqual(len(lines), 1)
            self.assertIsNone(memory_report.current())

    def test_ready_reconciles_weights_workspace_and_optional_snapshots(self):
        report = memory_report.MemoryReport(values=dict(
            context=211968, requests=1, chunk=8192, slots=3, weights=100,
            host_weights=10, cache=20, cache_100k=10, linear=9, other=4,
            snapshots=6, snapshot_count=2, snapshot_each=3, fixed_workspace=3,
            workspace=11, total=150, load_peak=110, room=200))
        report.ready(weights=90, host_weights=20, workspace=10, snapshot_count=4, device_used=170)
        self.assertEqual(report.values['total'], 148)
        self.assertEqual(report.values['snapshots'], 12)
        self.assertEqual(report.values['snapshot_count'], 4)
        self.assertEqual(report.values['workspace'], 13)
        text = '\n'.join(report.lines('ready'))
        self.assertIn('reqs=1, state snapshots=21.000 B, 3.000 B per snapshot, '
                      '3 live + 4 cached per req', text)
        self.assertNotIn('shared', text)
        self.assertIn('ctx=211968, KV=20.000 B, 10.000 B per 100K tok', text)
        report.values['per_request'] = 131072
        self.assertIn('ctx=211968, max/request=131072, KV=20.000 B', '\n'.join(report.lines('ready')))
        self.assertNotIn('layer', text)
        self.assertNotIn('|', text)
        self.assertIn('extra/request~4.000 B', text)
        self.assertIn('device=170.000 B', text)
        self.assertEqual(len(text.splitlines()), 6)
        report.max_host_offload = 15 << 30
        self.assertEqual(list(report.lines('ready'))[:2],
                         ['weights 110.000 B, max host offload=15.000 GiB',
                          'weights GPU=90.000 B, host=20.000 B'])
        self.assertEqual(len(list(report.lines('ready'))), 6)
        self.assertIn('weights 110.000 B, max host offload=15.000 GiB', list(report.lines('failed')))
        report.values['allocation'] = 'host'
        text = '\n'.join(report.lines('ready'))
        self.assertIn('weights GPU=0.000 B, host=110.000 B', text)
        self.assertIn('host tensors=', text)
        self.assertNotIn('GPU tensors=', text)
        self.assertNotIn('max host offload', text)
        report.shortfall(61 << 20)
        text = '\n'.join(report.lines('failed'))
        self.assertIn('short=61.000 MiB', text)
        report.suggestion = 211968
        self.assertIn('--max-model-len 211968 (est)', '\n'.join(report.lines('failed')))
        report.values.update(requests=2, snapshot_count=5)
        self.assertIn('reqs=2, state snapshots=21.000 B, 3.000 B per snapshot, 3 live + ~2.5 cached per req',
                      '\n'.join(report.lines('ready')))

    def test_draft_prefix_pages_are_counted_as_kv(self):
        report = memory_report.MemoryReport(values=dict(weights=100, host_weights=0,
            workspace=10, snapshot_each=3, cache_snapshot_each=5, snapshot_count=2,
            snapshots=6, cache=30, total=146))
        report.ready(weights=100, host_weights=0, workspace=10, snapshot_count=4, device_used=200)
        self.assertEqual(report.values['snapshots'], 12)
        self.assertEqual(report.values['cache'], 40)
        self.assertEqual(report.values['total'], 162)

    def test_available_memory_is_printed_before_planning(self):
        @memory_report.startup_report
        def start():
            memory_report.current().available(90 << 30, 20 << 30)
            print('planning')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            start()
        self.assertEqual(output.getvalue().splitlines()[:2],
                         ['[planner] available GPU=90.000 GiB, host=20.000 GiB', 'planning'])

    def test_constraint_hints_only_offer_applicable_offload(self):
        gpu = str(memory_shortage(61500000))
        self.assertIn('GPU-bound', gpu)
        self.assertNotIn('--max-host-mem', gpu)
        self.assertNotIn('--max-disk-offload', gpu)
        host = str(memory_shortage(61500000, host_help=True))
        self.assertIn('--max-host-mem', host)
        self.assertIn('host offload is limited', host)
        self.assertNotIn('--max-disk-offload', host)
        self.assertIn('--max-disk-offload', str(memory_shortage(1, disk_help=True)))
        host = str(memory_shortage(1, target='host'))
        self.assertIn('host memory budget is short', host)
        self.assertNotIn('--max-gpu-mem', host)

    def test_suppressed_search_preserves_report(self):
        @memory_report.startup_report
        def start():
            report = memory_report.current()
            with memory_report.suspended():
                self.assertIsNone(memory_report.current())
            self.assertIs(memory_report.current(), report)
        with contextlib.redirect_stdout(io.StringIO()):
            start()


if __name__ == '__main__':
    unittest.main()

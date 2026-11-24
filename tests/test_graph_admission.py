# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from snowllm.engine.graphs import GraphBudget
from snowllm.engine.runner import GraphRunner
from snowllm.engine.spec_decode import DFlashDecoder


class GraphAdmissionTests(unittest.TestCase):
    def test_draft_estimate_and_actual_capture_are_checked_separately(self):
        budget = GraphBudget()
        budget.measured = 28
        runner = SimpleNamespace(budget=budget, graphs={}, gpu_util=1., last_hidden='original')

        def capture(label, walk):
            runner.last_hidden = 'temporary'
            return object(), object()

        runner._capture = Mock(side_effect=capture)
        with patch.object(budget, '_in_use', side_effect=[820, 820, 1011]), \
             patch('snowllm.engine.graphs.gpu_mem_info', return_value=(180, 1000)), \
             patch('snowllm.engine.graphs.device_mem_info', return_value=(180, 1000)), \
             patch('torch.cuda.memory_reserved', return_value=0), \
             patch('snowllm.host_alloc.enabled', return_value=False), \
             patch('snowllm.engine.runner.release_device_cache'):
            self.assertIsNone(GraphRunner._try_capture(runner, 'draft', None, 191))
            runner._capture.assert_not_called()
            self.assertIsNone(GraphRunner._try_capture(runner, 'draft', None, 100))
        runner._capture.assert_called_once()
        self.assertEqual(runner.last_hidden, 'original')
        self.assertEqual(budget.retained_bytes, 0)

    def test_full_target_budget_skips_draft_staging(self):
        runner, draft = Mock(), Mock()
        runner.max_num_seqs = 4
        runner.budget.full = True
        draft.graphs = {}
        self.assertEqual(DFlashDecoder.capture(draft, runner), [])
        draft._stage_values.assert_not_called()
        runner._try_capture.assert_not_called()

    def test_fixed_placement_without_prefix_cache_probes_only_at_final_capture(self):
        runner, draft = Mock(), Mock()
        runner.prefix_memory_ratio = 0
        runner.model._automatic_placement = False
        self.assertEqual(DFlashDecoder.probe_graph_bytes(draft, runner, [1, 2]), 0)
        runner.capture.assert_not_called()
        draft.capture.assert_not_called()

    def test_draft_probe_measures_coexisting_graphs_and_releases_them(self):
        from contextlib import nullcontext
        runner = Mock()
        runner.prefix_memory_ratio = .08
        runner._graph_probe_scope.return_value = nullcontext()
        runner.budget._in_use.side_effect = [100, 400]
        draft = SimpleNamespace(graphs={})
        def capture(r):
            r.capture.assert_called_once_with([1, 2])
            draft.graphs[1] = object()
        draft.capture = capture
        self.assertEqual(DFlashDecoder.probe_graph_bytes(draft, runner, [1, 2]), 300)
        self.assertFalse(draft.graphs)
        runner._init_graphs.assert_called_once()


if __name__ == '__main__':
    unittest.main()

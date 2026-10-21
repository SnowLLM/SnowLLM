# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location(
    'dspark_schedule', Path(__file__).parents[1] / 'snowllm/engine/dspark_schedule.py')
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class Scheduling(unittest.TestCase):
    def test_algorithm_one(self):
        rng = random.Random(12)
        for _ in range(100):
            confidence = [[rng.random() for _ in range(5)] for _ in range(3)]
            limits = [rng.randrange(6) for _ in range(3)]
            cost = lambda rows: .01 + .001*rows**1.5
            ranked = schedule.ranked_prefixes(confidence, limits)
            expected, best, count = 3.0, 3/cost(3), 0
            for survival, _, _ in ranked:
                expected += survival
                throughput = expected/cost(4+count)
                if throughput <= best:
                    break
                best, count = throughput, count+1
            result = schedule.PrefixScheduler(256).choose([0, 1, 2], confidence, limits, cost)
            self.assertEqual(result, schedule.prefix_lengths(ranked, count, 3))

    def test_two_step_capacity_and_current_ranking(self):
        scheduler = schedule.PrefixScheduler(256)
        cost = lambda rows: {2: 1., 3: 1.3, 4: 1.5, 5: 2., 6: 3.}[rows]
        historical = [[.99, .99], [.01, .01]]
        current = [[.01, .01], [.99, .99]]
        scheduler.choose([0, 1], historical, [2, 2], cost)
        scheduler.choose([0, 1], [[.01]*2]*2, [2, 2], cost)
        result = scheduler.choose([0, 1], current, [2, 2], cost)
        self.assertEqual(result, [0, 2])

    def test_hardware_cliff_search_is_historical(self):
        cost = lambda rows: {1: 1., 2: 2., 3: 1.2}[rows]
        scheduler = schedule.PrefixScheduler(256)
        self.assertEqual(scheduler.choose([0], [[.8, 1.]], [2], cost), [0])
        self.assertEqual(scheduler.choose([0], [[.8, 1.]], [2], cost), [0])
        self.assertEqual(scheduler.choose([0], [[.01, .01]], [2], cost), [2])
        self.assertEqual(scheduler.choose([1], [[.8, 1.]], [2], cost), [0])

    def test_caps_ties_and_zero(self):
        scheduler = schedule.PrefixScheduler(5)
        self.assertEqual(scheduler.choose([0, 1], [[1.]*5]*2, [1, 5], lambda _: 1.), [1, 2])
        scheduler.reset()
        self.assertEqual(scheduler.choose([0], [[0., 1.]], [2], lambda _: 1.), [0])
        scheduler.reset()
        self.assertEqual(scheduler.choose([0], [[1.]*5], [0], lambda _: 1.), [0])

    def test_plan_does_not_advance_history(self):
        scheduler = schedule.PrefixScheduler(256)
        for _ in range(3):
            scheduler.choose([0], [[.9]], [1], lambda _: 1., remember=False)
        self.assertEqual(len(scheduler.history), 0)
        scheduler.choose([0], [[.9]], [1], lambda _: 1.)
        self.assertEqual(len(scheduler.history), 1)

    def test_cost_only_uses_graph_at_exact_bucket(self):
        cost = schedule.StepCost({2: 2., 4: 4., 6: 6.}, {4: .5}, .1)
        self.assertAlmostEqual(cost(4), .6)
        self.assertAlmostEqual(cost(3), 3.1)
        self.assertAlmostEqual(cost(5), 5.1)
        with self.assertRaises(ValueError):
            cost(7)

    def test_invalid_confidence(self):
        for value in (float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                schedule.ranked_prefixes([[value]], [1])


if __name__ == '__main__':
    unittest.main()

# SPDX-License-Identifier: Apache-2.0

import math
from collections import deque


def ranked_prefixes(confidence, limits):
    candidates = []
    for r, (scores, limit) in enumerate(zip(confidence, limits)):
        survival = 1.0
        for j, score in enumerate(scores[:limit]):
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'invalid DSpark confidence: {score}')
            survival *= score
            if survival <= 0:
                break
            candidates.append((survival, r, j))
    return sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))


def prefix_lengths(candidates, count, requests):
    lengths = [0] * requests
    for _, r, j in candidates[:count]:
        lengths[r] = j + 1
    return lengths


def best_capacity(candidates, requests, row_limit, cost, *, early_stop):
    expected = float(requests)
    best, capacity = expected / cost(requests), 0
    for k, (survival, _, _) in enumerate(candidates[:row_limit - requests], 1):
        expected += survival
        throughput = expected / cost(requests + k)
        if throughput > best:
            best, capacity = throughput, k
        elif early_stop:
            break
    return capacity


class PrefixScheduler:
    def __init__(self, row_limit):
        self.row_limit = row_limit
        self.history = deque(maxlen=2)
        self.cohort = ()

    def reset(self):
        self.cohort = ()
        self.history.clear()

    def choose(self, cohort, confidence, limits, cost, *, remember=True):
        cohort = tuple(cohort)
        if cohort != self.cohort:
            self.reset()
            self.cohort = cohort
        requests = len(cohort)
        current = ranked_prefixes(confidence, limits)
        historical = len(self.history) == 2
        # Global capacity search must not look ahead into the current proposal's future tokens.
        candidates = (ranked_prefixes(self.history[0], limits) if historical else current)
        capacity = best_capacity(candidates, requests, max(requests, self.row_limit), cost,
                                 early_stop=not historical)
        if remember:
            self.history.append(tuple(tuple(row) for row in confidence))
        return prefix_lengths(current, capacity, requests)


class StepCost:
    def __init__(self, eager, graphs, draft):
        if (not eager or not math.isfinite(draft) or draft < 0
                or any(not math.isfinite(t) or t <= 0 for t in (*eager.values(), *graphs.values()))):
            raise ValueError('DSpark profiling needs positive, finite step times')
        self.eager, self.graphs, self.draft = eager, graphs, draft
        self.points = sorted(eager)

    def __call__(self, rows):
        if rows in self.graphs:
            return self.draft + self.graphs[rows]
        if rows in self.eager:
            return self.draft + self.eager[rows]
        for lo, hi in zip(self.points, self.points[1:]):
            if lo < rows < hi:
                return self.draft + (self.eager[lo] * (hi - rows)
                                     + self.eager[hi] * (rows - lo)) / (hi - lo)
        raise ValueError(f'unprofiled DSpark batch size: {rows}')

# SPDX-License-Identifier: Apache-2.0

import math
from collections import deque
from statistics import median


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


class InklingVerifyPolicy:
    def __init__(self, width):
        self.widths = tuple(sorted({n for n in (2, 3, 4, 5, 6, width) if n <= width}))
        self.times = {n: deque(maxlen=21) for n in self.widths}
        self.by_confidence = [[0, 0] for _ in range(10)]
        self.by_depth = [[[0, 0] for _ in range(10)] for _ in range(width - 1)]
        self.pending = None

    def choose(self, request, scores):
        if not request.drafts or not all(math.isfinite(s) for s in scores):
            self.pending = None
            return self.widths[-1]
        self.pending = request, tuple(request.drafts), scores
        for width, samples in self.times.items():
            if len(samples) < 2:
                return width
        survival = expected = 1.
        rates = {}
        for depth, score in enumerate(scores):
            bucket = min(9, max(0, int(score * 10)))
            total, hits = self.by_confidence[bucket]
            prior = (hits + 6 * (0.4 + 0.6 * score)) / (total + 6)
            total, hits = self.by_depth[depth][bucket]
            survival *= (hits + 10 * prior) / (total + 10)
            expected += survival
            width = depth + 2
            if width in self.times:
                rates[width] = expected / median(self.times[width])
        return max(rates, key=rates.get)

    def observe(self, request, rows, kept):
        probe, self.pending = self.pending, None
        if probe is None or probe[0] is not request:
            return
        _, predictions, scores = probe
        for depth in range(min(kept, rows - 1, len(predictions))):
            if predictions[depth] != request.drafts[depth]:
                break
            bucket = min(9, max(0, int(scores[depth] * 10)))
            hit = int(depth < kept - 1)
            for counts in (self.by_confidence[bucket], self.by_depth[depth][bucket]):
                counts[0] += 1
                counts[1] += hit

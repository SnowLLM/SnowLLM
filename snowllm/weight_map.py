# SPDX-License-Identifier: Apache-2.0
import re
from dataclasses import dataclass

from .errors import SnowLLMError

PROJECTIONS = {f'{group}.{part}': group for group in ('experts', 'shared')
               for part in ('gate_up', 'down')}
GROUPS = ("experts", "dense", "shared", "head", *PROJECTIONS)
LAYERED = ("dense", "shared", "experts", *PROJECTIONS)


@dataclass(frozen=True)
class PlacementRule:
    group: str
    tier: str
    count: int | None
    span: tuple[int, int] | None


@dataclass(frozen=True)
class WeightMap:
    rules: tuple[PlacementRule, ...]

    @property
    def disk_experts(self):
        return any(rule.tier == 'disk' for rule in self.rules)

    @classmethod
    def parse(cls, spec):
        if not isinstance(spec, str):
            raise SnowLLMError('--device-map requires a placement description')
        if spec.strip() == 'gpu':
            return cls(())
        if '=' not in spec:
            return None
        rules = []
        for item in spec.split(','):
            name, _, tier = item.partition('=')
            tier = tier.strip()
            match = re.fullmatch(r'([a-z_.*]+)(?::([0-9]+)|\[([0-9]+)(?:-([0-9]+))?\])?', name.strip())
            if not match:
                raise SnowLLMError(f'invalid --device-map selector {name!r}')
            group, count, start, end = match.groups()
            span = (int(start), int(end or start)) if start is not None else None
            if (group not in (*GROUPS, '*') or tier not in ('gpu', 'host', 'disk')
                    or (group == '*' and tier != 'gpu')
                    or ((count is not None or span) and group not in LAYERED)
                    or (span and span[0] > span[1])):
                raise SnowLLMError(f'invalid --device-map rule {item!r}')
            if tier == 'disk' and group != 'experts':
                raise SnowLLMError(f'--device-map {item!r}: disk supports only routed experts (experts)')
            for prior in rules:
                if prior.group == group:
                    if span and prior.span:
                        if max(prior.span[0], span[0]) <= min(prior.span[1], span[1]):
                            raise SnowLLMError(f'overlapping --device-map layers in {item!r}')
                    elif count is None and prior.count is None and span == prior.span:
                        raise SnowLLMError(f'duplicate --device-map default for {group}')
                elif prior.group.startswith(group + '.') or group.startswith(prior.group + '.'):
                    raise SnowLLMError('overlapping weight groups in --device-map')
            rules.append(PlacementRule(group, tier, int(count) if count is not None else None, span))
        return cls(tuple(rules))

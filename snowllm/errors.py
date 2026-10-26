# SPDX-License-Identifier: Apache-2.0
"""Errors shared by CPU management commands and the inference runtime."""


class SnowLLMError(RuntimeError):
    pass


def is_capacity_error(message):
    text = message.casefold()
    return any(part in text for part in (
        'memory budget', 'gpu budget', 'out of memory', 'outofmemory', 'memoryerror',
        'kv pool cannot carry', 'persistent state and workspace need',
        'cannot allocate memory', 'not enough gpu memory'))

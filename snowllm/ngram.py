# SPDX-License-Identifier: Apache-2.0


def banned_ngram_tokens(tokens, size):
    if size <= 0 or len(tokens) + 1 < size:
        return []
    prefix = tuple(tokens[-(size - 1) :]) if size > 1 else ()
    return sorted(
        {
            tokens[i + size - 1]
            for i in range(len(tokens) - size + 1)
            if tuple(tokens[i : i + size - 1]) == prefix
        }
    )

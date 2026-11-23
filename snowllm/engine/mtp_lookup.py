# SPDX-License-Identifier: Apache-2.0
import json
import re
from array import array
from collections import Counter, defaultdict, deque


_NUMBERED_LINE = re.compile(r' *([0-9]{1,9})\t(.*)')


def _tool_source(text):
    if text.lstrip().startswith('{'):
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            return None
        content = value.get('content') if isinstance(value, dict) else None
        return content if isinstance(content, str) and '\n' in content else None

    lines = text.removesuffix('\n').split('\n')
    if len(lines) < 3:
        return None
    source = []
    first = None
    for line in lines:
        match = _NUMBERED_LINE.fullmatch(line)
        if match is None:
            return None
        number = int(match[1])
        if first is None:
            first = number
        if number != first + len(source):
            return None
        source.append(match[2])
    return '\n'.join(source) + ('\n' if text.endswith('\n') else '')


def tool_lookup_passages(messages, tokenizer):
    passages = []
    for message in messages:
        if message.get('role') != 'tool':
            continue
        content = message.get('content')
        parts = (content,)
        if isinstance(content, list):
            parts = (part.get('text') for part in content if part.get('type') == 'text')
        for text in parts:
            if not isinstance(text, str) or len(text) > 262144:
                continue
            source = _tool_source(text)
            if source and not re.search(r'[\ud800-\udfff]', source):
                passages.append(tokenizer.encode(source, add_special_tokens=False))
    return tuple(passages)


class MtpLookup:
    def __init__(self, request, vocab_size):
        self.tokens = array('I')
        self.previous = array('i')
        self.latest = array('i', [-1]) * vocab_size
        self.output_length = 0
        self.native_depth = 2
        self.suffix = None
        self.guard = None
        self._append(request.prompt)
        self.passages = defaultdict(lambda: deque(maxlen=64))
        for passage in request.lookup_passages:
            for at in range(11, len(passage) - 2):
                self.passages[passage[at]].append((passage, at))

    def _append(self, tokens):
        latest, previous, history = self.latest, self.previous, self.tokens
        for pos, token in enumerate(tokens, len(history)):
            previous.append(latest[token])
            latest[token] = pos
            history.append(token)

    def _raw_positions(self):
        end = len(self.tokens) - 1
        at = self.previous[end]
        for _ in range(64):
            if at < 11:
                break
            if end - at >= 3:
                yield self.tokens, at
            at = self.previous[at]

    def _match(self, positions, prefix, limit, first):
        best, matches = 12, []
        for passage, at in positions:
            length = 1
            while length < min(prefix, at + 1) and passage[at-length] == self.tokens[-length-1]:
                length += 1
            if length >= best:
                if length > best:
                    matches.clear()
                    best = length
                matches.append(passage[at+1:at+1+limit])
        result = []
        for depth in range(limit):
            votes = Counter(m[depth] for m in matches if depth < len(m))
            if not votes:
                break
            token, count = votes.most_common(1)[0]
            if 3 * count < 2 * len(matches) or (depth == 0 and first is not None and token != first):
                break
            result.append(token)
            matches = [m for m in matches if depth < len(m) and m[depth] == token]
        return result if len(result) >= 3 else []

    def continuation(self, limit, first=None):
        if limit < 3:
            return []
        if len(self.tokens) >= 15:
            draft = self._match(self._raw_positions(), 32, limit, first)
            if draft:
                return draft
        # A normalized source must match committed output wholly within the same file.
        if self.passages and self.output_length >= 12:
            return self._match(self.passages.get(self.tokens[-1], ()),
                               min(32, self.output_length), limit, first)
        return []

    def candidate(self, request, limit, first=None):
        self._append(request.out[self.output_length:])
        self.output_length = len(request.out)
        return self.continuation(limit, first)

    def overlay(self, request, k, sampling=None):
        width = min(k, len(request.drafts))
        if width < 2:
            return
        sampled = sampling is not None and request.params.temperature > 0
        first = None if sampled else request.drafts[0]
        draft = self.candidate(request, max(3, width), first)[:width]
        if sampled:
            if len(draft) != k:
                return
            # A lookup chosen independently of the sampled path has point-mass q, even if all IDs coincide.
            sampling.release(request)
            request.drafts = draft
        elif draft:
            request.drafts[:len(draft)] = draft

    def propose(self, request, limit, used_suffix):
        self.suffix = None
        self.guard = request.drafts if used_suffix else None
        if not used_suffix and request.drafts:
            self.native_depth = len(request.drafts)
        if limit >= 3 and request.drafts:
            draft = self.candidate(request, limit, request.drafts[0])
            if draft:
                self.suffix = request.drafts = draft
                self.guard = None

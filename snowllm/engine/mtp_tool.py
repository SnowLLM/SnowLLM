# SPDX-License-Identifier: Apache-2.0
from copy import copy


class MtpToolFormat:
    def __init__(self, tokenizer, opening, closing, eos_ids, functions, headers, required, ending,
                 param_end, param_confirm):
        self.tokenizer = tokenizer
        self.opening = opening
        self.closing = closing
        self.eos = frozenset(eos_ids)
        self.functions = functions
        self.headers = headers
        self.required = required
        self.ending = ending
        self.param_end = param_end
        self.param_confirm = param_confirm

    @classmethod
    def from_tools(cls, tokenizer, tools, eos_ids=()):
        if not tools:
            return None

        def encode(text):
            return tuple(tokenizer.encode(text, add_special_tokens=False))

        def valid_name(name):
            return isinstance(name, str) and bool(name) and not any(c in name for c in '<>\r\n')

        opening, closing = encode('<tool_call>'), encode('</tool_call>')
        if len(opening) != 1 or len(closing) != 1 or opening == closing:
            return None
        functions, headers, required = [], {}, {}
        for tool in tools:
            if not isinstance(tool, dict) or tool.get('type', 'function') != 'function':
                return None
            spec = tool.get('function', tool)
            name = spec.get('name') if isinstance(spec, dict) else None
            if not valid_name(name) or name in headers:
                return None
            params = spec.get('parameters', {})
            if not isinstance(params, dict) or any(key in params for key in
                    ('anyOf', 'oneOf', 'allOf', '$ref', 'patternProperties')):
                return None
            props, needed = params.get('properties', {}), params.get('required', [])
            if (not isinstance(props, dict) or not isinstance(needed, (list, tuple))
                    or not all(valid_name(p) for p in props)
                    or not all(isinstance(p, str) and p in props for p in needed)
                    or params.get('additionalProperties', False) is not False):
                return None
            tokens = encode(f'<tool_call>\n<function={name}>\n')
            if tokens[:1] != opening or len(tokens) < 2:
                return None
            functions.append((tokens[1:], name))
            headers[name] = {p: encode(f'<parameter={p}>') for p in props}
            required[name] = frozenset(needed)
        ending, param_end, recognized = (encode('</function>\n</tool_call>'),
                                          encode('</parameter>\n'), encode('</parameter'))
        if (ending[-1:] != closing or not recognized or param_end[:len(recognized)] != recognized
                or not all(tokens for fields in headers.values() for tokens in fields.values())):
            return None
        return cls(tokenizer, opening[0], closing[0], (eos_ids,) if isinstance(eos_ids, int) else eos_ids,
                   tuple(functions), headers, required, ending, param_end, len(recognized))

    def new_state(self):
        return _MtpToolState(self)


class _MtpToolState:
    def __init__(self, fmt):
        self.fmt = fmt
        self.phase = 'outside'
        self.choices = ()
        self.at = 0
        self.function = None
        self.seen = frozenset()
        self.previous = None

    def _choose(self, phase, choices):
        self.phase, self.choices, self.at = phase, choices, 0

    def _next_field(self):
        fmt, name = self.fmt, self.function
        choices = [(tokens, field) for field, tokens in fmt.headers[name].items() if field not in self.seen]
        if not (fmt.required[name] - self.seen):
            choices.append((fmt.ending, None))
        self._choose('field', tuple(choices))

    def _consume(self, token):
        fmt = self.fmt
        if token in fmt.eos or self.phase == 'ended':
            self._choose('ended', ())
        elif token == fmt.opening and self.phase != 'value':
            self.seen = frozenset()
            self._choose('function', fmt.functions)
        elif token == fmt.closing:
            self._choose('outside', ())
        elif self.phase == 'value':
            close = fmt.param_end
            if self.at:
                self.at = self.at + 1 if token == close[self.at] else 0
            elif token == close[0] and self.previous is not None:
                # The preceding BPE token may end in one or several newlines, possibly with spaces.
                if fmt.tokenizer.decode([self.previous]).endswith('\n'):
                    self.at = 1
            if self.at == len(close):
                self._next_field()
        elif self.choices:
            choices = tuple(c for c in self.choices if c[0][self.at] == token)
            self.at += 1
            finished = [label for ids, label in choices if len(ids) == self.at]
            if not choices or (finished and len(choices) != 1):
                self._choose('outside', ())
            elif not finished:
                self.choices = choices
            elif self.phase == 'function':
                self.function = finished[0]
                self._next_field()
            elif finished[0] is None:
                self._choose('outside', ())
            else:
                self.seen = self.seen | {finished[0]}
                self._choose('value', ())
        self.previous = token

    def advance(self, committed_ids):
        for token in committed_ids:
            self._consume(token)

    def propose(self, limit):
        if limit < 1 or not self.choices and self.phase != 'value':
            return []
        if self.phase == 'value' and self.at < self.fmt.param_confirm:
            return []
        state = copy(self)
        draft = []
        for _ in range(limit):
            if state.choices:
                first = state.choices[0][0][state.at]
                if any(ids[state.at] != first for ids, _ in state.choices[1:]):
                    break
            elif state.phase == 'value' and state.at >= state.fmt.param_confirm:
                first = state.fmt.param_end[state.at]
            else:
                break
            draft.append(first)
            state._consume(first)
        return draft

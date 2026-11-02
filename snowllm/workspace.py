# SPDX-License-Identifier: Apache-2.0
import contextlib
from contextvars import ContextVar
import weakref


_current = ContextVar('snowllm_workspace', default=None)


def current_workspace():
    return _current.get()


class SharedWorkspace:
    def __init__(self, target):
        self.target = target
        self.buffer = None
        self.nbytes = 0
        self.generation = 0
        self.allocation = target.upper()
        self.before_resize = None
        self.arenas = weakref.WeakSet()

    @contextlib.contextmanager
    def scope(self):
        token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(token)

    def bind(self, arena, nbytes, allocate):
        if nbytes > self.nbytes or self.buffer is None:
            if self.before_resize is not None:
                self.before_resize()
            buffer = allocate(nbytes)
            self.buffer, self.nbytes = buffer, nbytes
            self.generation += 1
            for other in self.arenas:
                other.buf = buffer
                if other.planner.buf is not None:
                    other.planner.buf = buffer
        self.arenas.add(arena)
        arena.buf = self.buffer
        return self.buffer

    def release(self, arena):
        self.arenas.discard(arena)
        if not self.arenas:
            self.buffer, self.nbytes = None, 0

    def snapshot(self):
        return dict(bytes=self.nbytes, generation=self.generation)

    def close(self):
        self.buffer, self.nbytes = None, 0
        self.arenas.clear()

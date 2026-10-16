"""Per-call diagnostics; nested phases are inclusive, never summed as wall time."""
from contextlib import contextmanager
from contextvars import ContextVar
from collections import defaultdict
import time

_current = ContextVar('alignment_performance', default=None)


class Metrics:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.counts = defaultdict(float)
        self.events = []

    def to_dict(self):
        return {'phase_seconds_inclusive': dict(self.seconds), 'counters': dict(self.counts),
                'decode_events': self.events, 'io_scope': 'logical application reads; not physical disk I/O'}


@contextmanager
def collecting():
    metrics = Metrics()
    token = _current.set(metrics)
    try:
        yield metrics
    finally:
        _current.reset(token)


@contextmanager
def phase(name):
    started = time.perf_counter()
    try:
        yield
    finally:
        if (metrics := _current.get()) is not None:
            metrics.seconds[name] += time.perf_counter() - started


def count(name, value=1):
    if (metrics := _current.get()) is not None:
        metrics.counts[name] += value


def event(**value):
    if (metrics := _current.get()) is not None:
        metrics.events.append(value)

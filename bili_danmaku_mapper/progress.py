"""Optional process-local progress reporting, without changing algorithm results."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_reporter = ContextVar("progress_reporter", default=None)
_phase = ContextVar("progress_phase", default="preparing")


@contextmanager
def reporting(callback):
    token = _reporter.set(callback)
    try:
        yield
    finally:
        _reporter.reset(token)


def report(phase=None, completed=None, total=None, **details):
    if phase:
        _phase.set(phase)
    callback = _reporter.get()
    if callback:
        callback({"phase": phase or _phase.get(), "completed": completed, "total": total, **details})

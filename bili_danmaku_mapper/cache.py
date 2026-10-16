"""Content identities and advisory interprocess locks for owned cache files."""
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import json
import math
import re
from pathlib import Path
from dataclasses import asdict

from .io import atomic_json
from .performance import count, phase

_session = ContextVar('media_identity_session', default=None)


class InputChangedError(RuntimeError):
    pass


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stamp(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def session():
    token = _session.set({})
    try:
        yield
        for path, value in _session.get().items():
            if stamp(path) != value['stat']:
                raise InputChangedError('Input media changed during alignment')
    finally:
        _session.reset(token)


def _sentinels(path, size):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for position in sorted(set((0, max(0, size // 2 - 524288), max(0, size - 1048576)))):
            f.seek(position)
            block = f.read(1048576)
            count('identity_sentinel_bytes', len(block))
            h.update(str(position).encode()); h.update(block)
    return h.hexdigest()


def identity(path, config):
    from .media import content_digest
    path = str(Path(path).resolve(strict=True))
    state = stamp(path)
    memo = _session.get()
    if memo is not None and path in memo:
        if memo[path]['stat'] != state:
            raise InputChangedError('Input media changed during alignment')
        count('identity_memory_hits')
        return memo[path]['digest']
    entry = Path(config.cache_dir) / 'identity-v1' / (key(path) + '.json') if config.cache_dir else None
    with phase('identity'):
        def resolve():
            sentinel = _sentinels(path, state[2])
            try:
                record = json.loads(entry.read_text()) if entry else {}
            except (ValueError, OSError):
                record = {}
            if not isinstance(record, dict):
                record = {}
            if (record.get('stat') == state and record.get('sentinels') == sentinel
                    and isinstance(record.get('digest'), str) and re.fullmatch(r"[0-9a-f]{64}",record['digest'])):
                count('identity_persistent_hits')
                digest = record['digest']
            else:
                count('digest_scans')
                count('digest_logical_bytes', state[2])
                digest = content_digest(path)
            if stamp(path) != state:
                raise InputChangedError('Input media changed during hashing')
            record = {'stat': state, 'sentinels': sentinel, 'digest': digest}
            if entry:
                atomic_json(entry, record, overwrite=True)
            return record
        if entry:
            with locked(entry):
                record = resolve()
        else:
            record = resolve()
    if memo is not None:
        memo[path] = record
    return record['digest']


def probe(path, config, decoder):
    from .media import VideoInfo
    path = str(Path(path).resolve(strict=True))
    digest = identity(path, config)
    entry = Path(config.cache_dir) / 'probe-v1' / (key([digest, config.ffprobe]) + '.json') if config.cache_dir else None
    def resolve():
        try:
            record = json.loads(entry.read_text()) if entry else {}
            result = VideoInfo(**{**record, 'path': path})
            if (not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (result.duration,result.origin,result.fps))
                    or result.duration <= 0 or result.fps < 0 or not isinstance(result.pixel_format,str)):
                raise ValueError('Invalid cached media probe')
            count('probe_cache_hits')
            return result
        except (ValueError, TypeError, OSError):
            with phase('probe'):
                result = decoder(path, config)
            if entry:
                atomic_json(entry, asdict(result), overwrite=True)
            return result
    if entry:
        with locked(entry):
            return resolve()
    return resolve()

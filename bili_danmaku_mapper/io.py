"""Atomic writes; refuse replacement unless explicitly requested."""
import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path, data, overwrite=False, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if private:
            os.chmod(name, 0o600)
        if overwrite:
            os.replace(name, path)
        else:
            os.link(name, path)  # Atomic creation; concurrent writers cannot clobber.
    finally:
        Path(name).unlink(missing_ok=True)


def save_mapping_outputs(output, result, alignment, overwrite=False):
    path = Path(output)
    paths = (path, path.with_suffix(".review.json"), path.with_suffix(".report.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError(f"Another writer owns output lock: {lock}") from None
    os.close(fd)
    try:
        if not overwrite:
            for p in paths:
                if p.exists():
                    raise FileExistsError(f"Output exists: {p}; use --overwrite")
        # Serialize and fsync every document before changing any visible output.
        with tempfile.TemporaryDirectory(prefix=".output-", dir=path.parent) as directory:
            staged = [Path(directory) / f"{i}.json" for i in range(3)]
            documents = [result.danmaku, result.review, {**result.report, "alignment": alignment.to_dict()}]
            for file, data in zip(staged, documents):
                atomic_json(file, data)
            backups, committed = {}, []
            try:
                for file, dest in zip(staged, paths):
                    if dest.exists():
                        backup = Path(directory) / f"{len(backups)}.backup"
                        os.link(dest, backup)
                        backups[dest] = backup
                    if overwrite:
                        os.replace(file, dest)
                    else:
                        os.link(file, dest)
                    committed.append(dest)
            except OSError:
                for dest in reversed(committed):
                    if dest in backups:
                        os.replace(backups[dest], dest)
                    else:
                        dest.unlink(missing_ok=True)
                raise
    finally:
        lock.unlink(missing_ok=True)
    return paths

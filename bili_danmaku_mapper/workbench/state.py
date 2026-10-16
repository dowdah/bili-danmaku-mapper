import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def identifier():
    return uuid.uuid4().hex


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory / "state.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY, path TEXT NOT NULL, kind TEXT NOT NULL, info TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, body TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def register_file(self, path, kind, info=None):
        path = str(Path(path).resolve(strict=True))
        with self.lock:
            row = self.db.execute("SELECT id FROM files WHERE path=? AND kind=?", (path, kind)).fetchone()
            key = row[0] if row else identifier()
            self.db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?)", (key, path, kind, json.dumps(info or {})))
            self.db.commit()
        return {"id": key, "name": Path(path).name, "kind": kind, **(info or {})}

    def file(self, key):
        with self.lock:
            row = self.db.execute("SELECT * FROM files WHERE id=?", (key,)).fetchone()
        if not row:
            raise KeyError("Unknown file ID")
        path = Path(row["path"])
        if not path.is_file() or path.resolve() != path:
            raise FileNotFoundError("Registered file is missing or replaced by a link")
        return {"id": key, "path": str(path), "kind": row["kind"], "info": json.loads(row["info"])}

    def create_job(self, source, target_id, options, anchors=None, parent_id=None):
        key = identifier()
        self.file(target_id)
        body = {"id": key, "source": source, "target_id": target_id, "options": options,
                "anchors": anchors or [], "parent_id": parent_id, "status": "queued", "phase": "queued",
                "checkpoint": {}, "progress": {}, "result": None, "error": None}
        stamp = time.time()
        with self.lock:
            self.db.execute("INSERT INTO jobs VALUES(?,?,?,?)", (key, json.dumps(body), stamp, stamp))
            self.db.commit()
        return self.job(key)

    def job(self, key):
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone()
        if not row:
            raise KeyError("Unknown task")
        return {**json.loads(row["body"]), "created": row["created"], "updated": row["updated"]}

    def jobs(self):
        with self.lock:
            keys = [r[0] for r in self.db.execute("SELECT id FROM jobs ORDER BY created DESC")]
        return [self.job(key) for key in keys]

    def update(self, key, **values):
        with self.lock:
            body = self.job(key)
            body.pop("created", None); body.pop("updated", None)
            body.update(values)
            self.db.execute("UPDATE jobs SET body=?,updated=? WHERE id=?", (json.dumps(body, allow_nan=False), time.time(), key))
            self.db.commit()
        return self.job(key)

    def setting(self, key, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, json.dumps(value)))
            self.db.commit()

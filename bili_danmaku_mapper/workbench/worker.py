"""One owned computation process; stdout is a JSON event stream."""
import json
import os
from pathlib import Path
import signal
import threading
import sys
import time

from ..config import AlignConfig
from ..cli import run_pair
from ..io import atomic_json, read_json
from ..progress import reporting, report


def main(path):
    job = read_json(path)
    if job.get('owner_pid'):
        def watch_owner():
            while True:
                if os.getppid() != job['owner_pid']:
                    os.killpg(os.getpid(), signal.SIGTERM)
                    return
                time.sleep(.2)
        threading.Thread(target=watch_owner, daemon=True).start()
    config = AlignConfig(**job['config'])
    directory = Path(job['directory'])
    target = job['target']
    last = 0
    def emit(value):
        nonlocal last
        now = time.monotonic()
        if value.get('completed') is not None and now - last < .25:
            return
        last = now
        print(json.dumps({'event': 'progress', **value}), flush=True)
    try:
        with reporting(emit):
            report('aligning')
            # Retry only the interrupted alignment stage in this task's private directory.
            for item in directory.glob('.mapped*.lock'):
                item.unlink()
            result = run_pair(job['source_video'], target, job['raw_danmaku'], directory/'mapped.json', config,
                              job['mode'], True, job.get('anchors'))
            result['target_video'] = target
            atomic_json(directory/'result.json', result, overwrite=True)
            print(json.dumps({'event':'result', 'result':result}), flush=True)
        return 0
    except Exception:
        # No raw SDK URLs, credentials, or subprocess paths in the browser event channel.
        print(json.dumps({'event':'error', 'error':'计算失败，请检查影片、音轨和锚点范围后重试。'}), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))

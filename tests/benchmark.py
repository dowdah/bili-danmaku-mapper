"""Reproducible isolated-process acceptance metrics: python -m tests.benchmark."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

from bili_danmaku_mapper import AlignConfig, align_videos, map_danmaku
from bili_danmaku_mapper.io import atomic_json
from tests.video_helpers import make_video


def worker(task):
    cache = Path(task["cache"])
    started = time.perf_counter()
    config = AlignConfig(cache_dir=str(cache), algorithm=task["algorithm"])
    alignment = align_videos(task["source"], task["target"], config)
    mapped = map_danmaku({"data": [{"time": t} for t in task["times"]]}, alignment)
    errors, wrong = [], 0
    for entry in mapped.report["entries"]:
        if entry["target_time"] is None:
            continue
        expected = task["expected"][entry["source_index"]]
        if expected is None:
            wrong += 1
        else:
            error = abs(entry["target_time"] - expected)
            errors.append(error)
            wrong += error > .2
    accepted = mapped.report["counts"]["reliable"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"algorithm": task["algorithm"], "scenario": task["scenario"],
            "counts": mapped.report["counts"], "rates": mapped.report["rates"],
            "precision_within_200ms": (accepted - wrong) / accepted if accepted else None,
            "p95_error_seconds": float(np.percentile(errors, 95)) if errors else None,
            "max_error_seconds": max(errors) if errors else None,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "cache_bytes": sum(p.stat().st_size for p in cache.glob("*.npz")),
            "segments": len(alignment.segments),
            "legacy_discarded_nonmonotonic": alignment.diagnostics.get("discarded_nonmonotonic", 0)}


def run_benchmark(output):
    metrics = []
    with tempfile.TemporaryDirectory(prefix="danmaku-benchmark-") as tmp:
        root = Path(tmp)
        for scenario in ("identity", "head_cut", "middle_cut", "speed", "overlay", "target_cut", "vfr", "reencode"):
            source_duration, target_duration = 8, 8
            source_timeline, target_timeline = lambda t: t, lambda t: t
            expected = lambda t: t
            filters = None
            if scenario == "head_cut":
                source_duration, source_timeline = 7.3, lambda t: t + .7
                expected = source_timeline
            elif scenario == "middle_cut":
                source_duration, source_timeline = 7, lambda t: t if t < 3 else t + 1
                expected = source_timeline
            elif scenario == "speed":
                source_duration, source_timeline = 6.4, lambda t: t * 1.25
                expected = source_timeline
            elif scenario == "overlay":
                filters = "drawbox=x=0:y=136:w=256:h=8:color=white:t=fill,pad=288:176:16:16:black"
            elif scenario == "target_cut":
                target_duration, target_timeline = 7, lambda t: t if t < 3 else t + 1
                expected = lambda t: t if t < 3 else (None if t < 4 else t - 1)
            elif scenario == "vfr":
                filters = "select='not(eq(mod(n,10),3)+eq(mod(n,10),4))'"
            elif scenario == "reencode":
                filters = "scale=192:108,eq=brightness=0.025:contrast=1.05"
            directory = root / scenario
            directory.mkdir()
            source = make_video(directory / "source.mp4", source_duration, fps=24, timeline=source_timeline)
            target = make_video(directory / "target.mp4", target_duration, fps=30, timeline=target_timeline,
                                filters=filters, vfr=scenario == "vfr")
            times = [float(t) for t in np.arange(.2, source_duration - .2, .1)]
            for algorithm in ("sequence", "legacy"):
                task = {"source": str(source), "target": str(target), "cache": str(directory / algorithm),
                        "algorithm": algorithm, "scenario": scenario, "times": times,
                        "expected": [expected(t) for t in times]}
                task_path = directory / "task.json"
                atomic_json(task_path, task, overwrite=True)
                process = subprocess.run([sys.executable, "-m", "tests.benchmark", "--worker", str(task_path)],
                                         capture_output=True, text=True, check=True)
                metrics.append(json.loads(process.stdout))
                print(f"{scenario}: {algorithm} measured", file=sys.stderr, flush=True)
    report = {"schema_version": 1, "generated_date": str(datetime.now(ZoneInfo("Asia/Shanghai")).date()), "python": platform.python_version(),
              "platform": platform.platform(), "ffmpeg": subprocess.check_output(["ffmpeg", "-version"]).decode().splitlines()[0],
              "scope": "Short deterministic moving-texture videos; not real anime acceptance or a long-video performance claim",
              "baseline": "Original AAP matching on full decoded frame hashes; PTS conversion and nonmonotonic filtering retained for safe comparison",
              "metrics": metrics}
    atomic_json(output, report, overwrite=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/acceptance-metrics.json")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(Path(args.worker).read_text()))))
    else:
        run_benchmark(args.output)

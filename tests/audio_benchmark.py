"""Save reproducible soundtrack alignment metrics without retaining decoded audio."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import platform
import tempfile
from zoneinfo import ZoneInfo

import numpy as np

from bili_danmaku_mapper import AlignConfig, align_videos, map_danmaku
from bili_danmaku_mapper.io import atomic_json
from tests.audio_helpers import mux_sound
from tests.video_helpers import make_video


def run(output="docs/audio-acceptance-metrics.json"):
    metrics = []
    with tempfile.TemporaryDirectory(prefix="danmaku-audio-benchmark-") as tmp:
        root = Path(tmp)
        for scenario in ("head_cut", "middle_cut", "target_cut", "volume", "silence", "origin", "fully_obscured_hybrid"):
            directory = root / scenario
            directory.mkdir()
            sd, td = 10, 10
            source_time = target_time = lambda t: t
            expected = lambda t: t
            mute, gain, origin, filters = None, 1, 0, None
            if scenario in ("head_cut", "fully_obscured_hybrid"):
                sd, source_time = 9.3, lambda t: t + .7
                expected = source_time
                if scenario == "fully_obscured_hybrid":
                    filters = "drawbox=color=black:t=fill"
            elif scenario == "middle_cut":
                sd, source_time = 9, lambda t: t if t < 4 else t + 1
                expected = source_time
            elif scenario == "target_cut":
                td, target_time = 9, lambda t: t if t < 4 else t + 1
                expected = lambda t: t if t < 4 else None if t < 5 else t - 1
            elif scenario == "volume":
                gain = .3
            elif scenario == "silence":
                mute = (4, 6)
            elif scenario == "origin":
                origin = 5
            sv = make_video(directory / "source-video.mp4", sd, filters=filters)
            tv = make_video(directory / "target-video.mp4", td, fps=30)
            extension = "mkv" if origin else "mp4"
            source = mux_sound(sv, directory / f"source.{extension}", sd, source_time,
                               rate=44100, gain=gain, mute=mute, offset=origin)
            target = mux_sound(tv, directory / f"target.{extension}", td, target_time, rate=48000, offset=origin)
            config = AlignConfig(cache_dir=str(directory / "cache"),
                                 evidence_mode="hybrid" if filters else "audio")
            alignment = align_videos(source, target, config)
            data = {"data": [{"time": float(t)} for t in np.arange(.5, 8.5 if filters else sd - .5, .1)]}
            mapped = map_danmaku(data, alignment)
            errors, wrong = [], 0
            for entry in mapped.report["entries"]:
                if entry["target_time"] is None:
                    continue
                correct = expected(entry["source_time"])
                if correct is None:
                    wrong += 1
                else:
                    error = abs(entry["target_time"] - correct)
                    errors.append(error)
                    wrong += error > .2
            accepted = mapped.report["counts"]["reliable"]
            record = {"scenario": scenario, "counts": mapped.report["counts"], "rates": mapped.report["rates"],
                      "precision_within_200ms": (accepted - wrong) / accepted if accepted else None,
                      "p95_error_seconds": float(np.percentile(errors, 95)) if errors else None,
                      "max_error_seconds": max(errors) if errors else None,
                      "elapsed_seconds": alignment.diagnostics["elapsed_seconds"],
                      "audio_anchors": alignment.diagnostics["audio"]["audio_anchors"],
                      "cache_bytes": sum(p.stat().st_size for p in (directory / "cache").glob("*.npz"))}
            if filters:
                visual = map_danmaku(data, align_videos(source, target, replace(config, evidence_mode="visual")))
                record["visual_only_counts"] = visual.report["counts"]
            metrics.append(record)
            print(f"{scenario}: measured", flush=True)
    report = {"schema_version": 1, "generated_date": str(datetime.now(ZoneInfo("Asia/Shanghai")).date()),
              "python": platform.python_version(), "scope": "Synthetic moving texture and chirp-mixture soundtracks; no real anime acceptance",
              "source_audio_rate": 44100, "target_audio_rate": 48000, "codec": "AAC 96k", "metrics": metrics}
    atomic_json(output, report, overwrite=True)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/audio-acceptance-metrics.json")
    run(parser.parse_args().output)

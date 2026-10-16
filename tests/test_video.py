import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from bili_danmaku_mapper import AlignConfig, align_videos, map_danmaku
from bili_danmaku_mapper.media import extract_features
from tests.video_helpers import make_video


@pytest.mark.parametrize("scenario", ["identity", "head_cut", "middle_cut", "speed", "overlay", "target_cut", "vfr", "reencode"])
def test_known_video_timeline(tmp_path, scenario):
    target = tmp_path / "target with 'quotes'.mp4"
    source = tmp_path / "source.mp4"
    source_duration, target_duration = 8, 8
    source_timeline = lambda t: t
    target_timeline = lambda t: t
    expected = lambda t: t
    filters = None
    if scenario == "head_cut":
        source_duration = 7.3
        source_timeline = lambda t: t + .7
        expected = source_timeline
    elif scenario == "middle_cut":
        source_duration = 7
        source_timeline = lambda t: t if t < 3 else t + 1
        expected = source_timeline
    elif scenario == "speed":
        source_duration = 6.4
        source_timeline = lambda t: t * 1.25
        expected = source_timeline
    elif scenario == "overlay":
        filters = "drawbox=x=0:y=136:w=256:h=8:color=white:t=fill,pad=288:176:16:16:black"
    elif scenario == "target_cut":
        target_duration = 7
        target_timeline = lambda t: t if t < 3 else t + 1
        expected = lambda t: t if t < 3 else (None if t < 4 else t - 1)
    elif scenario == "vfr":
        filters = "select='not(eq(mod(n,10),3)+eq(mod(n,10),4))'"
    elif scenario == "reencode":
        filters = "scale=192:108,eq=brightness=0.025:contrast=1.05"
    make_video(source, source_duration, fps=24, timeline=source_timeline)
    make_video(target, target_duration, fps=30, timeline=target_timeline, filters=filters, vfr=scenario == "vfr")
    config = AlignConfig(cache_dir=str(tmp_path / "cache"))
    result = align_videos(source, target, config)
    times = np.arange(.2, source_duration - .2, .1)
    mapped = map_danmaku({"data": [{"time": float(t)} for t in times]}, result)
    errors, false = [], 0
    for entry in mapped.report["entries"]:
        if entry["target_time"] is not None:
            correct = expected(entry["source_time"])
            if correct is None:
                false += 1
            else:
                errors.append(abs(entry["target_time"] - correct))
    print(json.dumps({"scenario": scenario, "counts": mapped.report["counts"],
                      "segments": len(result.segments), "p95": float(np.percentile(errors, 95)) if errors else None,
                      "false": false}))
    assert len(errors) >= len(times) * .75, "Do not pass accuracy by rejecting nearly everything"
    assert false == 0
    assert np.percentile(errors, 95) <= .2
    assert sum(e <= .2 for e in errors) / len(errors) >= .99
    assert all(a.target_time < b.target_time for a, b in zip(
        [a for s in result.segments for a in s.anchors], [a for s in result.segments for a in s.anchors][1:]))


def test_frozen_gap_is_reviewed_or_explicitly_estimated(tmp_path):
    freeze = lambda t: 3.0 if 3 <= t < 6 else t
    source = make_video(tmp_path / "source.mp4", 8.3, timeline=lambda t: freeze(t + .7))
    target = make_video(tmp_path / "target.mp4", 9, fps=30, timeline=freeze)
    a = align_videos(source, target, AlignConfig(cache_dir=str(tmp_path / "cache")))
    data = {"data": [{"time": 4.0}, {"time": 1.2}, {"time": 6.2}]}
    strict, lenient = map_danmaku(data, a), map_danmaku(data, a, "lenient")
    assert strict.report["entries"][0]["status"] == "unmapped"
    assert lenient.report["entries"][0]["status"] == "estimated"
    assert abs(lenient.report["entries"][0]["target_time"] - 4.7) <= .2
    assert strict.report["counts"]["reliable"] == 2


def test_entire_repeated_clip_requires_manual_disambiguation(tmp_path):
    source = make_video(tmp_path / "source.mp4", 4)
    target = make_video(tmp_path / "target.mp4", 8, fps=30, timeline=lambda t: t % 4)
    config = AlignConfig(cache_dir=str(tmp_path / "cache"))
    a = align_videos(source, target, config)
    data = {"data": [{"time": 1.2}, {"time": 2.2}]}
    automatic = map_danmaku(data, a)
    assert automatic.report["counts"]["unmapped"] == 2
    fixed = align_videos(source, target, config,
                         [{"source_time": 0, "target_time": 4}, {"source_time": 3.5, "target_time": 7.5}])
    mapped = map_danmaku(data, fixed)
    assert mapped.report["counts"]["reliable"] == 2
    assert all(abs(e["target_time"] - e["source_time"] - 4) <= .2 for e in mapped.report["entries"])


def test_vfr_actual_pts_and_cache_recovery(tmp_path):
    path = make_video(tmp_path / "vfr.mkv", 2, fps=25,
                      filters="select='not(eq(mod(n,10),3)+eq(mod(n,10),4))'")
    # make_video's normal muxing can duplicate frames, so create explicit VFR output.
    vfr = tmp_path / "explicit-vfr.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                    "select='not(eq(mod(n,10),3)+eq(mod(n,10),4))'", "-fps_mode", "vfr",
                    "-c:v", "ffv1", str(vfr)], check=True)
    config = AlignConfig(cache_dir=str(tmp_path / "cache"))
    data = extract_features(vfr, config, rate=0)
    stamps = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(vfr)]))
    assert np.allclose(data.times, [float(f["best_effort_timestamp_time"]) for f in stamps["frames"]])
    assert np.max(np.diff(data.times)) > .07
    cache = next((tmp_path / "cache").glob("*.npz"))
    cache.write_bytes(b"interrupted")
    assert np.allclose(extract_features(vfr, config, rate=0).times, data.times)


def test_offset_and_window_pts(tmp_path):
    path = make_video(tmp_path / "offset.mkv", 3, pts_offset=5)
    config = AlignConfig(cache_dir=None)
    all_frames = extract_features(path, config, rate=0)
    assert all_frames.info.origin == 5
    assert abs(all_frames.info.duration - 3) < .01
    assert abs(all_frames.times[0]) < 1e-6
    window = extract_features(path, config, rate=0, start=1.1, end=2.1)
    assert 1.1 <= window.times[0] < 1.15
    assert window.times[-1] < 2.1


def test_hardware_failure_software_retry_and_corrupt_video(tmp_path):
    path = make_video(tmp_path / "valid.mp4", 1)
    config = AlignConfig(cache_dir=None, hwaccel="nonexistent")
    assert len(extract_features(path, config)) >= 2
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"broken")
    with pytest.raises(subprocess.CalledProcessError):
        extract_features(bad, config)

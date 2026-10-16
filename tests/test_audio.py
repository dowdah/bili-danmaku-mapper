from dataclasses import replace
import json
import subprocess

import numpy as np
import pytest

from bili_danmaku_mapper import AlignConfig, align_videos, map_danmaku
from bili_danmaku_mapper.audio import AudioUnavailable, audio_anchors, fingerprint, fuse_alignment, select_streams
from bili_danmaku_mapper.models import Anchor
from tests.audio_helpers import mux_sound, sound
from tests.test_mapping import alignment
from tests.video_helpers import make_video


def test_audio_fingerprints_local_offsets():
    rate = 11025
    t = np.arange(rate * 12) / rate
    target = fingerprint(sound(t))
    source = fingerprint(sound(t + .7))
    anchors, diagnostics = audio_anchors(source, target, 12, 12)
    assert len(anchors) >= 16
    assert diagnostics["fingerprint_votes"] > 0
    assert np.percentile([abs(a.target_time - a.source_time - .7) for a in anchors], 95) < .08


def test_silence_tone_and_unrelated_sound_do_not_invent_mapping():
    rate = 11025
    t = np.arange(rate * 8) / rate
    for source, target in [(np.zeros_like(t), np.zeros_like(t)),
                            (np.sin(t * 440 * 2 * np.pi), np.sin(t * 440 * 2 * np.pi)),
                            (sound(t), sound(t, variant=5))]:
        anchors, _ = audio_anchors(fingerprint(source), fingerprint(target), 8, 8)
        assert not anchors


def test_repeated_sound_requires_disambiguation():
    t = np.arange(11025 * 6) / 11025
    source = fingerprint(sound(t))
    target = fingerprint(np.concatenate([sound(t), sound(t)]))
    anchors, _ = audio_anchors(source, target, 6, 12)
    interior = [a for a in anchors if 1 <= a.source_time <= 4]
    assert not interior
    fixed, _ = audio_anchors(source, target, 6, 12, [Anchor(0, 6, manual=True), Anchor(5.5, 11.5, manual=True)])
    assert len(fixed) >= 6
    assert all(abs(a.target_time - a.source_time - 6) <= .08 for a in fixed)


@pytest.mark.parametrize("scenario", ["head_cut", "middle_cut", "target_cut", "volume", "silence", "origin"])
def test_audio_only_encoded_media(tmp_path, scenario):
    source_duration, target_duration = 10, 10
    source_timeline = target_timeline = lambda t: t
    expected = lambda t: t
    mute, gain, offset = None, 1, 0
    if scenario == "head_cut":
        source_duration, source_timeline = 9.3, lambda t: t + .7
        expected = source_timeline
    elif scenario == "middle_cut":
        source_duration, source_timeline = 9, lambda t: t if t < 4 else t + 1
        expected = source_timeline
    elif scenario == "target_cut":
        target_duration, target_timeline = 9, lambda t: t if t < 4 else t + 1
        expected = lambda t: t if t < 4 else None if t < 5 else t - 1
    elif scenario == "volume":
        gain = .3
    elif scenario == "silence":
        mute = (4, 6)
    elif scenario == "origin":
        offset = 5
    source_video = make_video(tmp_path / "source-video.mp4", source_duration)
    target_video = make_video(tmp_path / "target-video.mp4", target_duration, fps=30)
    extension = "mkv" if offset else "mp4"
    source = mux_sound(source_video, tmp_path / f"source.{extension}", source_duration,
                       source_timeline, rate=44100, gain=gain, mute=mute, offset=offset)
    target = mux_sound(target_video, tmp_path / f"target.{extension}", target_duration,
                       target_timeline, rate=48000, offset=offset)
    config = AlignConfig(cache_dir=str(tmp_path / "cache"), evidence_mode="audio")
    a = align_videos(source, target, config)
    times = np.arange(.5, source_duration - .5, .1)
    mapped = map_danmaku({"data": [{"time": float(t)} for t in times]}, a)
    errors, wrong = [], 0
    for entry in mapped.report["entries"]:
        if entry["target_time"] is None:
            continue
        truth = expected(entry["source_time"])
        if truth is None:
            wrong += 1
        else:
            errors.append(abs(entry["target_time"] - truth))
    print(json.dumps({"scenario": scenario, "counts": mapped.report["counts"],
                      "p95_ms": float(np.percentile(errors, 95) * 1000) if errors else None, "wrong": wrong}))
    assert len(errors) >= len(times) * .5
    assert wrong == 0
    assert np.percentile(errors, 95) < .1
    assert max(errors) <= .2
    if mute:
        center = map_danmaku({"data": [{"time": 5.0}]}, a)
        assert center.report["counts"]["unmapped"] == 1


def test_audio_rescues_fully_obscured_picture(tmp_path):
    # Censor the source picture completely; preserve its soundtrack.
    source_video = make_video(tmp_path / "source-video.mp4", 9.3, filters="drawbox=color=black:t=fill")
    target_video = make_video(tmp_path / "target-video.mp4", 10, fps=30)
    source = mux_sound(source_video, tmp_path / "source.mp4", 9.3, lambda t: t + .7)
    target = mux_sound(target_video, tmp_path / "target.mp4", 10, rate=48000)
    config = AlignConfig(cache_dir=str(tmp_path / "cache"))
    data = {"data": [{"time": float(t)} for t in np.arange(.5, 8.5, .1)]}
    visual = map_danmaku(data, align_videos(source, target, replace(config, evidence_mode="visual")))
    hybrid = map_danmaku(data, align_videos(source, target, config))
    assert visual.report["counts"]["reliable"] == 0
    assert hybrid.report["counts"]["reliable"] >= 60
    errors = [abs(e["target_time"] - e["source_time"] - .7) for e in hybrid.report["entries"] if e["target_time"] is not None]
    assert np.percentile(errors, 95) < .1
    assert all("audio" in e["evidence"] for e in hybrid.report["entries"] if e["target_time"] is not None)
    print(json.dumps({"scenario": "fully_obscured_hybrid", "visual": visual.report["counts"],
                      "hybrid": hybrid.report["counts"], "p95_ms": float(np.percentile(errors, 95) * 1000)}))


def test_actual_delayed_audio_calibrates_against_picture(tmp_path):
    source_video = make_video(tmp_path / "sv.mp4", 9.3, timeline=lambda t: t + .7)
    target_video = make_video(tmp_path / "tv.mp4", 10, fps=30)
    source = mux_sound(source_video, tmp_path / "s.mp4", 9.3, lambda t: t + .7, delay=.3)
    target = mux_sound(target_video, tmp_path / "t.mp4", 10, rate=48000)
    a = align_videos(source, target, AlignConfig(cache_dir=str(tmp_path / "cache")))
    assert abs(a.diagnostics["audio_bias_seconds"] + .3) < .08
    assert a.diagnostics["audio_visual_conflicts"] == 0
    mapped = map_danmaku({"data": [{"time": 2.1}, {"time": 6.2}]}, a)
    assert mapped.report["counts"]["reliable"] == 2
    assert all(abs(e["target_time"] - e["source_time"] - .7) < .08 for e in mapped.report["entries"])


def test_audio_cache_corruption_and_multi_track_selection(tmp_path):
    from bili_danmaku_mapper.audio import extract_audio
    from bili_danmaku_mapper.media import probe_video
    video = make_video(tmp_path / "v.mp4", 3)
    first = mux_sound(video, tmp_path / "a.mp4", 3, metadata="jpn")
    second = mux_sound(video, tmp_path / "b.mp4", 3, variant=5, metadata="eng")
    combined = tmp_path / "multi.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(second), "-i", str(first),
                    "-map", "0:v:0", "-map", "0:a:0", "-map", "1:a:0", "-c", "copy", str(combined)], check=True)
    config = AlignConfig(cache_dir=str(tmp_path / "cache"))
    s, t, selection = select_streams(first, combined, config)
    assert s["ordinal"] == 0 and t["ordinal"] == 1 and selection == "matching_language"
    info = probe_video(first, config)
    original = extract_audio(info, s, config)
    cache = next((tmp_path / "cache").glob("*.audio.npz"))
    cache.write_bytes(b"interrupted")
    regenerated = extract_audio(info, s, config)
    assert np.array_equal(regenerated.keys, original.keys)
    with pytest.raises(AudioUnavailable, match="no_common_audio_language"):
        select_streams(first, second, config)
    _, forced, _ = select_streams(first, combined, replace(config, target_audio_stream=0))
    assert forced["ordinal"] == 0


def test_audio_missing_fallback_and_explicit_failure(tmp_path):
    video = make_video(tmp_path / "no-audio.mp4", 2)
    result = align_videos(video, video, AlignConfig(cache_dir=str(tmp_path / "cache")))
    assert result.segments
    assert result.diagnostics["audio"]["status"] == "unavailable"
    with pytest.raises(AudioUnavailable):
        align_videos(video, video, AlignConfig(evidence_mode="audio"))
    with pytest.raises(ValueError):
        select_streams(video, video, AlignConfig(source_audio_stream=7))


def test_fusion_calibrates_constant_av_delay():
    v = alignment([[(i / 2, i / 2) for i in range(19)]])
    audio = [Anchor(i / 2 + .25, i / 2 + .55, evidence="audio") for i in range(16)]
    result = fuse_alignment(v, audio, AlignConfig())
    assert abs(result.diagnostics["audio_bias_seconds"] - .3) < 1e-6
    assert result.diagnostics["audio_visual_conflicts"] == 0


def test_fusion_conflicts_are_reviewed_in_both_modes():
    v = alignment([[(i / 2, i / 2) for i in range(19)]])
    # Inconsistent offset, not a calibratable common track delay.
    audio = [Anchor(.75, 1.35, evidence="audio"), Anchor(3.25, 2.5, evidence="audio")]
    result = fuse_alignment(v, audio, AlignConfig())
    assert result.diagnostics["audio_visual_conflicts"] == 2
    for mode in ("strict", "lenient"):
        out = map_danmaku({"data": [{"time": .75}, {"time": 3.25}]}, result, mode)
        assert out.report["counts"]["unmapped"] == 2
        assert all(e["reason"] == "audio_visual_conflict" for e in out.report["entries"])


def test_fusion_does_not_bridge_visible_edit():
    v = alignment([[(0, 0), (1, 1)], [(2, 4), (3, 5)]])
    result = fuse_alignment(v, [Anchor(1.5, 2.5, evidence="audio")], AlignConfig())
    assert map_danmaku({"data": [{"time": 1.5}]}, result, "lenient").report["counts"]["unmapped"] == 1


def test_exact_manual_anchor_wins_over_audio_disagreement():
    v = alignment([[(0, 0), (.5, .5), (.75, .75), (1, 1), (1.5, 1.5)]])
    manual = v.segments[0].anchors[2]
    manual.manual, manual.evidence = True, "manual"
    result = fuse_alignment(v, [Anchor(.75, 1.35, evidence="audio")], AlignConfig())
    for mode in ("strict", "lenient"):
        out = map_danmaku({"data": [{"time": .75}, {"time": .8}]}, result, mode)
        assert out.report["entries"][0]["target_time"] == .75
        assert out.report["entries"][0]["evidence"] == ["manual"]
        assert out.report["entries"][1]["status"] == "unmapped"

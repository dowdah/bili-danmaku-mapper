import numpy as np
import pytest

from bili_danmaku_mapper.alignment import align_features, monotonic_path
from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.media import Features, VideoInfo, describe_frame


def features(labels, times=None):
    images = []
    for label in labels:
        if label == "flat":
            pixels = np.zeros((144, 256), dtype=np.uint8)
        else:
            # Smooth distinct luminance patterns retain identity through resize/encode.
            rng = np.random.default_rng(int(label))
            from PIL import Image
            pixels = np.asarray(Image.fromarray(rng.integers(25, 230, (9, 16), dtype=np.uint8)).resize((256, 144), Image.Resampling.BILINEAR))
        images.append(describe_frame(pixels, .8))
    times = np.asarray(times if times is not None else np.arange(len(labels)) / 2, dtype=float)
    return Features(times, np.array([x[0] for x in images]), np.array([x[1] for x in images]),
                    np.array([x[2] for x in images]), np.array([x[3] for x in images]),
                    VideoInfo("fixture", float(times[-1] + .5), 0, 24))


def test_monotonic_repeated_picture():
    source, target = features([2, 3, 1, 4]), features([0, 1, 2, 3, 1, 4])
    result = align_features(source, target, AlignConfig(refine=False))
    anchors = [a for s in result.segments for a in s.anchors]
    assert len(anchors) >= 3
    assert all(a.target_time < b.target_time for a, b in zip(anchors, anchors[1:]))
    assert all(abs(a.target_time - a.source_time - 1) < 1e-6 for a in anchors)


def test_flat_picture_has_no_independent_anchor():
    source = features(["flat"] * 12)
    assert not align_features(source, source, AlignConfig(refine=False)).segments


def test_frozen_detailed_picture_has_no_invented_time():
    source = features([1] * 12)
    assert not align_features(source, source, AlignConfig(refine=False)).segments


def test_identity_coverage():
    source = features(list(range(20)))
    result = align_features(source, source, AlignConfig(refine=False))
    anchors = [a for s in result.segments for a in s.anchors]
    assert len(anchors) >= 18
    assert all(a.source_time == a.target_time for a in anchors)


def test_all_unmatched():
    source, target = features(list(range(12))), features(list(range(50, 62)))
    assert not align_features(source, target, AlignConfig(refine=False)).segments


def test_edit_skips_and_time_scale():
    labels = list(range(30))
    source = features(labels[:10] + labels[14:])
    target = features(labels, np.arange(30) * .625)
    result = align_features(source, target, AlignConfig(refine=False))
    anchors = [a for s in result.segments for a in s.anchors]
    assert len(anchors) >= 20
    assert len(result.segments) >= 2
    for a in anchors:
        index = int(round(a.source_time * 2))
        original = index if index < 10 else index + 4
        assert abs(a.target_time - original * .625) < 1e-6


def test_dp_never_collapses_or_regresses():
    f = features(list(range(10)))
    rows = [[(3, 0)], [(4, .01), (1, 0)], [(2, 0), (5, .01)], [(6, 0)]] + [[]] * 6
    path = monotonic_path(rows, f, f, AlignConfig(refine=False))
    assert all(a[1] < b[1] for a, b in zip(path, path[1:]))


def test_manual_constraints_resolve_repeated_tracks():
    source = features(list(range(8)))
    target = features(list(range(8)) + list(range(8)))
    result = align_features(source, target, AlignConfig(refine=False),
                            [{"source_time": 0, "target_time": 4}, {"source_time": 3.5, "target_time": 7.5}])
    anchors = [a for s in result.segments for a in s.anchors]
    assert all(a.target_time >= 4 for a in anchors)
    assert sum(a.manual for a in anchors) == 2

import copy

import pytest

from bili_danmaku_mapper import map_danmaku
from bili_danmaku_mapper.alignment import build_segments, validate_manual_anchors
from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.models import AlignmentResult, Anchor, Segment


def alignment(groups):
    segments = []
    for group in groups:
        a = [Anchor(s, t) for s, t in group]
        first, last = a[0], a[-1]
        slope = (last.target_time - first.target_time) / (last.source_time - first.source_time) if len(a) > 1 else 1
        segments.append(Segment(first.source_time, last.source_time, first.target_time, last.target_time,
                                slope, first.target_time - slope * first.source_time, 1, a))
    return AlignmentResult("source", "target", 20, 20, segments, {"strict_gap": 1.0})


def test_no_missing_interval_absorption_or_extrapolation():
    data = {"code": 1, "data": [{"time": i / 2, "text": str(i)} for i in range(15)]}
    original = copy.deepcopy(data)
    result = map_danmaku(data, alignment([[(1, 2), (2, 3)], [(5, 4), (6, 5)]]))
    assert data == original
    assert [e["source_time"] for e in result.report["entries"] if e["status"] == "reliable"] == [1, 1.5, 2, 5, 5.5, 6]
    assert all(e["target_time"] is None for e in result.report["entries"] if e["status"] == "unmapped")


def test_empty_alignment_and_empty_danmaku():
    assert map_danmaku({"data": [{"time": 1}]}, alignment([])).report["counts"]["unmapped"] == 1
    assert map_danmaku({"data": []}, alignment([])).report["rates"] == {"reliable": 0, "estimated": 0, "unmapped": 0}


def test_strict_vs_lenient_and_roundtrip():
    a = alignment([[(1, 2), (4, 5)]])
    data = {"data": [{"time": 2.0, "text": "long gap"}]}
    assert map_danmaku(data, a).danmaku["data"] == []
    restored = AlignmentResult.from_dict(a.to_dict())
    result = map_danmaku(data, restored, "lenient")
    assert result.danmaku["data"][0]["time"] == 3
    assert result.report["counts"]["estimated"] == 1


def test_segments_split_at_both_edit_directions():
    config = AlignConfig(refine=False)
    a = [Anchor(0, 0), Anchor(.5, .5), Anchor(1, 3), Anchor(1.5, 3.5), Anchor(4, 4), Anchor(4.5, 4.5)]
    groups = build_segments(a, config)
    assert len(groups) == 3


@pytest.mark.parametrize("anchors", [
    [{"source_time": 2, "target_time": 1}, {"source_time": 1, "target_time": 2}],
    [{"source_time": 1, "target_time": 2}, {"source_time": 2, "target_time": 1}],
    [{"source_time": float("nan"), "target_time": 2}],
    [{"source_time": 50, "target_time": 2}],
])
def test_invalid_manual_anchors(anchors):
    with pytest.raises(ValueError):
        validate_manual_anchors(anchors, 20, 20)


@pytest.mark.parametrize("time", [None, True, "1", float("nan"), float("inf")])
def test_invalid_comment_times(time):
    with pytest.raises(ValueError):
        map_danmaku({"data": [{"time": time}]}, alignment([]))

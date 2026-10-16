import bisect
import copy
import math

from .models import AlignmentResult, MappingResult


def map_time(value, alignment, mode="strict", _index=None):
    if mode not in ("strict", "lenient"):
        raise ValueError("mode must be strict or lenient")
    if not math.isfinite(value) or not 0 <= value <= alignment.source_duration:
        return None, "unmapped", "outside_source_duration"
    starts = _index[0] if _index is not None else [s.source_start for s in alignment.segments]
    index = bisect.bisect_right(starts, value) - 1
    if index >= 0:
        segment = alignment.segments[index]
        times = _index[1][index] if _index is not None else [a.source_time for a in segment.anchors]
        right = bisect.bisect_left(times, value)
        if right < len(times) and abs(times[right] - value) <= 1e-7 and segment.anchors[right].manual:
            return segment.anchors[right].target_time, "reliable", "manual_anchor"
    for interval in alignment.diagnostics.get("unverified_intervals", []):
        if interval["source_start"] <= value <= interval["source_end"]:
            return None, "unmapped", interval["reason"]
    if index < 0:
        return None, "unmapped", "no_verified_segment"
    segment = alignment.segments[index]
    if value > segment.source_end:
        return None, "unmapped", "edit_or_unverified_interval"
    if right < len(times) and abs(times[right] - value) <= 1e-7:
        return segment.anchors[right].target_time, "reliable", "verified_anchor"
    if not 0 < right < len(times):
        return None, "unmapped", "no_extrapolation"
    before, after = segment.anchors[right - 1:right + 1]
    gap = after.source_time - before.source_time
    strict_gap = float(alignment.config.get("strict_gap", 1.0))
    if gap > strict_gap + 1e-7 and mode == "strict":
        return None, "unmapped", "gap_exceeds_strict_limit"
    target = before.target_time + (value - before.source_time) / gap * (after.target_time - before.target_time)
    if not 0 <= target <= alignment.target_duration:
        return None, "unmapped", "outside_target_duration"
    status = "reliable" if gap <= strict_gap + 1e-7 else "estimated"
    return target, status, "segment_interpolation" if status == "reliable" else "long_gap_interpolation"


def reason_category(reason):
    if reason in ("audio_visual_conflict", "audio_visual_segment_conflict"):
        return "evidence_conflict"
    if reason == "confirmed_absent_interval":return "confirmed_no_correspondence"
    if reason.startswith("outside_"):return "outside_timeline"
    return "insufficient_evidence"


def map_danmaku(data, alignment, mode="strict"):
    if isinstance(alignment, dict):
        alignment = AlignmentResult.from_dict(alignment)
    alignment.validate()
    if mode not in ("strict", "lenient"):
        raise ValueError("mode must be strict or lenient")
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("Danmaku must be an object with a data list")
    output, review, records = [], [], []
    time_index = ([s.source_start for s in alignment.segments],
                  [[a.source_time for a in s.anchors] for s in alignment.segments])
    counts = {"reliable": 0, "estimated": 0, "unmapped": 0}
    for index, item in enumerate(data["data"]):
        if not isinstance(item, dict):
            raise ValueError(f"Danmaku {index} must be an object")
        value = item.get("time")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Danmaku {index} time must be a finite number")
        target, status, reason = map_time(float(value), alignment, mode, time_index)
        counts[status] += 1
        record = {"source_index": index, "source_time": value, "target_time": target,
                  "status": status, "reason": reason}
        if target is None:record["reason_category"] = reason_category(reason)
        if target is not None:
            segment_index = bisect.bisect_right(time_index[0], value) - 1
            segment = alignment.segments[segment_index]
            position = bisect.bisect_left(time_index[1][segment_index], value)
            if position < len(segment.anchors) and abs(segment.anchors[position].source_time - value) <= 1e-7:
                supporting = [segment.anchors[position]]
            else:
                supporting = segment.anchors[max(0, position - 1):position + 1]
            evidence = sorted({a.evidence for a in supporting})
            record["evidence"] = evidence
        records.append(record)
        if target is None:
            review.append({"item": copy.deepcopy(item), **record})
        else:
            mapped = copy.deepcopy(item)
            mapped["time"] = target
            output.append(mapped)
    output.sort(key=lambda item: item["time"])
    total = len(data["data"])
    result = MappingResult({**copy.deepcopy(data), "data": output},
                         {"schema_version": 1, "data": review},
                         {"schema_version": 1, "mode": mode, "counts": {"total": total, **counts},
                          "rates": {k: v / total if total else 0.0 for k, v in counts.items()},
                          "entries": records})

    if alignment.config.get('coverage_recovery'):
        from collections import Counter
        categories=Counter(e['reason_category'] for e in records if e['status']=='unmapped')
        grid=Counter();step=.5;size=math.ceil(alignment.source_duration/step)
        for i in range(size):
            t=min(alignment.source_duration,(i+.5)*step)
            grid[map_time(t,alignment,mode,time_index)[1]]+=1
        result.report['unmapped_categories']=dict(categories)
        result.report['uniform_time_coverage']={'step_seconds':step,'samples':size,
            'rates':{k:grid[k]/size for k in ('reliable','estimated','unmapped')}}
        result.report['comparable_interval_coverage']=None
        result.report['real_accuracy_human_verified']=False
    return result

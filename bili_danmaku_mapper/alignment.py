"""Sparse, monotonic sequence alignment with explicit edit boundaries."""
import math
import time

import numpy as np

from .config import AlignConfig, ALGORITHM_VERSION, BASE_ALGORITHM_VERSION
from .media import Features, extract_features
from .models import AlignmentResult, Anchor, Segment

POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def frame_cost(source, i, target, indices):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return np.empty(0)
    full = POPCOUNT[np.bitwise_xor(source.hashes[i], target.hashes[indices])].sum(axis=1) / 64
    core = POPCOUNT[np.bitwise_xor(source.center_hashes[i], target.center_hashes[indices])].sum(axis=1) / 64
    a = source.gray[i].astype(np.float32)
    b = target.gray[indices].astype(np.float32)
    # Centered luminance reduces sensitivity to modest encoding/brightness changes.
    a = (a - a.mean()) / max(float(a.std()), 6.0)
    b = (b - b.mean(axis=(1, 2), keepdims=True)) / np.maximum(b.std(axis=(1, 2), keepdims=True), 6.0)
    gray = np.minimum(np.abs(a - b).mean(axis=(1, 2)) / 2, 1)
    edge = np.minimum(np.abs(np.diff(a, axis=1) - np.diff(b, axis=2)).mean(axis=(1, 2)) / 2, 1)
    return 0.2 * full + 0.5 * core + 0.2 * gray + 0.1 * edge


def _informative(features):
    valid = features.quality >= 6.0
    # A detailed but frozen image is not a temporal anchor either.
    for i in range(len(features)):
        lo = max(0, i - 2)
        hi = min(len(features), i + 3)
        motion = POPCOUNT[np.bitwise_xor(features.center_hashes[i], features.center_hashes[lo:hi])].sum(axis=1)
        if len(motion) >= 3 and np.max(motion) <= 1:
            valid[i] = False
        # Even the first/last sample of a frozen run has an ambiguous image time.
        adjacent = [k for k in (i - 1, i + 1) if 0 <= k < len(features)]
        if any(POPCOUNT[np.bitwise_xor(features.center_hashes[i], features.center_hashes[k])].sum() <= 1
               for k in adjacent):
            valid[i] = False
    return valid


def _context_cost(source, i, target, j, seconds, extra_scales=()):
    costs = []
    half = seconds / 2
    # Try a small set of local time scales. Final segment slopes are fitted from PTS.
    for scale in sorted(set((0.5, 0.8, 1.0, 1.25, 2.0) + tuple(extra_scales))):
        neighbors = []
        for delta in (-half, -half / 2, half / 2, half):
            def nearest(times, value):
                at = int(np.searchsorted(times, value))
                options = [k for k in (at - 1, at) if 0 <= k < len(times)]
                return min(options, key=lambda k: abs(times[k] - value))
            s = nearest(source.times, source.times[i] + delta)
            # Use the actual available source displacement. Clamping source
            # context at an endpoint must not favor the last repeated copy.
            actual_delta = float(source.times[s] - source.times[i])
            t = nearest(target.times, target.times[j] + actual_delta * scale)
            if s == i or t == j or abs(source.times[s] - source.times[i] - delta) > half / 2:
                continue
            neighbors.append(float(frame_cost(source, s, target, [t])[0]))
        if len(neighbors) >= 2:
            costs.append(float(np.mean(neighbors)))
    return min(costs) if costs else 0.5


def candidate_rows(source, target, config, manual=()):
    from .performance import count
    count("candidate_pairs", len(source) * len(target))
    from .evidence import rejected, call, scales
    call()
    rows = [[] for _ in range(len(source))]
    source_valid, target_valid = _informative(source), _informative(target)
    # No NxM Python objects or full-video distance matrix; blocks stay bounded.
    from .progress import report
    for start in range(0, len(source), 64):
        report("candidates", completed=start, total=len(source), unit="frames")
        block = source.center_hashes[start:start + 64]
        distances = POPCOUNT[np.bitwise_xor(block[:, None, :], target.center_hashes[None, :, :])].sum(axis=2)
        full_distances = POPCOUNT[np.bitwise_xor(source.hashes[start:start + 64, None, :], target.hashes[None, :, :])].sum(axis=2)
        for local, dist in enumerate(distances):
            i = start + local
            if not source_valid[i]:
                rejected("low_information", source.times[i])
                continue
            allowed = target_valid.copy()
            for anchor in manual:
                if source.times[i] < anchor.source_time:
                    allowed &= target.times < anchor.target_time
                elif source.times[i] > anchor.source_time:
                    allowed &= target.times > anchor.target_time
                else:
                    allowed &= np.abs(target.times - anchor.target_time) <= 0.6
            eligible = np.flatnonzero(allowed & (np.minimum(dist, full_distances[local]) <= config.max_hash_distance))
            if not len(eligible):
                rejected("no_hash_candidate", source.times[i])
                continue
            costs = frame_cost(source, i, target, eligible)
            order = np.argsort(costs, kind="stable")
            chosen = []
            # Keep distinct temporal alternatives, rather than five adjacent frames.
            for index in order:
                j = int(eligible[index])
                if any(abs(target.times[j] - target.times[k]) < 0.75 for k, _ in chosen):
                    continue
                if costs[index] > 0.25:
                    break
                chosen.append((j, float(costs[index])))
                if len(chosen) == config.top_k:
                    break
            if not chosen: rejected("appearance_rejected", source.times[i], hash_candidates=len(eligible))
            candidates = [(j, 0.65 * cost + 0.35 * _context_cost(source, i, target, j, config.context_seconds, scales()))
                          for j, cost in chosen]
            candidates.sort(key=lambda pair: pair[1])
            for j, cost in candidates:
                if cost > 0.22:
                    rejected("context_failed", source.times[i], target_time=float(target.times[j]))
                    continue
                # Similar independent tracks cannot be disambiguated by a hash threshold.
                competing = [c for k, c in candidates if abs(target.times[k] - target.times[j]) >= 2.0]
                if competing and min(competing) <= cost + 0.025:
                    rejected("candidate_ambiguity", source.times[i], target_time=float(target.times[j]))
                    continue
                rows[i].append((j, cost))
    return rows


def monotonic_path(rows, source, target, config):
    """Sparse DP: recent continuity transitions plus a prefix tree for edit skips."""
    nodes, by_row = [], []
    tree = [None] * (len(target) + 1)
    def better(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return a if nodes[a][3] >= nodes[b][3] else b
    def query(j):
        best = None
        while j:
            best = better(best, tree[j])
            j -= j & -j
        return best
    def update(j, node):
        j += 1
        while j < len(tree):
            tree[j] = better(tree[j], node)
            j += j & -j
    recent = max(4, int(config.sample_rate * 5))
    for i, row in enumerate(rows):
        current = []
        for j, cost in row:
            reward = 1.0 - 2 * cost
            score, parent, slope = reward, None, 1.0
            global_parent = query(j)  # Strictly earlier target frames only.
            if global_parent is not None:
                proposal = nodes[global_parent][3] + reward - 0.65
                if proposal > score:
                    score, parent = proposal, global_parent
            for group in by_row[max(0, i - recent):]:
                for previous in group:
                    pi, pj, _, previous_score, _, previous_slope = nodes[previous]
                    if pj >= j:
                        continue
                    ds, dt = source.times[i] - source.times[pi], target.times[j] - target.times[pj]
                    ratio = dt / ds
                    penalty = 0.08 * max(0, i - pi - 1)
                    if 0.5 <= ratio <= 2.0:
                        penalty += 0.15 * min(abs(math.log(ratio / previous_slope)), 2)
                    else:
                        penalty += 0.5  # An edit transition; not continuous interpolation.
                    proposal = previous_score + reward - penalty
                    if proposal > score:
                        score, parent, slope = proposal, previous, ratio if 0.5 <= ratio <= 2 else 1.0
            nodes.append((i, j, cost, score, parent, slope))
            current.append(len(nodes) - 1)
        by_row.append(current)
        for node in current:
            update(nodes[node][1], node)
    if not nodes:
        return []
    node = max(range(len(nodes)), key=lambda x: nodes[x][3])
    path = []
    while node is not None:
        i, j, cost, _, parent, _ = nodes[node]
        path.append((i, j, max(0.0, 1 - cost / 0.4)))
        node = parent
    return list(reversed(path))


def refine_path(path, source, target, config):
    """Merge all requested neighborhoods and reuse bounded dense tile features."""
    config = config.for_video("target")
    from .media import dense_windows, merge_windows
    from .performance import phase, count
    from .progress import report
    report("refining")
    context_halo = 2.5 if config.coverage_recovery else .6
    windows = [(max(0., float(target.times[j])-context_halo), min(target.info.duration, float(target.times[j])+context_halo+.05))
               for _, j, _ in path]
    if config.merge_refinement:
        blocks = dense_windows(target.info, windows, config)
    else:
        # Exact former 8-second scheduling, retained for controlled ablations.
        groups = []
        for point in path:
            if not groups or target.times[point[1]]-target.times[groups[-1][0][1]] > 8:
                groups.append([])
            groups[-1].append(point)
        blocks = (extract_features(target.info.path, config, rate=0,
                  start=max(0., float(target.times[g[0][1]])-.6),
                  end=min(target.info.duration, float(target.times[g[-1][1]])+.65),
                  use_cache=False, _info=target.info) for g in groups)
    best = {}
    alternatives = {}
    stamps = np.array([target.times[j] for _, j, _ in path])
    with phase("refinement"):
        for dense in blocks:
            left, right = np.searchsorted(stamps, [dense.times[0]-.6, dense.times[-1]+.6])
            for point in range(left, min(len(path), right+1)):
                i, j, confidence = path[point]
                # Only complete neighborhoods nominate an answer; halos make tile
                # and scheduler boundaries equivalent to the original windows.
                at = float(target.times[j])
                coverage = getattr(dense, "coverage", (float(dense.times[0]),float(dense.times[-1])+1/max(target.info.fps,1)))
                if (coverage[0] > max(0., at-.6)+1e-6
                        or coverage[1] < min(target.info.duration, at+.6)-1e-6):
                    continue
                near = np.flatnonzero(np.abs(dense.times-at) <= .6)
                costs = frame_cost(source, i, dense, near)
                if not len(costs): continue
                if config.coverage_recovery:
                    from .evidence import scales
                    shortlist=np.argsort(costs,kind='stable')[:3]
                    contextual=np.array([.8*float(costs[k])+.2*_context_cost(source,i,dense,int(near[k]),config.context_seconds,scales()) for k in shortlist])
                    for short,sequence_cost in zip(shortlist,contextual):
                        if costs[short] <= .25:
                            stamp=float(dense.times[int(near[short])])
                            choices=alternatives.setdefault(point,{})
                            choices[stamp]=min(choices.get(stamp,float('inf')),float(sequence_cost))
                    winner=int(shortlist[int(np.argmin(contextual))])
                    low=float(costs[winner])
                    if low>.25:continue
                    tied=near[shortlist[contextual<=float(np.min(contextual))+.001]]
                else:
                    low = float(np.min(costs))
                    tied = near[costs <= low+.003]
                k = int(tied[np.argmin(np.abs(dense.times[tied]-at))])
                value = (low, abs(float(dense.times[k])-at), float(dense.times[k]))
                if point not in best or value[:2] < best[point][:2]: best[point] = value
    if config.coverage_recovery:
        from .coverage import joint_path
        options=[Anchor(float(source.times[path[point][0]]),stamp,
                        max(0.,path[point][2]-.2*cost))
                 for point,choices in alternatives.items() for stamp,cost in choices.items()]
        valid=joint_path(options,[])
        count("refined_anchors",len(valid))
        return valid
    valid = []
    for point, (i, j, confidence) in enumerate(path):
        if point in best:
            anchor = Anchor(float(source.times[i]), best[point][2], confidence)
            if not valid or anchor.target_time > valid[-1].target_time: valid.append(anchor)
    count("refined_anchors", len(valid))
    return valid


def validate_manual_anchors(values, source_duration, target_duration):
    if values is not None and not isinstance(values, list):
        raise ValueError("Manual anchors must be a JSON list")
    anchors = []
    for pair in values or []:
        if not isinstance(pair, dict) or any(isinstance(pair.get(key), bool) or
                not isinstance(pair.get(key), (int, float)) for key in ("source_time", "target_time")):
            raise ValueError("Manual anchor times must be numbers")
        anchors.append(Anchor(float(pair["source_time"]), float(pair["target_time"]), 1.0, True, "manual"))
    previous_s = previous_t = -1.0
    for anchor in anchors:
        if not (math.isfinite(anchor.source_time) and math.isfinite(anchor.target_time)
                and 0 <= anchor.source_time <= source_duration and 0 <= anchor.target_time <= target_duration):
            raise ValueError("Manual anchor outside duration or non-finite")
        if anchor.source_time <= previous_s or anchor.target_time <= previous_t:
            raise ValueError("Manual anchors must be strictly increasing in both timelines")
        previous_s, previous_t = anchor.source_time, anchor.target_time
    return anchors


def build_segments(anchors, config):
    if not anchors:
        return []
    anchors = sorted(anchors, key=lambda a: a.source_time)
    ratios = [(b.target_time - a.target_time) / (b.source_time - a.source_time)
              for a, b in zip(anchors, anchors[1:]) if b.source_time > a.source_time and b.target_time > a.target_time]
    normal = [r for r in ratios if 0.5 <= r <= 2]
    typical = float(np.median(normal)) if normal else 1.0
    groups, current = [], [anchors[0]]
    for anchor in anchors[1:]:
        previous = current[-1]
        ds, dt = anchor.source_time - previous.source_time, anchor.target_time - previous.target_time
        if ds <= 0 or dt <= 0:
            raise ValueError("Non-monotonic alignment")
        ratio = dt / ds
        recent_ratios = [(b.target_time - a.target_time) / (b.source_time - a.source_time)
                         for a, b in zip(current[-5:], current[-4:])] if len(current) >= 5 else []
        expected = float(np.median(recent_ratios)) if recent_ratios else typical
        # One-frame PTS quantization is tolerated; edit offsets are not.
        continuous = (max(ds, dt) <= config.max_segment_gap and 0.5 <= ratio <= 2
                      and abs(dt - ds * expected) <= max(0.12, ds * 0.12))
        if not continuous:
            groups.append(current)
            current = []
        current.append(anchor)
    groups.append(current)
    segments = []
    for group in groups:
        first, last = group[0], group[-1]
        slope = ((last.target_time - first.target_time) / (last.source_time - first.source_time)
                 if len(group) > 1 else 1.0)
        segments.append(Segment(first.source_time, last.source_time, first.target_time, last.target_time,
                                slope, first.target_time - slope * first.source_time,
                                min(a.confidence for a in group), group))
    return segments


def align_features(source, target, config=None, anchors=None):
    config = config or AlignConfig()
    manual = validate_manual_anchors(anchors, source.info.duration, target.info.duration)
    from .performance import phase
    with phase("candidate_search"):
        rows = candidate_rows(source, target, config, manual)
    # Manual correspondences constrain the entire DP, rather than merely overriding output.
    for i, row in enumerate(rows):
        s = source.times[i]
        rows[i] = [(j, c) for j, c in row if all(
            (s < a.source_time and target.times[j] < a.target_time) or
            (s > a.source_time and target.times[j] > a.target_time) or
            abs(s - a.source_time) <= 1e-6 for a in manual)]
    path = monotonic_path(rows, source, target, config)
    from .evidence import rejected
    selected_rows = {i for i,j,c in path}
    for i,row in enumerate(rows):
        if row and i not in selected_rows: rejected("path_conflict", source.times[i])
    automatic = [Anchor(float(source.times[i]), float(target.times[j]), confidence) for i, j, confidence in path]
    if config.refine and path:
        automatic = refine_path(path, source, target, config)
        selected_times = {round(a.source_time, 6) for a in automatic}
        for i,j,c in path:
            if round(float(source.times[i]),6) not in selected_times: rejected("precision_failed", source.times[i])
    from .evidence import primary_rows
    primary_rows(source,target,rows,path,automatic)
    # Keep manual anchors exact; automatic samples close to them must not contradict them.
    automatic = [a for a in automatic if all(abs(a.source_time - m.source_time) > 0.15 and
                 ((a.source_time < m.source_time and a.target_time < m.target_time) or
                  (a.source_time > m.source_time and a.target_time > m.target_time)) for m in manual)]
    matched = sorted(automatic + manual, key=lambda a: a.source_time)
    result = AlignmentResult(source.info.path, target.info.path, source.info.duration, target.info.duration,
                             build_segments(matched, config), config.to_dict(),
                             {"algorithm_version": ALGORITHM_VERSION, "source_samples": len(source),
                              "target_samples": len(target), "automatic_anchors": len(automatic),
                              "manual_anchors": len(manual), "confidence_is_probability": False})
    result.validate()
    return result


def _align_uncached(source, target, config=None, anchors=None):
    from .progress import report
    config = config or AlignConfig.resolve()
    started = time.perf_counter()
    if config.coverage_recovery and config.evidence_mode != "audio":
        from .coverage import align_recovered
        return align_recovered(source, target, config, anchors)
    if config.algorithm != "legacy" and config.evidence_mode == "audio":
        from .media import probe_video
        from .audio import align_audio
        source_info, target_info = probe_video(source, config), probe_video(target, config)
        manual = validate_manual_anchors(anchors, source_info.duration, target_info.duration)
        points, diagnostics = align_audio(source_info, target_info, config, manual)
        points = [a for a in points if all(abs(a.source_time - m.source_time) > .08 for m in manual)]
        result = AlignmentResult(source_info.path, target_info.path, source_info.duration, target_info.duration,
                                 build_segments(sorted(points + manual, key=lambda a: a.source_time), config),
                                 config.to_dict(), {"algorithm_version": ALGORITHM_VERSION,
                                                    "audio": diagnostics, "confidence_is_probability": False})
        result.validate()
        result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return result
    if config.algorithm == "sequence" and config.evidence_mode == "hybrid" and config.hybrid_strategy == "audio_first":
        from .audio_first import align_audio_first
        return align_audio_first(source, target, config, anchors)
    report("source_features")
    source_features = extract_features(source, config.for_video("source"), rate=0 if config.algorithm == "legacy" else None)
    report("target_features")
    target_features = extract_features(target, config.for_video("target"), rate=0 if config.algorithm == "legacy" else None)
    if config.algorithm == "legacy":
        if anchors:
            raise ValueError("Manual anchors require the sequence algorithm")
        from mapper_frame import AAPConfig, adaptive_anchor_propagation
        class Hashes:
            def __init__(self, features):
                self.hashes = features.hashes
            def __len__(self):
                return len(self.hashes)
            def __getitem__(self, i):
                import imagehash
                return imagehash.ImageHash(np.unpackbits(self.hashes[i]).reshape(8, 8).astype(bool))
        mapping, _ = adaptive_anchor_propagation(Hashes(target_features), Hashes(source_features),
                                               AAPConfig(max_hamming_distance=config.max_hash_distance))
        candidates = [Anchor(float(source_features.times[i]), float(target_features.times[j]), 0.5)
                      for i, j in sorted(mapping.items()) if j is not None]
        # Preserve the baseline's raw map for audits; do not export invalid time regression.
        valid = []
        for anchor in candidates:
            if not valid or anchor.target_time > valid[-1].target_time:
                valid.append(anchor)
        result = AlignmentResult(str(source), str(target), source_features.info.duration, target_features.info.duration,
                                 build_segments(valid, config), config.to_dict(),
                                 {"algorithm_version": "legacy-aap", "raw_anchors": [vars(a) for a in candidates],
                                  "discarded_nonmonotonic": len(candidates) - len(valid)})
        result.validate()
    else:
        report("sequence_matching")
        result = align_features(source_features, target_features, config, anchors)
        if config.evidence_mode == "hybrid":
            from .audio import AudioUnavailable, align_audio, fuse_alignment
            manual = validate_manual_anchors(anchors, source_features.info.duration, target_features.info.duration)
            try:
                report("audio_matching")
                points, diagnostics = align_audio(source_features.info, target_features.info, config, manual)
                result = fuse_alignment(result, points, config)
                result.diagnostics["audio"] = diagnostics
            except AudioUnavailable as error:
                result.diagnostics["audio"] = {"status": "unavailable", "reason": str(error)}
    result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
    return result


def align_videos(source, target, config=None, anchors=None):
    """Public API; input identity, validated result cache and per-run diagnostics."""
    import json
    from pathlib import Path
    from .cache import session, identity, key, locked
    from .media import probe_video
    from .io import atomic_json
    from .performance import collecting, phase, count
    from .progress import report
    config = config or AlignConfig.resolve()
    started = time.perf_counter()
    with collecting() as metrics, session():
        report("media_identity")
        source_info, target_info = probe_video(source, config), probe_video(target, config)
        validate_manual_anchors(anchors, source_info.duration, target_info.duration)
        options = {k:v for k,v in config.to_dict().items() if k not in ("cache_dir", "alignment_cache")}
        # Preserve the old cache signature when the experimental verifier is off.
        if not config.coverage_recovery:
            options.pop("coverage_recovery", None); options.pop("recovery_stage", None)
        streams = None
        if config.algorithm != "legacy" and config.evidence_mode != "visual":
            from .audio import select_streams, AudioUnavailable
            try:
                a,b,selection = select_streams(source, target, config)
                streams = [a["index"], b["index"], selection]
            except AudioUnavailable as error:
                streams = str(error)
        signature = key({"version": ALGORITHM_VERSION if config.coverage_recovery else BASE_ALGORITHM_VERSION, "source": identity(source, config),
                         "target": identity(target, config), "config": options, "streams": streams,
                         "anchors": anchors or []})
        cache = Path(config.cache_dir)/"alignment-v1"/(signature+".json") if config.cache_dir and config.alignment_cache else None
        def compute():
            result = None
            if cache:
                try:
                    document = json.loads(cache.read_text())
                    if document["signature"] != signature: raise ValueError("Cache signature mismatch")
                    result = AlignmentResult.from_dict(document["alignment"])
                    if result.source_duration != source_info.duration or result.target_duration != target_info.duration:
                        raise ValueError("Cache duration mismatch")
                    result.source, result.target = source_info.path, target_info.path
                    count("alignment_cache_hits"); report("alignment_cache", cache_hit=True)
                except (OSError, ValueError, KeyError, TypeError): result = None
            if result is None:
                count("alignment_cache_misses")
                with phase("alignment_compute"):
                    result = _align_uncached(source, target, config, anchors)
                result.validate()
                if cache and not metrics.counts.get("hardware_fallbacks",0):
                    atomic_json(cache, {"signature":signature, "alignment":result.to_dict()}, overwrite=True)
            return result
        if cache:
            with locked(cache): result = compute()
        else: result = compute()
    result.diagnostics.update(input_signature=signature, performance=metrics.to_dict(),
                              elapsed_seconds=time.perf_counter()-started)
    result.config = config.to_dict()
    if config.algorithm == "sequence":
        result.diagnostics["algorithm_version"] = ALGORITHM_VERSION if config.coverage_recovery else BASE_ALGORITHM_VERSION
    result.validate()
    return result

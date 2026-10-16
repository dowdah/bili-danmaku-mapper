"""Local spectral-landmark audio anchors and conservative audiovisual fusion.

Inspired by landmark fingerprinting (https://github.com/dpwe/audfprint).
This implementation is independent; it does not contact an identification service.
"""
from collections import defaultdict
import bisect
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
import zlib

import numpy as np
from scipy.signal import find_peaks

from .media import content_digest
from .models import Anchor, AlignmentResult

RATE, FFT_SIZE, HOP = 11025, 2048, 512
AUDIO_VERSION = "landmark-3"


class AudioUnavailable(RuntimeError):
    pass


@dataclass
class AudioFeatures:
    times: np.ndarray
    keys: np.ndarray
    frame_times: np.ndarray
    rms: np.ndarray
    spectra: np.ndarray


def audio_streams(path, config):
    data = subprocess.run([config.ffprobe, "-v", "error", "-select_streams", "a",
                           "-show_entries", "stream=index,channels,channel_layout,sample_rate:stream_tags=language,title",
                           "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=60)
    return [{**s, "ordinal": i} for i, s in enumerate(json.loads(data.stdout).get("streams", []))]


def select_streams(source, target, config):
    a, b = audio_streams(source, config), audio_streams(target, config)
    def explicit(streams, ordinal):
        if ordinal is None:
            return None
        if ordinal >= len(streams):
            raise ValueError(f"Audio stream ordinal {ordinal} does not exist")
        return streams[ordinal]
    chosen_a = explicit(a, config.source_audio_stream)
    chosen_b = explicit(b, config.target_audio_stream)
    if not a or not b:
        raise AudioUnavailable("missing_audio_stream")
    def language(stream):
        value = stream.get("tags", {}).get("language", "und").lower()
        return {"ja": "jpn", "japanese": "jpn", "en": "eng", "english": "eng", "zh": "zho", "chi": "zho"}.get(value, value)
    for s in ([chosen_a] if chosen_a else a):
        for t in ([chosen_b] if chosen_b else b):
            if language(s) == language(t) and language(s) not in ("und", ""):
                return s, t, "matching_language"
    if chosen_a or chosen_b:
        return chosen_a or a[0], chosen_b or b[0], "explicit_or_first"
    if all(language(s) not in ("und", "") for s in a + b):
        raise AudioUnavailable("no_common_audio_language")
    return a[0], b[0], "first_audio_stream_no_language_metadata"


def _fingerprint(samples, start=0.0):
    """Bounded FFT blocks; no full-duration spectrogram in memory."""
    samples = np.asarray(samples, dtype=np.float32)
    window = np.hanning(FFT_SIZE).astype(np.float32)
    peaks, frame_times, rms, descriptors = [], [], [], []
    frame_count = max(0, (len(samples) - FFT_SIZE) // HOP + 1)
    from .progress import report
    for begin in range(0, frame_count, 128):
        report("audio_fingerprints", completed=begin, total=frame_count, unit="windows")
        count = min(128, frame_count - begin)
        frames = np.lib.stride_tricks.sliding_window_view(samples[begin * HOP:], FFT_SIZE)[::HOP][:count]
        energy = np.sqrt(np.mean(frames * frames, axis=1))
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
        compressed = spectrum[:, :1024].reshape(count, 128, 8).sum(axis=2)
        compressed /= np.maximum(np.linalg.norm(compressed, axis=1, keepdims=True), 1e-12)
        descriptors.extend(compressed.astype(np.float32))
        for local in range(count):
            n = begin + local
            frame_times.append(start + (n * HOP + FFT_SIZE / 2) / RATE)
            rms.append(float(energy[local]))
            if energy[local] < 1e-4:
                continue
            spec = spectrum[local]
            bins, _ = find_peaks(spec, distance=6)
            bins = bins[(bins >= 12) & (bins < 930)]
            if not len(bins):
                continue
            bins = bins[np.argsort(spec[bins])[-3:]]
            for frequency in bins:
                if spec[frequency] >= np.max(spec) * 0.15:
                    peaks.append((n, int(frequency)))
    events, keys = [], []
    # Pair landmarks across a bounded time neighborhood, never the entire clip.
    by_frame = defaultdict(list)
    for n, f in peaks:
        by_frame[n].append(f)
    for n, f1 in peaks:
        neighbors = [(m, f2) for m in range(n + 3, n + 21) for f2 in by_frame.get(m, [])]
        # Spread the two paired peaks across time, so local transients remain useful.
        if neighbors:
            for index in sorted({len(neighbors) // 3, 2 * len(neighbors) // 3}):
                m, f2 = neighbors[index]
                events.append(start + (n * HOP + FFT_SIZE / 2) / RATE)
                keys.append((f1, f2, int(m - n)))
    return AudioFeatures(np.asarray(events, dtype=np.float64), np.asarray(keys, dtype=np.uint16).reshape(-1, 3),
                         np.asarray(frame_times, dtype=np.float64), np.asarray(rms, dtype=np.float32),
                         np.asarray(descriptors, dtype=np.float32).reshape(-1, 128))


def fingerprint(samples, start=0.0):
    from .performance import phase
    with phase('audio_descriptors'):
        return _fingerprint(samples, start)


def _fingerprint_decode(info, stream, config, time_scale=1.0, start=0., end=None, mixdown="mono"):
    from .progress import report
    report("audio_extract")
    # ashowinfo is after resampling/downmixing: its sample count and PTS describe
    # exactly the PCM bytes we consume, including codec priming and track delays.
    command = [config.ffmpeg, "-hide_banner", "-nostdin", "-copyts"]
    if start: command += ["-ss",str(max(0.,start-.5))]
    if end is not None: command += ["-t",str(end-max(0.,start-.5))]
    mixes={"mono":"aformat=sample_fmts=flt:channel_layouts=mono",
           "front":"pan=mono|c0=0.5*FL+0.5*FR", "center":"pan=mono|c0=FC",
           "left":"pan=mono|c0=FL", "right":"pan=mono|c0=FR"}
    if mixdown not in mixes:raise ValueError("Unknown audio downmix")
    command += ["-i", info.path,
               "-map", f"0:a:{stream['ordinal']}", "-vn", "-sn", "-dn", "-af",
               f"aresample={RATE},{mixes[mixdown]},"
               + (f"atrim=start={info.origin+start}:end={info.origin+end}," if end is not None else "")
               + (f"asetrate={round(RATE/time_scale)},aresample={RATE}," if time_scale!=1 else "")
               + "asetnsamples=n=4096:p=0,ashowinfo",
               "-f", "f32le", "pipe:1"]
    # PCM is spooled to an owned temporary file; long episodes do not accumulate
    # stdout/stderr copies or full spectrograms in RAM.
    with tempfile.TemporaryDirectory(prefix="danmaku-audio-") as directory:
        directory = Path(directory)
        pcm, log = directory / "decoded.f32", directory / "decode.log"
        with pcm.open("wb") as output, log.open("wb") as error:
            from .performance import count
            count("ffmpeg_starts"); count("audio_decode_starts")
            from .performance import phase
            with phase("audio_decode"):
                process = subprocess.run(command, stdout=output, stderr=error)
        if process.returncode:
            raise AudioUnavailable("audio_decode_failed")
        chunks = []
        pattern = re.compile(r"\bn:(\d+)\s+.*?pts_time:([^\s]+).*?nb_samples:(\d+)")
        with log.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    if int(match[1]) != len(chunks):
                        raise AudioUnavailable("audio_pts_order_mismatch")
                    chunks.append((float(match[2]) - info.origin*time_scale, int(match[3])))
        if not chunks or pcm.stat().st_size != sum(n for _, n in chunks) * 4:
            raise AudioUnavailable("audio_pts_sample_count_mismatch")
        if any(not math.isfinite(t) or n <= 0 for t, n in chunks) or any(
                b[0] < a[0] + a[1] / RATE - .003 for a, b in zip(chunks, chunks[1:])):
            raise AudioUnavailable("nonmonotonic_audio_timestamps")
        features = []
        # Split timestamp discontinuities. Never infer timestamps across packet gaps.
        begin, offset = 0, 0
        runs = []
        for i, (stamp, count) in enumerate(chunks):
            if i and abs(stamp - chunks[i - 1][0] - chunks[i - 1][1] / RATE) > .003:
                runs.append((chunks[begin][0], offset, sum(n for _, n in chunks[begin:i])))
                offset += runs[-1][2]
                begin = i
        runs.append((chunks[begin][0], offset, sum(n for _, n in chunks[begin:])))
        samples = np.memmap(pcm, dtype="<f4", mode="r")
        try:
            for stamp, index, count in runs:
                features.append(fingerprint(samples[index:index + count], stamp))
        finally:
            samples._mmap.close()
        return AudioFeatures(**{name: np.concatenate([getattr(f, name) for f in features])
                                for name in ("times", "keys", "frame_times", "rms", "spectra")})


def _extract_audio(info, stream, config, time_scale=1.0, start=0., end=None, mixdown="mono"):
    from .cache import identity, stamp, InputChangedError
    from .performance import count, phase
    before = stamp(info.path)
    path = None
    if config.cache_dir:
        signature = {"version": AUDIO_VERSION, "content": identity(info.path, config),
                     "stream": stream["index"], "rate": RATE, "fft": FFT_SIZE, "hop": HOP}
        if mixdown!="mono": signature.update(mixdown=mixdown, mixdown_version=1)
        if time_scale!=1: signature.update(normalization="asetrate-1",time_scale=time_scale)
        if start or end is not None: signature.update(start=start,end=end)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = Path(config.cache_dir) / f"{digest}.audio.npz"
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    features = AudioFeatures(**{key: arrays[key] for key in ("times", "keys", "frame_times", "rms", "spectra")})
                if (features.keys.shape == (len(features.times), 3) and features.keys.dtype == np.uint16
                        and features.times.ndim == 1 and features.frame_times.ndim == 1
                        and features.rms.shape == features.frame_times.shape
                        and features.spectra.shape == (len(features.frame_times), 128)
                        and features.spectra.dtype == np.float32
                        and features.times.dtype.kind == features.frame_times.dtype.kind == features.rms.dtype.kind == "f"
                        and all(np.all(np.isfinite(getattr(features, k))) for k in ("times", "frame_times", "rms", "spectra"))
                        and np.all(np.diff(features.times) >= 0) and np.all(np.diff(features.frame_times) > 0)):
                    count("audio_cache_hits")
                    return features
            except (ValueError, TypeError, KeyError, OSError, EOFError, zipfile.BadZipFile, zlib.error):
                pass
    count("audio_cache_misses")
    with phase("audio_decode_fingerprint_inclusive"):
        features = (_fingerprint_decode(info,stream,config,time_scale,start,end,mixdown) if time_scale!=1 or start or end is not None or mixdown!="mono"
                    else _fingerprint_decode(info, stream, config))
    if stamp(info.path) != before:
        raise InputChangedError("Input media changed during audio extraction")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".audio-", suffix=".npz", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **{k: getattr(features, k) for k in ("times", "keys", "frame_times", "rms", "spectra")})
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    return features


def extract_audio(info, stream, config, time_scale=1.0, start=0., end=None, mixdown="mono"):
    if not math.isfinite(time_scale) or not .5<=time_scale<=2:raise ValueError("Invalid audio normalization scale")
    if time_scale!=1:time_scale=RATE/round(RATE/time_scale)
    from .cache import identity, key, locked
    if not config.cache_dir:
        return _extract_audio(info, stream, config,time_scale,start,end,mixdown)
    lock = Path(config.cache_dir)/"audio-locks"/key([identity(info.path, config), stream["index"],time_scale,start,end,mixdown])
    with locked(lock):
        return _extract_audio(info, stream, config,time_scale,start,end,mixdown)


def _verify_audio_position(source, target, stamp, offset):
    # Verify the actual anchor neighborhood; matching landmarks from the sides
    # of a deleted interval must not justify a point inside that interval.
    similarities = []
    for delta in (-.15, 0, .15):
        value = stamp + delta
        s = int(np.searchsorted(source.frame_times, value))
        candidates = [k for k in (s - 1, s) if 0 <= k < len(source.frame_times)]
        if not candidates:
            continue
        s = min(candidates, key=lambda k: abs(source.frame_times[k] - value))
        if abs(source.frame_times[s] - value) > .06 or source.rms[s] < 1e-4:
            continue
        t = int(np.searchsorted(target.frame_times, source.frame_times[s] + offset))
        near = [k for k in range(t - 2, t + 2) if 0 <= k < len(target.frame_times)
                and abs(target.frame_times[k] - source.frame_times[s] - offset) <= .075
                and target.rms[k] >= 1e-4]
        if near:
            similarities.append(float(np.max(target.spectra[near] @ source.spectra[s])))
    return len(similarities) == 3 and min(similarities) >= .8 and np.median(similarities) >= .9


def audio_anchors(source, target, source_duration, target_duration, manual=(), stamp_range=None):
    from .progress import report
    report("audio_index")
    index = defaultdict(list)
    for ordinal,(time,key) in enumerate(zip(target.times,target.keys)):
        if ordinal%8192==0:report("audio_index",completed=ordinal,total=len(target.times),unit="fingerprints")
        index[tuple(int(v) for v in key)].append(float(time))
    votes = []
    report("audio_matching")
    for ordinal,(time,key) in enumerate(zip(source.times,source.keys)):
        if ordinal%8192==0:report("audio_matching",completed=ordinal,total=len(source.times),unit="fingerprints")
        f1, f2, dt = (int(v) for v in key)
        found = set()
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    bucket = index.get((f1 + a, f2 + b, dt + c), [])
                    if len(bucket) > 40:
                        continue  # Repeated tones/music cannot independently locate a scene.
                    for t in bucket:
                        if not 0 <= time <= source_duration or not 0 <= t <= target_duration:
                            continue
                        if all((time < m.source_time and t < m.target_time) or
                               (time > m.source_time and t > m.target_time) or
                               abs(time - m.source_time) < 1e-6 for m in manual):
                            found.add(t)
        if len(found) > 32:
            continue  # Bound work and discard globally over-common landmarks.
        for t in sorted(found):
            votes.append((float(time), t, f1 * 1024 + f2))
    if not votes:
        return [], {"fingerprint_votes": 0, "ambiguous_windows": 0}
    votes = np.asarray(votes, dtype=float)
    votes = votes[np.argsort(votes[:, 0], kind="stable")]
    anchors, ambiguous = [], 0
    stamp_start,stamp_end=(.25, source_duration-.1) if stamp_range is None else stamp_range
    stamp_start=max(.25,math.ceil((stamp_start-.25)/.5)*.5+.25)
    for stamp in np.arange(stamp_start, min(source_duration-.1,stamp_end), .5):
        left, right = np.searchsorted(votes[:, 0], [stamp - .75, stamp + .75])
        local = votes[left:right]
        if len(local) < 8 or not len(source.frame_times):
            continue
        at = int(np.searchsorted(source.frame_times, stamp))
        frame = min((k for k in (at-1, at) if 0 <= k < len(source.frame_times)),
                    key=lambda k: abs(source.frame_times[k]-stamp))
        if source.rms[frame] < 1e-4:
            continue
        offsets = local[:, 1] - local[:, 0]
        bins, counts = np.unique(np.round(offsets / .1).astype(int), return_counts=True)
        groups = []
        for pos in np.argsort(counts)[::-1][:8]:
            selected = np.abs(offsets - bins[pos] * .1) <= .08
            points = local[selected]
            if len(points) < 8:
                continue
            center = float(np.median(points[:, 1] - points[:, 0]))
            if any(abs(center - old[0]) <= .16 for old in groups):
                continue
            unique = len(np.unique(points[:, [0, 2]], axis=0))
            groups.append((center, unique, points))
        groups.sort(key=lambda g: g[1], reverse=True)
        if not groups:
            continue
        offset, support, points = groups[0]
        if support < 8 or len(np.unique(points[:, 2])) < 5 or np.ptp(points[:, 0]) < .45:
            continue
        if len(groups) > 1 and groups[1][1] >= support * .7:
            ambiguous += 1
            continue
        if not np.min(points[:, 0]) <= stamp <= np.max(points[:, 0]):
            continue
        residual = np.abs(points[:, 1] - points[:, 0] - offset)
        # A cut can move the two STFT grids by half a hop; correct fingerprint
        # votes then occupy adjacent offset bins rather than one exact value.
        if np.median(residual) > .055 or np.percentile(residual, 90) > .1:
            continue
        target_stamp = float(stamp + offset)
        if not 0 <= target_stamp <= target_duration:
            continue
        if not _verify_audio_position(source, target, float(stamp), offset):
            continue
        confidence = min(.95, .65 + support / 200)
        anchors.append(Anchor(float(stamp), target_stamp, confidence, evidence="audio"))
    return monotonic_audio_path(anchors), {"fingerprint_votes": len(votes), "ambiguous_windows": ambiguous,
                                          "local_audio_candidates": len(anchors)}



def monotonic_audio_path(points):
    """Weighted strict LIS: an early accidental late hit cannot block a movie."""
    if not points:return []
    targets=sorted({a.target_time for a in points});tree=[None]*(len(targets)+1)
    scores=[];parents=[]
    def better(a,b):
        if a is None:return b
        if b is None:return a
        return a if scores[a]>=scores[b] else b
    for i,a in enumerate(points):
        rank=int(np.searchsorted(targets,a.target_time));at=rank;parent=None
        while at:
            parent=better(parent,tree[at]);at-=at&-at
        scores.append((scores[parent] if parent is not None else 0.)+1.+a.confidence*1e-6)
        parents.append(parent);at=rank+1
        while at<len(tree):tree[at]=better(tree[at],i);at+=at&-at
    at=max(range(len(points)),key=lambda i:scores[i]);path=[]
    while at is not None:path.append(points[at]);at=parents[at]
    return list(reversed(path))

def supported_audio(points, source_duration, target_duration, config):
    """A few isolated sound coincidences must not constrain the whole movie."""
    from .alignment import build_segments
    span=min(2.5,max(1.,min(source_duration,target_duration)-1.))
    needed=max(3,math.ceil(span*2)+1)
    runs=[segment.anchors for segment in build_segments(points,config)]
    kept=[run for run in runs if len(run)>=needed and run[-1].source_time-run[0].source_time>=span-1e-6]
    # Sequence support can span a short evidence hole; strict mapping still does
    # not interpolate across it. Coverage counts only <= strict_gap neighbors.
    coverage=sum(b.source_time-a.source_time for run in kept for a,b in zip(run,run[1:])
                 if b.source_time-a.source_time<=config.strict_gap+1e-7)
    return [a for run in kept for a in run],coverage


def _original_audio_points(points,scale):
    return [Anchor(a.source_time/scale,a.target_time,a.confidence,evidence='audio') for a in points]


def _align_audio(source_info, target_info, config, manual=()):
    from .performance import phase
    from .progress import report
    s,t,selection=select_streams(source_info.path,target_info.path,config)
    source,target=extract_audio(source_info,s,config),extract_audio(target_info,t,config)
    report('audio_matching')
    with phase('audio_candidate_search'):
        raw,diagnostics=audio_anchors(source,target,source_info.duration,target_info.duration,manual)
    points,coverage=supported_audio(raw,source_info.duration,target_info.duration,config)
    original_count=len(raw);raw_coverage=coverage
    normalization={'status':'not_needed','scale':1.0}
    ratio=source_info.fps/target_info.fps if target_info.fps and source_info.fps else 1.
    pal=any(abs(ratio-rate)<.003 for rate in (25/24,24/25))
    if config.audio_speed_search and pal and min(source_info.duration,target_info.duration)>=6 and coverage/source_info.duration<.5:
        report('audio_speed_search')
        scale=RATE/round(RATE/ratio)
        from .media import merge_windows
        d=source_info.duration
        windows=merge_windows([(max(0.,min(d-90,60.)),min(d,150.)),
                               (max(0.,d/2-45),min(d,d/2+45)),
                               (max(0.,d-180),max(0.,d-90))],d,maximum=120)
        windows=[(a,b) for a,b in windows if b-a>=1]
        # Short media uses its full available duration; long media tests disjoint
        # windows before paying for an additional complete audio normalization.
        if d<180:windows=[(0.,d)]
        parts=[extract_audio(source_info,s,config,scale,a,b) for a,b in windows]
        fields=('times','keys','frame_times','rms','spectra')
        sampled=AudioFeatures(**{k:np.concatenate([getattr(f,k) for f in parts]) for k in fields})
        normalized_manual=[Anchor(a.source_time*scale,a.target_time,a.confidence,True,'manual') for a in manual]
        with phase('audio_speed_probe'):
            probe,probe_diag=audio_anchors(sampled,target,d*scale,target_info.duration,normalized_manual)
        probe=_original_audio_points(probe,scale)
        probe,probe_coverage=supported_audio(probe,d,target_info.duration,config)
        sample_duration=sum(b-a for a,b in windows)
        normalization={'status':'probe_rejected','scale':scale,'probe_anchors':len(probe),
                       'probe_supported_seconds':probe_coverage,'sample_seconds':sample_duration,
                       'method':'asetrate_resample','fps_ratio_is_only_a_hypothesis':True}
        if probe_coverage>=max(2.5,sample_duration*.1) and probe_coverage/sample_duration>coverage/d+.05:
            report('audio_speed_normalization')
            normalized=extract_audio(source_info,s,config,scale)
            with phase('audio_normalized_candidate_search'):
                corrected,corrected_diag=audio_anchors(normalized,target,d*scale,target_info.duration,normalized_manual)
            corrected=_original_audio_points(corrected,scale)
            corrected,corrected_coverage=supported_audio(corrected,d,target_info.duration,config)
            if corrected_coverage>coverage:
                points,coverage,diagnostics=corrected,corrected_coverage,corrected_diag
                normalization['status']='verified_sequence'
                normalization['supported_seconds']=coverage
    diagnostics.update({'status':'matched' if points else 'no_reliable_audio_match',
                        'source_stream':s,'target_stream':t,'selection':selection,
                        'source_fingerprints':len(source.times),'target_fingerprints':len(target.times),
                        'raw_audio_anchors':original_count,'audio_anchors':len(points),
                        'supported_audio_seconds':coverage,'unscaled_supported_seconds':raw_coverage,
                        'normalization':normalization,'version':AUDIO_VERSION})
    if not points:diagnostics['reason']='insufficient_audio_sequence_support'
    report('audio_coarse_result',reliable_audio_anchors=len(points),supported_seconds=coverage,
           normalization_status=normalization['status'])
    return points,diagnostics


def fuse_alignment(visual, audio, config):
    from .alignment import build_segments
    from .mapping import map_time
    automatic = [a for s in visual.segments for a in s.anchors if not a.manual]
    manual = [a for s in visual.segments for a in s.anchors if a.manual]
    time_index = ([s.source_start for s in visual.segments],
                  [[a.source_time for a in s.anchors] for s in visual.segments])
    overlaps = [(a, map_time(a.source_time, visual, _index=time_index)[0]) for a in audio]
    overlaps = [(a, t) for a, t in overlaps if t is not None]
    deltas = np.array([a.target_time - t for a, t in overlaps])
    bias = 0.0
    if len(deltas) >= 5 and np.ptp([a.source_time for a, _ in overlaps]) >= 3:
        median = float(np.median(deltas))
        if abs(median) <= 1 and np.percentile(np.abs(deltas - median), 90) <= .08:
            bias = median  # Consistent track A/V delay, calibrated with visible matches.
    accepted, conflicts = [], []
    all_visual = sorted(automatic + manual, key=lambda a: a.source_time)
    visual_times = [a.source_time for a in all_visual]
    for a in audio:
        adjusted = Anchor(a.source_time, a.target_time - bias, a.confidence, evidence="audio")
        if not 0 <= adjusted.target_time <= visual.target_duration:
            continue
        predicted, _, _ = map_time(a.source_time, visual, "lenient", time_index)
        if predicted is not None and abs(adjusted.target_time - predicted) > .2:
            conflicts.append(a.source_time)
            continue
        # Do not fill an already identified visual edit boundary from audio.
        in_edit = any(left.source_end < a.source_time < right.source_start and
                      abs((right.target_start - left.target_end) -
                          (right.source_start - left.source_end)) > .2
                      for left, right in zip(visual.segments, visual.segments[1:]))
        if in_edit:
            continue
        position = bisect.bisect_left(visual_times, a.source_time)
        near = [v for v in all_visual[max(0, position - 1):position + 2] if abs(v.source_time - a.source_time) < .08]
        if near:
            continue
        if ((position and adjusted.target_time <= all_visual[position - 1].target_time) or
                (position < len(all_visual) and adjusted.target_time >= all_visual[position].target_time)):
            conflicts.append(a.source_time)
            continue
        accepted.append(adjusted)
    # Conflicting intervals are held for review in both modes, rather than choosing
    # a modality silently. A single disagreement contaminates only its local window.
    forbidden = [{"source_start": max(0, t - .25), "source_end": min(visual.source_duration, t + .25),
                  "reason": "audio_visual_conflict"} for t in conflicts]
    combined = sorted(automatic + manual + accepted, key=lambda a: a.source_time)
    valid = []
    for a in combined:
        if not valid or (a.source_time > valid[-1].source_time and a.target_time > valid[-1].target_time):
            valid.append(a)
    visual.segments = build_segments(valid, config)
    visual.diagnostics.update({"audio_bias_seconds": bias, "accepted_audio_anchors": len(accepted),
                               "audio_visual_conflicts": len(conflicts), "unverified_intervals": forbidden})
    visual.validate()
    return visual


def align_audio(source_info,target_info,config,manual=()):
    """Cache verified acoustic anchors independently from visual scheduling."""
    if not config.coverage_recovery or not config.cache_dir:
        return _align_audio(source_info,target_info,config,manual)
    from .cache import identity,key,locked
    from .performance import count
    from .io import atomic_json
    a,b,selection=select_streams(source_info.path,target_info.path,config)
    signature=key({'version':AUDIO_VERSION,'anchor_verifier':1,
        'source':identity(source_info.path,config),'target':identity(target_info.path,config),
        'streams':[a['index'],b['index'],selection],'speed_search':config.audio_speed_search,
        'strict_gap':config.strict_gap,'max_segment_gap':config.max_segment_gap,
        'manual':[vars(m) for m in manual]})
    path=Path(config.cache_dir)/'audio-anchors-v1'/(signature+'.json')
    with locked(path):
        try:
            document=json.loads(path.read_text())
            if document['signature']!=signature:raise ValueError('Invalid audio signature')
            points=[Anchor(**a) for a in document['anchors']]
            if any(not (math.isfinite(a.source_time) and math.isfinite(a.target_time) and
                        math.isfinite(a.confidence) and 0<=a.source_time<=source_info.duration and
                        0<=a.target_time<=target_info.duration and 0<=a.confidence<=1 and
                        a.evidence=='audio' and not a.manual) for a in points):raise ValueError('Invalid acoustic point')
            if any(b.source_time<=a.source_time or b.target_time<=a.target_time for a,b in zip(points,points[1:])):
                raise ValueError('Invalid acoustic sequence')
            if not isinstance(document['diagnostics'],dict):raise ValueError('Invalid audio diagnostics')
            count('audio_anchor_cache_hits')
            return points,document['diagnostics']
        except (OSError,ValueError,KeyError,TypeError):pass
        points,diagnostics=_align_audio(source_info,target_info,config,manual)
        atomic_json(path,{'signature':signature,'anchors':[vars(a) for a in points],'diagnostics':diagnostics},True)
        count('audio_anchor_cache_misses')
        return points,diagnostics

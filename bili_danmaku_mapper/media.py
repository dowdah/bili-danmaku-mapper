"""Pair decoded pixels with their own presentation timestamps, including VFR."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import zipfile
import zlib

import imagehash
import numpy as np
from PIL import Image

from .config import AlignConfig


@dataclass
class VideoInfo:
    path: str
    duration: float
    origin: float
    fps: float
    pixel_format: str = ""


@dataclass
class Features:
    times: np.ndarray
    hashes: np.ndarray
    center_hashes: np.ndarray
    gray: np.ndarray
    quality: np.ndarray
    info: VideoInfo

    def __len__(self):
        return len(self.times)


def _probe_video(path, config=None):
    config = config or AlignConfig.resolve()
    path = str(Path(path).resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    process = subprocess.run([
        config.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "format=duration,start_time:stream=duration,start_time,avg_frame_rate,r_frame_rate,pix_fmt", "-of", "json", path,
    ], capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(process.stdout)
    if not data.get("streams"):
        raise ValueError(f"No video stream: {path}")
    stream, fmt = data["streams"][0], data.get("format", {})
    def number(value, fallback=0.0):
        try:
            result = float(value)
            return result if np.isfinite(result) else fallback
        except (ValueError, TypeError):
            return fallback
    origin = number(fmt.get("start_time"), number(stream.get("start_time")))
    try:
        num, den = stream.get("avg_frame_rate", "0/1").split("/")
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    stream_duration = number(stream.get("duration"))
    if stream_duration > 0:
        duration = number(stream.get("start_time"), origin) + stream_duration - origin
    else:
        # Matroska's format duration can be an absolute end timestamp when start_time
        # is nonzero. Derive the visible end from packet PTS instead of subtracting
        # blindly from every container's duration convention.
        packets = subprocess.run([config.ffprobe, "-v", "error", "-select_streams", "v:0",
                                  "-show_entries", "packet=pts_time,duration_time", "-of", "json", path],
                                 capture_output=True, text=True, check=True, timeout=60)
        ends = [number(p.get("pts_time"), float("nan")) + number(p.get("duration_time"), 1 / fps if fps > 0 else 0)
                for p in json.loads(packets.stdout).get("packets", [])]
        ends = [t for t in ends if np.isfinite(t)]
        duration = max(ends) - origin if ends else 0.0
    if duration <= 0:
        raise ValueError(f"Missing positive video duration: {path}")
    return VideoInfo(path, duration, origin, fps, stream.get("pix_fmt", ""))


def probe_video(path, config=None):
    from .cache import probe
    return probe(path, config or AlignConfig.resolve(), _probe_video)


def content_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_frame(pixels, crop_ratio):
    # Remove solid letterbox borders, but never crop an all-dark image.
    bright = pixels > 12
    rows = np.flatnonzero(bright.mean(axis=1) > 0.05)
    cols = np.flatnonzero(bright.mean(axis=0) > 0.05)
    if len(rows) > pixels.shape[0] * 0.35 and len(cols) > pixels.shape[1] * 0.35:
        pixels = pixels[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    image = Image.fromarray(pixels)
    height = pixels.shape[0]
    margin = int(height * (1 - crop_ratio) / 2)
    center = image.crop((0, margin, pixels.shape[1], height - margin))
    full_hash = np.packbits(imagehash.phash(image).hash.ravel())
    core_hash = np.packbits(imagehash.phash(center).hash.ravel())
    gray = np.asarray(center.resize((32, 24), Image.Resampling.BILINEAR), dtype=np.uint8)
    quality = float(gray.std())
    return full_hash, core_hash, gray, quality


def _decode(info, config, rate, start, end, hwaccel):
    from .progress import report
    from .performance import count, event, phase
    from .cache import stamp, InputChangedError
    before = stamp(info.path)
    started = time.perf_counter()
    count("ffmpeg_starts")
    seek = max(0.0, start - 0.5)
    command = [config.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-copyts"]
    if seek:
        command += ["-ss", str(seek)]
    if end is not None:
        command += ["-t", str(end - seek)]  # Input duration, not absolute output PTS.
    if hwaccel:
        command += ["-hwaccel", hwaccel]
        if hwaccel == "videotoolbox":
            command += ["-hwaccel_output_format", "videotoolbox_vld"]
    command += ["-i", info.path, "-map", "0:v:0", "-an", "-sn", "-dn"]
    selection = f"gte(t,{info.origin + start:.9f})"
    if end is not None:
        selection += f"*lt(t,{info.origin + end:.9f})"
    if rate:
        selection += f"*(isnan(prev_selected_t)+gte(t-prev_selected_t,{1 / rate - 1e-7:.9f}))"
    # Timestamp selection is metadata-only and can discard hardware surfaces
    # before copying pixels to the CPU. Download only frames actually requested.
    transfer = ""
    if hwaccel == "videotoolbox":
        download = "p010le" if "10" in info.pixel_format else "nv12"
        transfer = f"hwdownload,format={download},"
    filters = (f"select='{selection}',{transfer}scale=256:144:force_original_aspect_ratio=decrease,"
               "pad=256:144:(ow-iw)/2:(oh-ih)/2,format=gray,showinfo")
    command += ["-vf", filters, "-fps_mode", "passthrough", "-threads", "1",
                "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stamps, tail = {}, deque(maxlen=25)
    pattern = re.compile(r"\bn:\s*(\d+).*?\bpts_time:([^\s]+)")
    def read_metadata():
        for raw in iter(process.stderr.readline, b""):
            line = raw.decode("utf-8", "replace")
            match = pattern.search(line)
            if match:
                stamps[int(match[1])] = float(match[2]) - info.origin
            elif "showinfo" not in line:
                tail.append(line.rstrip())
    thread = threading.Thread(target=read_metadata, daemon=True)
    thread.start()
    hashes, centers, gray, quality = [], [], [], []
    try:
        frame_size = 256 * 144
        while True:
            frame = bytearray()
            while len(frame) < frame_size:
                with phase("visual_pipe_read_wait"):
                    block = process.stdout.read(frame_size - len(frame))
                if not block:
                    break
                frame.extend(block)
            if not frame:
                break
            if len(frame) != frame_size:
                raise ValueError("Truncated raw frame")
            pixels = np.frombuffer(frame, dtype=np.uint8).reshape(144, 256)
            with phase("visual_descriptors"):
                h, c, g, q = describe_frame(pixels, config.crop_ratio)
            hashes.append(h)
            centers.append(c)
            gray.append(g)
            quality.append(q)
            report(completed=len(hashes), unit="frames")
        code = process.wait(timeout=60)
        thread.join(timeout=10)
        if code:
            raise RuntimeError(f"FFmpeg decoding failed ({code}): " + "\n".join(tail))
        if thread.is_alive() or len(stamps) != len(hashes) or set(stamps) != set(range(len(hashes))):
            raise ValueError("Decoded frames and PTS metadata disagree")
        if not hashes:
            raise ValueError("No decoded frames in requested interval")
        times = np.array([stamps[i] for i in range(len(hashes))], dtype=np.float64)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0) or times[0] < -1e-5:
            raise ValueError("Invalid or non-increasing presentation timestamps")
        if stamp(info.path) != before:
            raise InputChangedError("Input media changed during decoding")
        count("decoded_selected_frames", len(hashes))
        count("decoded_requested_seconds", (end if end is not None else info.duration) - start)
        event(start=start, end=end if end is not None else info.duration, rate=rate,
              requested_backend=hwaccel or "software",
              actual_backend="videotoolbox" if hwaccel == "videotoolbox" else ("software" if not hwaccel else "unverified"),
              backend_verification="required_hardware_surface_download" if hwaccel == "videotoolbox" else "software_path" if not hwaccel else "unverified",
              elapsed_seconds=time.perf_counter()-started, frames=len(hashes))
        return Features(times, np.asarray(hashes), np.asarray(centers), np.asarray(gray),
                        np.asarray(quality, dtype=np.float32), info)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        thread.join(timeout=10)
        process.stderr.close()


FEATURE_FIELDS = ("times", "hashes", "center_hashes", "gray", "quality")


def subset(features, mask):
    return Features(**{k: getattr(features, k)[mask] for k in FEATURE_FIELDS}, info=features.info)


def concatenate(parts, info):
    if not parts:
        return Features(np.empty(0), np.empty((0, 8), dtype=np.uint8), np.empty((0, 8), dtype=np.uint8),
                        np.empty((0, 24, 32), dtype=np.uint8), np.empty(0, dtype=np.float32), info)
    combined = Features(**{k: np.concatenate([getattr(f, k) for f in parts]) for k in FEATURE_FIELDS}, info=info)
    order = np.argsort(combined.times, kind="stable")
    combined = subset(combined, order)
    return subset(combined, np.r_[True, np.diff(combined.times) > 1e-6])


def _load(path, info):
    try:
        with np.load(path, allow_pickle=False) as arrays:
            f = Features(**{k: arrays[k] for k in FEATURE_FIELDS}, info=info)
        n = len(f)
        if (n and f.times.shape == (n,) and f.times.dtype.kind == "f"
                and f.hashes.shape == f.center_hashes.shape == (n, 8)
                and f.gray.shape == (n, 24, 32) and f.quality.shape == (n,)
                and f.hashes.dtype == f.center_hashes.dtype == f.gray.dtype == np.uint8
                and f.quality.dtype.kind == "f" and np.all(np.isfinite(f.quality))
                and np.all(np.isfinite(f.times)) and np.all(np.diff(f.times) > 0)
                and 0 <= f.times[0] <= f.times[-1] <= info.duration + .1):
            return f
    except (ValueError, KeyError, OSError, EOFError, TypeError, zipfile.BadZipFile, zlib.error):
        pass
    return None


def _save(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".features-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **{k: getattr(features, k) for k in FEATURE_FIELDS})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _decode_retry(info, config, rate, start, end):
    from .performance import phase, count, event
    from .cache import InputChangedError
    with phase("visual_decode_inclusive"):
        try:
            result = _decode(info, config, rate, start, end, config.hwaccel)
            result.decoded_backend = config.hwaccel or "software"
            return result
        except (RuntimeError, ValueError) as error:
            if not config.hwaccel or isinstance(error, InputChangedError):
                raise
            count("hardware_fallbacks")
            event(requested_backend=config.hwaccel, actual_backend="failed", reason=str(error)[-2000:])
            result = _decode(info, config, rate, start, end, None)
            result.decoded_backend = "software"
            return result


def extract_features(path, config=None, *, rate=None, start=0.0, end=None, use_cache=True, _info=None):
    from .cache import identity, key, locked
    from .performance import count
    config = config or AlignConfig.resolve()
    info = _info or probe_video(path, config)
    rate = config.sample_rate if rate is None else rate
    if start < 0 or (end is not None and (end <= start or end > info.duration + .1)):
        raise ValueError("Invalid extraction interval")
    cache_path = None
    if use_cache and config.cache_dir:
        signature = {**config.feature_signature(rate), "content": identity(path, config)}
        if config.hwaccel:
            signature["backend"] = config.hwaccel
        if start != 0 or end is not None:
            signature.update(start=start, end=end)
            cache_path = Path(config.cache_dir) / "windows-v1" / (key(signature) + ".npz")
        else:
            cache_path = Path(config.cache_dir) / (key(signature) + ".npz")
    def obtain():
        cached = _load(cache_path, info) if cache_path else None
        if cached is not None:
            if cached.times[0] < start-1e-6 or cached.times[-1] >= (end if end is not None else info.duration)+.1:
                cached = None
            else:
                count("visual_cache_hits")
                if rate == 0 and end is not None:
                    from .progress import report
                    count("dense_interval_cache_hits"); report("dense_cache", cache_hit=True)
                cached.coverage = (start, end if end is not None else info.duration)
                return cached
        count("visual_cache_misses")
        features = _sample_cached_dense(info, config, start, end, rate) if end is not None and use_cache else None
        if features is None:
            features = _decode_retry(info, config, rate, start, end)
        if cache_path and getattr(features, "decoded_backend", config.hwaccel or "software") == (config.hwaccel or "software"):
            _save(cache_path, features)
        features.coverage = (start, end if end is not None else info.duration)
        return features
    if cache_path:
        with locked(cache_path):
            return obtain()
    return obtain()


def merge_windows(windows, duration, gap=2.0, maximum=120.0):
    merged = []
    for start, end in sorted((max(0., a), min(duration, b)) for a, b in windows if b > a):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return [(start, min(end, start + maximum)) for left, end in merged
            for start in np.arange(left, end, maximum)]


def dense_windows(info, windows, config):
    """Yield bounded feature blocks; 30-second tiles have a 0.75-second halo."""
    from .cache import identity, key, locked
    from .performance import count
    from .progress import report
    if not config.cache_dir or not config.dense_cache:
        for start, end in merge_windows(windows, info.duration, maximum=118.5):
            yield extract_features(info.path, config, rate=0, start=max(0.,start-.75),
                                   end=min(info.duration,end+.75), use_cache=False, _info=info)
        return
    tiles = sorted({int(t) for a, b in windows for t in range(int(max(0, a) // 30), int(max(a, b-1e-7) // 30) + 1)})
    # Sparse anchor neighborhoods must not force decoding an entire 30-second
    # tile. Cache their merged real intervals instead; dense coverage still uses
    # fixed tiles for parameter-independent reuse.
    merged = merge_windows(windows, info.duration, maximum=118.5)
    requested = sum(end-start for start,end in merged)
    if requested < .5 * sum(min(30.,info.duration-t*30) for t in tiles):
        count('dense_sparse_interval_requests')
        for start,end in merged:
            yield extract_features(info.path,config,rate=0,start=max(0.,start-.75),
                                   end=min(info.duration,end+.75),use_cache=True,_info=info)
        return
    digest = identity(info.path, config)
    def tile_path(tile):
        return Path(config.cache_dir) / "dense-v1" / (key({**config.feature_signature(0), "content": digest,
                "backend": config.hwaccel or "software", "tile": tile, "seconds": 30, "halo": .75}) + ".npz")
    # Lock each run, then publish only complete tiles. Locks release on process death.
    cursor = 0
    while cursor < len(tiles):
        tile = tiles[cursor]
        cached = _load(tile_path(tile), info)
        if cached is not None:
            count("dense_cache_hits"); report("dense_cache", cache_hit=True)
            cached.coverage = (max(0.,tile*30-.75), min(info.duration,(tile+1)*30+.75))
            yield cached; cursor += 1; continue
        run = [tile]
        while (cursor + len(run) < len(tiles) and len(run) < 3
               and tiles[cursor + len(run)] == run[-1] + 1
               and _load(tile_path(tiles[cursor + len(run)]), info) is None):
            run.append(tiles[cursor + len(run)])
        producer = Path(config.cache_dir)/"dense-v1"/(key([digest,config.feature_signature(0),config.hwaccel])+".producer")
        with locked(producer):
            # A competing producer may have completed this run while we waited.
            missing = [t for t in run if _load(tile_path(t), info) is None]
            if missing:
                start, end = max(0., missing[0]*30-.75), min(info.duration, (missing[-1]+1)*30+.75)
                report("refining", completed=cursor, total=len(tiles), unit="tiles")
                features = extract_features(info.path, config, rate=0, start=start, end=end,
                                            use_cache=False, _info=info)
                for t in missing:
                    part = subset(features, (features.times >= max(0, t*30-.75)) &
                                  (features.times < min(info.duration, (t+1)*30+.75)))
                    if len(part):
                        if getattr(features,"decoded_backend",config.hwaccel or "software") == (config.hwaccel or "software"):
                            _save(tile_path(t), part)
                        count("dense_cache_misses")
            for t in run:
                part = _load(tile_path(t), info)
                if part is not None:
                    part.coverage = (max(0.,t*30-.75), min(info.duration,(t+1)*30+.75))
                    yield part
                elif missing:
                    part = subset(features, (features.times >= max(0,t*30-.75)) &
                                  (features.times < min(info.duration,(t+1)*30+.75)))
                    part.coverage = (max(0.,t*30-.75), min(info.duration,(t+1)*30+.75))
                    yield part
        cursor += len(run)


def _sample_cached_dense(info, config, start, end, rate):
    """A cached dense tile also answers sparse requests without decoding again."""
    from .cache import identity, key
    from .performance import count
    if not config.cache_dir or not config.dense_cache or end-start>120:
        return None
    digest=identity(info.path,config)
    parts=[]
    for tile in range(int(start//30),int(max(start,end-1e-7)//30)+1):
        signature={**config.feature_signature(0),"content":digest,"backend":config.hwaccel or "software",
                   "tile":tile,"seconds":30,"halo":.75}
        part=_load(Path(config.cache_dir)/"dense-v1"/(key(signature)+".npz"),info)
        if part is None:return None
        parts.append(part)
    f=concatenate(parts,info)
    f=subset(f,(f.times>=start-1e-9)&(f.times<end-1e-9))
    if not len(f):return None
    if rate == 0:
        count('dense_slice_cache_hits')
        return f
    chosen=[0]
    for i in range(1,len(f)):
        if f.times[i]-f.times[chosen[-1]]>=1/rate-1e-7:chosen.append(i)
    count('dense_to_sparse_cache_hits')
    return subset(f,np.asarray(chosen,dtype=int))

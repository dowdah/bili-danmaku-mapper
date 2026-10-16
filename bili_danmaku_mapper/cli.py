import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

from . import AlignConfig, align_videos, map_danmaku
from .io import atomic_json, read_json, save_mapping_outputs


def _algorithm_options(parser):
    parser.add_argument("--sample-rate", type=float)
    parser.add_argument("--crop-ratio", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-hash-distance", type=int)
    parser.add_argument("--strict-gap", type=float)
    parser.add_argument("--max-segment-gap", type=float)
    parser.add_argument("--cache-dir")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--hwaccel")
    parser.add_argument("--source-hwaccel", choices=["software", "videotoolbox"])
    parser.add_argument("--target-hwaccel", choices=["software", "videotoolbox"])
    parser.add_argument("--algorithm", choices=["sequence", "legacy"])
    parser.add_argument("--evidence-mode", choices=["hybrid", "visual", "audio"])
    parser.add_argument("--no-audio-speed-search", dest="audio_speed_search", action="store_false", default=None, help="Disable verified PAL audio normalization")
    parser.add_argument("--hybrid-strategy", choices=["audio_first", "visual_first"])
    parser.add_argument("--recovery-stage", choices=["fusion", "visual", "audio"], help="Recovery ablation stage; default includes audio rescue")
    parser.add_argument("--coverage-recovery", action=argparse.BooleanOptionalAction, default=None, help="Experimental measured-evidence coverage recovery")
    parser.add_argument("--no-alignment-cache", action="store_true", help="Bypass final alignment cache; keep feature caches")
    parser.add_argument("--source-audio-stream", type=int, help="Source audio stream ordinal, starting at 0")
    parser.add_argument("--target-audio-stream", type=int, help="Target audio stream ordinal, starting at 0")
    parser.add_argument("--mode", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--no-refine", action="store_true", help="Coarse alignment only, for profiling")
    parser.add_argument("--overwrite", action="store_true")


def parser():
    root = argparse.ArgumentParser(prog="bili-danmaku-mapper")
    root.add_argument("--env-file", help="Defaults to the project's .env")
    commands = root.add_subparsers(dest="command", required=True)
    align = commands.add_parser("align", help="Align source danmaku to a target video")
    align.add_argument("--source", required=True)
    align.add_argument("--target", required=True)
    align.add_argument("--danmaku", help="Source nplayer JSON; omit to produce alignment only")
    align.add_argument("--output", required=True)
    align.add_argument("--anchors", help="JSON list of source_time/target_time pairs")
    _algorithm_options(align)
    batch = commands.add_parser("batch", help="Match actual episode numbers, not directory length")
    batch.add_argument("--source-dir", required=True)
    batch.add_argument("--target-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--source-pattern", default=r"(?P<no>\d+)$")
    batch.add_argument("--target-pattern", default=r"(?P<no>\d+)$")
    batch.add_argument("--danmaku-pattern", default="dms_raw_{no}.json")
    batch.add_argument("--offset", type=int, default=0)
    batch.add_argument("--pairs", help='JSON list of {"source": episode_number, "target": episode_number}')
    batch.add_argument("--workers", type=int, default=2)
    batch.add_argument("--start", type=int)
    batch.add_argument("--end", type=int, help="Exclusive source episode number")
    _algorithm_options(batch)
    download = commands.add_parser("download", help="Interactive Bilibili login and episode download")
    download.add_argument("--media-id", required=True, type=int)
    download.add_argument("--data-dir")
    download.add_argument("--ffmpeg")
    download.add_argument("--ffprobe")
    download.add_argument("--overwrite", action="store_true")
    commands.add_parser("doctor", help="Offline executable/dependency and decoding checks")
    return root


def _config(args):
    keys = ("sample_rate", "crop_ratio", "top_k", "max_hash_distance", "strict_gap", "max_segment_gap",
            "cache_dir", "ffmpeg", "ffprobe", "hwaccel", "source_hwaccel", "target_hwaccel", "algorithm", "evidence_mode",
            "source_audio_stream", "target_audio_stream", "hybrid_strategy", "coverage_recovery", "recovery_stage")
    return AlignConfig.resolve(args.env_file, **{k: getattr(args, k, None) for k in keys},
                               refine=not getattr(args, "no_refine", False),
                               alignment_cache=not getattr(args, "no_alignment_cache", False),
                               audio_speed_search=getattr(args,"audio_speed_search",None))


def discover_videos(directory, pattern):
    result, regex = {}, re.compile(pattern)
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".mp4", ".mkv", ".avi", ".flv", ".mov", ".webm"):
            continue
        match = regex.search(path.stem)
        if not match or "no" not in match.groupdict():
            raise ValueError(f"Cannot extract episode number: {path.name}")
        no = int(match["no"])
        if no in result:
            raise ValueError(f"Duplicate episode number {no}: {path.name}")
        result[no] = path
    return result


def run_pair(source, target, danmaku, output, config, mode, overwrite, anchors=None):
    output = Path(output)
    paths = [output] if danmaku is None else [output, output.with_suffix(".review.json"), output.with_suffix(".report.json")]
    protected = {Path(p).resolve() for p in (source, target, danmaku) if p is not None}
    if any(p.resolve() in protected for p in paths):
        raise ValueError("Output must not replace input video or original danmaku")
    if not overwrite and any(p.exists() for p in paths):
        raise FileExistsError(f"Output exists: {output}; use --overwrite")
    data = read_json(danmaku) if danmaku else None
    alignment = align_videos(source, target, config, anchors)
    if data is None:
        atomic_json(output, alignment.to_dict(), overwrite)
        return {"segments": len(alignment.segments), "diagnostics": alignment.diagnostics}
    result = map_danmaku(data, alignment, mode)
    save_mapping_outputs(output, result, alignment, overwrite)
    return {"segments": len(alignment.segments), "counts": result.report["counts"],
            "rates": result.report["rates"], "diagnostics": alignment.diagnostics}


def run_batch(args, config):
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    source = discover_videos(args.source_dir, args.source_pattern)
    target = discover_videos(args.target_dir, args.target_pattern)
    if args.pairs:
        pairs = read_json(args.pairs)
        if not isinstance(pairs, list) or any(not isinstance(p, dict) for p in pairs):
            raise ValueError("pairs must be a JSON list of source/target episode numbers")
        numbers = [(int(p["source"]), int(p["target"])) for p in pairs]
    else:
        numbers = [(no, no + args.offset) for no in sorted(source)]
    if len({b for _, b in numbers}) != len(numbers) or len({a for a, _ in numbers}) != len(numbers):
        raise ValueError("Episode pairing must be one-to-one")
    summary_path = Path(args.output_dir) / "batch.summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {summary_path}; use --overwrite")
    reports, jobs = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for no, target_no in numbers:
            if (args.start is not None and no < args.start) or (args.end is not None and no >= args.end):
                continue
            record = {"source_episode": no, "target_episode": target_no}
            danmaku = Path(args.source_dir) / args.danmaku_pattern.format(no=no)
            output = Path(args.output_dir) / f"dms_ep_{target_no}.json"
            if no not in source or target_no not in target or not danmaku.is_file():
                reports.append({**record, "status": "skipped", "reason": "missing_video_or_danmaku"})
                continue
            jobs.append((pool.submit(run_pair, source[no], target[target_no], danmaku, output,
                                     config, args.mode, args.overwrite), record))
        lookup = {future: record for future, record in jobs}
        for future in as_completed(lookup):
            record = lookup[future]
            try:
                reports.append({**record, "status": "success", **future.result()})
            except Exception as error:
                reports.append({**record, "status": "failed", "reason": str(error)})
    reports.sort(key=lambda r: r["source_episode"])
    summary = {"schema_version": 1, "episodes": reports,
               "counts": {status: sum(r["status"] == status for r in reports) for status in ("success", "failed", "skipped")}}
    atomic_json(summary_path, summary, args.overwrite)
    return summary


def doctor(config):
    checks = {}
    for name in ("numpy", "scipy", "PIL", "imagehash", "dotenv", "xmltodict", "tqdm", "bilibili_api"):
        try:
            module = importlib.import_module(name)
            checks[name] = {"ok": True, "version": str(getattr(module, "__version__", "available"))}
        except ImportError as error:
            checks[name] = {"ok": False, "error": str(error)}
    for name, path in (("ffmpeg", config.ffmpeg), ("ffprobe", config.ffprobe)):
        try:
            text = subprocess.check_output([path, "-version"], stderr=subprocess.STDOUT, timeout=10).decode()
            checks[name] = {"ok": True, "version": text.splitlines()[0]}
        except (OSError, subprocess.SubprocessError) as error:
            checks[name] = {"ok": False, "error": str(error)}
    try:
        subprocess.run([config.ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                        "testsrc2=size=32x32:rate=2:duration=1", "-fps_mode", "passthrough", "-f", "null", "-"],
                       check=True, capture_output=True, timeout=15)
        checks["software_decode"] = {"ok": True}
    except (OSError, subprocess.SubprocessError) as error:
        checks["software_decode"] = {"ok": False, "error": str(error)}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = _config(args)
        if args.command == "doctor":
            report = doctor(config)
            code = 0 if report["ok"] else 1
        elif args.command == "align":
            anchors = read_json(args.anchors) if args.anchors else None
            if args.anchors and Path(args.output).resolve() == Path(args.anchors).resolve():
                raise ValueError("Output must not replace manual anchors")
            report = run_pair(args.source, args.target, args.danmaku, args.output, config,
                              args.mode, args.overwrite, anchors)
            code = 0
        elif args.command == "batch":
            report = run_batch(args, config)
            code = 1 if report["counts"]["failed"] or report["counts"]["skipped"] else 0
        else:
            from bili_downloader import run_download
            run_download(args.media_id, args.data_dir or str(Path(config.cache_dir).parent),
                         config.ffmpeg, config.ffprobe, args.overwrite)
            report, code = {"status": "complete"}, 0
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return code
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

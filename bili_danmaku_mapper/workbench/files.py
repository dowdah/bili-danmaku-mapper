"""Native selections grant explicit file access; no arbitrary path API."""
import asyncio
import contextlib
import json
from pathlib import Path
import re
import subprocess
import sys

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v"}


def episode_number(name):
    if re.search(r"\b(?:SP|OVA|PV|OP|ED)\d*\b", name, re.I):
        return None
    for pattern in (r"S\d+E(\d+)", r"第\s*(\d+)\s*[集话話]", r"\[(\d{1,3})\]", r"(?:^|[ ._-])(\d{1,3})(?=$|[ ._-])"):
        found = [int(n) for n in re.findall(pattern, name, flags=re.I) if int(n) not in (264, 265)]
        if found:
            return found[0] if len(set(found)) == 1 else None
    return None


def inspect_file(path, ffprobe="ffprobe"):
    p = Path(path).resolve(strict=True)
    if not p.is_file() or p.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("请选择视频文件")
    process = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                              "format=duration,format_name,start_time:stream=index,codec_type,codec_name,width,height:stream_tags=language",
                              "-of", "json", str(p)], capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(process.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("文件没有视频流")
    audios = [s for s in data["streams"] if s.get("codec_type") == "audio"]
    # Conservative default; browser can reject additional unusual profiles at play time.
    compatible = p.suffix.lower() in (".mp4", ".m4v", ".mov") and video["codec_name"] == "h264" and all(
        s["codec_name"] in ("aac", "mp3") for s in audios)
    return {"origin": float(data.get("format", {}).get("start_time", 0)), "duration": float(data.get("format", {}).get("duration", 0)), "number": episode_number(p.stem),
            "browser_compatible": compatible, "video_codec": video["codec_name"],
            "audio_streams": [{"ordinal": i, "codec": s["codec_name"], "language": s.get("tags", {}).get("language", "und")}
                              for i, s in enumerate(audios)], "bytes": p.stat().st_size}


_pickers = set()

async def stop_pickers():
    for process in list(_pickers):
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


async def native_pick(kind):
    if sys.platform != "darwin":
        raise ValueError("原生文件选择首版仅支持 macOS")
    if kind not in ("files", "directory", "playback_video", "danmaku", "subtitle"):
        raise ValueError("Unknown picker kind")
    script = Path(__file__).with_name("native_picker.js")
    process = await asyncio.create_subprocess_exec("osascript", "-l", "JavaScript", str(script), kind,
                       stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _pickers.add(process)
    try:
        output, error = await process.communicate()
        if process.returncode:
            raise ValueError("系统文件选择器已关闭，请重新选择")
        values = json.loads(output or "[]")
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        _pickers.discard(process)
    if kind == "directory" and values:
        return sorted(p for p in Path(values[0]).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    return [Path(p) for p in values]


def suggest_pairs(sources, files, offset=0):
    assignments = []
    for source in sources:
        number = source.get("number")
        candidates = [f for f in files if number is not None and f.get("number") == number + offset]
        if len(sources) == len(files) == 1:
            candidates = files
        assignments.append({"source_key": source["key"], "file_id": candidates[0]["id"] if len(candidates) == 1 else None,
                            "reason": "matched" if len(candidates) == 1 else "ambiguous" if candidates else "missing"})
    # Never silently pair one local file to multiple source episodes.
    used = [a["file_id"] for a in assignments if a["file_id"]]
    for a in assignments:
        if a["file_id"] and used.count(a["file_id"]) > 1:
            a.update(file_id=None, reason="ambiguous")
    return assignments

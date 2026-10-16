"""Isolated playback preparation. Alignment inputs and mapped comments are never edited."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from ..io import read_json
from .storage import check_space


def identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def probe(path, ffprobe):
    result = subprocess.run([ffprobe, '-v', 'error', '-show_entries',
        'format=start_time:stream=codec_type,codec_name,start_time', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout)


def origin(info):
    value = float(info.get('format', {}).get('start_time', 0))
    if not math.isfinite(value):
        raise ValueError('时间基准无效')
    return value


def timestamps(path, ffprobe, frames=False):
    # Packet scans do not decode video for stream-copy paths. Full transcodes need
    # decoded frame PTS because source and destination packetization may differ.
    entry = 'frame=best_effort_timestamp_time' if frames else 'packet=pts_time'
    result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', entry, '-of', 'csv=p=0', str(path)], check=True,
        capture_output=True, text=True, timeout=7200)
    values = []
    for line in result.stdout.splitlines():
        first = line.split(',', 1)[0].strip()
        if not first:
            continue
        try:
            value = float(first)
        except ValueError:
            raise ValueError('视频时间戳缺失，无法验证播放文件') from None
        if not math.isfinite(value):
            raise ValueError('视频时间戳缺失')
        values.append(value)
    if not values:
        raise ValueError('视频没有有效时间戳')
    return np.sort(np.asarray(values, dtype=np.float64))


def verify_timeline(source, output, ffprobe, strategy, audio_stream=0):
    a, b = probe(source, ffprobe), probe(output, ffprobe)
    def compare(frames):
        av = timestamps(source, ffprobe, frames) - origin(a)
        bv = timestamps(output, ffprobe, frames) - origin(b)
        offset = float(bv[0]-av[0])
        error = float(np.max(np.abs((av+offset)-bv))) if len(av)==len(bv) else float('inf')
        return offset,error,len(av)
    # Some AVI streams have DTS but no packet PTS. Decode their frame timestamps
    # before rejecting a lossless remux; the video itself still stays stream-copy.
    if strategy == 'transcode':
        offset,error,count = compare(True)
    else:
        try:
            offset,error,count = compare(False)
        except ValueError:
            offset,error,count = compare(True)
        if error > .002:
            offset,error,count = compare(True)
    if error > .002:
        raise ValueError('播放文件时间轴与原片不一致，已拒绝使用；原片与弹幕保留')
    aa = [s for s in a['streams'] if s['codec_type'] == 'audio']
    ba = [s for s in b['streams'] if s['codec_type'] == 'audio']
    if aa:
        if len(ba) != 1:
            raise ValueError('播放音轨不完整')
        if 'start_time' in aa[audio_stream] and 'start_time' in ba[0]:
            difference = (float(aa[audio_stream]['start_time'])-origin(a)) - (float(ba[0]['start_time'])-origin(b)-offset)
            if abs(difference) > .05:
                raise ValueError('播放音轨起点变化过大，已拒绝使用')
    return {'timeline_verified':True, 'timeline_offset':offset, 'max_video_error':error, 'timestamps_checked':count}


def build_variant(request):
    source = request['source']
    if identity(source) != request['source_identity']:
        raise ValueError('原片已经变化，请重新选择影片')
    directory = Path(request['directory'])
    check_space(directory, request['source_identity'][2])
    directory.mkdir(parents=True, exist_ok=True)
    partial, final = directory/'playback.partial.mp4', directory/'playback.mp4'
    strategy, audio = request['strategy'], request['audio_stream']
    ffmpeg, ffprobe = request['ffmpeg'], request['ffprobe']
    source_info = probe(source, ffprobe)
    streams = [s for s in source_info['streams'] if s['codec_type'] == 'audio']
    if audio >= max(1, len(streams)):
        raise ValueError('播放音轨序号超出范围')
    if strategy == 'audio' and not streams:
        raise ValueError('影片没有音频，不需要转换音频')
    if strategy not in ('remux','audio','transcode'):
        raise ValueError('不支持的播放准备方式')
    command = [ffmpeg, '-v', 'error', '-nostdin', '-y', '-copyts', '-start_at_zero', '-i', source, '-map', '0:v:0']
    if streams:
        command += ['-map', f'0:a:{audio}']
    command += ['-sn', '-dn', '-map_chapters', '-1']
    if strategy == 'transcode':
        command += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-fps_mode:v', 'passthrough', '-enc_time_base:v', '1:1000000']
    else:
        command += ['-c:v', 'copy', '-copytb', '1']
    if strategy != 'transcode' and next(s for s in source_info['streams'] if s['codec_type'] == 'video')['codec_name'] == 'hevc':
        command += ['-tag:v', 'hvc1']
    command += ['-c:a', 'copy'] if strategy == 'remux' else ['-c:a', 'aac', '-ar', '48000']
    command += ['-avoid_negative_ts', 'disabled', '-use_editlist', '1', '-video_track_timescale', '1000000',
                '-movflags', '+faststart', str(partial)]
    try:
        # FFmpeg may write an output larger than a CRF estimate. Guard the
        # destination while it writes, and keep stderr in the private task folder.
        with (directory/'ffmpeg.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log)
            try:
                while process.poll() is None:
                    check_space(directory, admission=False)
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                check_space(directory, admission=False)
                if process.returncode:
                    raise RuntimeError('FFmpeg failed')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        verified = verify_timeline(source, partial, ffprobe, strategy, audio)
        if identity(source) != request['source_identity']:
            raise ValueError('准备播放期间原片发生变化，已拒绝使用结果')
        os.replace(partial, final)
        return {**verified, 'path':str(final), 'output_identity':identity(final)}
    finally:
        partial.unlink(missing_ok=True)


def main(path):
    request = read_json(path)
    def watch_owner():
        while True:
            if os.getppid() != request['owner_pid']:
                os.killpg(os.getpid(), signal.SIGTERM)
                return
            time.sleep(.2)
    threading.Thread(target=watch_owner, daemon=True).start()
    try:
        print(json.dumps({'result':build_variant(request)}), flush=True)
        return 0
    except Exception as error:
        # Never expose FFmpeg commands, private paths or raw stderr to the UI.
        message = str(error) if isinstance(error, ValueError) else '播放准备失败，原片与弹幕保留'
        print(json.dumps({'error':message}), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))

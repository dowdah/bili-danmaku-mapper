"""Deterministic moving textures with a known timeline, not real show footage."""
import subprocess
import numpy as np
from PIL import Image


def picture(t):
    step = t * 2
    index, alpha = int(np.floor(step)), step % 1
    def grid(seed):
        return np.random.default_rng(seed + 700).integers(25, 230, (9, 16), dtype=np.uint8)
    pixels = ((1 - alpha) * grid(index) + alpha * grid(index + 1)).astype(np.uint8)
    return np.asarray(Image.fromarray(pixels).resize((256, 144), Image.Resampling.BILINEAR))


def make_video(path, duration=8, fps=24, timeline=lambda t: t, filters=None, pts_offset=0, vfr=False):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pixel_format", "gray", "-video_size", "256x144",
               "-framerate", str(fps), "-i", "pipe:0", "-an"]
    if filters:
        command += ["-vf", filters]
    if vfr:
        command += ["-fps_mode", "vfr"]
    command += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    if pts_offset:
        command += ["-output_ts_offset", str(pts_offset)]
    command.append(str(path))
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(round(duration * fps)):
            process.stdin.write(picture(timeline(i / fps)).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode()
        if process.wait():
            raise RuntimeError(stderr)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stderr.close()
    return path

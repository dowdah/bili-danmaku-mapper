import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_a = os.path.join(BASE_DIR, "age.mp4")
path_b = os.path.join(BASE_DIR, "bilibili.mp4")
frames_a = os.path.join(BASE_DIR, "frames_a")
frames_b = os.path.join(BASE_DIR, "frames_b")


# Step 1: 使用ffmpeg导出裁剪+缩放后的小帧
def export_frames_ffmpeg(video_path, output_dir, crop_ratio=0.6, width=256, height=256):
    """
    使用ffmpeg导出视频所有帧，裁剪并缩放到指定大小
    """
    os.makedirs(output_dir, exist_ok=True)
    crop_start = (1 - crop_ratio) / 2
    crop_expr = f"in_w:in_h*{crop_ratio}:0:in_h*{crop_start}"
    out_pattern = os.path.join(output_dir, "%05d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "videotoolbox",
        "-i", video_path,
        "-vf", f"crop={crop_expr},scale={width}:{height}",
        "-qscale:v", "3",
        out_pattern
    ]
    print(f"运行命令：{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"帧已导出到 {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"导出失败: {video_path}\n错误: {e}")


if __name__ == "__main__":
    video_paths = [path_a, path_b]
    frame_dirs = [frames_a, frames_b]
    fps = 1
    crop_ratio = 0.6
    width, height = 256, 256

    # 1. 导出帧(如果不存在)
    if not all([os.path.exists(frame_dir) for frame_dir in frame_dirs]):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(export_frames_ffmpeg, video_path, frame_dir, crop_ratio, width, height)
                       for video_path, frame_dir in zip(video_paths, frame_dirs)]
            for future in futures:
                future.result()

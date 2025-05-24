import subprocess
import os
import csv
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_a = os.path.join(BASE_DIR, "age.mp4")
path_b = os.path.join(BASE_DIR, "bilibili.mp4")
frames_a = os.path.join(BASE_DIR, "frames_a")
frames_b = os.path.join(BASE_DIR, "frames_b")


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


def load_frame_paths(frame_dir):
    """按帧文件名排序，返回所有帧路径列表"""
    files = os.listdir(frame_dir)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    return [os.path.join(frame_dir, f) for f in files]


class PhashCache:
    """缓存帧的phash，仅在首次访问时计算并缓存"""
    def __init__(self, frame_paths):
        self.frame_paths = frame_paths
        self.cache = {}
        import PIL.Image
        import imagehash
        self.PIL = PIL
        self.imagehash = imagehash

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        path = self.frame_paths[idx]
        with self.PIL.Image.open(path) as img:
            phash = self.imagehash.phash(img)
        self.cache[idx] = phash
        return phash


def adaptive_anchor_propagation(phashes_a, phashes_b, anchor_interval, max_search_range, max_hamming_distance):
    """自适应锚点推进匹配，返回b2a_map和锚点b帧索引列表"""
    b2a_map = {}
    anchor_b_indices = list(range(0, len(phashes_b), anchor_interval))
    print("建立锚点匹配并进行自适应锚点推进...")
    # 先建立初始锚点匹配
    for b_idx in tqdm(anchor_b_indices):
        best_dist = 256
        best_a_idx = None
        start_a = max(0, b_idx - max_search_range)
        end_a = min(len(phashes_a), b_idx + max_search_range + 1)
        for a_idx in range(start_a, end_a):
            dist = phashes_b[b_idx] - phashes_a[a_idx]
            if dist < best_dist:
                best_dist = dist
                best_a_idx = a_idx
        if best_dist <= max_hamming_distance:
            b2a_map[b_idx] = best_a_idx
        else:
            b2a_map[b_idx] = None

    # 自适应推进
    b_len = len(phashes_b)
    i = 0
    while i < b_len:
        if i in b2a_map and b2a_map[i] is not None:
            start_b = i
            start_a = b2a_map[i]
            # 推进直到失配
            j = i + 1
            while j < b_len:
                # 线性估计a索引
                est_a_idx = start_a + (j - start_b)
                search_start = max(0, est_a_idx - 3)
                search_end = min(len(phashes_a), est_a_idx + 4)
                best_dist = 256
                best_a_idx = None
                for a_idx in range(search_start, search_end):
                    dist = phashes_b[j] - phashes_a[a_idx]
                    if dist < best_dist:
                        best_dist = dist
                        best_a_idx = a_idx
                if best_dist <= max_hamming_distance:
                    b2a_map[j] = best_a_idx
                    j += 1
                else:
                    # 失配，重新在窗口内查找最佳锚点
                    found_anchor = False
                    window_end = min(j + anchor_interval, b_len)
                    for k in range(j, window_end):
                        best_dist = 256
                        best_a_idx = None
                        start_a_search = max(0, k - max_search_range)
                        end_a_search = min(len(phashes_a), k + max_search_range + 1)
                        for a_idx in range(start_a_search, end_a_search):
                            dist = phashes_b[k] - phashes_a[a_idx]
                            if dist < best_dist:
                                best_dist = dist
                                best_a_idx = a_idx
                        if best_dist <= max_hamming_distance:
                            b2a_map[k] = best_a_idx
                            i = k
                            found_anchor = True
                            break
                        else:
                            b2a_map[k] = None
                    if not found_anchor:
                        i = window_end
                    break
            else:
                # 推进到末尾
                i = j
        else:
            # 当前帧无锚点，尝试全局搜索匹配
            best_dist = 256
            best_a_idx = None
            for a_idx in range(len(phashes_a)):
                dist = phashes_b[i] - phashes_a[a_idx]
                if dist < best_dist:
                    best_dist = dist
                    best_a_idx = a_idx
                if best_dist == 0:
                    break
            if best_dist <= max_hamming_distance:
                b2a_map[i] = best_a_idx
            else:
                b2a_map[i] = None
            i += 1

    return b2a_map, anchor_b_indices


def save_mapping_csv(b2a_map, num_b_frames, out_path):
    print(f"保存映射结果到 {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_b_index", "frame_a_index"])
        for b_idx in range(num_b_frames):
            a_idx = b2a_map.get(b_idx)
            if a_idx is None:
                writer.writerow([b_idx, "None"])
            else:
                writer.writerow([b_idx, a_idx])


def save_mapping_json(b2a_map, out_path):
    print(f"保存映射结果到 {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(b2a_map, f, ensure_ascii=False, indent=4)


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

    # 2. 帧级匹配主流程
    frames_a_paths = load_frame_paths(frames_a)
    frames_b_paths = load_frame_paths(frames_b)

    phashes_a = PhashCache(frames_a_paths)
    phashes_b = PhashCache(frames_b_paths)

    anchor_interval = 50  # 锚点间隔帧数
    max_search_range = int(len(phashes_a) * 0.2)  # 搜索范围，避免匹配过远帧
    max_hamming_distance = 10  # 最大允许的汉明距离，判断是否匹配成功

    b2a_map, anchor_b_indices = adaptive_anchor_propagation(phashes_a, phashes_b, anchor_interval, max_search_range, max_hamming_distance)

    csv_path = os.path.join(BASE_DIR, "b2a_mapping.csv")
    json_path = os.path.join(BASE_DIR, "b2a_mapping.json")
    save_mapping_csv(b2a_map, len(phashes_b), csv_path)
    save_mapping_json(b2a_map, json_path)

    print("匹配完成。")

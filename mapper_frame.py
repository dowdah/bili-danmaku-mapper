"""
A tool for mapping frames between two videos based on perceptual hash (phash) values.
Copyright (C) 2025 dowdah

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import csv
import json
import PIL.Image
import imagehash
import subprocess
import bisect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm


def export_frames_ffmpeg(video_path, output_dir, crop_ratio=0.6, width=256, height=256):
    """
    使用ffmpeg导出视频帧，裁剪并缩放到指定大小
    :param video_path: str，视频文件路径
    :param output_dir: str，输出目录
    :param crop_ratio: float，比例裁剪，0.6表示裁剪为60%的高度
    :param width: int，缩放后的宽度
    :param height: int，缩放后的高度
    :return: None
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
    """
    加载指定目录下的所有图像帧路径，按文件名排序
    :param frame_dir: str，图像帧目录
    :return: list，图像帧文件路径列表
    """
    files = os.listdir(frame_dir)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    return [os.path.join(frame_dir, f) for f in files]


def adaptive_anchor_propagation(phashes_a, phashes_b, anchor_interval=50, search_range_factor=0.2,
                                max_hamming_distance=10, fill=True):
    """
    使用自适应锚点推进算法在两组感知哈希值之间建立映射关系。

    该算法首先在B视频帧中每隔anchor_interval帧选取锚点，并在A视频帧的指定搜索范围内寻找最佳匹配（汉明距离最小）。
    随后从每个锚点出发，尝试线性推进匹配，若推进失败则在局部窗口内重新寻找锚点，直到所有B帧都被处理。
    可选地对未匹配的帧填充None。

    :param phashes_a: list-like，A视频所有帧的感知哈希值（支持下标访问和减法运算，返回汉明距离）
    :param phashes_b: list-like，B视频所有帧的感知哈希值（同上）
    :param anchor_interval: int，锚点间隔帧数
    :param search_range_factor: float，搜索范围系数，表示在A视频帧数的多少倍范围内搜索匹配
    :param max_hamming_distance: int，最大允许的汉明距离，超过则视为不匹配
    :param fill: bool，是否对未匹配帧填充None（默认True）
    :return: tuple (b2a_map, anchor_b_indices)
        b2a_map: dict，B视频帧到A视频帧的映射，key为B帧索引，value为A帧索引或None
        anchor_b_indices: list，所有锚点B帧的索引列表
    """
    b2a_map = {}
    anchor_b_indices = list(range(0, len(phashes_b), anchor_interval))
    max_search_range = int(len(phashes_a) * search_range_factor)  # 根据A视频帧数计算搜索范围
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
    if fill:
        # 填充None值
        for b_idx in range(len(phashes_b)):
            if b_idx not in b2a_map:
                b2a_map[b_idx] = None
    return b2a_map, anchor_b_indices


def save_mapping_csv(out_path, b2a_map):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_b_index", "frame_a_index"])
        for b_idx in range(len(b2a_map)):
            a_idx = b2a_map.get(b_idx)
            if a_idx is None:
                writer.writerow([b_idx, "None"])
            else:
                writer.writerow([b_idx, a_idx])


def get_video_fps(video_path):
    """
    使用ffprobe获取视频的帧率（frames per second, FPS）。

    该函数调用ffprobe命令行工具，提取视频的r_frame_rate字段（如"24000/1001"），
    并将其转换为浮点数表示的帧率。

    :param video_path: str，视频文件路径
    :return: float，视频帧率
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    # r_frame_rate返回如"24000/1001"
    if "/" in out:
        num, denom = out.split("/")
        return float(num) / float(denom)
    else:
        return float(out)


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_nearest_key(keys, target):
    """
    在有序列表keys中，使用二分查找找到最接近目标值target的key。

    :param keys: list[int]，有序的整数key列表
    :param target: int，目标值
    :return: int，最接近target的key
    """
    idx = bisect.bisect_left(keys, target)
    if idx == 0:
        return keys[0]
    if idx == len(keys):
        return keys[-1]
    before = keys[idx - 1]
    after = keys[idx]
    if abs(before - target) <= abs(after - target):
        return before
    else:
        return after


def map_danmaku_times(danmaku_json, b2a_map, fps_b, fps_a):
    """
    将弹幕时间从B视频映射到A视频时间轴。

    遍历弹幕JSON中的每条弹幕，根据B视频时间计算对应的B帧号，查找最近有映射的B帧，
    然后通过b2a_map找到对应的A帧号，并根据A视频帧率换算为A视频时间，替换弹幕的'time'字段。

    :param danmaku_json: dict，包含弹幕数据的JSON对象，需有'data'键，值为弹幕列表
    :param b2a_map: dict，B帧号到A帧号的映射（key为B帧号str/int，value为A帧号或None）
    :param fps_b: float，B视频帧率
    :param fps_a: float，A视频帧率
    :return: dict，时间已映射到A视频的弹幕JSON对象
    """
    b_keys = sorted(int(k) for k in b2a_map.keys() if b2a_map[k] is not None)
    for item in danmaku_json['data']:
        time_b = item['time']       # B视频时间（秒）
        b_frame = int(round(time_b * fps_b))
        # 找最近有映射的b帧
        nearest_b = find_nearest_key(b_keys, b_frame)
        a_frame = b2a_map[nearest_b]
        # A视频帧号换算为时间
        time_a = int(a_frame) / fps_a
        item['time'] = time_a
    return danmaku_json


class PhashCache:
    """
    图像帧的感知哈希值缓存类。

    该类用于缓存一组图像帧的感知哈希（phash）值，避免重复计算。
    使用PIL加载图像并通过imagehash库计算phash，支持下标访问和len操作。

    属性:
        frame_paths (list): 图像帧文件路径列表。
        cache (dict): 已计算的phash缓存，key为帧索引，value为phash对象。
    """
    def __init__(self, frame_paths):
        """
        初始化PhashCache。

        :param frame_paths: 图像帧文件路径列表
        """
        self.frame_paths = frame_paths
        self.cache = {}

    def __len__(self):
        """
        返回帧的数量。

        :return: 帧数量
        """
        return len(self.frame_paths)

    def __getitem__(self, idx):
        """
        获取指定索引帧的phash值，若已缓存则直接返回，否则计算后缓存并返回。

        :param idx: 帧索引
        :return: phash对象
        """
        if idx in self.cache:
            return self.cache[idx]
        path = self.frame_paths[idx]
        with PIL.Image.open(path) as img:
            phash = imagehash.phash(img)
        self.cache[idx] = phash
        return phash


# class Options:
#     """
#     配置选项类，用于存储和管理各种参数。
#
#     该类用于集中管理脚本运行所需的各种配置选项，便于修改和维护。
#     """
#     def __init__(self, save_frames=True, save_b2a_json=False,
#                  mapped_dms_path=None, frame_dirs=None, crop_ratio=0.6, width=256, height=256,
#                  anchor_interval=50, search_factor=0.2, max_hamming_distance=10):
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         self.save_frames = save_frames
#         self.save_b2a_json = save_b2a_json
#         self.mapped_dms_path = mapped_dms_path or os.path.join(base_dir, "dms_mapped.json")
#         self.frame_dirs = frame_dirs or [os.path.join(base_dir, "frames_a"), os.path.join(base_dir, "frames_b")]
#         self.crop_ratio = crop_ratio
#         self.width = width
#         self.height = height
#         self.anchor_interval = anchor_interval
#         self.search_factor = search_factor
#         self.max_hamming_distance = max_hamming_distance


if __name__ == "__main__":
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    SAVE_FRAMES = True  # 是否保存帧图像
    SAVE_B2A_JSON = True  # 是否保存B到A的映射JSON
    VIDEO_PATHS = [os.path.join(BASE_DIR, "ep13_a.mp4"), os.path.join(BASE_DIR, "ep13_b.mp4")]
    RAW_DMS_PATH = os.path.join(BASE_DIR, "dms.json")  # 原始弹幕文件路径
    MAPPED_DMS_PATH = os.path.join(BASE_DIR, "dms_mapped.json")
    FRAME_DIRS = [os.path.join(BASE_DIR, "frames_a"), os.path.join(BASE_DIR, "frames_b")]
    CROP_RATIO = 0.6 # 视频的纵向裁剪比例，0.6表示仅保留中间60%。可以根据视频字幕、水印等情况调整。
    WIDTH, HEIGHT = 256, 256 # 输出帧的宽度和高度
    ANCHOR_INTERVAL = 50  # 锚点间隔帧数
    SEARCH_FACTOR = 0.2  # 搜索范围系数(占被搜索视频帧数的比例)。若视频间最大帧数差异较大，可适当增大。
    MAX_HAMMING_DISTANCE = 10  # 最大允许的汉明距离，判断是否匹配成功。该参数越大，匹配越宽松，但可能导致错误匹配。
    EXPORT_B2A_CSV_PATH = os.path.join(BASE_DIR, "b2a_mapping.csv")
    EXPORT_B2A_JSON_PATH = os.path.join(BASE_DIR, "b2a_mapping.json")

    # 1. 导出帧(如果不存在)
    if not all([os.path.exists(frame_dir) for frame_dir in FRAME_DIRS]):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(export_frames_ffmpeg, video_path, frame_dir, CROP_RATIO, WIDTH, HEIGHT)
                       for video_path, frame_dir in zip(VIDEO_PATHS, FRAME_DIRS)]
            for future in futures:
                future.result()

    # 2. 帧级匹配主流程
    frames_a_paths = load_frame_paths(FRAME_DIRS[0])
    frames_b_paths = load_frame_paths(FRAME_DIRS[1])

    phashes_a = PhashCache(frames_a_paths)
    phashes_b = PhashCache(frames_b_paths)

    b2a_map, anchor_b_indices = adaptive_anchor_propagation(phashes_a, phashes_b, ANCHOR_INTERVAL, SEARCH_FACTOR,
                                                            MAX_HAMMING_DISTANCE)

    print("匹配完成。")

    print("自动获取帧率...")
    fps_a = get_video_fps(VIDEO_PATHS[0])
    fps_b = get_video_fps(VIDEO_PATHS[1])
    print(f"A视频帧率: {fps_a:.3f}  B视频帧率: {fps_b:.3f}")

    danmaku = load_json(RAW_DMS_PATH)
    danmaku_mapped = map_danmaku_times(danmaku, b2a_map, fps_b, fps_a)
    danmaku_mapped['data'] = sorted(danmaku_mapped['data'], key=lambda x: x['time'])
    save_json(MAPPED_DMS_PATH, danmaku_mapped)
    print(f"弹幕时间已映射，结果保存到 {MAPPED_DMS_PATH}")

    if not SAVE_FRAMES:
        # 删除帧目录
        for frame_dir in FRAME_DIRS:
            if os.path.exists(frame_dir):
                for file in os.listdir(frame_dir):
                    os.remove(os.path.join(frame_dir, file))
                os.rmdir(frame_dir)
        print("已删除帧图像目录。")

    if SAVE_B2A_JSON:
        save_json(EXPORT_B2A_JSON_PATH, b2a_map)
        print(f"B到A的映射已保存到 {EXPORT_B2A_JSON_PATH}")

    print("所有操作完成。")

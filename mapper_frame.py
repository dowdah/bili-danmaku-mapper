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
from dataclasses import dataclass


BASE_DIR = os.environ.get("BASE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
# 视频的纵向裁剪比例，0.6表示仅保留中间60%。可以根据视频字幕、水印等情况调整。
CROP_RATIO = int(os.environ.get("CROP_RATIO", 0.6))
# 视频帧的宽度和高度，默认为256x256。可以根据需要调整。
WIDTH, HEIGHT = int(os.environ.get("WIDTH", 256)), int(os.environ.get("HEIGHT", 256))
# 锚点间隔帧数
ANCHOR_INTERVAL = int(os.environ.get("ANCHOR_INTERVAL", 50))
# 搜索范围系数(占被搜索视频帧数的比例)。若视频间最大帧数差异较大，可适当增大。
RANGE_FACTOR = float(os.environ.get("RANGE_FACTOR", 0.2))
# 最大允许的汉明距离，判断是否匹配成功。该参数越大，匹配越宽松，但可能导致错误匹配。
MAX_HAMMING_DISTANCE = int(os.environ.get("MAX_HAMMING_DISTANCE", 10))


@dataclass
class FrameExportConfig:
    """
    视频帧导出配置类。

    包含裁剪比例、缩放宽度和高度等参数。
    """
    crop_ratio: float = CROP_RATIO
    width: int = WIDTH
    height: int = HEIGHT


@dataclass
class AAPConfig:
    """
    自适应锚点推进算法配置类。

    包含锚点间隔、搜索范围系数和最大汉明距离等参数。
    """
    anchor_interval: int = ANCHOR_INTERVAL
    range_factor: float = RANGE_FACTOR
    max_hamming_distance: int = MAX_HAMMING_DISTANCE
    filled: bool = True  # 是否对未匹配的B帧填充None


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


def export_frames_ffmpeg(video_path, output_dir, config: FrameExportConfig=None):
    """
    使用ffmpeg导出视频帧，裁剪并缩放到指定大小
    :param video_path: str，视频文件路径
    :param output_dir: str，输出目录
    :param config: FrameExportConfig，导出配置，包括裁剪比例、宽度和高度
    :return: None
    """
    config = config or FrameExportConfig()
    crop_ratio = config.crop_ratio
    width = config.width
    height = config.height
    os.makedirs(output_dir, exist_ok=True)
    crop_start = (1 - crop_ratio) / 2
    crop_expr = f"in_w:in_h*{crop_ratio}:0:in_h*{crop_start}"
    out_pattern = os.path.join(output_dir, "%05d.jpg")
    cmd = [
        FFMPEG_PATH,
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


def loading_video_paths(video_dir):
    """
    加载指定目录下的视频(.mp4)文件路径，按文件名排序
    :param video_dir: str，视频文件目录
    :return: list，视频文件路径列表
    """
    files = os.listdir(video_dir)
    files = [f for f in files if f.lower().endswith('.mp4')]
    files.sort()
    return [os.path.join(video_dir, f) for f in files]


def adaptive_anchor_propagation(phashes_a, phashes_b, config: AAPConfig=None):
    """
    使用自适应锚点推进算法在两组感知哈希值之间建立映射关系。

    该算法首先在B视频帧中每隔anchor_interval帧选取锚点，并在A视频帧的指定搜索范围内寻找最佳匹配（汉明距离最小）。
    随后从每个锚点出发，尝试线性推进匹配，若推进失败则在局部窗口内重新寻找锚点，直到所有B帧都被处理。
    可选地对未匹配的帧填充None。

    :param phashes_a: list-like，A视频所有帧的感知哈希值（支持下标访问和减法运算，返回汉明距离）
    :param phashes_b: list-like，B视频所有帧的感知哈希值（同上）
    :param config: AAPConfig，自适应锚点推进配置，包括锚点间隔、搜索范围系数、最大汉明距离和是否填充None等参数。
    :return: tuple (b2a_map, anchor_b_indices)
        b2a_map: dict，B视频帧到A视频帧的映射，key为B帧索引，value为A帧索引或None
        anchor_b_indices: list，所有锚点B帧的索引列表
    """
    config = config or AAPConfig()
    anchor_interval = config.anchor_interval
    range_factor = config.range_factor
    max_hamming_distance = config.max_hamming_distance
    b2a_map = {}
    anchor_b_indices = list(range(0, len(phashes_b), anchor_interval))
    max_search_range = int(len(phashes_a) * range_factor)  # 根据A视频帧数计算搜索范围
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
    if config.filled:
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
        FFPROBE_PATH, "-v", "error",
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


def map_danmaku_times(danmaku_json, b2a_map, video_path_a, video_path_b):
    """
    将弹幕时间从B视频映射到A视频时间轴。

    遍历弹幕JSON中的每条弹幕，根据B视频时间计算对应的B帧号，查找最近有映射的B帧，
    然后通过b2a_map找到对应的A帧号，并根据A视频帧率换算为A视频时间，替换弹幕的'time'字段。

    :param danmaku_json: dict，包含弹幕数据的JSON对象，需有'data'键，值为弹幕列表
    :param b2a_map: dict，B帧号到A帧号的映射（key为B帧号str/int，value为A帧号或None）
    :param video_path_a: str，A视频文件路径
    :param video_path_b: str，B视频文件路径
    :param fps_a: float，A视频帧率
    :return: dict，时间已映射到A视频的弹幕JSON对象
    """
    fps_a = get_video_fps(video_path_a)
    fps_b = get_video_fps(video_path_b)
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
    danmaku_json['data'] = sorted(danmaku_json['data'], key=lambda x: x['time'])
    return danmaku_json


def mp4s_2_frames(mp4s_dir, max_workers=4, delete_mp4=True, config: FrameExportConfig=None):
    config = config or FrameExportConfig()
    mp4_paths = loading_video_paths(mp4s_dir)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for mp4_path in mp4_paths:
            mp4_filename = os.path.splitext(os.path.basename(mp4_path))[0]
            frame_dir = os.path.join(mp4s_dir, mp4_filename)
            os.makedirs(frame_dir, exist_ok=True)
            futures.append(pool.submit(export_frames_ffmpeg, mp4_path, frame_dir, config))
        for future in tqdm(futures, desc="导出帧"):
            future.result()
    if delete_mp4:
        for mp4_path in mp4_paths:
            os.remove(mp4_path)
        else:
            print(f"已删除原始视频文件：{mp4s_dir}/*.mp4")


def calc_b2a_map_with_frames(frames_a_dir, frames_b_dir, config: AAPConfig=None, export_csv_path=None,
                             export_json_path=None):
    """
    计算B视频帧到A视频帧的映射关系。

    :param frames_a_dir: str，A视频帧目录
    :param frames_b_dir: str，B视频帧目录
    :param anchor_interval: int，锚点间隔帧数
    :param range_factor: float，搜索范围系数
    :param max_hamming_distance: int，最大允许的汉明距离
    :param export_csv_path: str，可选，保存映射关系到CSV文件路径
    :param export_json_path: str，可选，保存映射关系到JSON文件路径
    :return: dict，B到A的映射关系（key为B帧索引，value为A帧索引或None）
    """
    config = config or AAPConfig()
    frames_a_paths = load_frame_paths(frames_a_dir)
    frames_b_paths = load_frame_paths(frames_b_dir)
    phashes_a = PhashCache(frames_a_paths)
    phashes_b = PhashCache(frames_b_paths)

    b2a_map, anchor_b_indices = adaptive_anchor_propagation(phashes_a, phashes_b, config)

    if export_csv_path is not None:
        save_mapping_csv(export_csv_path, b2a_map)
        print(f"B到A的映射已保存到 {export_csv_path}")

    if export_json_path is not None:
        save_json(export_json_path, b2a_map)
        print(f"B到A的映射已保存到 {export_json_path}")

    return b2a_map


def map_danmaku_by_video_paths(video_path_a, video_path_b, raw_dms_path, export_dms_path,
                               f_config: FrameExportConfig=None, aap_config: AAPConfig=None):
    temp_dir = os.path.join(BASE_DIR, "tmp")
    frames_a_dir = os.path.join(temp_dir, "frames_a")
    frames_b_dir = os.path.join(temp_dir, "frames_b")
    os.makedirs(frames_a_dir, exist_ok=True)
    os.makedirs(frames_b_dir, exist_ok=True)
    f_config = f_config or FrameExportConfig()
    aap_config = aap_config or AAPConfig()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(export_frames_ffmpeg, video_path_a, frames_a_dir, f_config),
            pool.submit(export_frames_ffmpeg, video_path_b, frames_b_dir, f_config)
        ]
        for future in tqdm(futures, desc="导出视频帧"):
            future.result()
    b2a_map = calc_b2a_map_with_frames(frames_a_dir, frames_b_dir, aap_config)
    danmaku = load_json(raw_dms_path)
    danmaku_mapped = map_danmaku_times(danmaku, b2a_map, video_path_a, video_path_b)
    save_json(export_dms_path, danmaku_mapped)


def map_danmaku_by_frame_dirs(frames_a_dir, frames_b_dir, raw_dms_path, export_dms_path, config: AAPConfig=None):
    config = config or AAPConfig()
    b2a_map = calc_b2a_map_with_frames(frames_a_dir, frames_b_dir, config)
    danmaku = load_json(raw_dms_path)
    danmaku_mapped = map_danmaku_times(danmaku, b2a_map, frames_a_dir, frames_b_dir)
    save_json(export_dms_path, danmaku_mapped)


if __name__ == "__main__":
    mp4s_2_frames("/Users/sheldon/Documents/github_projects/bili-danmaku-mapper/data/bangumi/28235123",
                  delete_mp4=False)

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
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass


BASE_DIR = os.environ.get("BASE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
# 视频的纵向裁剪比例，0.6表示仅保留中间60%。可以根据视频字幕、水印等情况调整。
CROP_RATIO = float(os.environ.get("CROP_RATIO", 0.6))
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


class PhashCachedList:
    """
    图像帧的感知哈希值缓存类。

    该类用于缓存一组图像帧的感知哈希（phash）值，避免重复计算。
    使用PIL加载图像并通过imagehash库计算phash，支持下标访问和len操作。

    属性:
        frame_paths (list): 图像帧文件路径列表。
        cache (dict): 已计算的phash缓存，key为帧索引，value为phash对象。
    """
    def __init__(self, frame_paths: list[str]):
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

    def __getitem__(self, idx: int):
        """
        获取指定索引帧的phash值，若已缓存则直接返回，否则计算后缓存并返回。

        :param idx: int，帧索引
        :return: phash对象
        """
        if idx in self.cache:
            return self.cache[idx]
        path = self.frame_paths[idx]
        with PIL.Image.open(path) as img:
            phash = imagehash.phash(img)
        self.cache[idx] = phash
        return phash


class VideoFile:
    def __init__(self, path: str, template_str: str):
        self.path = path
        self.basename = os.path.basename(path)
        self.stem, self.ext = os.path.splitext(self.basename)
        self.no = None  # 默认无序号

        match = re.search(template_str, self.stem)
        if match and "no" in match.groupdict():
            self.no = int(match.group("no"))
        else:
            raise ValueError("Unable to extract 'no' from filename using template: " + template_str)

    def __repr__(self):
        return f"<VideoFile no='{self.no}' basename='{self.basename}'>"


def export_frames_ffmpeg(video_path: str, output_dir: str, config: FrameExportConfig | None = None) -> None:
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


def load_frame_paths(frame_dir: str) -> list[str]:
    """
    加载指定目录下的所有图像帧路径，按文件名排序
    :param frame_dir: str，图像帧目录
    :return: list，图像帧文件路径列表
    """
    files = os.listdir(frame_dir)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    return [os.path.join(frame_dir, f) for f in files]


def loading_videos_from_dir(video_dir: str, template_str: str = r"(?P<no>\d+)$") -> dict:
    """
    加载指定目录下的视频文件路径，按文件名排序
    :param video_dir: str，视频文件目录
    :param template_str: str，正则表达式模板，用于提取视频文件名中的序号部分
    :return: dict，视频文件路径字典
    """
    files = os.listdir(video_dir)
    files = [f for f in files if f.lower().endswith(('.mp4', '.mkv', '.avi', '.flv'))]
    files.sort()
    videos_dict = {}
    for f in files:
        video_file = VideoFile(os.path.join(video_dir, f), template_str)
        if video_file.no in videos_dict:
            raise ValueError(f"从文件夹加载视频时出现重复的序号，模板{template_str}可能不正确。")
        videos_dict[video_file.no] = video_file
    return videos_dict


def adaptive_anchor_propagation(phash_a: PhashCachedList, phash_b: PhashCachedList,
                                config: AAPConfig | None = None) -> tuple[dict, list]:
    """
    使用自适应锚点推进算法在两组感知哈希值之间建立映射关系。

    该算法首先在B视频帧中每隔anchor_interval帧选取锚点，并在A视频帧的指定搜索范围内寻找最佳匹配（汉明距离最小）。
    随后从每个锚点出发，尝试线性推进匹配，若推进失败则在局部窗口内重新寻找锚点，直到所有B帧都被处理。
    可选地对未匹配的帧填充None。

    :param phash_a: PhashCachedList，A视频帧的感知哈希值列表
    :param phash_b: PhashCachedList，B视频帧的感知哈希值列表
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
    anchor_b_indices = list(range(0, len(phash_b), anchor_interval))
    max_search_range = int(len(phash_a) * range_factor)  # 根据A视频帧数计算搜索范围
    # 先建立初始锚点匹配
    for b_idx in anchor_b_indices:
        best_dist = 256
        best_a_idx = None
        start_a = max(0, b_idx - max_search_range)
        end_a = min(len(phash_a), b_idx + max_search_range + 1)
        for a_idx in range(start_a, end_a):
            dist = phash_b[b_idx] - phash_a[a_idx]
            if dist < best_dist:
                best_dist = dist
                best_a_idx = a_idx
        if best_dist <= max_hamming_distance:
            b2a_map[b_idx] = best_a_idx
        else:
            b2a_map[b_idx] = None

    # 自适应推进
    b_len = len(phash_b)
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
                search_end = min(len(phash_a), est_a_idx + 4)
                best_dist = 256
                best_a_idx = None
                for a_idx in range(search_start, search_end):
                    dist = phash_b[j] - phash_a[a_idx]
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
                        end_a_search = min(len(phash_a), k + max_search_range + 1)
                        for a_idx in range(start_a_search, end_a_search):
                            dist = phash_b[k] - phash_a[a_idx]
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
            for a_idx in range(len(phash_a)):
                dist = phash_b[i] - phash_a[a_idx]
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
        for b_idx in range(len(phash_b)):
            if b_idx not in b2a_map:
                b2a_map[b_idx] = None
    return b2a_map, anchor_b_indices


def save_mapping_csv(out_path: str, b2a_map: dict) -> None:
    """
    将B到A的映射关系保存为CSV文件。
    :param out_path: str，输出CSV文件路径
    :param b2a_map: dict，B帧到A帧的映射关系（key为B帧索引，value为A帧索引或None）
    :return: None
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_b_index", "frame_a_index"])
        for b_idx in range(len(b2a_map)):
            a_idx = b2a_map.get(b_idx)
            if a_idx is None:
                writer.writerow([b_idx, "None"])
            else:
                writer.writerow([b_idx, a_idx])


def get_video_fps(video_path: str) -> float:
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


def load_json(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filename: str, data: dict | list) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_nearest_key(keys: list[int], target: int) -> int:
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


def map_danmaku_json(danmaku_json: dict, b2a_map: dict, video_path_a: str, video_path_b: str) -> dict:
    """
    将弹幕时间从B视频映射到A视频时间轴。

    遍历弹幕JSON中的每条弹幕，根据B视频时间计算对应的B帧号，查找最近有映射的B帧，
    然后通过b2a_map找到对应的A帧号，并根据A视频帧率换算为A视频时间，替换弹幕的'time'字段。

    :param danmaku_json: dict，包含弹幕数据的JSON对象，需有'data'键，值为弹幕列表
    :param b2a_map: dict，B帧号到A帧号的映射（key为B帧号str/int，value为A帧号或None）
    :param video_path_a: str，A视频文件路径
    :param video_path_b: str，B视频文件路径
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


def videos_2_frames(videos_dir: str, max_workers: int = 4, delete_video: bool = False,
                    config: FrameExportConfig | None = None, template_str: str = r"(?P<no>\d+)$") -> None:
    """
    将指定目录下的视频文件导出为帧图像。帧图像将保存在视频文件所在目录下的子目录中，子目录名为视频序号。
    :param videos_dir: str，视频文件目录
    :param max_workers: int，最大工作线程数
    :param delete_video: bool，是否在导出后删除视频文件
    :param config: FrameExportConfig，视频帧导出配置
    :param template_str: str，正则表达式模板，用于提取视频文件名中的序号部分
    :return: None
    """
    config = config or FrameExportConfig()
    videos_dict = loading_videos_from_dir(videos_dir, template_str)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for video_no, video_file in videos_dict.items():
            frame_dir = os.path.join(videos_dir, str(video_no))
            os.makedirs(frame_dir, exist_ok=True)
            futures.append(pool.submit(export_frames_ffmpeg, video_file.path, frame_dir, config))
        for future in tqdm(futures, desc="导出帧"):
            future.result()
    if delete_video:
        for video_file in videos_dict.values():
            os.remove(video_file.path)
        else:
            print(f"已删除该文件夹下的视频文件:\n{videos_dir}")


def calc_b2a_map_with_frames(frames_a_dir: str, frames_b_dir: str, config: AAPConfig | None = None,
                             export_csv_path: str | None = None, export_json_path: str | None = None) -> dict:
    """
    计算B视频帧到A视频帧的映射关系。

    :param frames_a_dir: str，A视频帧目录
    :param frames_b_dir: str，B视频帧目录
    :param config: AAPConfig，自适应锚点推进配置，包括锚点间隔、搜索范围系数、最大汉明距离和是否填充None等参数
    :param export_csv_path: str，可选，保存映射关系到CSV文件路径
    :param export_json_path: str，可选，保存映射关系到JSON文件路径
    :return: dict，B到A的映射关系（key为B帧索引，value为A帧索引或None）
    """
    config = config or AAPConfig()
    frames_a_paths = load_frame_paths(frames_a_dir)
    frames_b_paths = load_frame_paths(frames_b_dir)
    phash_a = PhashCachedList(frames_a_paths)
    phash_b = PhashCachedList(frames_b_paths)

    b2a_map, anchor_b_indices = adaptive_anchor_propagation(phash_a, phash_b, config)

    if export_csv_path is not None:
        save_mapping_csv(export_csv_path, b2a_map)
        print(f"B到A的映射已保存到 {export_csv_path}")

    if export_json_path is not None:
        save_json(export_json_path, b2a_map)
        print(f"B到A的映射已保存到 {export_json_path}")

    return b2a_map


def map_danmaku_main(video_path_a: str, video_path_b: str, raw_dms_path: str, export_dms_path: str,
                     frames_a_dir: str | None = None, frames_b_dir: str | None = None,
                     f_config: FrameExportConfig | None = None, aap_config: AAPConfig | None = None) -> None:
    """
    根据两个视频文件路径，导出视频帧并计算B视频帧到A视频帧的映射关系，然后将弹幕时间从B视频映射到A视频。
    如果提供了帧目录，则直接使用这些目录中的帧。
    :param video_path_a: str，A视频文件路径
    :param video_path_b: str，B视频文件路径
    :param frames_a_dir: str，可选，A视频帧目录
    :param frames_b_dir: str，可选，B视频帧目录
    :param raw_dms_path: str，原始弹幕JSON文件路径（B视频的弹幕）
    :param export_dms_path: str，导出后的弹幕JSON文件路径（A视频的弹幕）
    :param f_config: FrameExportConfig，视频帧导出配置，包括裁剪比例、宽度和高度等参数
    :param aap_config: AAPConfig，自适应锚点推进配置，包括锚点间隔、搜索范围系数、最大汉明距离和是否填充None等参数
    :return: None
    """
    aap_config = aap_config or AAPConfig()
    temp_dir = os.path.join(BASE_DIR, "tmp")
    f_config = f_config or FrameExportConfig()
    # 如果没有提供帧目录，则创建临时目录导出帧
    def prepare_and_export(video_path, frames_dir):
        if os.path.exists(frames_dir):
            for filename in os.listdir(frames_dir):
                file_path = os.path.join(frames_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(frames_dir)
        export_frames_ffmpeg(video_path, frames_dir, f_config)
    if frames_a_dir is None and frames_b_dir is None:
        frames_a_dir = os.path.join(temp_dir, "frames_a")
        frames_b_dir = os.path.join(temp_dir, "frames_b")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(prepare_and_export, video_path_a, frames_a_dir),
                pool.submit(prepare_and_export, video_path_b, frames_b_dir)
            ]
            for future in tqdm(futures, desc="导出视频帧"):
                future.result()
    elif frames_a_dir is None:
        frames_a_dir = os.path.join(temp_dir, "frames_a")
        prepare_and_export(video_path_a, frames_a_dir)
    elif frames_b_dir is None:
        frames_b_dir = os.path.join(temp_dir, "frames_b")
        prepare_and_export(video_path_b, frames_b_dir)
    b2a_map = calc_b2a_map_with_frames(frames_a_dir, frames_b_dir, aap_config)
    danmaku = load_json(raw_dms_path)
    danmaku_mapped = map_danmaku_json(danmaku, b2a_map, video_path_a, video_path_b)
    save_json(export_dms_path, danmaku_mapped)


def map_bili_danmaku_with_frames(media_id: int, videos_a_name: str, template_str: str = r"(?P<no>\d+)$",
                                 index_start: int = 0, index_end: int | None = None, offset: int = 0,
                                 max_workers: int = 4, config: AAPConfig | None = None) -> None:
    """
    针对B站番剧的弹幕映射，使用自适应锚点推进算法。

    :param media_id: int，B站番剧ID
    :param videos_a_name: str，A视频目录名
    :param index_start: int，起始索引，默认为0
    :param index_end: int，结束索引，默认为None（表示到最后一个目录）
    :param offset: int，A视频帧文件夹索引名偏移量，offset = <index_a> - <index_b>，默认为0
    :param max_workers: int，最大线程数，默认为4
    :param config: AAPConfig，自适应锚点推进配置
    :return: None
    """
    config = config or AAPConfig()
    frames_b_parent_dir = os.path.join(BASE_DIR, "bangumi", str(media_id))
    videos_dict_b = loading_videos_from_dir(frames_b_parent_dir)
    frames_a_parent_dir = os.path.join(BASE_DIR, "bangumi", videos_a_name)
    videos_dict_a = loading_videos_from_dir(frames_a_parent_dir, template_str=template_str)
    videos_count_b = len(videos_dict_b)
    index_end = videos_count_b if index_end is None else min(index_end, videos_count_b)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i in range(index_start, index_end):
            video_file_b = videos_dict_b.get(i)
            video_file_a = videos_dict_a.get(i + offset)
            if video_file_a is None or video_file_b is None:
                print(f"跳过索引 {i}，缺少视频文件：A({i + offset})或B({i})")
                continue
            frames_b_dir = os.path.join(frames_b_parent_dir, str(video_file_b.no))
            frames_a_dir = os.path.join(frames_a_parent_dir, str(video_file_a.no))
            if not os.path.exists(frames_b_dir) or not os.path.exists(frames_a_dir):
                print(f"跳过索引 {i}，缺少帧目录：{frames_b_dir} 或 {frames_a_dir}")
                continue
            raw_dms_path = os.path.join(frames_b_parent_dir, f"dms_raw_{video_file_b.no}.json")
            export_dms_path = os.path.join(frames_a_parent_dir, f"dms_ep_{video_file_a.no}.json")
            futures.append(pool.submit(
                map_danmaku_main, video_file_a.path, video_file_b.path,
                raw_dms_path, export_dms_path, frames_a_dir, frames_b_dir, aap_config=config))
        for future in tqdm(futures, desc="映射弹幕"):
            future.result()


if __name__ == "__main__":
    map_bili_danmaku_with_frames(28235123,
                                 "[UHA-WINGS&JOJO][JoJo's Bizarre Adventure - Stone Ocean][25-38][x264 1080p][sc_jp]",
                                 offset=1, template_str=r"^\[.*?\]\[.*?\]\[(?P<no>\d+)\]")

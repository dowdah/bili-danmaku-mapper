# bili-danmaku-mapper

![Status: Developing](https://img.shields.io/badge/Status-Developing-blue?style=for-the-badge)
![Version: null](https://img.shields.io/badge/Version-null-brightgreen?style=for-the-badge)
[![English_Doc](https://img.shields.io/badge/English_Doc-brightgreen?style=for-the-badge)](README.md)

**bili-danmaku-mapper** 是一款基于 Python 的弹幕映射工具，支持将**B站番剧/动画等视频的弹幕（danmaku）精准迁移到另一版本的视频**（如：将有删减的版本弹幕同步到未删减版本）。本项目专为 Bilibili 番剧设计，对**多集动画和高删改场景有特别优化**。

目前已支持的弹幕映射算法：

- **基于图像帧的感知哈希（phash）比对（AAP）**

  基于帧感知哈希（phash）比对不同视频版本的图像帧，自动构建帧到帧映射关系，再批量映射弹幕时间轴。

- **未来将支持更多算法（如音频指纹、特征点匹配等）**

---

## 功能特性

- 基于感知哈希（phash）自动对齐不同版本视频帧
- 支持未来扩展多种对齐算法（架构预留）
- 支持批量多集番剧弹幕迁移和对齐
- 支持将弹幕从有删减版本批量转移到无删减版本
- 支持 macOS / Linux / Windows
- 附带便捷的本地 HTML 播放器（支持 nplayer + 弹幕 JSON 导入）

---

## 安装与环境

- **系统要求**：macOS / Linux / Windows
- **Python 版本**：3.11+
- **依赖管理**：pipenv

**安装依赖：**

```bash
pip install pipenv
pipenv install
```

或者直接用 requirements.txt：

```bash
pip install -r requirements.txt
```

---

## 文件结构

```
.
├── bili_downloader.py      # B站番剧视频及弹幕批量下载与转换
├── mapper_frame.py         # 基于帧感知哈希的弹幕迁移核心逻辑
├── player.html             # 本地 HTML 视频弹幕播放器（支持 JSON 弹幕）
├── Pipfile / Pipfile.lock  # pipenv 环境配置
├── requirements.txt        # 依赖列表
├── README-zh.md / README.md
├── LICENSE
└── ...
```

---

## 快速上手

### 1. 下载视频与弹幕

使用 `bili_downloader.py` 批量下载番剧视频和弹幕：

```bash
pipenv run python bili_downloader.py
# 按照提示登录并输入番剧 media_id，程序自动下载所有剧集及原始弹幕为 JSON
```

### 2. 帧导出与弹幕映射

**推荐：直接用 `mapper_frame.py` 提供的主流程函数**，例如：

```python
# 示例：将 B 站番剧弹幕从被删减版批量迁移到无删减版
if __name__ == "__main__":
    from mapper_frame import map_bili_danmaku_with_frames

    map_bili_danmaku_with_frames(
        media_id=28235123,  # B站番剧ID（与下载目录保持一致）
        videos_a_name="[UHA-WINGS&JOJO][JoJo's Bizarre Adventure - Stone Ocean][25-38][x264 1080p][sc_jp]",  # 未删减版视频文件夹名
        offset=1,           # 若集数对应有偏移（如第1集删减，需手动调整）
        template_str=r"^\[.*?\]\[.*?\]\[(?P<no>\d+)\]"
    )
```

弹幕映射后会自动在目标目录生成新的 JSON 弹幕文件（如 `dms_ep_25.json`）。

### 3. 本地 HTML 播放器预览

直接用浏览器打开 `player.html`，选择本地视频和 JSON 弹幕文件即可预览效果。

---

## 进阶用法

### 1. 自定义帧参数/映射算法

在 `mapper_frame.py` 中，所有帧裁剪、哈希、锚点推进、匹配参数均可灵活配置（见第 4 点）。未来支持算法切换配置。

### 2. 支持多集批量自动化

可通过循环、并行等方式处理整个番剧季，极大提升效率。

### 3. 支持各种弹幕格式转化

目前输出为 nplayer 兼容的弹幕 JSON，后续可拓展 ASS、XML 等格式。

### 4. `.env` 环境变量文件参数解释

| 参数                   | 描述                                           | 必须 | 默认值     |
|----------------------|----------------------------------------------|:--:|---------|
| BASE_DIR             | 程序的数据目录                                      | ❌  | 项目根路径   |
| FFMPEG_PATH          | ffmpeg 可执行文件路径。如果已经加入到PATH环境变量，则无需配置。        | ❌  | ffmpeg  |
| FFPROBE_PATH         | ffprobe 可执行文件路径。如果已经加入到PATH环境变量，则无需配置。       | ❌  | ffprobe |
| CROP_RATIO           | 视频的纵向裁剪比例，可以根据视频字幕、水印等情况调整。                  | ❌  | 0.6     |
| WIDTH                | 提取的视频帧宽度                                     | ❌  | 256     |
| HEIGHT               | 提取的视频帧高度                                     | ❌  | 256     |
| ANCHOR_INTERVAL      | AAP: 锚点间隔帧数                                  | ❌  | 50      |
| RANGE_FACTOR         | AAP: 搜索范围系数(占被搜索视频帧数的比例)。若视频间最大帧数差异较大，可适当增大。 | ❌  | 0.2     |
| MAX_HAMMING_DISTANCE | AAP: 最大汉明距离，控制匹配精度。该参数越大，匹配越宽松，但可能导致错误匹配。    | ❌  | 10      |


> ⚠️ 由于视频导出帧后产生数以万计的图像文件可能会导致 IDE 卡顿，**强烈建议将`BASE_DIR`设置为项目根目录外的独立目录**。

---

## FA♂Q

- **Q: 支持哪些 B站番剧？**

  - 只要两个版本画面基本一致（删减/重编码不影响主视觉），都可映射。高度删改或马赛克需适当调整参数。

- **Q: 是否需要会员或登录？**

  - 需要登录 B 站账号下载番剧及弹幕。

- **Q: 输出弹幕格式？**

  - 默认输出为 JSON，兼容 nplayer，本地预览和自定义转格式均可。

- **Q: 可以切换算法吗？**

  - 当前版本仅支持图像帧 phash 算法，未来会增加算法切换接口。

---

## 许可协议

本项目基于 GNU 通用公共许可证第 3 版（GPLv3）发布。
详情请查看 [LICENSE](LICENSE) 文件。

---

本项目为个人兴趣开发，如有问题或建议欢迎 Issue/PR。

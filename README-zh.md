# bili-danmaku-mapper

支持从工作台一键在 IINA 原音质观看，原始外挂 ASS 与独立弹幕同时显示，无需媒体副本。见 [原生播放器说明](docs/native-player.md) 与 [验收记录](docs/native-player-acceptance.md)。

> 新版观影流程：已有成品默认进入“观看”，单片自动配对，人工复核收进高级功能；原片不兼容时使用有上限的按需分段播放，不自动复制全片。详见 [观影工作台说明](docs/viewing-upgrade.md)。

将 B 站番剧弹幕迁移到同一剧情的另一视频版本，并调整弹幕时间轴。适用于删减、目标新增片段、重编码、不同帧率和适度变速。默认使用本地 CPU，无需 GPU、模型下载或上传视频。

新版以实际展示时间戳为基准，默认结合画面与音频证据建立分段时间映射。默认保留视觉优先的混合流程；可通过 `--hybrid-strategy audio_first` 先用音频指纹定位，再按需验证画面及精定位，音频不可用时回退完整视觉序列对齐。音频优先在真实长片上尚未通过覆盖率验收，暂不作为默认。共享文件摘要、粗细特征及已验证的对齐结果，单独更换弹幕无需重新解码。单张相似画面或整段音频的固定偏移不会直接成为跨全片的时间依据。源、目标分别指**原弹幕对应的视频**和**希望观看的视频**。

## 双击使用（macOS，0.11.2）

双击项目根目录的 **`启动.command`**：粘贴 B站链接 → 选择本地影片或目录 → 确认配对 → 开始处理 → 点击已完成集数播放。缺少项目依赖会先显示网页安装确认；扫码登录也在网页进行。多集的下载与计算并行，任务按阶段保存，网页关闭后继续运行。

支持 BV/AV、多P、普通合集、番剧 ep/ss/md 和短链接，默认严格模式、最高 720P H.264 参考视频。对齐直接使用原片，包括 MKV；网页播放先尝试直放，音频不兼容时自动切换有上限的分段播放，保留视频编码、将音频按需转换为 AAC 双声道，并明确提醒音质变化。普通观影流程不转码视频或生成整片副本。希望保留原音轨和 ASS 样式时使用 IINA。原片和旧结果保留。详细说明见 [本地工作台](docs/workbench.md)，验证结果与人工验收边界见 [验收记录](docs/workbench-acceptance.md)。CLI 仍可继续使用。

顶部「观看」是独立的成品观看标签页：既可选择已完成任务，也可直接导入本地影片和已对齐弹幕；支持倍速、全屏、弹幕设置与观看进度恢复。网页播放切回「准备影片」时自动暂停，后台任务继续。双视频复核收进高级功能，进入前提示可能剧透。正片控件采用本地固定版本 NPlayer，支持外挂 SRT/VTT/ASS/SSA 对白字幕；字幕和弹幕分别支持 -5～+5 秒微调（0.1 秒步进，按影片保存）。网页将 ASS/SSA 转为文本显示，IINA 保留原始 ASS 样式。见 [成品播放器说明](docs/finished-player.md)。

## 安装

需要 Python 3.11+ 和支持 `-fps_mode` 的 FFmpeg/FFprobe（建议 FFmpeg 6+），并加入 PATH。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m bili_danmaku_mapper doctor
```

`pyproject.toml` 是直接依赖的唯一来源，`requirements.lock` 记录本次测试使用的完整依赖版本；旧 Pipenv 清单已移除。仅使用映射功能可安装 `pip install -e .`；下载器使用 `pip install -e '.[download]'`。完整工作台测试环境为 `pip install -e '.[download,web,test]'`。`doctor` 的下载器依赖检查在仅安装核心时可能显示缺失，不影响离线映射。

可以复制 `.env.example` 为 `.env`。配置优先级为命令行 > 进程环境变量 > `.env` > 默认值；`.env` 在运行时主动读取。`--env-file` 放在子命令之前。FFmpeg 和 FFprobe 已在 PATH 中时，可不设置 `FFMPEG_PATH`、`FFPROBE_PATH`，不要赋为空字符串。旧 `.env` 的裁剪比例继续生效。密码、Cookie 和验证码不应放在 `.env` 或 Git 中。

## 单集对齐

```bash
python -m bili_danmaku_mapper align \
  --source 'B站版.mp4' --target '本地完整版.mkv' \
  --danmaku 'dms_raw_0.json' --output 'dms_ep_1.json'
```

输出三个文件：

- `dms_ep_1.json`：保持 nplayer 格式的可播放弹幕。
- `dms_ep_1.review.json`：未映射弹幕，保留原内容、原时间及原因。
- `dms_ep_1.report.json`：带 `schema_version: 1` 的分段映射、逐条状态、统计和运行信息。

原视频和原始弹幕不修改；已有输出默认拒绝覆盖，确认要替换时加 `--overwrite`。没有 `--danmaku` 时，`--output` 保存对齐结果。输入和输出即使使用 `--overwrite` 也不能为同一文件。

### 严格与宽松模式

默认 `--mode strict`：输出可靠锚点和同一连续片段内不超过 1 秒的插值，其他弹幕进入待复核文件。

`--mode lenient`：还允许同一已确认片段内较长间隙插值，报告中标为 `estimated`。两种模式都禁止跨编辑断点插值、向无证据区域外推或把失配弹幕吸附到附近时间点。`reliable` 表示通过规则验证，可信度分数不是正确概率。

静止画面和完全相同的重复片段可能无法唯一定位。严格模式的拒绝结果是正常现象，应同时查看可靠覆盖率、估算率和未映射率。

### 音频辅助（0.3.0）

默认 `--evidence-mode hybrid`：在画面被打码、遮挡或失配时，允许经验证的音频锚点补充映射。无音轨、已知语言不对应、解码失败或没有可靠声音匹配时，保留视觉流程，原因写入报告。纯视觉使用 `--evidence-mode visual`；排查音频时可用 `--evidence-mode audio`，缺少音轨时会报错。

```bash
python -m bili_danmaku_mapper align \
  --source source.mp4 --target target.mp4 --danmaku raw.json --output mapped.json \
  --evidence-mode hybrid --source-audio-stream 0 --target-audio-stream 1
```

音轨编号是**音频流内从 0 开始的序号**，不是所有媒体流的绝对编号。未指定时优先选择相同语言音轨；缺少语言标签时选首个音轨并记录这一决定。声音版本不同、重复配乐、静音以及明显变速/变调仍可能失配。

音频先按实际 PTS 解码，再使用局部频谱地标投票和当前位置的短时频谱复核。支持将不同剪辑区间分开处理；不会把一个全片偏移应用到所有弹幕。稳定的音画延迟只有得到足够视觉证据后才校准；无法解释的音画冲突在两种输出模式下都进入复核。报告记录选中音轨、音频状态、锚点数、校准偏移，以及逐条弹幕的 `visual` / `audio` / `manual` 支持来源。

音频指纹和频谱缓存以视频内容、音轨与算法版本区分。解码 PCM 只写独立临时目录，处理后移除，不上传音频。说明与验收范围见 [docs/audio-alignment.md](docs/audio-alignment.md)。

## 批量处理

```bash
python -m bili_danmaku_mapper batch \
  --source-dir 'data/bangumi/28235123' --target-dir 'data/bangumi/完整版' \
  --output-dir 'data/mapped' --offset 1 --workers 2
```

按实际文件集数处理，默认正则 `(?P<no>\d+)$` 提取文件名末尾数字。源弹幕默认 `dms_raw_{no}.json`，可通过 `--danmaku-pattern` 修改。缺少对应视频或弹幕时记录 `skipped`，单集错误记录 `failed`，汇总写入 `batch.summary.json`；有失败或跳过时进程返回非零退出码。

目标文件名复杂时可指定 `--target-pattern '^\[.*?\]\[.*?\]\[(?P<no>\d+)\]'`。`--start` 和 `--end` 是源集数范围，结束值不包含在内。也可用 `--pairs 配对.json`，文件内容为：

```json
[{"source": 0, "target": 25}, {"source": 2, "target": 27}]
```

显式配对优先于 `--offset`，每边集数必须唯一。无需事先导出帧图片。

## 下载视频和原始弹幕

```bash
python -m bili_danmaku_mapper download --media-id 28235123 --data-dir ./data
```

CLI 支持扫码、密码和短信登录，工作台使用扫码登录；验证网页仍由人手动操作。密码输入隐藏，Cookie 在数据目录中以限制权限的文件保存，刷新后原子更新。下载和合流先写任务临时目录，完整性和视频检查通过后再发布；现有剧集默认拒绝覆盖，可显式使用 `--overwrite`。普通视频和番剧的匿名真实下载已验证；扫码成功登录、Cookie 刷新及受限账号仍需人工验收，详见 [来源验证记录](docs/workbench-acceptance.md)。

## 双视频复核与人工锚点

直接打开 `player.html`。它使用浏览器原生视频与本地弹幕渲染器，无 CDN 依赖，仍接受 nplayer JSON。选择源视频、目标视频、映射弹幕和 `.report.json`；原始弹幕可选，用于显示待复核文本。

- 点击待复核弹幕定位源时间；对应关系缺失时暂停目标视频。
- 启用“源视频控制同步播放”可跟随分段映射播放；关闭后分别定位两边画面。
- 记录人工锚点并导出 JSON，再重新运行：

```bash
python -m bili_danmaku_mapper align \
  --source source.mp4 --target target.mp4 --danmaku raw.json \
  --anchors anchors.json --output corrected.json
```

锚点格式为 `[{"source_time": 1.25, "target_time": 3.5}]`，必须在视频时长内，并在两条时间轴共同严格递增。锚点约束整个序列搜索，不仅覆盖最终结果。不要跨删减边界强行设置连续对应关系。

浏览器通常不能播放 MKV 或某些音视频编码，可先转换为 H.264/AAC MP4；改变内容或时间轴的转换需重新对齐。播放器只读取本地文件，不上传媒体。

## Python 接口与参数

```python
from bili_danmaku_mapper import AlignConfig, align_videos, map_danmaku

alignment = align_videos("source.mp4", "target.mp4", AlignConfig.resolve())
result = map_danmaku({"code": 1, "data": [{"time": 1.2, "text": "弹幕"}]}, alignment, mode="strict")
# result.danmaku / result.review / result.report
# alignment.to_dict(); AlignmentResult.from_dict(...) 可保存并重新读取
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--sample-rate` / `SAMPLE_RATE` | 2 | 每秒粗采样帧数 |
| `--crop-ratio` / `CROP_RATIO` | 0.8 | 中心辅助特征保留高度；全画面也参与比较 |
| `--top-k` / `TOP_K` | 5 | 保留的不同时间位置候选数 |
| `CONTEXT_SECONDS` | 3 | 前后画面上下文长度 |
| `--max-hash-distance` | 10 | 64 位哈希候选距离上限 |
| `--strict-gap` / `STRICT_GAP` | 1 | 严格模式最大插值间隔，秒 |
| `--max-segment-gap` | 10 | 允许构成同一连续片段的最大锚点间隔，秒 |
| `--cache-dir` | `BASE_DIR/features` | 特征缓存目录 |
| `--hwaccel` / `HWACCEL` | 未启用 | 可选硬件解码；失败后重试软件路径 |
| `--algorithm` / `ALGORITHM` | `sequence` | 新算法；`legacy` 显式选择 AAP 基线 |
| `--evidence-mode` / `EVIDENCE_MODE` | `hybrid` | 画面与声音联合；另有 `visual`、`audio` |
| `--source-audio-stream` | 自动 | 源音频流序号 |
| `--target-audio-stream` | 自动 | 目标音频流序号 |

`--no-refine` 仅用于分析粗对齐性能，不建议用其结果验收精度。特征缓存在视频内容摘要、采样参数或算法版本变化后自动失效；损坏缓存会重新计算。默认不导出全量 JPEG。长视频仍需完整解码获得采样画面，候选检索也并非严格线性复杂度。

## 兼容与验证边界

`mapper_frame.py` 的主要单集、批量入口转接新版严格模式，兼容传入的裁剪比例与哈希距离；其余新版设置使用 `AlignConfig` 或 CLI。原 AAP 函数仍保留供基线研究。旧帧目录不作为新版时间依据，会从视频重新提取；旧帧映射 JSON 包装函数支持字符串键，只保留直接有效且递增的匹配，不再做无限距离吸附。旧自动删除视频选项已禁用。帧导出可用于调试，会附带 PTS 清单并拒绝非空目录。

```bash
python -m pytest -q
node tests/test_danmaku_renderer.mjs
node tests/test_player.mjs
node tests/test_watch.mjs
node tests/test_subtitle_renderer.mjs
node tests/test_segment_player.mjs
python -m tests.benchmark --output docs/acceptance-metrics.json
python -m tests.audio_benchmark --output docs/audio-acceptance-metrics.json
```

验收记录见 [docs/acceptance.md](docs/acceptance.md)，详细指标见 [docs/acceptance-metrics.json](docs/acceptance-metrics.json)。合成样例通过不能替代真实番剧验收；镜头调序、严重裁切、重绘画面和无法分辨的重复片段不在自动保证范围内。

映射输出提交过程中采用独占锁、暂存和回滚，报告最后发布。进程被强制终止或系统崩溃仍可能留下 `.输出文件名.lock` 或部分输出；确认没有任务运行后检查文件，再移除该任务的锁并重新运行。跨文件集合不具备数据库级崩溃事务保证。

## 许可

GPL-3.0-or-later，详见 [LICENSE](LICENSE)。


音频优先、精细缓存、硬件解码与受监护基准测试见 [性能优化说明](docs/performance-optimization.md)。


### 实验：恢复严格模式覆盖率

工作台高级设置新增“补充证据以恢复覆盖”，默认关闭、按任务保存。该模式增加局部画面与音频验证，仍保持最长 1 秒严格插值；可能增加处理时间，真实影片结果需要复核。CLI 使用 `--coverage-recovery`，显式关闭使用 `--no-coverage-recovery`。两种音画调度共用验证规则。已有任务和成品不自动重算。

使用方法、缓存与复核工具见 [覆盖率优化说明](docs/coverage-optimization.md)，分项结果与限制见 [验收报告](docs/coverage-acceptance.md)。

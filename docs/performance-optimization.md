# 音频优先与缓存优化（0.8）

音频优先流程先建立音频时间锚点，再验证画面及局部精定位；没有可靠音频时回退完整视觉对齐。由于真实影片的严格可靠覆盖率从旧版35.9%降至音频优先23.5%，本轮不将它提升为默认。默认仍使用 `visual_first` 混合流程，解码合并、共享摘要及分层缓存同时生效。严格模式仍只在同一可靠片段的最长 1 秒间隔内插值；打码或低纹理画面不能单独否定可靠音频，无法解释的音画冲突进入复核。

## 使用与兼容

原有网页、CLI 和播放器格式保留。现有任务不迁移、不重做，也不会自动替换历史结果。完整加载新版服务配置需在空闲时正常退出后再次双击启动；开发和验收不会代用户重启。

```sh
.venv/bin/python -m bili_danmaku_mapper align --source source.mp4 --target original.mkv --danmaku raw.json --output mapped.json --hybrid-strategy audio_first
```

- `--hybrid-strategy audio_first`：显式启用音频优先实验策略；`visual_first` 为当前默认。真实准确性仍待人工标注，不将覆盖率差异当作正确率。
- `--no-alignment-cache`：绕过最终对齐缓存，保留粗、精细和音频特征缓存。
- `--hwaccel videotoolbox`：显式使用 macOS 硬件解码；默认软件解码，异常时自动回退。报告注明请求与实际后端；回退结果不会发布到硬件缓存中。
- `--source-hwaccel software --target-hwaccel videotoolbox`：分别指定两个输入的后端；分输入设置优先于 `--hwaccel`。环境变量为 `SOURCE_HWACCEL`、`TARGET_HWACCEL`。720P H.264 参考片可能软件更快，不能因为 4K HEVC 原片受益就强制两边硬解。
- `.env` / 环境变量 `HYBRID_STRATEGY=audio_first|visual_first`，优先级沿用现有配置规则。

只更换弹幕 JSON 或严格/宽松输出模式时复用已验证对齐，不需要重新解码。修改锚点、音轨、内容或对齐参数会使对齐缓存失效，但不无故废弃仍有效的特征。

## 缓存与时间轴

文件身份及探测信息共享；首次计算完整摘要，以后依据文件标识、大小、纳秒时间和分布式校验块确认复用。读写期间输入发生变化则失败，不发布有效结果。这是普通本地文件变化检查，不是针对恶意伪造文件元数据的安全校验。

粗特征继续兼容现有格式。精细特征采用 30 秒分块和 0.75 秒边缘，连续缺失分块合并解码，每次不超过 120 秒；非缓存区间也合并处理。稀疏请求覆盖不足所涉分块一半时，直接缓存合并后的实际区间，避免为了少量锚点过取整块。保存的是 PTS 和特征，没有全量图片或视频副本。已缓存精细特征也可回答稀疏采样请求。

对齐结果按两部视频内容、实际音轨、参数、版本和人工锚点校验；旧工作台结果文件存在不再代表结果仍然有效。并发生产者使用文件锁，完整分块和结果原子发布。

## 重复基准

先保存待比较代码的完整工作区副本，不能拿尚未包含现有功能的 Git HEAD 冒充基线。准备 `inputs.json`：

```json
{"source":"/absolute/reference.mp4","target":"/absolute/original.mkv","danmaku":"/absolute/raw.json","config":{"evidence_mode":"hybrid","sample_rate":2.0,"refine":true}}
```

所有输出目录使用本次实验专用目录。生产数据库只读监护；出现排队或运行任务时终止实验自己的进程组。监护默认读取本项目 `data/workbench/state.sqlite3`；隔离副本必须显式指定实际数据库。

```sh
# 硬件组件对照：实验目录内放 inputs.json
.venv/bin/python -m tests.optimization_benchmark --output-dir data/bench-hardware --guard-command .venv/bin/python -m tests.hardware_benchmark data/bench-hardware

# 原片自身窗口对照：时间真值是自身 PTS，不是 B站与原片的真实准确性
.venv/bin/python -m tests.optimization_benchmark --output-dir data/bench-windows --guard-command .venv/bin/python -m tests.window_benchmark --inputs data/bench-hardware/inputs.json --baseline-dir /absolute/baseline --output-dir data/bench-windows

# 整片对比；硬件通过片段门槛后才添加 --hardware
.venv/bin/python -m tests.optimization_benchmark --inputs data/bench-hardware/inputs.json --baseline-dir /absolute/baseline --output-dir data/bench-full --hardware --hardware-source-software
```

整片比较分别运行首次处理、暖特征重新对齐、对齐结果复用；旧版没有最终对齐缓存，跳过该项。空应用缓存不等于物理磁盘冷缓存；不清理系统缓存。首次处理目录必须为空，防止误标冷启动。

上例硬件候选保留参考片软件解码，只让目标原片使用 VideoToolbox；去掉 `--hardware-source-software` 可测两边硬解。当前硬件路径先按 PTS 选择请求帧，再 `hwdownload`，减少不需要的帧搬运；不使用依赖像素的场景检测表达式。[FFmpeg 的 select 实现](https://github.com/FFmpeg/FFmpeg/blob/n8.0/libavfilter/f_select.c)提供元数据筛选路径，实际兼容性仍以本机 PTS、特征和映射对照为准。此前中止的两边硬解试验单独归档，不计作已完成整片结果。

监护限制实验目录总量 4 GiB、可用磁盘至少 5 GiB；临时音频进入实验目录。进程树内存按秒采样，可能漏掉更短的峰值；无法读取进程信息时记为不可用。报告的摘要字节数是逻辑读取量，不能描述为实测物理磁盘 I/O。阶段计时有嵌套，不能直接求和。

准确性验收使用 `python -m tests.optimization_acceptance --baseline-dir /absolute/baseline --output-dir data/bench-synthetic`，同样应通过监护入口运行。真实影片没有人工标注时只报告覆盖率、一致性、时间差与待复核材料，不能将一致性写成正确率。

窗口组件组含义：`baseline` 的局部粗特征也需重提取；`merged` 复用局部粗特征、重新精定位；`dense_cache` 复用粗细特征。冷组用于观察解码调度变化，两个新版暖组用于观察精细缓存收益。音频先行的检索量和阶段耗时单列，整片提速作为组合结果，不能把各项倍数相乘。

## 长片音频与检查调度

音频指纹使用更细的频率及时间差索引（landmark-3），以免长片中大量语音键被误当成重复背景丢弃；旧音频特征需按版本重新提取，旧视觉特征继续兼容。局部候选通过全局单调路径选择及至少一组稳定序列支持，零散巧合不能约束整片。

对于常见25/24及其相邻PAL比例，只有原始音频支持不足时才抽样验证。通过后对参考音轨做临时采样率校正并重新计算指纹；不改变影片、播放速度或弹幕原时间。所有源锚点换回原播放器秒数。不同帧率也可能只是补帧，因此不能仅凭帧率比例决定映射。`--no-audio-speed-search` 或 `AUDIO_SPEED_SEARCH=false` 可禁用，配置优先级保持命令行、进程环境、.env、默认值。

当至少20个检查窗口覆盖一部影片40%以上时，复用或一次提取全片稀疏特征，避免许多重叠窗口反复解码。上下文只用来验证；稳定抽检只精定位代表点。真正需要精定位的工作集合并后统一提取，再由各检查复用，已缓存的密集特征也可回答新的精细子窗口。

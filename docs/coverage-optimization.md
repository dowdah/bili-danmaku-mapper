# 严格模式覆盖率恢复（实验）

本功能通过补充实际音画证据恢复覆盖，默认关闭。严格模式仍只在同一可靠片段内、最长 1 秒的锚点间隔内插值；不向无证据区域外推。`reliable` 是规则判定状态，不是真实正确概率。真实影片准确率需要独立人工标注。

## 使用

工作台重新正常启动后，高级设置提供“实验：补充证据以恢复覆盖”。此选项按任务保存，处理会更久。当前正在运行的旧服务没有该能力时，网页隐藏此选项；无需为部署本次代码打断任务。

CLI 示例：

```sh
.venv/bin/python -m bili_danmaku_mapper align \
  --source '/path/reference.mp4' --target '/path/original.mkv' \
  --danmaku '/path/raw.json' --output '/path/new-result.json' \
  --coverage-recovery --mode strict
```

`--no-coverage-recovery` 显式关闭；也可设置 `COVERAGE_RECOVERY=true`。命令行优先于进程环境、`.env` 和默认值。更换输出路径保留旧成品；原片直接参与对齐，本功能不生成播放副本。

`--recovery-stage fusion|visual|audio` 用于分项实验，默认 `audio`（全部步骤）。`--hybrid-strategy audio_first|visual_first` 只改变准备顺序，共用同一候选验证、分段和冲突政策；为覆盖优先，实验恢复流程使用整片粗特征作为安全回退，因此不承诺保留旧音频优先流程的稀疏解码优势。

## 如何恢复

1. 根据实际 PTS、上下文和经过音频验证的时间比例检索候选；精定位保留多个实际帧候选，由单调序列求解选择。
2. 鲁棒模型识别稳定时间关系和孤立离群点。模型只用于检索和验证，输出锚点保持原始测量时间；不能用拟合线伪造锚点。
3. 音画反证需要连续序列支持。疑点范围由两侧重新一致的证据界定；没有边界时保留未确认范围，不盲目缩小屏蔽。已识别的视觉编辑间隙禁止用音频直接填过。
4. 先在现有粗特征中重检短空洞与编辑两侧，再合并缺失源区间，以 4 fps 实际帧补证。精定位复用固定分块缓存，重复、无新增证据的窗口不重算。
5. 对仍有空洞的区间，短窗口比较单声道、前置声道、左右声道及可用中心声道。只有指纹与两秒频谱序列均支持的位置才接入；重复声音和静音继续拒绝。PAL 比例只作为候选，必须经过声音序列验证。

全部采用本地 CPU/DSP；没有新增深度模型、云识别或依赖。硬件解码仍使用现有显式配置与失败回退。

## 缓存与报告

粗细特征格式保持兼容。实验对齐使用独立结果签名；关闭实验时兼容原有默认流程的结果签名。新增音频锚点缓存包含两部影片身份、实际音轨、速度搜索、严格间隔、分段参数和人工锚点，损坏时重建。

报告增加：候选过滤事件、完整首轮帧候选与路径状态、有限数量的拒绝示例、补证轮次及耗时、声道方案、均匀时间覆盖率。事件次数可能包含不同候选和局部重复检验，不能直接当作唯一失配帧数。未映射区分证据不足、证据冲突、时间轴外及人工确认无对应；未确认的编辑不自动声称没有对应剧情。

人工标注不存在时，`comparable_interval_coverage` 为 `null`，`real_accuracy_human_verified` 为 `false`。历史报告与播放器 JSON 继续兼容。

## 可重复验证

```sh
# 读取已有报告，审计拒绝原因；不解码影片。
.venv/bin/python tests/coverage_benchmark.py --audit old.report.json \
  --output audit.json

# 使用现有本地输入和独立缓存运行整片；inputs 为 source/target/danmaku/config。
.venv/bin/python tests/coverage_benchmark.py --inputs inputs.json \
  --seed-cache '/path/existing-features' --output-dir '/path/new-experiment' \
  --database data/workbench/state.sqlite3 --stage audio

# 比较并导出 discovery/validation 分层复核清单。
.venv/bin/python tests/coverage_benchmark.py --compare old.report.json new.report.json \
  --output comparison.json
.venv/bin/python tests/coverage_review.py --review comparison.review.json \
  --alignment new.report.json --images 5

# 人工确认后，只评价 validation 样本，不把预测当真值。
.venv/bin/python tests/coverage_benchmark.py --annotations coverage-annotations.json \
  --alignment new.report.json --output human-validation.json
```

实验守护器只读取生产队列。生产任务进入队列、空间不足或实验超预算时，停止本实验拥有的进程组，保留已经发布的阶段结果。预算默认 4 GiB，可用空间必须至少 5 GiB；多个实验共享预算时传 `--budget-root`。种子 NPZ 使用硬链接，后续原子发布不会覆盖种子数据；逻辑容量会重复计算链接。

已知时间线验收：`python -m tests.coverage_acceptance --fixtures /path/synthetic-fixtures --output-dir /path/new-output`。完整统计与真实影片限制见 [coverage-acceptance.md](coverage-acceptance.md)。

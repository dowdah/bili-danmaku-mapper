# bili-danmaku-mapper

Play the original movie in IINA with its original audio, external ASS subtitles and an independent danmaku overlay, without creating a movie copy. See the [native player guide](docs/native-player.md) and [validation record](docs/native-player-acceptance.md).

> Viewing-first workbench: resume finished films, prepare unambiguous single-film pairs, and keep manual review under advanced tools. Direct playback falls back to bounded stream-copy fragments instead of whole-film copies. See [viewing workbench notes](docs/viewing-upgrade.md).

A local CPU tool that transfers Bilibili anime danmaku to another edition of the same story. It handles cuts, inserted footage, re-encoding, differing frame rates, VFR and moderate playback speed changes. [中文文档](README-zh.md)

Hybrid mode retains the visual-first schedule by default. Opt in with `--hybrid-strategy audio_first` to build local spectral audio anchors before selected visual checks and original-frame refinement; unavailable or unreliable audio falls back to full visual alignment. Audio-first has not passed real-film coverage acceptance and is not promoted to the default. Actual presentation timestamps define every mapping. File identities, coarse and fine features, and validated alignments are cached; changing danmaku alone reuses the alignment. Edit boundaries remain separate, and ambiguous images or repeated sounds do not establish independent anchors.

## Double-click workbench (macOS, 0.11.2)

Double-click **`启动.command`** in Finder, paste a Bilibili link, select local movies, confirm episode pairing and start. Finished episodes can play immediately while later episodes download and align. QR sign-in, stage recovery, playback recovery, review and anchor revisions are available in the local UI. Missing project dependencies have a browser confirmation step; Python 3.11+ and FFmpeg remain system prerequisites.

Supports BV/AV, multi-part videos, UGC collections, ep/ss/md and b23.tv links. References default to H.264 at no more than 720P. All UI assets are local; the service binds only to loopback. Alignment uses original movies, including MKV. Web playback tries the original first; incompatible audio automatically selects bounded fragments that copy the video and convert audio to AAC stereo, with a quality warning. The normal viewing flow does not transcode video or create a whole-film copy. Use IINA to retain the original audio and native ASS rendering. Existing CLI and standalone player remain available. See [workbench guide](docs/workbench.md) and [acceptance boundaries](docs/workbench-acceptance.md).

The **Watch** tab plays a completed result or independently imports a local movie and aligned danmaku JSON. Web playback pauses when returning to **Prepare**, preserving progress and settings; background preparation continues. Video comparison and anchors are advanced tools with a spoiler warning. Locally bundled NPlayer controls support subtitles and independent subtitle/danmaku offsets from -5 to +5 seconds in 0.1s steps. Browser ASS/SSA display uses converted text; IINA preserves the original ASS styles. See the [viewer guide](docs/finished-player.md).

## Install

Python 3.11+ and FFmpeg/FFprobe supporting `-fps_mode` are required (FFmpeg 6+ recommended).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m bili_danmaku_mapper doctor
```

`pyproject.toml` owns the direct dependencies; `requirements.lock` pins the tested full environment. The former Pipenv files have been replaced. For mapping alone use `pip install -e .`; add `.[download]` for Bilibili downloads and `.[test]` for pytest. Doctor also checks download dependencies, so a core-only install can report missing optional modules while mapping remains usable.

Copy `.env.example` to `.env` if desired. Precedence: command line > process environment > dotenv > defaults. Put `--env-file` before the subcommand. If FFmpeg and FFprobe are already in PATH, leave `FFMPEG_PATH` and `FFPROBE_PATH` unset rather than assigning empty strings. Credentials must not be placed in dotenv or committed to Git.

## Commands

```bash
python -m bili_danmaku_mapper align \
  --source source.mp4 --target target.mkv --danmaku raw.json --output mapped.json

python -m bili_danmaku_mapper batch \
  --source-dir data/source --target-dir data/target --output-dir data/mapped \
  --offset 1 --workers 2

python -m bili_danmaku_mapper download --media-id 28235123 --data-dir ./data
```

Source means the video corresponding to the original danmaku. Target means the edition you want to watch. Outputs preserve nplayer JSON compatibility and include `.review.json` with unresolved comments and `.report.json` with versioned alignment, per-comment status, counts and diagnostics. Omitting `--danmaku` saves alignment alone.

Inputs remain unchanged. Existing outputs require `--overwrite`. Batch jobs use actual episode numbers and automatically extract missing features. Missing files are reported as skipped; failures do not silently pass. Both skips and failures produce a nonzero exit code. Filename patterns must contain a named `no` group (default `(?P<no>\d+)$`). `--pairs` accepts a JSON list such as `[{"source":0,"target":25}]` and overrides the offset. See `batch --help` for range and pattern options.

Strict mode is the default. It accepts verified anchors and interpolation across gaps no longer than one second within verified continuous segments. Lenient mode (`--mode lenient`) also exports longer within-segment interpolation as `estimated`. Neither mode extrapolates or interpolates across detected edits. Confidence is a rule score, not a probability. Check coverage alongside accuracy.

## Audio assistance (0.3.0)

The default `--evidence-mode hybrid` combines visual and locally verified audio anchors. Use `visual` to disable audio, or `audio` to diagnose soundtrack correspondence without image matching. Hybrid mode retains the visual result if audio is missing, unusable, in a different known language, or has no reliable matches. Audio-only mode requires audio streams.

`--source-audio-stream 0 --target-audio-stream 1` selects audio-stream ordinals, not absolute media stream indices. Automatic selection prefers a shared language; otherwise untagged files use their first audio stream with the choice recorded. Fingerprints use decoded PCM PTS, per-window offset consensus and short-time spectrum verification. Stable A/V track delays are calibrated only with enough visual evidence; unresolved modality conflicts are held for review in both output modes.

Reports include stream choices, audio match/fallback status, bias calibration and per-comment `visual`/`audio`/`manual` support. Decoded PCM is confined to task-owned temporary storage and deleted after fingerprinting. Audio is never uploaded. Re-dubbing, replaced music, repeated soundtrack segments and significant pitch/tempo changes remain limitations. See [audio notes](docs/audio-alignment.md).

## Review and manual corrections

Open `player.html` and select the local videos, mapped JSON and report. The native browser player and danmaku renderer have no CDN dependency. Original JSON is optional for reviewing comment text. Navigate unresolved comments, synchronize the target to the source, hide estimated comments, or independently seek both videos and export manual anchors.

```json
[{"source_time":1.25,"target_time":3.5}]
```

```bash
python -m bili_danmaku_mapper align \
  --source source.mp4 --target target.mp4 --danmaku raw.json \
  --anchors anchors.json --output corrected.json
```

Manual anchors must increase strictly in both timelines and lie within the media durations. They constrain candidate search. Do not bridge edits with an assumed continuous segment. Browser support for MKV and some codecs varies; use H.264/AAC MP4 for review. Media files are read locally, never uploaded.

The CLI supports interactive password/SMS/QR login, while the workbench uses QR login; verification remains human-operated. Cookies are saved privately and refreshed atomically. Downloads and merges are staged and checked before publishing. Anonymous ordinary-video and bangumi downloads were validated; successful QR sign-in, cookie refresh and restricted accounts still require human acceptance. See the [source validation record](docs/workbench-acceptance.md).

## API and tuning

```python
from bili_danmaku_mapper import AlignConfig, align_videos, map_danmaku

alignment = align_videos("source.mp4", "target.mp4", AlignConfig.resolve())
result = map_danmaku({"data": [{"time": 1.2, "text": "hello"}]}, alignment, "strict")
# result.danmaku, result.review, result.report
```

Default coarse sampling is 2 fps, five distinct temporal candidates and three seconds of context. Full images and central crops both contribute to verification. `--sample-rate`, `--crop-ratio`, `--top-k`, `--max-hash-distance`, `--strict-gap`, `--max-segment-gap` and `--cache-dir` are configurable. Hardware decoding is optional and falls back to software. `--no-refine` is for coarse profiling, not accuracy acceptance.

Caches store compact arrays keyed by media content, preprocessing and algorithm version; corrupted caches are recomputed. Default operation does not export full-frame JPEG collections. Sampling still requires video decoding, and candidate search is not guaranteed to scale linearly.

`--algorithm legacy` retains the original AAP matcher as an explicit baseline using full-frame PTS. Unsafe regressions are filtered before playable output, and the raw map remains in diagnostics. Legacy main/batch Python entrypoints forward to the new strict pipeline. Untimed legacy frame directories are regenerated from video. Automatic source video deletion is disabled.

## Validation

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

See [acceptance notes](docs/acceptance.md) and [measured metrics](docs/acceptance-metrics.json). Synthetic success is not real anime acceptance. Reordered shots, severe cropping/redrawing and visually indistinguishable repeated footage may require manual review.

Output sets are staged under a per-output lock with rollback on ordinary write errors. The report is published last. Forced termination or a system crash can leave a lock or partial set; inspect the files after confirming no writer is running before removing that task's lock and rerunning. This is not a database-level multi-file crash transaction.

GPL-3.0-or-later; see [LICENSE](LICENSE).


See the [performance guide](docs/performance-optimization.md) for audio-first alignment, fine-feature caching, hardware decoding and guarded benchmarks.


### Experimental strict coverage recovery

Enable **Recover coverage with additional evidence** in the workbench advanced settings, or pass `--coverage-recovery` (`--no-coverage-recovery` disables it). It is off by default. Both hybrid schedules share one verifier; recovered points use measured timestamps, and strict interpolation remains limited to one second. Processing can take longer. Existing jobs are not recomputed, and real-film accuracy remains subject to human review. See [usage and validation tools](docs/coverage-optimization.md) and [acceptance results](docs/coverage-acceptance.md).

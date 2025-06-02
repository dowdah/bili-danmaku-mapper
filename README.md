# bili-danmaku-mapper

![Status: Developing](https://img.shields.io/badge/Status-Developing-blue?style=for-the-badge)
![Version: null](https://img.shields.io/badge/Version-null-brightgreen?style=for-the-badge)
[![中文文档](https://img.shields.io/badge/中文文档-brightgreen?style=for-the-badge)](README-zh.md)

**bili-danmaku-mapper** is a Python-based danmaku (comment) mapping tool that **accurately migrates Bilibili anime/video danmaku to another version of the same video** (e.g., syncing danmaku from a cut/edited version to an uncut version). This project is specially optimized for **multi-episode anime and heavily edited scenarios** in Bilibili.

Currently supported danmaku mapping algorithms:

- **Image frame-based perceptual hash (phash) matching (AAP)**

  Automatically builds a frame-to-frame mapping by comparing image frames of different video versions using perceptual hash (phash), then batch-maps danmaku timelines accordingly.

- **More algorithms to be supported in the future (e.g., audio fingerprinting, feature point matching, etc.)**

---

## Features

- Automatic frame alignment between different versions using perceptual hash (phash)
- Extensible architecture for future alignment algorithms
- Batch migration and alignment for multi-episode anime
- Batch transfer of danmaku from cut/edited versions to uncut versions
- Supports macOS / Linux / Windows
- Includes a convenient local HTML player (supports nplayer + JSON danmaku import)

---

## Installation & Environment

- **System Requirements**: macOS / Linux / Windows
- **Python Version**: 3.11+
- **Dependency Management**: pipenv

**Install dependencies:**

```bash
pip install pipenv
pipenv install
```

Or use `requirements.txt` directly:

```bash
pip install -r requirements.txt
```

---

## File Structure

```
.
├── bili_downloader.py      # Batch download and convert Bilibili anime videos & danmaku
├── mapper_frame.py         # Core logic for frame-based danmaku migration
├── player.html             # Local HTML video danmaku player (supports JSON danmaku)
├── Pipfile / Pipfile.lock  # pipenv environment config
├── requirements.txt        # Dependency list
├── README-zh.md / README.md
├── LICENSE
└── ...
```

---

## Quick Start

### 1. Download Videos & Danmaku

Use `bili_downloader.py` to batch download anime videos and danmaku:

```bash
pipenv run python bili_downloader.py
# Follow the prompt to log in and input the anime media_id, the script will auto-download all episodes and original danmaku as JSON
```

### 2. Frame Export & Danmaku Mapping

**Recommended: Use the main process function provided by `mapper_frame.py`**, for example:

```python
# Example: Batch-migrate Bilibili danmaku from cut versions to uncut versions
if __name__ == "__main__":
    from mapper_frame import map_bili_danmaku_with_frames

    map_bili_danmaku_with_frames(
        media_id=28235123,  # Bilibili anime ID (keep consistent with download directory)
        videos_a_name="[UHA-WINGS&JOJO][JoJo's Bizarre Adventure - Stone Ocean][25-38][x264 1080p][sc_jp]",  # Uncut video folder name
        offset=1,           # If episodes are offset (e.g., ep1 is cut, adjust manually)
        template_str=r"^\[.*?\]\[.*?\]\[(?P<no>\d+)\]"
    )
```

After mapping, new JSON danmaku files (like `dms_ep_25.json`) will be automatically generated in the target directory.

### 3. Preview with Local HTML Player

Just open `player.html` in your browser, select a local video and JSON danmaku file to preview.

---

## Advanced Usage

### 1. Custom Frame Parameters / Mapping Algorithm

In `mapper_frame.py`, all frame cropping, hashing, anchor pushing, and matching parameters are configurable (see #4 below). Algorithm switching will be supported in the future.

### 2. Batch Automation for Multi-episode Series

You can process an entire anime season efficiently via loop or parallel processing.

### 3. Danmaku Format Conversion Support

Currently outputs nplayer-compatible JSON danmaku; can be extended to support ASS, XML, etc., in the future.

### 4. `.env` Environment Variable Parameters

| Parameter            | Description                                                                                        | Req. | Default      |
|----------------------|----------------------------------------------------------------------------------------------------|:----:|--------------|
| BASE_DIR             | Data directory for the program                                                                     |  ❌   | Project root |
| FFMPEG_PATH          | Path to ffmpeg executable. If already in PATH, leave blank.                                        |  ❌   | ffmpeg       |
| FFPROBE_PATH         | Path to ffprobe executable. If already in PATH, leave blank.                                       |  ❌   | ffprobe      |
| CROP_RATIO           | Vertical cropping ratio, adjust for subtitles/watermarks, etc.                                     |  ❌   | 0.6          |
| WIDTH                | Extracted video frame width                                                                        |  ❌   | 256          |
| HEIGHT               | Extracted video frame height                                                                       |  ❌   | 256          |
| ANCHOR_INTERVAL      | AAP: Frame interval for anchors                                                                    |  ❌   | 50           |
| RANGE_FACTOR         | AAP: Search range factor (proportion of searched video frames).                                    |  ❌   | 0.2          |
| MAX_HAMMING_DISTANCE | AAP: Max Hamming distance, controls matching precision. Higher = looser, but may cause mismatches. |  ❌   | 10           |

> ⚠️ Exporting tens of thousands of frames may cause IDE lag. **It's highly recommended to set `BASE_DIR` to a directory outside the project root**.

---

## FAQ

- **Q: Which Bilibili anime are supported?**
  - As long as both versions are visually similar (cuts/re-encoding don't change the main visuals), mapping works. For heavily edited or mosaic content, adjust parameters as needed.

- **Q: Is a Bilibili account/login required?**
  - Yes, you must log in to download anime and danmaku.

- **Q: Output danmaku format?**
  - Default is JSON, compatible with nplayer. Local preview and custom format conversion are both supported.

- **Q: Can I switch algorithms?**
  - Only frame-based phash algorithm is currently supported; future releases will add switching interfaces.

---

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).  
See the [LICENSE](LICENSE) file for details.

---

This project is a personal hobby project. Questions or suggestions are welcome via Issue/PR.

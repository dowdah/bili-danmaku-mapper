"""
A tool for downloading videos and danmaku from Bilibili
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
import asyncio
import getpass
import json
import os
from pathlib import Path
import subprocess
import tempfile

import xmltodict
from tqdm import tqdm
from bilibili_api import Geetest, GeetestType, login_v2, Credential, bangumi, HEADERS, get_client, video

from bili_danmaku_mapper.config import ROOT, AlignConfig, settings
from bili_danmaku_mapper.io import atomic_json
from bili_danmaku_mapper.media import probe_video

_ENV = settings()
BASE_DIR = _ENV.get("BASE_DIR", str(ROOT / "data"))
FFMPEG_PATH = _ENV.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = _ENV.get("FFPROBE_PATH", "ffprobe")
cookies_path = os.path.join(BASE_DIR, "cookies.json")


async def _geetest(kind=None):
    gee = Geetest()
    if kind is None:
        await gee.generate_test()
    else:
        await gee.generate_test(type_=kind)
    gee.start_geetest_server()
    try:
        print("请在浏览器中手动完成验证：", gee.get_geetest_server_url())
        while not gee.has_done():
            await asyncio.sleep(.1)
    finally:
        gee.close_geetest_server()
    return gee


async def login() -> Credential:
    choice = input("pwd(p) / sms(s) / qr code(q):").strip()
    if choice not in ("p", "s", "q"):
        raise ValueError("Invalid login choice")
    if choice == "p":
        gee = await _geetest()
        cred = await login_v2.login_with_password(username=input("username:"),
                  password=getpass.getpass("password:"), geetest=gee)
    elif choice == "s":
        gee = await _geetest()
        phone = login_v2.PhoneNumber(input("phone:"), "+86")
        captcha_id = await login_v2.send_sms(phonenumber=phone, geetest=gee)
        cred = await login_v2.login_with_sms(phonenumber=phone, code=input("code:"), captcha_id=captcha_id)
    else:
        qr = login_v2.QrCodeLogin(platform=login_v2.QrCodeLoginChannel.WEB)
        await qr.generate_qrcode()
        print(qr.get_qrcode_terminal())
        while not qr.has_done():
            await qr.check_state()
            await asyncio.sleep(1)
        cred = qr.get_credential()
    if isinstance(cred, login_v2.LoginCheck):
        gee = await _geetest(GeetestType.VERIFY)
        await cred.send_sms(gee)
        cred = await cred.complete_check(input("code:"))
    return cred


def xml_2_json(xml_str: str) -> tuple[dict, int]:
    parsed = xmltodict.parse(xml_str)
    root = parsed.get("i") or {}
    entries = root.get("d", [])
    if isinstance(entries, dict):
        entries = [entries]
    if entries is None:
        entries = []
    output = {"code": 1, "data": []}
    for entry in entries:
        values = entry["@p"].split(",")
        if len(values) < 7:
            raise ValueError("Malformed danmaku parameters")
        timestamp = float(values[0])
        import math
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid danmaku timestamp")
        output["data"].append({"author": values[6], "time": timestamp, "text": entry.get("#text", ""),
                               "color": f"#{int(values[3]):06x}",
                               "type": {"4": "bottom", "5": "top"}.get(values[1], "scroll")})
    output["data"].sort(key=lambda item: item["time"])
    return output, len(output["data"])


async def download(url: str, out: str, intro: str, retries: int = 3, overwrite=False) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}")
    for attempt in range(retries):
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
        try:
            client = get_client()
            download_id = await client.download_create(url, dict(HEADERS))
            total = max(0, client.download_content_length(download_id) or 0)
            written = 0
            with os.fdopen(fd, "wb") as file, tqdm(total=total or None, unit="B", unit_scale=True, desc=intro) as bar:
                fd = None
                while True:
                    try:
                        chunk = await client.download_chunk(download_id)
                    except StopAsyncIteration:
                        break
                    if not chunk:
                        break
                    written += file.write(chunk)
                    bar.update(len(chunk))
                    if total and written >= total:
                        break
                if total and written != total:
                    raise IOError("Incomplete download")
                if not written:
                    raise IOError("Empty download")
                file.flush()
                os.fsync(file.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
            return
        except Exception:
            # Signed stream URLs and account/session data must not appear in logs.
            print(f"下载失败（第 {attempt + 1} 次）")
            if attempt + 1 == retries:
                raise RuntimeError("Download failed after retries") from None
            await asyncio.sleep(1)
        finally:
            if fd is not None:
                os.close(fd)
            Path(temporary).unlink(missing_ok=True)


async def download_episode(ep: bangumi.Episode, out: str, overwrite=False) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}")
    detecter = video.VideoDownloadURLDataDetecter(await ep.get_download_url())
    streams = detecter.detect_best_streams(video_max_quality=video.VideoQuality._8K,
                  audio_max_quality=video.AudioQuality._192K, no_dolby_audio=False,
                  no_dolby_video=False, no_hdr=False, no_hires=False)
    with tempfile.TemporaryDirectory(prefix=".episode-", dir=path.parent) as directory:
        directory = Path(directory)
        merged = directory / "merged.mp4"
        if detecter.check_video_and_audio_stream():
            v, a = directory / "video.mp4", directory / "audio.mp4"
            await download(streams[0].url, str(v), "视频流")
            await download(streams[1].url, str(a), "音频流")
            args = ["-i", str(v), "-i", str(a), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy"]
        else:
            stream = directory / "stream.flv"
            await download(streams[0].url, str(stream), "音视频流")
            args = ["-i", str(stream), "-c", "copy"]
        subprocess.run([FFMPEG_PATH, "-hide_banner", "-nostdin", "-loglevel", "error", *args, str(merged)], check=True)
        probe_video(merged, AlignConfig(ffmpeg=FFMPEG_PATH, ffprobe=FFPROBE_PATH))
        if overwrite:
            os.replace(merged, path)
        else:
            os.link(merged, path)
    print(f"已下载为：{path}")


async def download_ep_danmaku(ep: bangumi.Episode, out: str, overwrite=False) -> None:
    data, count = xml_2_json(await ep.get_danmaku_xml())
    atomic_json(out, data, overwrite=overwrite)
    print(f"已下载弹幕：{out}，共 {count} 条")


async def download_bangumi(media_id: int, cred: Credential, overwrite=False) -> None:
    output = Path(BASE_DIR) / "bangumi" / str(media_id)
    output.mkdir(parents=True, exist_ok=True)
    series = bangumi.Bangumi(media_id=media_id, credential=cred)
    for index, episode in enumerate(await series.get_episodes()):
        path = output / f"{index}.mp4"
        comments = output / f"dms_raw_{index}.json"
        if not overwrite and (path.exists() or comments.exists()):
            raise FileExistsError(f"Episode {index} already has output; use --overwrite")
        await download_episode(episode, str(path), overwrite)
        await download_ep_danmaku(episode, str(comments), overwrite)


async def _run(media_id, overwrite):
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    cred = None
    if Path(cookies_path).exists():
        with open(cookies_path, encoding="utf-8") as file:
            cookies = json.load(file)
        cred = Credential(sessdata=cookies.get("SESSDATA"), bili_jct=cookies.get("bili_jct"),
                          ac_time_value=cookies.get("ac_time_value"),
                          buvid3=cookies.get("buvid3"), buvid4=cookies.get("buvid4"), dedeuserid=cookies.get("DedeUserID"))
    if cred is None or not await cred.check_valid():
        cred = await login()
    elif await cred.check_refresh():
        await cred.refresh()
    atomic_json(cookies_path, cred.get_cookies(), overwrite=True, private=True)
    await download_bangumi(media_id, cred, overwrite)


def run_download(media_id, base_dir=None, ffmpeg=None, ffprobe=None, overwrite=False):
    global BASE_DIR, FFMPEG_PATH, FFPROBE_PATH, cookies_path
    if int(media_id) <= 0:
        raise ValueError("media_id must be positive")
    env = settings()
    BASE_DIR = str(base_dir or env.get("BASE_DIR", ROOT / "data"))
    FFMPEG_PATH = ffmpeg or env.get("FFMPEG_PATH", "ffmpeg")
    FFPROBE_PATH = ffprobe or env.get("FFPROBE_PATH", "ffprobe")
    cookies_path = str(Path(BASE_DIR) / "cookies.json")
    asyncio.run(_run(int(media_id), overwrite))


if __name__ == "__main__":
    run_download(int(input("输入番剧media_id:")))

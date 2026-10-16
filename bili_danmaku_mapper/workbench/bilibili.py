"""URL/catalog, in-memory QR login and source downloads on one async loop."""
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.parse import parse_qs, urlparse

from bilibili_api import Credential, HEADERS, bangumi, video, channel_series, get_client
from bilibili_api.utils.network import Api
from bilibili_api.utils.utils import get_api
import qrcode
import httpx

from ..io import atomic_json, read_json
from ..media import content_digest, probe_video
from .files import episode_number, inspect_file

ALLOWED_HOSTS = {"www.bilibili.com", "bilibili.com", "m.bilibili.com", "b23.tv"}


def danmaku_color(value):
    if isinstance(value,int) and not isinstance(value,bool) and 0<=value<=0xffffff:
        return f'#{value:06x}'
    if isinstance(value,str):
        match=re.fullmatch(r'#?([0-9a-fA-F]{1,6})',value)
        if match:return '#'+match[1].lower().zfill(6)
    # Special member styles have no ordinary RGB value in this API.
    return '#ffffff'


def parse_source(text):
    text = text.strip()
    short = re.fullmatch(r"(BV[0-9A-Za-z]{10}|av\d+|ep\d+|ss\d+|md\d+)", text, re.I)
    if short:
        name = short[1]
        return {"kind": name[:2].lower(), "value": name[2:] if name[:2].lower() != "bv" else "BV" + name[2:], "page": 1}
    match = re.search(r"https?://[^\s<>\"]+", text)
    url = match[0].rstrip("，。；!！)") if match else "https://" + text
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.port not in (None, 443):
        raise ValueError("请输入 B站视频、番剧或 b23.tv 分享链接")
    if parsed.hostname == "b23.tv":
        return {"kind": "short", "url": url}
    part = re.fullmatch(r"/(?:video/(BV[0-9A-Za-z]{10}|av\d+)|bangumi/(?:play/(ep\d+|ss\d+)|media/(md\d+)))/?", parsed.path, re.I)
    if not part:
        raise ValueError("首版不支持直播、收藏夹、空间或互动视频链接")
    value = next(v for v in part.groups() if v)
    page = int(parse_qs(parsed.query).get("p", ["1"])[0])
    if page < 1:
        raise ValueError("分P序号必须从 1 开始")
    return {"kind": value[:2].lower(), "value": "BV" + value[2:] if value[:2].lower() == "bv" else value[2:], "page": page}


async def expand_short(url):
    async with httpx.AsyncClient(follow_redirects=False, timeout=15) as client:
        current = url
        for _ in range(5):
            parsed = urlparse(current)
            if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.port not in (None, 443):
                raise ValueError("短链接跳转到了不支持的站点")
            response = await client.get(current)
            if response.status_code in (301, 302, 303, 307, 308):
                from urllib.parse import urljoin
                current = urljoin(current, response.headers["location"])
            else:
                response.raise_for_status()
                return current
    raise ValueError("短链接跳转次数过多")


class Auth:
    def __init__(self, cookies_path):
        self.cookies_path = Path(cookies_path)
        self.credential = Credential()
        self.loaded = False
        self.checked_at = 0
        self.qr_key = None
        self.qr_state = "logged_out"
        self.qr_created = 0
        self.lock = asyncio.Lock()

    async def current(self):
        async with self.lock:
            if not self.loaded or (self.credential.sessdata and time.monotonic()-self.checked_at > 600):
                self.loaded = True
                self.checked_at = time.monotonic()
                if self.cookies_path.is_file():
                    try:
                        self.credential = Credential.from_cookies(read_json(self.cookies_path))
                        if not await self.credential.check_valid():
                            self.credential = Credential()
                        elif self.credential.ac_time_value and await self.credential.check_refresh():
                            await self.credential.refresh()
                            self.save()
                    except (ValueError, OSError):
                        self.credential = Credential()
                    except Exception:
                        self.loaded = False
                        raise ValueError("暂时无法验证已有登录，请检查网络后重试") from None
            return self.credential

    def save(self):
        atomic_json(self.cookies_path, self.credential.get_cookies(), overwrite=True, private=True)

    async def status(self):
        cred = await self.current()
        return {"logged_in": bool(cred.sessdata), "qr_state": self.qr_state, "state": self.qr_state}

    async def generate_qr(self):
        # Mirrors the pinned SDK's WEB flow, without its shared /tmp/qrcode.png.
        api = get_api("login")["qrcode"]["web"]["get_qrcode_and_token"]
        data = await Api(credential=Credential(), **api).result
        self.qr_key, self.qr_state = data["qrcode_key"], "waiting_scan"
        import time
        self.qr_created = time.monotonic()
        image = qrcode.make(data["url"])
        output = io.BytesIO(); image.save(output, format="PNG")
        return {"image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(), "state": self.qr_state}

    async def poll_qr(self):
        if not self.qr_key:
            return await self.status()
        import time
        if time.monotonic() - self.qr_created > 180:
            self.qr_key, self.qr_state = None, "expired"
            return await self.status()
        api = get_api("login")["qrcode"]["web"]["get_events"]
        data = await Api(credential=Credential(), **api).update_params(qrcode_key=self.qr_key).result
        code = data["code"]
        if code == 86101:
            self.qr_state = "waiting_scan"
        elif code == 86090:
            self.qr_state = "waiting_confirm"
        elif code == 86038:
            self.qr_key, self.qr_state = None, "expired"
        elif code == 0:
            values = parse_qs(urlparse(data["url"]).query)
            self.credential = Credential(sessdata=values.get("SESSDATA", [None])[0],
                bili_jct=values.get("bili_jct", [None])[0], dedeuserid=values.get("DedeUserID", [None])[0],
                ac_time_value=data.get("refresh_token"))
            if not await self.credential.check_valid():
                raise ValueError("扫码完成但登录未验证，请刷新二维码")
            self.loaded, self.qr_key, self.qr_state = True, None, "logged_in"
            self.save()
        else:
            self.qr_state = "verification_required"
        return await self.status()


def safe_error(error):
    code = getattr(error, "code", None)
    if code in (-101, -111):
        return "需要重新扫码登录"
    if code in (-403, -10403, 10015002, 10015001):
        return "当前账号可能没有该影片的会员或地域访问权限"
    if code in (-412, -509):
        return "B站请求受限，请暂停后续任务并稍后重试"
    if isinstance(error, (ValueError, FileNotFoundError, FileExistsError)):
        return str(error)
    return "B站接口或网络暂时不可用，请稍后重试"


class BiliService:
    def __init__(self, auth, directory, config):
        self.auth, self.directory, self.config = auth, Path(directory), config

    async def _video_parts(self, bvid=None, aid=None, selected_page=1):
        cred = await self.auth.current()
        obj = video.Video(bvid=bvid, aid=aid, credential=cred)
        info, pages = await obj.get_info(), await obj.get_pages()
        if info.get("is_stein_gate"):
            raise ValueError("首版不支持互动视频")
        if selected_page > len(pages):
            raise ValueError("链接指定的分P不存在")
        bvid = info["bvid"]
        result = []
        for i, page in enumerate(pages):
            title = page.get("part") or info["title"]
            result.append({"key": f"video:{bvid}:{page['cid']}", "kind": "video", "bvid": bvid,
                           "aid": info["aid"], "cid": int(page["cid"]), "page_index": i,
                           "title": title if len(pages) > 1 else info["title"], "duration": page.get("duration", 0),
                           "number": i + 1 if len(pages) > 1 else episode_number(info["title"]),
                           "selected": i + 1 == selected_page, "focused": i + 1 == selected_page, "url": f"https://www.bilibili.com/video/{bvid}?p={i + 1}"})
        return info, result

    async def resolve(self, text, include_collection=True, _depth=0):
        if _depth > 3:
            raise ValueError("来源跳转次数过多")
        parsed = parse_source(text)
        if parsed["kind"] == "short":
            parsed = parse_source(await expand_short(parsed["url"]))
        cred = await self.auth.current()
        if parsed["kind"] in ("bv", "av"):
            info, parts = await self._video_parts(bvid=parsed["value"] if parsed["kind"] == "bv" else None,
                                                  aid=int(parsed["value"]) if parsed["kind"] == "av" else None,
                                                  selected_page=parsed["page"])
            if info.get("redirect_url"):
                return await self.resolve(info["redirect_url"], include_collection, _depth+1)
            collection = info.get("ugc_season")
            if include_collection and collection:
                series = channel_series.ChannelSeries(uid=info["owner"]["mid"], type_=channel_series.ChannelSeriesType.SEASON,
                                                       id_=collection["id"], credential=cred)
                archives, seen, page = [], set(), 1
                while True:
                    response = await series.get_videos(pn=page, ps=100)
                    batch = response.get("archives", [])
                    for item in batch:
                        if item["bvid"] not in seen:
                            archives.append(item); seen.add(item["bvid"])
                    total = response.get("page", {}).get("total")
                    if total is not None and len(archives) >= total:
                        break
                    if not batch:
                        if total is not None and len(archives) < total:
                            raise ValueError("合集成员未完整返回，请稍后重试")
                        break
                    if total is None and len(batch) < 100:
                        break
                    page += 1
                    if page > 100:
                        raise ValueError("合集过大，请改用单个 BV 链接")
                expanded = []
                for index, archive in enumerate(archives):
                    if archive["bvid"] == info["bvid"]:
                        members = parts
                    else:
                        _, members = await self._video_parts(bvid=archive["bvid"])
                        for m in members:
                            m["selected"] = False
                            m["focused"] = False
                    expanded.extend(members)
                if not expanded:
                    raise ValueError("合集列表为空，请稍后重试")
                parts = expanded
            return {"title": collection.get("title", info["title"]) if collection and include_collection else info["title"],
                    "kind": "collection" if collection and include_collection else "video", "items": parts}
        key = {"ep": "epid", "ss": "ssid", "md": "media_id"}[parsed["kind"]]
        series = bangumi.Bangumi(**{key: int(parsed["value"])}, credential=cred)
        listing = await series.get_episode_list()
        overview = await series.get_overview()
        metadata = {item.get("id"): item for item in overview.get("episodes", [])}
        sections = [listing.get("main_section", {})] + listing.get("section", [])
        parts, seen = [], set()
        for section in sections:
            for item in section.get("episodes", []):
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                item = {**metadata.get(item["id"], {}), **item}
                number = int(item["title"]) if str(item.get("title", "")).isdigit() else None
                parts.append({"key": f"episode:{item['id']}", "kind": "episode", "epid": item["id"],
                              "cid": item.get("cid"), "bvid": item.get("bvid"), "title": " ".join(
                                  str(x) for x in (item.get("title"), item.get("long_title")) if x),
                              "number": number, "duration": item.get("duration", 0) / 1000,
                              "selected": item["id"] == int(parsed["value"]) if parsed["kind"] == "ep" else number is not None,
                              "focused": parsed["kind"] == "ep" and item["id"] == int(parsed["value"]),
                              "url": f"https://www.bilibili.com/bangumi/play/ep{item['id']}"})
        meta = await series.get_meta()
        return {"title": meta.get("media", meta).get("title", "番剧 / 影视"), "kind": "bangumi", "items": parts}

    async def _request_object(self, source, quality=64):
        cred = await self.auth.current()
        if source["kind"] == "episode":
            service = self
            class ReferenceEpisode(bangumi.Episode):
                playback = None
                async def get_download_url(inner):
                    if inner.playback is None:
                        inner.playback = await service._download_info(source, quality)
                    return inner.playback
            return ReferenceEpisode(epid=source["epid"], credential=cred)
        return video.Video(bvid=source["bvid"], credential=cred)

    async def _download_info(self, source, quality):
        credential = await self.auth.current()
        params = {"qn": quality, "fnval": 16, "fnver": 0, "fourk": 0, "otype": "json", "prefer_codec_type": 7}
        if source["kind"] == "episode":
            api = get_api("bangumi")["info"]["playurl"]
            params["ep_id"] = source["epid"]
            return await Api(**api, credential=credential).update_params(**params).result
        api = get_api("video")["info"]["playurl"]
        params.update(bvid=source["bvid"], cid=source["cid"], avid=source.get("aid"))
        return await Api(**api, credential=credential, wbi=True).update_params(**params).result

    async def _bytes(self, url, destination, notify):
        written = 0
        # Own the response context so stop/cancel releases the connection immediately.
        async with httpx.AsyncClient(headers=dict(HEADERS), follow_redirects=True, max_redirects=5, timeout=30) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with open(destination, "wb") as file:
                    async for block in response.aiter_bytes(1024*1024):
                        written += file.write(block)
                        notify("downloading", written, total or None)
        if not written or (total and written != total):
            raise ValueError("参考视频下载不完整")

    async def download(self, source, job_directory, quality, notify):
        directory = Path(job_directory); directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{source['key']}:{quality}:avc".encode()).hexdigest()
        cache = self.directory / key
        cache.mkdir(parents=True, exist_ok=True)
        destination, manifest = cache / "source.mp4", cache / "manifest.json"
        obj = await self._request_object(source, quality)
        valid = False
        if destination.is_file() and manifest.is_file():
            try:
                valid = read_json(manifest).get("sha256") == await asyncio.to_thread(content_digest, destination)
            except (OSError, ValueError):
                pass
        if not valid:
            data = await self._download_info(source, quality)
            if source["kind"] == "episode":
                obj.playback = data
            data = data.get("video_info", data)
            if data.get("code", 0) != 0:
                raise ValueError("当前账号无法获取该影片，请检查登录、会员或地域权限")
            if data.get("is_drm"):
                raise ValueError("该参考流受 DRM 保护，首版不支持")
            if data.get("is_preview"):
                raise ValueError("当前账号只能获取试看片段，请检查登录、会员或地域权限")
            with tempfile.TemporaryDirectory(prefix=".source-", dir=cache) as tmp:
                tmp = Path(tmp); output = tmp / "merged.mp4"
                inputs = []
                if data.get("dash"):
                    choices = [s for s in data["dash"]["video"] if s.get("codecs", "").startswith("avc") and s["id"] <= quality]
                    if not choices:
                        raise ValueError("当前没有可访问的 H.264 参考视频，请检查权限或选择其他清晰度")
                    chosen = max(choices, key=lambda s: s["id"])
                    if chosen["id"] < quality:
                        notify("downloading", note="已选择可访问的较低清晰度")
                    await self._bytes(chosen.get("baseUrl") or chosen["base_url"], tmp / "video.mp4", notify)
                    inputs = ["-i", str(tmp / "video.mp4")]
                    audios = [s for s in data["dash"].get("audio", []) if s["id"] in (30216, 30232, 30280)]
                    if audios:
                        chosen_audio = max(audios, key=lambda s: s["id"])
                        await self._bytes(chosen_audio.get("baseUrl") or chosen_audio["base_url"], tmp / "audio.mp4", notify)
                        inputs += ["-i", str(tmp / "audio.mp4"), "-map", "0:v:0", "-map", "1:a:0"]
                    else:
                        inputs += ["-map", "0:v:0"]
                else:
                    if data.get("quality", 0) > quality:
                        raise ValueError("服务返回的清晰度超过所选上限，未下载")
                    segments = data.get("durl", [])
                    if not segments:
                        raise ValueError("没有可访问的下载流")
                    files = []
                    for i, segment in enumerate(segments):
                        file = tmp / f"part-{i}.bin"
                        await self._bytes(segment["url"], file, notify); files.append(file)
                    if len(files) == 1:
                        inputs = ["-i", str(files[0])]
                    else:
                        listing = tmp / "concat.txt"
                        listing.write_text("\n".join(f"file '{p.name}'" for p in files))
                        inputs = ["-f", "concat", "-safe", "1", "-i", str(listing)]
                notify("merging")
                process = await asyncio.create_subprocess_exec(self.config.ffmpeg, "-v", "error", "-nostdin", *inputs,
                    "-c", "copy", str(output), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
                try:
                    _, error = await process.communicate()
                    if process.returncode:
                        raise ValueError("参考视频合流失败")
                finally:
                    if process.returncode is None:
                        process.kill(); await process.wait()
                notify("verifying")
                info = await asyncio.to_thread(probe_video, output, self.config)
                codecs = await asyncio.to_thread(inspect_file, output, self.config.ffprobe)
                if codecs["video_codec"] != "h264":
                    raise ValueError("参考视频不是 H.264 编码，未发布为有效来源")
                if source.get("duration", 0) > 30 and info.duration < source["duration"] * .8:
                    raise ValueError("下载内容可能只有试看片段，未作为完整参考视频使用")
                import os
                os.replace(output, destination)
                atomic_json(manifest, {"source_key": source["key"], "sha256": await asyncio.to_thread(content_digest, destination),
                                       "duration": info.duration, "quality_limit": quality}, overwrite=True)
        notify("danmaku")
        comments = await obj.get_danmakus() if source["kind"] == "episode" else await obj.get_danmakus(cid=source["cid"])
        items, unsupported = [], []
        for dm in comments:
            mode = dm.mode.value if hasattr(dm.mode, "value") else dm.mode
            item = {"time": float(dm.dm_time), "text": dm.text, "author": dm.crc32_id,
                    "color": danmaku_color(dm.color), "type": {4: "bottom", 5: "top"}.get(mode, "scroll")}
            if mode in (1, 2, 3, 4, 5, 6):
                items.append(item)
            else:
                unsupported.append({**item, "original_mode": mode})
        items.sort(key=lambda i: i["time"])
        raw = directory / "raw.json"
        atomic_json(raw, {"code": 1, "data": items}, overwrite=True)
        atomic_json(directory / "danmaku-meta.json", {"fetched_at": datetime.now(timezone.utc).isoformat(),
                    "scope": "current_accessible_segmented_danmaku", "unsupported": unsupported}, overwrite=True)
        return {"source_video": str(destination), "raw_danmaku": str(raw)}

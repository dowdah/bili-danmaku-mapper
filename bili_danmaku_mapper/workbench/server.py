"""Loopback-only workbench; file capabilities and per-launch session auth."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import inspect
import json
import mimetypes
from pathlib import Path
import secrets
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..config import AlignConfig, ROOT, settings
from .bilibili import Auth, BiliService, safe_error
from .files import inspect_file, native_pick, suggest_pairs, stop_pickers
from .jobs import JobManager
from .state import Store
from .playback import PlaybackLibrary, inspect_danmaku
from .subtitles import inspect_subtitles, read_subtitles
from .playback_prepare import PlaybackPreparer
from .segments import SegmentManager
from .native import NativeManager


class LinkInput(BaseModel):
    url: str = Field(min_length=2, max_length=2048)

class PickInput(BaseModel):
    kind: str = 'files'

class PairInput(BaseModel):
    source_keys: list[str] = Field(max_length=1000)
    file_ids: list[str] = Field(max_length=1000)
    offset: int = Field(default=0, ge=-10000, le=10000)

class Options(BaseModel):
    mode: str = 'strict'
    coverage_recovery: bool = False
    quality: int = 64
    audio_stream: int | None = Field(default=None, ge=0)
    convert: bool = False  # Legacy input only; alignment never prepares a playback copy.

class JobInput(BaseModel):
    pairs: list[dict[str,str]] = Field(min_length=1, max_length=1000)
    options: Options = Options()

class AnchorInput(BaseModel):
    anchors: list[dict[str,float]] = Field(max_length=10000)
    file_id: str | None = None
    convert: bool | None = None


class PlaybackImport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    video_id: str
    danmaku_id: str | None = None
    item_id: str | None = None

class PlaybackState(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    position: float = Field(default=0, ge=0)
    volume: float = Field(default=1, ge=0, le=1)
    muted: bool = False
    playback_rate: float = Field(default=1, ge=.25, le=4)
    danmaku_enabled: bool = True
    font_size: int = Field(default=24, ge=12, le=48)
    opacity: float = Field(default=1, ge=0, le=1)
    show_estimated: bool = False
    danmaku_offset: float = Field(default=0, ge=-5, le=5)
    subtitle_id: str | None = None
    subtitle_enabled: bool = True
    subtitle_offset: float = Field(default=0, ge=-5, le=5)
    playback_audio_stream: int = Field(default=0, ge=0)
    preferred_engine: str = Field(default='web', pattern='^(web|iina)$')
    write_revision: int | None = Field(default=None, ge=0, le=9007199254740991)


class NativeSessionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    item_id: str
    subtitle_id: str | None = None


class PlaybackFeedback(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    position: float = Field(ge=0)
    kind: str = Field(pattern='^(early|late|mismatch)$')

class SegmentSessionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    item_id: str
    audio_stream: int = Field(default=0, ge=0)
    audio_copy: bool = False

class SegmentWindowInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    position: float = Field(ge=0)
    generation: int = Field(ge=0)

class PlaybackPrepareInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    strategy: str
    audio_stream: int = Field(default=0, ge=0)
    confirmed: bool = False


def public_job(job):
    return {k:v for k,v in job.items() if k not in ('checkpoint',)}


def create_app(directory=None, token=None, config=None, downloader=None, compute=None, picker=None, exit_callback=None):
    directory = Path(directory or ROOT/'data'/'workbench')
    token = token or secrets.token_urlsafe(32)
    config = config or AlignConfig.resolve()
    store = Store(directory)
    library = PlaybackLibrary(store)
    preparer = PlaybackPreparer(store, library, config)
    segments = SegmentManager(store, library, config)
    native = NativeManager(store, library, config)
    auth = Auth(Path(settings().get('BASE_DIR',ROOT/'data'))/'cookies.json')
    downloader = downloader or BiliService(auth, directory/'sources', config)
    manager = JobManager(store, downloader, config, compute)
    manager.playback_busy = lambda: segments.busy
    picker = picker or native_pick
    ui_root = ROOT if (ROOT/'web').is_dir() else Path(sys.prefix)/'share'/'bili-danmaku-mapper'
    assets = ui_root/'web'

    @asynccontextmanager
    async def lifespan(app):
        await native.start()
        await segments.start()
        await manager.start()
        try:
            yield
        finally:
            await native.close()
            await stop_pickers()
            await segments.close()
            await preparer.close()
            await manager.close()
            store.close()
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.manager, app.state.token = store, manager, token
    app.state.preparer = preparer
    app.state.segments = segments
    app.state.native = native

    @app.middleware('http')
    async def protect(request, call_next):
        host = request.headers.get('host','')
        expected = f'127.0.0.1:{request.url.port}' if request.url.port else '127.0.0.1'
        if host != expected:
            return JSONResponse({'detail':'Invalid local host'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != f'http://{expected}':
            return JSONResponse({'detail':'Invalid origin'}, status_code=403)
        if request.url.path.startswith('/api/'):
            bearer = request.headers.get('authorization','').removeprefix('Bearer ')
            cookie = request.cookies.get('bdm_session_'+str(request.url.port),'')
            if not (hmac.compare_digest(bearer, token) or hmac.compare_digest(cookie, token)):
                return JSONResponse({'detail':'请通过启动脚本打开工作台'}, status_code=401)
        try:
            response = await call_next(request)
        except Exception as error:
            response = JSONResponse({"detail":safe_error(error)}, status_code=400)
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.exception_handler(Exception)
    async def failure(request, error):
        return JSONResponse({'detail':safe_error(error)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({'detail':'文件或任务不存在'}, status_code=404)

    @app.get('/health')
    async def health():
        return {'service':'bili-danmaku-workbench', 'version':1,
                'alignment_target':'original', 'playback_disk_guard':True,
                'segmented_playback':True, 'watch_privacy':True,
                'coverage_recovery_experimental':True}

    @app.post('/api/session')
    async def session(request: Request):
        result = JSONResponse({'ok':True})
        result.set_cookie('bdm_session_'+str(request.url.port), token, httponly=True, samesite='strict')
        return result

    @app.get('/')
    async def index():
        return FileResponse(assets/'index.html')

    @app.get('/assets/{name}')
    async def asset(name: str):
        if name not in ('app.js','style.css','player-session.js','watch.js','watch.css','segment-player.js','subtitle-renderer.js','danmaku-renderer.js','danmaku-renderer.css'):
            raise HTTPException(404)
        return FileResponse((ui_root if name.startswith('danmaku-renderer.') else assets)/name)

    @app.get('/assets/vendor/nplayer-1.0.15/{name}')
    async def nplayer_asset(name: str):
        if name not in ('index.min.js','LICENSE','provenance.json'):
            raise HTTPException(404)
        return FileResponse(assets/'vendor'/'nplayer-1.0.15'/name)

    @app.get('/player/{name}')
    async def player(name: str):
        if name not in ('player.html','player.js','player.css','player-session.js','danmaku-renderer.js','danmaku-renderer.css'):
            raise HTTPException(404)
        return FileResponse((assets if name == 'player-session.js' else ui_root)/name)

    @app.get('/api/status')
    async def status():
        return {'paused':manager.paused,'jobs':[public_job(j) for j in store.jobs()]}

    @app.get('/api/auth')
    async def auth_status():
        return await auth.status()

    @app.post('/api/auth/qr')
    async def qr():
        return await auth.generate_qr()

    @app.get('/api/auth/qr')
    async def poll_qr():
        return await auth.poll_qr()

    @app.post('/api/resolve')
    async def resolve(body: LinkInput):
        result = await downloader.resolve(body.url)
        entries = store.setting('catalog', {})
        entries.update({s['key']:s for s in result['items']})
        store.set_setting('catalog', entries)
        return result

    picker_lock = asyncio.Lock()

    @app.post('/api/pick')
    async def pick(body: PickInput):
        async with picker_lock:
            paths = await picker(body.kind) if inspect.iscoroutinefunction(picker) else await asyncio.to_thread(picker, body.kind)
        if body.kind not in ('files','directory','playback_video','danmaku','subtitle'):
            raise ValueError('不支持的文件选择类型')
        if body.kind in ('playback_video','danmaku','subtitle') and len(paths) > 1:
            raise ValueError('一次请选择一个文件')
        files, errors = [], []
        for path in paths:
            try:
                if body.kind == 'subtitle':
                    info = await asyncio.to_thread(inspect_subtitles, path)
                else:
                    info = await asyncio.to_thread(inspect_danmaku, path) if body.kind == 'danmaku' else await asyncio.to_thread(inspect_file, path, config.ffprobe)
                kind = body.kind if body.kind in ('danmaku','playback_video','subtitle') else 'selected'
                files.append(store.register_file(path, kind, info))
            except Exception as error:
                reason = str(error) if body.kind in ('danmaku','subtitle') and isinstance(error, ValueError) else '无法读取视频，请检查文件和 FFmpeg'
                errors.append({'name':Path(path).name,'reason':reason})
        return {'files':files,'errors':errors}

    def sources(keys):
        catalog = store.setting('catalog', {})
        return [catalog[key] for key in keys]

    @app.post('/api/pairs')
    async def pairs(body: PairInput):
        files = [{**store.file(k)['info'],'id':k} for k in body.file_ids]
        return suggest_pairs(sources(body.source_keys), files, body.offset)

    def validate_options(options, file):
        if options.mode not in ('strict','lenient') or options.quality not in (16,32,64):
            raise ValueError('模式或清晰度无效')
        if file['kind'] != 'selected':
            raise ValueError('请从系统对话框选择目标影片')
        streams = file['info']['audio_streams']
        if options.audio_stream is not None and options.audio_stream >= len(streams):
            raise ValueError('音轨序号超出范围')
    @app.post('/api/jobs')
    async def create_jobs(body: JobInput):
        if len({p['source_key'] for p in body.pairs}) != len(body.pairs) or len({p['file_id'] for p in body.pairs}) != len(body.pairs):
            raise ValueError('本次集数与本地文件必须一一配对')
        prepared = []
        for pair in body.pairs:
            source = sources([pair['source_key']])[0]
            target = store.file(pair['file_id'])
            validate_options(body.options, target)
            prepared.append((source,target))
        result = [store.create_job(source,target['id'],body.options.model_dump()) for source,target in prepared]
        for job in result:
            manager.publish(job)
        return [public_job(j) for j in result]

    @app.get('/api/jobs/{key}')
    async def read_job(key: str):
        return public_job(store.job(key))

    @app.post('/api/jobs/{key}/{action}')
    async def control(key: str, action: str):
        if action == 'stop':
            await manager.stop(key)
        elif action == 'retry':
            manager.retry(key)
        else:
            raise HTTPException(404)
        return public_job(store.job(key))

    @app.post('/api/revise/{key}')
    async def revise(key: str, body: AnchorInput):
        from ..alignment import validate_manual_anchors
        previous = store.job(key)
        target = store.file(body.file_id or previous['target_id'])
        options = Options(**{**previous['options'], **({'convert':body.convert} if body.convert is not None else {})})
        validate_options(options, target)
        source_duration = previous['source']['duration']
        if manager.ready(previous):
            from ..media import probe_video
            source_duration = (await asyncio.to_thread(probe_video, previous['checkpoint']['source_video'], config)).duration
        validate_manual_anchors(body.anchors, source_duration, target['info']['duration'])
        job = store.create_job(previous['source'], target['id'], options.model_dump(), body.anchors, key)
        if manager.ready(previous):
            checkpoint = {k:v for k,v in previous['checkpoint'].items() if k != 'playback'}
            job = store.update(job['id'], checkpoint=checkpoint)
        manager.publish(job)
        return public_job(job)

    @app.post('/api/pipeline/{action}')
    async def pipeline(action: str):
        if action not in ('pause','resume'):
            raise HTTPException(404)
        manager.pause(action == 'pause')
        return {'paused':manager.paused}

    @app.get('/api/playback/items')
    async def playback_items():
        return {'items':library.items(), 'last_item_id':store.setting('playback_last_item')}

    @app.post('/api/playback/items')
    async def playback_import(body: PlaybackImport):
        return await asyncio.to_thread(library.import_item, body.video_id, body.danmaku_id, body.item_id)

    @app.patch('/api/playback/items/{key}/state')
    async def playback_state(key: str, body: PlaybackState):
        values=body.model_dump(exclude_unset=True)
        if key in library.native_owners and 'subtitle_id' in values and not values['subtitle_id']:
            raise ValueError('原音质观看必须保留外挂字幕，可替换字幕或先返回网页')
        saved=library.save_state(key, values)
        await native.apply(key,saved,values)
        return saved

    @app.get('/api/playback/native/capabilities')
    async def native_capabilities():
        return native.capabilities()

    @app.get('/api/playback/native/items/{key}')
    async def native_item(key: str):
        library.item(key)
        return {'session':native.item_session(key),'subtitle_suggestion':native.subtitle_suggestion(key)}

    @app.post('/api/playback/native/sessions')
    async def native_open(body: NativeSessionInput):
        return await native.create(body.item_id,body.subtitle_id)

    @app.get('/api/playback/native/sessions/{key}')
    async def native_status(key: str):
        return native.public(native.sessions[key])

    @app.delete('/api/playback/native/sessions/{key}')
    async def native_end(key: str):
        return await native.end(key)

    @app.post('/api/playback/items/{key}/feedback')
    async def playback_feedback(key: str, body: PlaybackFeedback):
        return library.feedback(key, body.position, body.kind)

    @app.post('/api/playback/sessions')
    async def segment_session(body: SegmentSessionInput):
        return await segments.create(body.item_id, body.audio_stream, body.audio_copy)

    @app.post('/api/playback/sessions/{sid}/windows')
    async def segment_window(sid: str, body: SegmentWindowInput):
        return await segments.request(sid, body.position, body.generation)

    @app.get('/api/playback/sessions/{sid}/windows/{wid}')
    async def segment_manifest(sid: str, wid: str):
        return segments.public(sid, wid)

    @app.get('/api/playback/sessions/{sid}/windows/{wid}/parts/{name}')
    async def segment_fragment(sid: str, wid: str, name: str):
        return FileResponse(segments.fragment(sid, wid, name), media_type='video/mp4')

    @app.post('/api/playback/sessions/{sid}/heartbeat')
    async def segment_heartbeat(sid: str):
        segments.get(sid)
        return {'ok':True}

    @app.post('/api/playback/sessions/{sid}/pause')
    async def segment_pause(sid: str):
        return await segments.pause(sid)

    @app.delete('/api/playback/sessions/{sid}')
    async def segment_close(sid: str):
        return await segments.delete(sid)

    @app.get('/api/playback/items/{key}/media')
    async def playback_media_info(key: str):
        item = library.item(key)
        file = store.file(item['video_id'])
        return await asyncio.to_thread(inspect_file, file['path'], config.ffprobe)

    @app.get('/api/playback/items/{key}/prepared')
    async def playback_prepared(key: str, audio_stream: int = 0):
        return preparer.cached(key, audio_stream)

    @app.post('/api/playback/items/{key}/prepare')
    async def playback_prepare(key: str, body: PlaybackPrepareInput):
        return await preparer.prepare(key, body.strategy, body.audio_stream, body.confirmed)

    @app.get('/api/playback/preparations/{key}')
    async def playback_preparation(key: str):
        return preparer.public(preparer.read(key))

    @app.get('/api/playback/subtitles/{key}')
    async def subtitle_data(key: str):
        file = store.file(key)
        if file['kind'] != 'subtitle':
            raise ValueError('该文件未作为字幕授权')
        return await asyncio.to_thread(read_subtitles, file['path'])

    @app.get('/api/files/{key}')
    async def media(key: str):
        file = store.file(key)
        return FileResponse(file['path'], media_type=mimetypes.guess_type(file['path'])[0] or 'application/octet-stream')

    @app.get('/api/events')
    async def events(request: Request):
        async def stream():
            queue = asyncio.Queue(maxsize=64)
            manager.events.add(queue)
            try:
                yield 'event: snapshot\ndata: '+json.dumps([public_job(j) for j in store.jobs()])+'\n\n'
                while not manager.closed and not await request.is_disconnected():
                    try:
                        job = await asyncio.wait_for(queue.get(), 15)
                        if job is None:
                            yield 'event: shutdown\ndata: {}\n\n'
                            break
                        yield 'event: job\ndata: '+json.dumps(public_job(job))+'\n\n'
                    except asyncio.TimeoutError:
                        yield ': keepalive\n\n'
            finally:
                manager.events.discard(queue)
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'X-Accel-Buffering':'no'})

    @app.post('/api/exit')
    async def exit_service():
        await native.close()
        await segments.close()
        await preparer.close()
        await stop_pickers()
        await manager.close()
        if exit_callback:
            asyncio.get_running_loop().call_later(.2, exit_callback)
        return {'stopped':True}
    return app

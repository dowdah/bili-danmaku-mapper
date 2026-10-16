"""Owned IINA sessions: original media, original subtitles, separate Lua OSD."""
import asyncio
import contextlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
import sys
import httpx

from ..cache import stamp
from ..io import atomic_json, read_json
from .playback import inspect_danmaku
from .subtitles import inspect_subtitles
from .segments import command_json


async def ipc(path, commands, timeout=4):
    """A short private Unix connection; never expose arbitrary RPC over HTTP."""
    reader,writer=await asyncio.wait_for(asyncio.open_unix_connection(str(path),limit=2*1024**2),timeout)
    try:
        for i,command in enumerate(commands):
            writer.write((json.dumps({'command':command,'request_id':i},ensure_ascii=False)+'\n').encode())
        await writer.drain();results={}
        async with asyncio.timeout(timeout):
            while len(results)<len(commands):
                line=await reader.readline()
                if not line:raise ValueError('IINA 控制连接已关闭')
                value=json.loads(line)
                if value.get('request_id') in range(len(commands)):results[value['request_id']]=value
        return [results[i] for i in range(len(commands))]
    finally:
        writer.close();await writer.wait_closed()


def native_items(data,report=None):
    """Mirror web status association, including duplicate timestamps in order."""
    from collections import defaultdict,deque
    states=defaultdict(deque)
    for e in (report or {}).get('entries',[]):
        t=e.get('target_time')
        if isinstance(t,(int,float)) and math.isfinite(t):states[f'{t:.6f}'].append(e.get('status','reliable'))
    items=[]
    for item in data['data']:
        queue=states[f"{item['time']:.6f}"]
        items.append({**item,'status':queue.popleft() if queue else item.get('status','reliable')})
    return {'data':sorted(items,key=lambda i:i['time'])}


class NativeManager:
    def __init__(self,store,library,config,launch=None):
        self.store,self.library,self.config=store,library,config
        self.root=store.directory/'native-v1';self.root.mkdir(mode=0o700,exist_ok=True)
        if self.root.resolve()!=self.root:raise ValueError('原生会话目录不能是链接')
        self.sessions={};self.processes={};self.sampling={};self.monitor=None;self.closed=False;self.lock=asyncio.Lock()
        self.launch=launch or self._launch

    def executable(self):
        for app in (Path('/Applications/IINA.app'),Path.home()/'Applications/IINA.app'):
            path=app/'Contents/MacOS/IINA'
            if path.is_file():return path
        return None

    def capabilities(self):
        return {'available':bool(self.executable()),'requires_subtitle':True,'danmaku_engine':'mpv_lua_osd','lossy_transcode':False}

    async def _launch(self,args):
        with open(os.devnull,'wb') as log:
            p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost'})
        return p

    async def _start_media(self,s,source,directory):
        config=directory/'media-source.json'
        s['media_token']=secrets.token_hex(24)
        atomic_json(config,{'source':source,'identity':s['identity'],'token':s['media_token']},True)
        with open(os.devnull,'wb') as log:
            worker=subprocess.Popen([sys.executable,'-m','bili_danmaku_mapper.workbench.native_media',str(config)],cwd=Path(__file__).parents[2],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        self.processes[s['id']+':media']=worker;s['media_pid']=worker.pid;self.save(s)
        for _ in range(100):
            try:
                ready=read_json(directory/'media-ready.json')
                if ready['pid']==worker.pid:
                    url=f"http://127.0.0.1:{ready['port']}/{s['media_token']}/media"
                    async with httpx.AsyncClient(trust_env=False) as client:
                        response=await client.head(url,timeout=1)
                        if response.status_code==200:return url
            except (OSError,ValueError,httpx.HTTPError):pass
            if worker.poll() is not None:break
            await asyncio.sleep(.05)
        worker.terminate();raise ValueError('本机原片读取接口未能启动')

    async def _stop_media(self,s):
        url=s.get('media_url','')
        if url.startswith('http://127.0.0.1:'):
            with contextlib.suppress(httpx.HTTPError):
                async with httpx.AsyncClient(trust_env=False) as client:await client.post(url.rsplit('/',1)[0]+'/stop',timeout=2)
        process=self.processes.get(s['id']+':media')
        if process:
            with contextlib.suppress(subprocess.TimeoutExpired):await asyncio.to_thread(process.wait,2)

    def save(self,s):
        atomic_json(self.root/s['id']/'session.json',s,True)

    def public(self,s):
        return {k:s.get(k) for k in ('id','item_id','status','error','video_ready','subtitle_ready','danmaku_ready','comment_count','subtitle_name','source_audio','output_audio','render_sync','state')}

    def item_session(self,item_id):
        return next((self.public(s) for s in self.sessions.values() if s['item_id']==item_id and s['status'] not in ('closed',)),None)

    def subtitle_suggestion(self,item_id):
        try:
            item=self.library.item(item_id);directory=Path(self.store.file(item['video_id'])['path']).parent
            candidates=[]
            for index,path in enumerate(directory.iterdir()):
                if index>=2000:break
                if path.is_file() and path.suffix.lower() in ('.ass','.ssa','.srt','.vtt') and any(x in path.name.lower() for x in ('.zh.','.chs.','简体','中文')):candidates.append(path.name)
            return sorted(candidates)[0] if len(candidates)==1 else None
        except OSError:return None

    async def start(self):
        for path in self.root.glob('*/session.json'):
            try:
                s=read_json(path)
                if s['status'] in ('closed','failed'):continue
                self._validate_owned(s)
                reply=await ipc(s['socket'],[['get_property','path']],1)
                if reply[0].get('data')!=s.get('media_url',self.store.file(s['video_id'])['path']):continue
                self.sessions[s['id']]=s
                self.library.native_owners[s['item_id']]=s['id']
                # Reconcile sessions kept alive across a workbench upgrade.
                # display-vdrop leaves the audio clock and samples unchanged.
                await ipc(s['socket'],[['set_property','video-sync','display-vdrop']],1)
                await self.sample(s,True)
                if not s['danmaku_ready']:raise ValueError('无法确认本工具的弹幕会话')
                s['status']='ready' if s['subtitle_ready'] else 'needs_attention'
            except Exception:
                if 's' in locals() and isinstance(s,dict):
                    self.sessions.pop(s.get('id'),None);self.library.native_owners.pop(s.get('item_id'),None)
                continue
        async def monitor():
            while not self.closed:
                for s in list(self.sessions.values()):
                    if s['status'] not in ('ready','needs_attention'):continue
                    try:await self.sample(s)
                    except (OSError,ValueError,asyncio.TimeoutError):
                        s.update(status='closed',error='IINA 会话已结束或控制连接断开');self.library.native_owners.pop(s['item_id'],None);self.save(s)
                await asyncio.sleep(1)
        self.monitor=asyncio.create_task(monitor())

    def _validate_owned(self,s):
        parent=Path(s['socket']).parent
        if parent.parent.resolve()!=Path(tempfile.gettempdir()).resolve() or not parent.name.startswith('bdm-native-') or parent.is_symlink():raise ValueError('会话路径无效')
        st=parent.stat()
        if st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o700:raise ValueError('会话目录不安全')
        os.kill(s['pid'],0)
        source=self.store.file(s['video_id']);subtitle=self.store.file(s['subtitle_id'])
        if list(stamp(source['path']))[:4]!=s['identity'][:4] or list(stamp(subtitle['path']))[:4]!=s['subtitle_identity'][:4]:raise ValueError('会话输入已变化')

    async def create(self,item_id,subtitle_id):
        async with self.lock:
            current=self.item_session(item_id)
            if current and current['status'] not in ('failed','closed'):return current
            for other in list(self.sessions.values()):
                if other['item_id']!=item_id and other['status'] not in ('closed','failed'):
                    await self.end(other['id'])
            item=self.library.item(item_id);state=self.library.state(item_id)
            subtitle_id=subtitle_id or state.get('subtitle_id')
            if not subtitle_id:raise ValueError('本片必须先选择并加载中文外挂字幕，再进行原音质观看')
            source=self.store.file(item['video_id']);subtitle=self.store.file(subtitle_id)
            if subtitle['kind']!='subtitle':raise ValueError('请通过系统对话框选择外挂字幕')
            info=await asyncio.to_thread(inspect_subtitles,subtitle['path'])
            if not info['count']:raise ValueError('外挂字幕没有可用对白，未启动无字幕播放')
            executable=self.executable()
            if not executable:raise ValueError('未找到 IINA，请安装后重试；网页播放器仍可使用')
            media=await command_json([self.config.ffprobe,'-v','error','-show_streams','-of','json',source['path']])
            audios=[x for x in media['streams'] if x['codec_type']=='audio'];ordinal=state.get('playback_audio_stream',0)
            if audios and ordinal>=len(audios):raise ValueError('所选音轨已不存在')
            audio=audios[ordinal] if audios else None
            comments={'data':[]}
            if item.get('danmaku_id'):
                path=self.store.file(item['danmaku_id'])['path'];await asyncio.to_thread(inspect_danmaku,path)
                report=read_json(self.store.file(item['report_id'])['path']) if item.get('report_id') else None
                comments=native_items(read_json(path),report)
            key=uuid.uuid4().hex;directory=self.root/key;directory.mkdir(mode=0o700)
            atomic_json(directory/'comments.json',comments,True)
            private=Path(tempfile.mkdtemp(prefix='bdm-native-'));os.chmod(private,0o700)
            s=dict(id=key,item_id=item_id,video_id=item['video_id'],subtitle_id=subtitle_id,identity=list(stamp(source['path'])),
                   subtitle_identity=list(stamp(subtitle['path'])),socket=str(private/'ipc'),token=secrets.token_hex(16),status='starting',error=None,
                   video_ready=False,subtitle_ready=False,danmaku_ready=False,comment_count=len(comments['data']),subtitle_name=Path(subtitle['path']).name,
                   audio_streams=audios,source_audio={k:audio.get(k) for k in ('codec_name','profile','channels','channel_layout','sample_rate')} if audio else None,output_audio=None,state=state,last_saved=0)
            self.sessions[key]=s
            try:s['media_url']=await self._start_media(s,source['path'],directory)
            except Exception as error:
                s.update(status='failed',error=str(error));self.save(s);raise
            args=[str(executable),'-recordRecentFiles','NO','-recordPlaybackHistory','NO','-enableThumbnailPreview','NO','-resumeLastPosition','NO','-playlistAutoAdd','NO','-pauseWhenOpen','YES','--no-stdin','--mpv-input-ipc-server='+s['socket'],'--mpv-pause=yes','--mpv-resume-playback=no',
                  '--mpv-sub-auto=no','--mpv-video-sync=display-vdrop','--mpv-audio-spdif=','--mpv-audio-channels=auto-safe','--mpv-rebase-start-time=yes','--mpv-force-media-title='+Path(source['path']).name,s['media_url']]
            try:
                p=await self.launch(args);self.processes[key]=p;s['pid']=p.pid;self.save(s)
                for _ in range(150):
                    try:
                        reply=await ipc(s['socket'],[['get_property','path'],['get_property','track-list']],.5)
                        if reply[0].get('data')==s['media_url'] and reply[1].get('data'):break
                    except (OSError,ValueError,asyncio.TimeoutError):pass
                    await asyncio.sleep(.1)
                else:raise ValueError('IINA 启动或媒体加载超时，未将会话标记为就绪')
                os.chmod(s['socket'],0o600)
                commands=[['set_property','pause',True],['set_property','volume',state['volume']*100],['set_property','mute',state['muted']],
                          ['set_property','speed',state['playback_rate']],['set_property','sub-ass-override','no'],['set_property','sub-delay',state.get('subtitle_offset',0)],
                          ['set_property','sub-visibility',True],['sub-add',subtitle['path'],'select']]
                tracks=reply[1]['data']
                if audio:
                    track=next((t for t in tracks if t['type']=='audio' and t.get('ff-index')==audio['index']),None)
                    if not track:raise ValueError('无法确认原片音轨，未静默切换其他语言')
                    s['audio_tracks']={str(i):t['id'] for i,a in enumerate(audios) for t in tracks if t['type']=='audio' and t.get('ff-index')==a['index']}
                    commands.append(['set_property','aid',track['id']])
                commands += [['seek',state['position'],'absolute+exact'],['load-script',str(Path(__file__).with_name('native_danmaku.lua'))]]
                result=await ipc(s['socket'],commands)
                if any(r.get('error')!='success' for r in result):raise ValueError('IINA 字幕、弹幕脚本或播放设置加载失败')
                self._configuration(s,state)
                result=await ipc(s['socket'],[['script-message-to','native_danmaku','configure',str(directory/'config.json')]])
                for _ in range(40):
                    await ipc(s['socket'],[['script-message-to','native_danmaku','configure',str(directory/'config.json')]])
                    await self.sample(s,initial=True)
                    if s['video_ready'] and s['subtitle_ready'] and s['danmaku_ready']:break
                    await asyncio.sleep(.1)
                else:raise ValueError('影片、中文外挂和弹幕没有同时就绪，未进入无字幕播放')
                s['status']='ready';self.library.native_owners[item_id]=key
                self.library.save_state(item_id,{'subtitle_id':subtitle_id,'subtitle_enabled':True,'preferred_engine':'iina'},owner=key)
                s['state']=self.library.state(item_id);self.save(s)
                return self.public(s)
            except Exception as e:
                s.update(status='failed',error=str(e));self.save(s)
                with contextlib.suppress(Exception):await ipc(s['socket'],[['quit']])
                await self._stop_media(s)
                raise

    def _configuration(self,s,state):
        directory=self.root/s['id']
        atomic_json(directory/'config.json',{'token':s['token'],'comments':str(directory/'comments.json'),'settings':state,'media':s.get('media_url')},True)

    async def sample(self,s,initial=False):
        async with self.sampling.setdefault(s['id'],asyncio.Lock()):
            if s['status'] in ('closed','stopping'):return
            await self._sample(s,initial)

    async def _sample(self,s,initial=False):
        self._validate_owned(s)
        names=['path','track-list','time-pos','pause','volume','mute','speed','sub-delay','sub-visibility','user-data/bdm-native','audio-out-params','video-params','video-sync','display-sync-active','estimated-display-fps','audio-speed-correction']
        values=await ipc(s['socket'],[['get_property',n] for n in names]);v={n:r.get('data') for n,r in zip(names,values)}
        if v['path']!=s.get('media_url',self.store.file(s['video_id'])['path']):raise ValueError('原生窗口已换片，已停止此条目的进度同步')
        video=v.get('video-params') or {}
        s['video_ready']=bool(video.get('w',0)>0 and video.get('h',0)>0)
        selected=next((t for t in (v['track-list'] or []) if t['type']=='sub' and t.get('selected')),None)
        s['subtitle_ready']=bool(selected and selected.get('external-filename')==self.store.file(s['subtitle_id'])['path'] and v['sub-visibility'])
        try:overlay=v['user-data/bdm-native'] if isinstance(v['user-data/bdm-native'],dict) else json.loads(v['user-data/bdm-native'] or '{}')
        except ValueError:overlay={}
        pulse=overlay.get('heartbeat')
        if pulse!=s.get('last_overlay_heartbeat'):
            s['last_overlay_heartbeat']=pulse;s['overlay_seen']=time.monotonic()
        alive=pulse is not None and time.monotonic()-s.get('overlay_seen',0)<4
        s['overlay_diagnostic']=overlay
        s['danmaku_ready']=alive and overlay.get('token')==s['token'] and overlay.get('count')==s['comment_count'] and not overlay.get('error')
        s['output_audio']=v['audio-out-params']
        s['render_sync']={'mode':v.get('video-sync'),'active':bool(v.get('display-sync-active')),'paused':bool(v['pause']),'display_fps':v.get('estimated-display-fps'),'audio_speed_correction':v.get('audio-speed-correction')}
        if not initial:
            s['status']='ready' if s['video_ready'] and s['subtitle_ready'] and s['danmaku_ready'] else 'needs_attention'
            s['error']=None if s['status']=='ready' else '影片画面、外挂字幕或弹幕层未就绪，请恢复后继续；不会标记全部就绪'
            updates={}
            for name,prop in [('position','time-pos'),('volume','volume'),('muted','mute'),('playback_rate','speed'),('subtitle_offset','sub-delay')]:
                value=v[prop]
                if isinstance(value,(bool,int,float)) and (isinstance(value,bool) or math.isfinite(value)):
                    if name=='volume':value=max(0,min(1,value/100))
                    if name=='subtitle_offset' and abs(value)>5:continue
                    updates[name]=value
            chosen=next((t for t in (v['track-list'] or []) if t['type']=='audio' and t.get('selected')),None)
            if chosen:
                ordinal=next((int(k) for k,aid in s.get('audio_tracks',{}).items() if aid==chosen['id']),None)
                if ordinal is not None:
                    updates['playback_audio_stream']=ordinal
                    source=s.get('audio_streams',[])[ordinal]
                    s['source_audio']={k:source.get(k) for k in ('codec_name','profile','channels','channel_layout','sample_rate')}
            updates['subtitle_enabled']=bool(v['sub-visibility'])
            if s['danmaku_ready']:updates['danmaku_enabled']=overlay.get('enabled',True)
            if time.monotonic()-s.get('last_saved',0)>5 or v['pause']!=s.get('paused'):
                self.library.save_state(s['item_id'],updates,owner=s['id']);s['last_saved']=time.monotonic()
            s['paused']=v['pause'];s['state']={**self.library.state(s['item_id']),**updates};self.save(s)

    async def apply(self,item_id,state,changed):
        s=next((x for x in self.sessions.values() if x['item_id']==item_id and x['status'] not in ('failed','closed')),None)
        if not s:return
        commands=[]
        for field,prop in [('volume','volume'),('muted','mute'),('playback_rate','speed'),('subtitle_offset','sub-delay'),('subtitle_enabled','sub-visibility')]:
            if field in changed:commands.append(['set_property',prop,state[field]*100 if field=='volume' else state[field]])
        if 'playback_audio_stream' in changed:
            aid=s.get('audio_tracks',{}).get(str(state['playback_audio_stream']))
            if aid is None:raise ValueError('原生播放音轨不存在')
            commands.append(['set_property','aid',aid])
        if 'subtitle_id' in changed and state.get('subtitle_id')!=s['subtitle_id']:
            file=self.store.file(state['subtitle_id']);inspect_subtitles(file['path']);commands.append(['sub-add',file['path'],'select']);s['subtitle_id']=file['id'];s['subtitle_name']=Path(file['path']).name;s['subtitle_identity']=list(stamp(file['path']))
        if any(k in changed for k in ('danmaku_enabled','danmaku_offset','font_size','opacity','show_estimated')):
            self._configuration(s,state);commands.append(['script-message-to','native_danmaku','configure',str(self.root/s['id']/'config.json')])
        if commands:
            result=await ipc(s['socket'],commands)
            if any(r.get('error')!='success' for r in result):raise ValueError('原生播放设置未全部生效')
        await self.sample(s)

    async def end(self,key):
        s=self.sessions[key]
        async with self.sampling.setdefault(key,asyncio.Lock()):
            with contextlib.suppress(Exception):
                await self._sample(s);self.library.save_state(s['item_id'],s['state'],owner=key)
            s['status']='stopping';self.save(s)
            with contextlib.suppress(Exception):await ipc(s['socket'],[['quit']])
            process=self.processes.get(key)
            if process:
                with contextlib.suppress(subprocess.TimeoutExpired):await asyncio.to_thread(process.wait,2)
            await self._stop_media(s)
            s['status']='closed';self.library.native_owners.pop(s['item_id'],None);self.save(s)
        return self.public(s)

    async def close(self):
        if self.closed:return
        self.closed=True
        if self.monitor:self.monitor.cancel();await asyncio.gather(self.monitor,return_exceptions=True)
        for s in self.sessions.values():
            if s['status'] in ('closed','failed'):continue
            with contextlib.suppress(Exception):
                await self.sample(s);self.library.save_state(s['item_id'],s['state'],owner=s['id'])
        # Do not quit IINA. Persistent manifests allow reattachment after restart.

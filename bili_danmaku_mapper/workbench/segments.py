"""Bounded on-demand stream-copy windows. No whole-film files or video encoder."""
import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import struct
import time
import uuid

from ..io import atomic_json
from ..cache import stamp as identity

GIB = 1024**3
CACHE_LIMIT = GIB
FREE_FLOOR = 5*GIB
MAX_PART = 128*1024**2
SCRATCH = 16*1024**2


async def command_json(command):
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
    try:
        output, _ = await asyncio.wait_for(process.communicate(),30)
        if process.returncode:
            raise ValueError('无法读取影片分段信息')
        return json.loads(output)
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGKILL)
            await process.communicate()


def rows(path,stream=0):
    tb, result = None, []
    for line in path.read_text().splitlines():
        if line.startswith(f'#tb {stream}:'):
            a,b = line.split(':',1)[1].strip().split('/');tb=float(a)/float(b)
        elif not line.startswith('#') and line.strip():
            values = line.split(',')
            if tb is not None and len(values)==6 and int(values[0])==stream:
                result.append((float(values[2])*tb,values[5].strip().lower()))
    return result


def boxes(data,start=0,end=None):
    end=len(data) if end is None else end
    while start<end:
        size,kind=struct.unpack_from('>I4s',data,start);header=8
        if size==1:size=struct.unpack_from('>Q',data,start+8)[0];header=16
        if size<header or start+size>end:raise ValueError('无效播放容器结构')
        yield kind,start,start+header,start+size
        start+=size


def retime_tracks(path,init,delta):
    """Preserve relative audio/video clocks lost by empty-moov track normalization.

    The second muxer's encoded packet PTS supplies the common clock; video bytes
    stay untouched. Only tfdt counters change, rounded to each track timebase.
    """
    tracks={}
    for kind,_,a,b in boxes(init):
        if kind!=b'moov':continue
        for kind,_,a,b in boxes(init,a,b):
            if kind!=b'trak':continue
            track=None;scale=None;handler=None
            for kind,_,c,d in boxes(init,a,b):
                if kind==b'tkhd':track=struct.unpack_from('>I',init,c+(20 if init[c] else 12))[0]
                if kind==b'mdia':
                    for sub,_,e,f in boxes(init,c,d):
                        if sub==b'mdhd':scale=struct.unpack_from('>I',init,e+(20 if init[e] else 12))[0]
                        if sub==b'hdlr':handler=init[e+8:e+12]
            if track is not None and scale:tracks[track]=(scale,handler)
    with path.open('r+b') as handle:
        handle.seek(len(init));header=handle.read(8);size,kind=struct.unpack('>I4s',header)
        if kind!=b'moof' or size>2*1024**2:raise ValueError('无效片段时间轴结构')
        data=bytearray(header+handle.read(size-8))
        for kind,_,a,b in boxes(data,8):
            if kind!=b'traf':continue
            track=None;tfdt=None
            for sub,_,c,d in boxes(data,a,b):
                if sub==b'tfhd':track=struct.unpack_from('>I',data,c+4)[0]
                if sub==b'tfdt':tfdt=c
            if track not in tracks or tfdt is None:raise ValueError('播放片段缺少轨道时间基准')
            scale,handler=tracks[track];shift=max(delta,0) if handler==b'soun' else max(-delta,0)
            fmt='>Q' if data[tfdt] else '>I';at=tfdt+4
            value=struct.unpack_from(fmt,data,at)[0]+round(shift*scale)
            struct.pack_into(fmt,data,at,value)
        handle.seek(len(init));handle.write(data);handle.flush()


def video_mime(stream):
    codec=stream['codec_name']
    if codec=='h264':
        # avcC profile, compatibility and level bytes are present in the extradata dump.
        raw=extradata(stream.get('extradata',''))
        if len(raw)<4:raise ValueError('无法读取 H.264 编码配置')
        return 'avc1.'+raw[1:4].hex().upper()
    if codec=='hevc':
        raw=extradata(stream.get('extradata',''))
        if len(raw)<13:raise ValueError('无法读取 HEVC 编码配置')
        profile=raw[1]&31;space=['','A','B','C'][raw[1]>>6];tier='H' if raw[1]&32 else 'L'
        compat=int.from_bytes(raw[2:6],'big');compat=int(f'{compat:032b}'[::-1],2)
        constraints=raw[6:12].rstrip(b'\0')
        return f'hvc1.{space}{profile}.{compat:X}.{tier}{raw[12]}'+''.join(f'.{b:02X}' for b in constraints)
    raise ValueError('该视频编码不支持保留画质的分段播放，请使用支持原片的浏览器')


def extradata(value):
    data=bytearray()
    for line in value.splitlines():
        if ':' not in line:continue
        hexpart=line.split(':',1)[1].split('  ',1)[0].strip().replace(' ','')
        try:data.extend(bytes.fromhex(hexpart))
        except ValueError:pass
    return bytes(data)


class SegmentManager:
    def __init__(self,store,library,config):
        self.store,self.library,self.config=store,library,config
        self.root=store.directory/'segments-v1';self.root.mkdir(exist_ok=True)
        if self.root.resolve()!=self.root:raise ValueError('分段缓存目录不能是链接')
        self.stats={'ffmpeg_starts':0,'windows_started':0,'cache_hits':0,'bytes_written':0}
        self.janitor=None
        self.sessions={};self.windows={};self.tasks={};self.processes={};self.last_request={}
        self.slot=asyncio.Semaphore(1);self.closed=False;self.requests=0
        # Only this namespace's incomplete files are removed. Complete fragments survive restart.
        for p in self.root.glob('*/part-*.partial'):p.unlink(missing_ok=True)
        for p in self.root.glob('*/packets.sha256'):p.unlink(missing_ok=True)
        self.used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    async def start(self):
        async def sweep():
            while not self.closed:
                await asyncio.sleep(5)
                for sid,value in list(self.sessions.items()):
                    if time.monotonic()-value['updated']>30:await self.delete(sid)
        self.janitor=asyncio.create_task(sweep())

    @property
    def busy(self):return bool(self.tasks)

    def reserve(self,size,protected=None):
        if shutil.disk_usage(self.root).free-size<FREE_FLOOR:
            raise ValueError('可用空间低于 5 GiB，已停止新增播放片段；原片与弹幕保留')
        while self.used+size+SCRATCH>CACHE_LIMIT:
            candidates=[]
            for directory in self.root.iterdir():
                if directory.is_dir() and directory!=protected and directory.name not in self.tasks and directory.name not in self.last_request.values():
                    candidates.append(directory)
            if not candidates:raise ValueError('分段缓存已达 1 GiB 上限，请稍后重试')
            victim=min(candidates,key=lambda p:p.stat().st_mtime_ns)
            count=sum(p.stat().st_size for p in victim.rglob('*') if p.is_file())
            shutil.rmtree(victim);self.used=max(0,self.used-count)
            self.windows.pop(victim.name,None)

    async def create(self,item_id,audio_stream=0,audio_copy=False):
        if self.closed:raise ValueError('服务正在退出')
        item=self.library.item(item_id);source=self.store.file(item['video_id']);stamp=identity(source['path'])
        info=await command_json([self.config.ffprobe,'-v','error','-show_streams','-show_format','-show_data','-of','json',source['path']])
        video=next((s for s in info['streams'] if s['codec_type']=='video'),None)
        if video is None:raise ValueError('影片没有视频流')
        audios=[s for s in info['streams'] if s['codec_type']=='audio']
        if audio_stream>=max(1,len(audios)):raise ValueError('播放音轨序号超出范围')
        audio=audios[audio_stream] if audios else None
        # AAC LC is the portable copy path. Other audio is transcoded, never the video.
        copy=bool(audio_copy and audio and audio['codec_name']=='aac' and audio.get('profile')=='LC')
        mime='video/mp4; codecs="'+video_mime(video)+(',mp4a.40.2' if audio else '')+'"'
        origin=float(info['format'].get('start_time',0));duration=float(info['format']['duration'])
        if video.get('duration') not in (None,'N/A'):
            duration=float(video.get('start_time',origin))+float(video['duration'])-origin
        elif origin!=0:
            # Only inspect the tail, never scan every packet of the original.
            end_guess=duration if 'matroska' in info['format'].get('format_name','') else duration+origin
            tail=await command_json([self.config.ffprobe,'-v','error','-read_intervals',f'{max(origin,end_guess-4):.6f}%+8',
                '-select_streams','v:0','-show_packets','-show_entries','packet=pts_time,duration_time','-of','json',source['path']])
            ends=[float(p['pts_time'])+float(p.get('duration_time',0)) for p in tail.get('packets',[]) if 'pts_time' in p]
            if not ends:raise ValueError('无法确认原片结束时间')
            duration=max(ends)-origin
        if not math.isfinite(origin) or not math.isfinite(duration) or duration<=0:raise ValueError('影片时间轴无效')
        sid=uuid.uuid4().hex
        signature=hashlib.sha256(json.dumps([source['id'],stamp,audio_stream,copy,'segments-6']).encode()).hexdigest()
        self.sessions[sid]=dict(id=sid,item_id=item_id,source_id=source['id'],source=source['path'],identity=stamp,
            duration=duration,origin=origin,audio_stream=audio_stream,audio=bool(audio),audio_copy=copy,mime=mime,
            signature=signature,video_colors={k:video[k] for k in ('color_primaries','color_transfer','color_space','color_range') if video.get(k) not in (None,'unknown','unspecified')},video_codec=video['codec_name'],generation=-1,updated=time.monotonic())
        return {k:self.sessions[sid][k] for k in ('id','duration','mime','audio_copy','video_codec')}

    def get(self,sid):
        if sid not in self.sessions:raise KeyError('Unknown playback session')
        session=self.sessions[sid]
        if identity(session['source'])!=session['identity']:raise ValueError('原片已经变化，请重新载入')
        session['updated']=time.monotonic();return session

    async def request(self,sid,position,generation):
        session=self.get(sid)
        if position>session['duration']:raise ValueError('播放位置超出影片范围')
        if generation<session['generation']:raise ValueError('播放请求已过期')
        previous=self.last_request.get(sid)
        # Fixed windows allow reusing a seek result without reading the whole original.
        start=math.floor(min(position,max(0,session['duration']-.000001))/24)*24
        wid=hashlib.sha256(json.dumps([session['signature'],start]).encode()).hexdigest()[:32]
        self.last_request[sid]=wid
        if previous and previous!=wid and previous not in self.last_request.values():await self.cancel(previous)
        session['generation']=generation;self.last_request[sid]=wid
        directory=self.root/wid
        if wid not in self.windows and (directory/'manifest.json').is_file():
            try:
                cached=json.loads((directory/'manifest.json').read_text())
                if (directory.resolve()==directory and cached['status']=='complete' and cached.get('signature')==session['signature']
                        and all(re.fullmatch(r'part-\d{4}\.mp4',p['name']) and (directory/p['name']).is_file()
                                and self.valid_part(directory,p) for p in cached['parts'])):
                    self.windows[wid]=cached
            except (OSError,ValueError,KeyError):pass
        if wid in self.windows and self.windows[wid]['status']=='complete' and not all(self.valid_part(directory,p) for p in self.windows[wid]['parts']):
            self.windows.pop(wid)
        if wid in self.windows and self.windows[wid]['status']=='complete':self.stats['cache_hits']+=1
        if wid not in self.windows or self.windows[wid]['status'] in ('failed','cancelled'):
            self.windows[wid]=dict(id=wid,status='queued',parts=[],start=start,error=None,signature=session['signature'])
            task=asyncio.create_task(self.run(wid,dict(session),start));self.tasks[wid]=task
            task.add_done_callback(lambda t:self.tasks.pop(wid,None))
        if directory.exists():os.utime(directory,None)
        return self.public(sid,wid)

    def public(self,sid,wid):
        self.get(sid)
        if self.last_request.get(sid)!=wid:raise KeyError('Expired playback window')
        value=self.windows[wid]
        return {k:v for k,v in value.items() if k!='signature'}

    @staticmethod
    def valid_part(directory,part):
        path=directory/part['name']
        if path.resolve()!=path or not path.is_file():return False
        values=(part.get('start'),part.get('end'),part.get('offset'))
        if any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in values):return False
        return (part['end']>part['start'] and 0<part['bytes']<=MAX_PART
                and identity(path)==part.get('identity') and path.stat().st_size==part['bytes'])

    def fragment(self,sid,wid,name):
        info=self.public(sid,wid)
        part=next((p for p in info['parts'] if p['name']==name),None)
        if part is None or not re.fullmatch(r'part-\d{4}\.mp4',name):raise KeyError('Unknown fragment')
        path=self.root/wid/name
        if not self.valid_part(path.parent,part):raise KeyError('Expired or changed fragment')
        os.utime(path.parent,None);return path

    async def cancel(self,wid):
        task=self.tasks.get(wid)
        if task:
            if not task.cancelling():task.cancel()
            with contextlib.suppress(asyncio.CancelledError):await asyncio.shield(task)

    async def pause(self,sid):
        self.get(sid)
        wid=self.last_request.get(sid)
        if wid and sum(v==wid for v in self.last_request.values())<=1:await self.cancel(wid)
        return {'ok':True}

    async def delete(self,sid):
        if sid not in self.sessions:return {'ok':True}
        wid=self.last_request.pop(sid,None)
        self.sessions.pop(sid,None)
        if wid and wid not in self.last_request.values():await self.cancel(wid)
        return {'ok':True}

    async def close(self):
        self.closed=True
        if self.janitor:
            self.janitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):await self.janitor
        for task in list(self.tasks.values()):
            if not task.cancelling():task.cancel()
        await asyncio.gather(*list(self.tasks.values()),return_exceptions=True)
        self.sessions.clear()

    async def run(self,wid,session,start):
        directory=self.root/wid;process=None;stderr_task=None
        try:
            async with self.slot:
                self.reserve(0,directory)
                started=time.monotonic();self.stats['windows_started']+=1
                if directory.exists():
                    size=sum(p.stat().st_size for p in directory.iterdir() if p.is_file());shutil.rmtree(directory);self.used=max(0,self.used-size)
                directory.mkdir();state=self.windows[wid];state.update(status='running',parts=[])
                anchor_pts=None;audio_delta=None
                hashes=directory/'packets.sha256'
                duration=min(30,session['duration']-start)
                command=[self.config.ffmpeg,'-debug_ts','-v','verbose','-nostdin','-y','-noaccurate_seek','-ss',str(start),'-i',session['source'],'-t',str(duration),'-map','0:v:0']
                if session['audio']:command+=['-map',f'0:a:{session["audio_stream"]}']
                command+=['-c:v','copy','-copytb','1']
                if session['video_codec']=='hevc':command+=['-tag:v','hvc1']
                for name,option in [('color_primaries','-color_primaries:v'),('color_transfer','-color_trc:v'),('color_space','-colorspace:v'),('color_range','-color_range:v')]:
                    if name in session['video_colors']:command+=[option,session['video_colors'][name]]
                if session['audio']:
                    command+=['-c:a','copy'] if session['audio_copy'] else ['-c:a','aac','-ac','2','-ar','48000','-b:a','256k','-threads:a','1']
                command+=['-sn','-dn','-map_chapters','-1','-avoid_negative_ts','disabled','-use_editlist','0',
                    '-movflags','+frag_keyframe+empty_moov+default_base_moof+write_colr','-min_frag_duration','4000000','-f','mp4','pipe:1',
                    '-t',str(duration),'-map','0:v:0','-c:v','copy']
                command+=['-f','framehash','-hash','sha256','-flush_packets','1',str(hashes)]
                audio_hashes=directory/'audio.sha256'
                if session['audio_copy']:
                    command+=['-t',str(duration),'-map',f'0:a:{session["audio_stream"]}','-c:a','copy','-f','framehash','-hash','sha256','-flush_packets','1',str(audio_hashes)]
                process=await asyncio.create_subprocess_exec(*command,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True)
                self.processes[wid]=process;self.stats['ffmpeg_starts']+=1
                first_audio_pts=None
                async def drain():
                    nonlocal first_audio_pts
                    while True:
                        line=await process.stderr.readline()
                        if not line:break
                        if first_audio_pts is None and b'encoder -> type:audio' in line:
                            match=re.search(rb'pkt_pts_time:([-0-9.e+]+)',line)
                            if match:first_audio_pts=float(match[1])
                stderr_task=asyncio.create_task(drain())
                init=bytearray();partial=None;handle=None;part_size=0;part_no=0;matched=0
                try:
                    while True:
                        try:header=await asyncio.wait_for(process.stdout.readexactly(8),15)
                        except asyncio.IncompleteReadError as e:
                            if e.partial:raise ValueError('分段数据中断')
                            break
                        size,kind=struct.unpack('>I4s',header)
                        if size==1:
                            tail=await process.stdout.readexactly(8);header+=tail;size=struct.unpack('>Q',tail)[0]
                        if size<len(header) or size>MAX_PART:raise ValueError('关键帧区间过大，无法在缓存限额内分段播放')
                        if kind in (b'ftyp',b'moov'):
                            if size+len(init)>2*1024**2:raise ValueError('影片初始化信息过大')
                            init.extend(header+await process.stdout.readexactly(size-len(header)));continue
                        if kind==b'moof':
                            if handle:raise ValueError('分段结构不完整')
                            partial=directory/f'part-{part_no:04d}.partial';handle=partial.open('wb');part_size=0
                            self.reserve(len(init),directory);handle.write(init);part_size+=len(init);self.used+=len(init)
                        if handle:
                            self.reserve(len(header),directory);handle.write(header);self.used+=len(header);part_size+=len(header)
                        remaining=size-len(header)
                        while remaining:
                            chunk=await asyncio.wait_for(process.stdout.readexactly(min(65536,remaining)),15);remaining-=len(chunk)
                            if handle:
                                if part_size+len(chunk)>MAX_PART:raise ValueError('播放片段超过 128 MiB，已停止处理')
                                self.reserve(len(chunk),directory);handle.write(chunk);self.used+=len(chunk);part_size+=len(chunk)
                        if kind==b'mdat' and handle:
                            handle.close();handle=None
                            if hashes.stat().st_size>8*1024**2:raise ValueError('片段校验信息超过限额')
                            source_rows=rows(hashes)
                            if not source_rows:raise ValueError('无法读取时间轴校验信息')
                            if anchor_pts is None:
                                guess=start+source_rows[0][0]+session['origin']
                                for delta in (.000001,.1,-.1):
                                    anchor=await command_json([self.config.ffprobe,'-v','error','-read_intervals',f'{max(0,guess+delta):.6f}%+#4',
                                        '-select_streams','v:0','-show_packets','-show_data_hash','sha256','-show_entries','packet=pts_time,data_hash,flags','-of','json',session['source']])
                                    packets=anchor.get('packets',[])
                                    if packets and abs(float(packets[0].get('pts_time','inf'))-guess)<=.002 and 'K' in packets[0].get('flags','') and packets[0]['data_hash'].split(':')[-1].lower()==source_rows[0][1]:
                                        anchor_pts=float(packets[0]['pts_time'])-session['origin'];break
                                if anchor_pts is None:raise ValueError('关键帧来源校验失败，已拒绝播放片段')
                            part_info=await command_json([self.config.ffprobe,'-v','error','-show_packets','-show_data_hash','sha256',
                                '-show_entries','packet=stream_index,pts_time,duration_time,data_hash,flags','-of','json',str(partial)])
                            video=[v for v in part_info.get('packets',[]) if v['stream_index']==0]
                            audio=[v for v in part_info.get('packets',[]) if v['stream_index']==1]
                            if not video or 'K' not in video[0].get('flags',''):raise ValueError('播放片段缺少独立关键帧')
                            positions=[]
                            for packet in video:
                                digest=packet['data_hash'].split(':')[-1].lower()
                                while matched<len(source_rows) and source_rows[matched][1]!=digest:matched+=1
                                if matched>=len(source_rows):raise ValueError('视频流复制校验失败')
                                positions.append(source_rows[matched][0]+anchor_pts-source_rows[0][0]);matched+=1
                            offset=positions[0]-float(video[0]['pts_time'])
                            if session['audio'] and audio_delta is None and audio:
                                audio_rows=rows(audio_hashes) if session['audio_copy'] else []
                                audio_pts=audio_rows[0][0] if audio_rows else first_audio_pts
                                if audio_pts is None:raise ValueError('缺少音频时间轴校验')
                                audio_delta=audio_pts+anchor_pts-source_rows[0][0]-float(audio[0]['pts_time'])-offset
                            if audio_delta is not None:
                                retime_tracks(partial,init,audio_delta)
                                shift=max(-audio_delta,0)
                                for v in video:v['pts_time']=str(float(v['pts_time'])+shift)
                                offset-=shift
                            if any(abs(expected-float(packet['pts_time'])-offset)>.002 for expected,packet in zip(positions,video)):
                                raise ValueError('片段时间轴不一致，已拒绝使用')
                            if identity(session['source'])!=session['identity']:raise ValueError('原片已经变化')
                            name=f'part-{part_no:04d}.mp4';final=directory/name;os.replace(partial,final);partial=None
                            end=max(float(v['pts_time'])+float(v.get('duration_time',0)) for v in video)+offset
                            state['parts'].append(dict(name=name,start=min(positions),end=end,offset=offset,bytes=part_size,identity=identity(final)))
                            self.stats['bytes_written']+=part_size
                            if part_no==0:state['first_part_seconds']=time.monotonic()-started
                            part_no+=1
                    if await process.wait():raise ValueError('无法准备播放片段，请更换支持原片的浏览器')
                    if identity(session['source'])!=session['identity']:raise ValueError('处理期间原片发生变化')
                    if not state['parts']:raise ValueError('没有生成可播放片段')
                    state['status']='complete';state['elapsed_seconds']=time.monotonic()-started;atomic_json(directory/'manifest.json',state,True)
                    self.used+=(directory/'manifest.json').stat().st_size
                finally:
                    if handle:handle.close()
                    if partial and partial.exists():self.used=max(0,self.used-partial.stat().st_size);partial.unlink()
        except asyncio.CancelledError:
            self.windows[wid].update(status='cancelled');raise
        except Exception as error:
            self.windows[wid].update(status='failed',error=str(error) if isinstance(error,ValueError) else '分段播放准备失败，原片与弹幕保留')
        finally:
            if process:
                async def discard_stdout():
                    while await process.stdout.read(65536):pass
                stdout_cleanup=asyncio.create_task(discard_stdout())
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGTERM)
                    try:await asyncio.wait_for(process.wait(),2)
                    except asyncio.TimeoutError:
                        with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGKILL)
                        await process.wait()
                await stdout_cleanup
                self.processes.pop(wid,None)
            if stderr_task:await stderr_task
            (directory/'packets.sha256').unlink(missing_ok=True)
            self.used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

"""Synthetic-only segmented playback tests; never load the user's film."""
import asyncio
import json
from pathlib import Path
import struct
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
from scipy.signal import correlate

from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.workbench.segments import SegmentManager, CACHE_LIMIT, SCRATCH
from bili_danmaku_mapper.workbench.state import Store
from bili_danmaku_mapper.workbench.playback import PlaybackLibrary
from tests.test_playback import client_for, import_video
from bili_danmaku_mapper.workbench.server import create_app


@pytest.fixture(scope='module')
def movie(tmp_path_factory):
    path=tmp_path_factory.mktemp('segments')/"movie 'with spaces'.mkv"
    subprocess.run(['ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc2=size=160x90:rate=24:duration=62',
        '-f','lavfi','-i','anoisesrc=sample_rate=16000:duration=62:seed=43:amplitude=0.15',
        '-c:v','libx264','-threads','1','-g','48','-c:a','ac3','-output_ts_offset','1.5',str(path)],check=True)
    return path


async def setup(tmp_path,movie):
    store=Store(tmp_path/'state');library=PlaybackLibrary(store)
    file=store.register_file(movie,'playback_video');item=library.import_item(file['id'])
    manager=SegmentManager(store,library,AlignConfig())
    return store,library,item,manager


def test_windows_preserve_pts_copy_bytes_audio_and_cache(tmp_path,movie):
    async def run():
        store,library,item,manager=await setup(tmp_path,movie)
        session=await manager.create(item['id']);assert session['video_codec']=='h264' and session['audio_copy'] is False
        assert abs(session['duration']-62)<.05
        results=[]
        for generation,pos in enumerate([0,26,51]):
            w=await manager.request(session['id'],pos,generation)
            await asyncio.gather(*list(manager.tasks.values()))
            result=manager.public(session['id'],w['id']);assert result['status']=='complete',result
            assert any(p['start']<=pos<p['end'] for p in result['parts'])
            assert all(abs(p['start']*24-round(p['start']*24))<.05 for p in result['parts'])
            for a,b in zip(result['parts'],result['parts'][1:]):assert abs(a['end']-b['start'])<1/24
            results.append(result)
        # An actual waveform check catches seconds of encoder seek/preroll drift.
        w=results[1]
        for part in w['parts'][:2]:
            path=manager.root/w['id']/part['name'];offset=part['offset']
            def pcm(path):
                return np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']),dtype='<f4')
            ref=pcm(movie);audio=pcm(path)
            meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_packets','-show_entries','packet=pts_time','-of','json',str(path)]))
            audio_start=float(meta['packets'][0]['pts_time'])+offset
            # Search against reference; compression may change samples, not temporal position.
            clip=audio[1600:17600]
            left=max(0,int((audio_start-1)*16000));right=min(len(ref),int((audio_start+3)*16000))
            correlation=correlate(ref[left:right],clip,method='fft',mode='valid')
            found=(left+int(np.argmax(correlation)))/16000-.1
            assert abs(found-audio_start)<.05,(found,audio_start)
        assert manager.used<CACHE_LIMIT
        again=await manager.request(session['id'],session['duration'],4);assert again['status']=='complete'
        assert not manager.tasks
        with pytest.raises(ValueError):await manager.request(session['id'],1,0)
        with pytest.raises(KeyError):manager.fragment(session['id'],again['id'],'../../movie.mkv')
        await manager.close();store.close()
    asyncio.run(run())


def test_sessions_validate_sources_limits_and_stop_owned_work(tmp_path,movie):
    async def run():
        store,library,item,manager=await setup(tmp_path,movie)
        with pytest.raises(ValueError):await manager.create(item['id'],audio_stream=99)
        session=await manager.create(item['id'])
        with pytest.raises(ValueError):await manager.request(session['id'],1000,0)
        with patch('bili_danmaku_mapper.workbench.segments.shutil.disk_usage',return_value=type('Usage',(),{'free':4*1024**3})()):
            w=await manager.request(session['id'],0,0);await asyncio.gather(*list(manager.tasks.values()))
            assert manager.public(session['id'],w['id'])['status']=='failed'
            assert not manager.processes
        # The quota is enforced before writing, with scratch space reserved.
        manager.used=CACHE_LIMIT-SCRATCH
        with pytest.raises(ValueError):manager.reserve(1,manager.root/'nonexistent')
        manager.used=0
        await manager.request(session['id'],25,1);await manager.delete(session['id'])
        assert not manager.processes
        await manager.close();store.close()
    asyncio.run(run())


def test_state_feedback_api_and_no_full_copy_requests(tmp_path,movie):
    app=create_app(tmp_path/'state',token='test',picker=lambda kind:[movie],config=AlignConfig())
    with client_for(app) as client:
        item=import_video(client)
        assert item['state']['show_estimated'] is False
        result=client.patch('/api/playback/items/'+item['id']+'/state',json={'danmaku_offset':-1.2}).json()
        assert result['danmaku_offset']==-1.2
        assert client.patch('/api/playback/items/'+item['id']+'/state',json={'danmaku_offset':5.1}).status_code==422
        assert client.post('/api/playback/items/'+item['id']+'/feedback',json={'position':4,'kind':'early'}).status_code==200
        assert client.post('/api/playback/items/'+item['id']+'/feedback',json={'position':4,'kind':'invalid'}).status_code==422
        assert app.state.store.jobs()==[]
        result=client.post('/api/playback/sessions',json={'item_id':item['id']}).json()
        assert 'path' not in str(result)
        assert client.post('/api/playback/sessions',json={'item_id':item['id'],'path':'/etc/hosts'}).status_code==422
        sid=result['id']
        assert client.post('/api/playback/sessions/'+sid+'/heartbeat').status_code==200
        assert client.post('/api/playback/sessions/'+sid+'/windows',content='{"position":NaN,"generation":0}',headers={'Content-Type':'application/json'}).status_code in (400,422)
        assert client.delete('/api/playback/sessions/'+sid).status_code==200
        assert not (tmp_path/'state'/'playback').exists()
        assert client.get('/api/playback/sessions/'+sid+'/windows/invalid').status_code==404


def test_corrupt_part_is_rebuilt_not_served_and_source_changes_reject(tmp_path,movie):
    async def run():
        local=tmp_path/'local.mkv';local.write_bytes(movie.read_bytes())
        store,library,item,manager=await setup(tmp_path,local)
        session=await manager.create(item['id']);wid=(await manager.request(session['id'],0,0))['id']
        await asyncio.gather(*list(manager.tasks.values()))
        part=manager.windows[wid]['parts'][0];path=manager.root/wid/part['name'];path.write_bytes(b'broken')
        with pytest.raises(KeyError):manager.fragment(session['id'],wid,part['name'])
        await manager.request(session['id'],0,1);await asyncio.gather(*list(manager.tasks.values()))
        assert manager.windows[wid]['status']=='complete'
        local.touch()
        with pytest.raises(ValueError):manager.get(session['id'])
        await manager.close();store.close()
    asyncio.run(run())


def test_playback_reservations_do_not_cancel_another_session(tmp_path,movie):
    async def run():
        store,library,item,manager=await setup(tmp_path,movie)
        a=await manager.create(item['id']);b=await manager.create(item['id'])
        first=await manager.request(a['id'],0,0)
        other=await manager.request(b['id'],0,0);assert first['id']==other['id']
        await manager.pause(a['id'])
        assert manager.busy
        await manager.delete(a['id'])
        await asyncio.gather(*list(manager.tasks.values()))
        assert manager.public(b['id'],other['id'])['status']=='complete'
        await manager.delete(b['id']);await manager.close();store.close()
    asyncio.run(run())


def test_hevc_main10_copy_and_unsupported_video_are_explicit(tmp_path):
    path=tmp_path/'main10.mkv'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc2=size=128x72:rate=24:duration=6',
        '-pix_fmt','yuv420p10le','-c:v','libx265','-threads','1','-x265-params','pools=1:frame-threads=1:log-level=error:keyint=24:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc',
        '-color_primaries','bt2020','-color_trc','smpte2084','-colorspace','bt2020nc',str(path)],check=True,stderr=subprocess.DEVNULL)
    async def run():
        store,library,item,manager=await setup(tmp_path,path)
        session=await manager.create(item['id']);assert 'hvc1.2.' in session['mime'] and 'mp4a' not in session['mime']
        wid=(await manager.request(session['id'],1,0))['id'];await asyncio.gather(*list(manager.tasks.values()))
        result=manager.public(session['id'],wid);assert result['status']=='complete',result
        output=manager.fragment(session['id'],wid,result['parts'][0]['name'])
        metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(output)]))['streams'][0]
        assert metadata['codec_name']=='hevc' and metadata['pix_fmt']=='yuv420p10le'
        # HEVC VUI is exposed on decoded frames by this FFmpeg build, not all stream fields.
        frame=json.loads(subprocess.check_output(['ffprobe','-v','error','-read_intervals','%+#1','-show_frames','-show_entries','frame=color_primaries,color_transfer','-of','json',str(output)]))['frames'][0]
        assert frame['color_transfer']=='smpte2084' and frame['color_primaries']=='bt2020'
        await manager.close();store.close()
    asyncio.run(run())
    from bili_danmaku_mapper.workbench.segments import video_mime
    with pytest.raises(ValueError,match='不支持'):video_mime({'codec_name':'ffv1'})


@pytest.mark.parametrize('vfr',[False,True])
def test_aac_copy_vfr_and_restart_reuse(tmp_path,vfr):
    path=tmp_path/'aac.mkv'
    command=['ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc2=size=128x72:rate=24:duration=6',
        '-f','lavfi','-i','sine=frequency=600:duration=6']
    if vfr:command+=['-vf',"select='not(eq(mod(n,5),0))'",'-fps_mode','vfr']
    command+=['-c:v','libx264','-threads','1','-g','24','-c:a','aac',str(path)]
    subprocess.run(command,check=True)
    async def run():
        store,library,item,manager=await setup(tmp_path,path)
        session=await manager.create(item['id'],audio_copy=True);assert session['audio_copy']
        wid=(await manager.request(session['id'],1,0))['id'];await asyncio.gather(*list(manager.tasks.values()))
        assert manager.public(session['id'],wid)['status']=='complete'
        await manager.close()
        replacement=SegmentManager(store,library,AlignConfig());new=await replacement.create(item['id'],audio_copy=True)
        hit=await replacement.request(new['id'],1,0);assert hit['status']=='complete' and not replacement.tasks
        assert replacement.stats['ffmpeg_starts']==0 and replacement.stats['cache_hits']==1
        await replacement.close();store.close()
    asyncio.run(run())


def test_hard_cache_limit_includes_unpublished_bytes(tmp_path,movie):
    async def run():
        store,library,item,manager=await setup(tmp_path,movie)
        session=await manager.create(item['id'])
        with patch('bili_danmaku_mapper.workbench.segments.CACHE_LIMIT',512*1024),patch('bili_danmaku_mapper.workbench.segments.SCRATCH',0):
            wid=(await manager.request(session['id'],25,0))['id'];await asyncio.gather(*list(manager.tasks.values()))
            assert manager.public(session['id'],wid)['status']=='failed'
            assert sum(p.stat().st_size for p in manager.root.rglob('*') if p.is_file())<=512*1024
            assert not manager.processes and not list(manager.root.rglob('*.partial'))
        await manager.close();store.close()
    asyncio.run(run())


def test_playback_defers_new_compute_without_interrupting_running_job(tmp_path,movie):
    from bili_danmaku_mapper.workbench.jobs import JobManager
    from bili_danmaku_mapper.io import atomic_json
    async def run():
        store=Store(tmp_path/'state');file=store.register_file(movie,'selected');raw=tmp_path/'raw.json';raw.write_text('{"data":[]}')
        entered=asyncio.Event();release=asyncio.Event();calls=[]
        async def compute(job,directory,notify):
            calls.append(job['id']);entered.set();await release.wait()
            for name in ('mapped.json','mapped.report.json','mapped.review.json'):atomic_json(directory/name,{'data':[]},True)
            return {'target_video':str(movie),'counts':{'reliable':0}}
        manager=JobManager(store,None,AlignConfig(),compute=compute)
        jobs=[store.create_job({'title':'synthetic'},file['id'],{'mode':'strict','quality':64}) for _ in range(2)]
        for job in jobs:store.update(job['id'],checkpoint={'source_video':str(movie),'raw_danmaku':str(raw)})
        await manager.start();await asyncio.wait_for(entered.wait(),2)
        manager.playback_busy=lambda:True;release.set()
        for _ in range(100):
            if store.job(jobs[0]['id'])['status']=='complete':break
            await asyncio.sleep(.01)
        assert store.job(jobs[0]['id'])['status']=='complete'
        assert calls==[jobs[0]['id']] and store.job(jobs[1]['id'])['status']=='queued'
        manager.playback_busy=lambda:False;manager.wakeup.set()
        for _ in range(100):
            if store.job(jobs[1]['id'])['status']=='complete':break
            await asyncio.sleep(.01)
        assert store.job(jobs[1]['id'])['status']=='complete'
        await manager.close();store.close()
    asyncio.run(run())


def test_movie_experiment_guard_refuses_a_live_workbench(tmp_path):
    import os
    from tests.viewing_benchmark import require_idle, Deferred
    (tmp_path/'instance.json').write_text(json.dumps({'pid':os.getpid(),'port':1}))
    with pytest.raises(Deferred):require_idle(tmp_path)
    assert not (tmp_path/'state').exists()


def test_fragment_http_range_and_session_authorization(tmp_path,movie):
    import time
    app=create_app(tmp_path/'state',token='test',picker=lambda kind:[movie],config=AlignConfig())
    with client_for(app) as client:
        item=import_video(client);session=client.post('/api/playback/sessions',json={'item_id':item['id']}).json();sid=session['id']
        assert client.post('/api/playback/sessions',json={'item_id':item['id']},headers={'Authorization':''}).status_code==401
        window=client.post('/api/playback/sessions/'+sid+'/windows',json={'position':26,'generation':0}).json();wid=window['id']
        deadline=time.monotonic()+10
        while not window['parts']:
            assert window['status']!='failed',window
            assert time.monotonic()<deadline
            time.sleep(.02);window=client.get('/api/playback/sessions/'+sid+'/windows/'+wid).json()
        url='/api/playback/sessions/'+sid+'/windows/'+wid+'/parts/'+window['parts'][0]['name']
        response=client.get(url,headers={'Range':'bytes=0-127'})
        assert response.status_code==206 and len(response.content)==128
        assert client.get(url,headers={'Authorization':''}).status_code==401
        assert 'path' not in str(window)
        assert client.post('/api/playback/sessions/'+sid+'/pause').status_code==200
        assert client.delete('/api/playback/sessions/'+sid).status_code==200
        assert client.get(url).status_code==404


@pytest.mark.parametrize('audio_codec',['ac3','aac'])
def test_reordered_video_preserves_audio_clock(tmp_path,audio_codec):
    """Two B-frame reorder periods must not become an artificial audio lead."""
    import re
    path=tmp_path/'reordered.mkv'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc2=size=128x72:rate=24:duration=10',
        '-f','lavfi','-i','anoisesrc=sample_rate=48000:duration=10:seed=61:amplitude=.15',
        '-c:v','libx264','-bf','8','-g','96','-threads','1','-c:a',audio_codec,str(path)],check=True)
    def pcm(path):
        r=subprocess.run(['ffmpeg','-v','info','-copyts','-i',str(path),'-map','0:a:0','-vn',
            '-af','aresample=16000,aformat=channel_layouts=mono,ashowinfo','-f','f32le','pipe:1'],capture_output=True,check=True)
        pts=float(re.search(rb'n:0 .*?pts_time:([-0-9.e+]+)',r.stderr).group(1))
        return np.frombuffer(r.stdout,dtype='<f4'),pts
    async def run():
        store,library,item,manager=await setup(tmp_path,path)
        session=await manager.create(item['id'],audio_copy=audio_codec=='aac');wid=(await manager.request(session['id'],1,0))['id']
        await asyncio.gather(*list(manager.tasks.values()));result=manager.public(session['id'],wid)
        assert result['status']=='complete',result
        ref,refstart=pcm(path)
        for part in result['parts'][:2]:
            audio,pts=pcm(manager.fragment(session['id'],wid,part['name']));clip=audio[1600:17600]
            found=refstart+int(np.argmax(correlate(ref,clip,method='fft',mode='valid')))/16000-.1
            expected=pts+part['offset']+manager.sessions[session['id']]['origin']
            assert abs(found-expected)<.05,(found,expected)
        await manager.close();store.close()
    asyncio.run(run())

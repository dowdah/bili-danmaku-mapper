import json
from pathlib import Path
import subprocess
import time

import pytest

from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.workbench.playback_worker import build_variant, identity, timestamps, verify_timeline
from bili_danmaku_mapper.workbench.files import inspect_file
from bili_danmaku_mapper.workbench.server import create_app
from tests.test_playback import client_for, import_video
from tests.video_helpers import make_video
from tests.test_workbench import FakeDownload


def ffmpeg(*args):
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y',*map(str,args)],check=True,capture_output=True)


def sample(tmp_path,audio='aac',offset=0,vfr=False):
    movie=make_video(tmp_path/'video.mp4',3,vfr=vfr,filters='select=not(mod(n\\,3))+not(mod(n\\,7))' if vfr else None)
    path=tmp_path/"original with 'quotes'.mkv"
    ffmpeg('-i',movie,'-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=3','-map','0:v','-map','1:a',
           '-c:v','copy','-c:a',audio,'-output_ts_offset',offset,path)
    return path


def build(path,tmp_path,strategy,audio=0):
    return build_variant(dict(source=str(path),source_identity=identity(path),directory=str(tmp_path/strategy),
                              strategy=strategy,audio_stream=audio,ffmpeg='ffmpeg',ffprobe='ffprobe'))


def video_hash(path):
    return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-an','-f','hash','-hash','sha256','-']).strip()


def test_mkv_lossless_remux_preserves_pixels_and_timestamps(tmp_path):
    path=sample(tmp_path)
    result=build(path,tmp_path,'remux')
    assert result['timeline_verified'] and result['max_video_error']<=.002
    assert result['timestamps_checked']==72
    assert video_hash(path)==video_hash(result['path'])
    assert inspect_file(result['path'])['audio_streams'][0]['codec']=='aac'
    assert not (tmp_path/'remux'/'playback.partial.mp4').exists()


def test_only_audio_conversion_preserves_video_with_pcm_input(tmp_path):
    path=sample(tmp_path,'pcm_s16le')
    result=build(path,tmp_path,'audio')
    assert result['timeline_verified']
    assert video_hash(path)==video_hash(result['path'])
    assert inspect_file(result['path'])['audio_streams'][0]['codec']=='aac'


@pytest.mark.parametrize('strategy',['remux','audio','transcode'])
def test_nonzero_start_vfr_timeline_unchanged(tmp_path,strategy):
    path=sample(tmp_path,'aac',offset=5,vfr=True)
    result=build(path,tmp_path,strategy)
    assert result['max_video_error']<=.002
    assert len(timestamps(path,'ffprobe'))==result['timestamps_checked']


def test_changed_or_truncated_timeline_is_rejected(tmp_path):
    a=make_video(tmp_path/'a.mp4',3);b=make_video(tmp_path/'b.mp4',2)
    with pytest.raises(ValueError,match='时间轴'):
        verify_timeline(a,b,'ffprobe','remux')
    with pytest.raises(ValueError,match='原片已经变化'):
        build_variant(dict(source=str(a),source_identity=[0,0,0,0],directory=str(tmp_path/'out'),strategy='remux',audio_stream=0,ffmpeg='ffmpeg',ffprobe='ffprobe'))


def wait_result(client,key,timeout=20):
    deadline=time.monotonic()+timeout
    while True:
        result=client.get('/api/playback/preparations/'+key).json()
        if result['status'] not in ('queued','running'):return result
        assert time.monotonic()<deadline,result
        time.sleep(.05)


def test_api_preparation_independent_cached_authorized_and_durable(tmp_path):
    path=sample(tmp_path)
    config=AlignConfig(cache_dir=str(tmp_path/'features'))
    def app():return create_app(tmp_path/'state',token='test',config=config,picker=lambda kind:[path])
    with client_for(app()) as c:
        item=import_video(c);url='/api/playback/items/'+item['id']+'/prepare'
        first=c.post(url,json={'strategy':'remux'}).json()
        assert c.post(url,json={'strategy':'remux'}).json()['id']==first['id']
        result=wait_result(c,first['id'])
        assert result['status']=='complete',result
        assert result['timeline_verified']
        assert c.get('/api/status').json()['jobs']==[]
        assert c.get('/api/playback/items').json()['items'][0]['video_id']==item['video_id']
        assert c.get('/api/files/'+result['file_id'],headers={'Range':'bytes=0-99'}).status_code==206
        full=c.post(url,json={'strategy':'transcode','confirmed':True}).json()
        full_result=wait_result(c,full['id'])
        assert full_result['status']=='complete',full_result
        assert any(item['strategy']=='transcode' for item in c.get('/api/playback/items/'+item['id']+'/prepared').json())
        assert 'path' not in str(result)
        assert c.post(url,json={'strategy':'transcode'}).status_code==400
        assert c.post(url,json={'strategy':'remux','audio_stream':9}).status_code==400
        assert c.post(url,json={'strategy':'remux','source':'/etc/passwd'}).status_code==422
        assert c.post('/api/playback/items/unknown/prepare',json={'strategy':'remux'}).status_code==404
    with client_for(app()) as c:
        cached=c.post(url,json={'strategy':'remux'}).json()
        assert cached['status']=='complete' and cached['file_id']==result['file_id']
        assert c.get('/api/playback/items').json()['items'][0]['video_id']==item['video_id']
        assert any(copy['file_id']==full_result['file_id'] for copy in c.get('/api/playback/items/'+item['id']+'/prepared').json())


def test_mkv_can_create_alignment_without_conversion_even_legacy_convert_true(tmp_path):
    path=sample(tmp_path)
    app=create_app(tmp_path/'state',token='test',config=AlignConfig(cache_dir=str(tmp_path/'features')),
                   downloader=FakeDownload(path),picker=lambda kind:[path])
    with client_for(app) as c:
        picked=c.post('/api/pick',json={'kind':'files'}).json()['files'][0]
        catalog=c.post('/api/resolve',json={'url':'BV1xx411c7mD'}).json()
        for legacy in [False,True]:
            response=c.post('/api/jobs',json={'pairs':[{'source_key':catalog['items'][0]['key'],'file_id':picked['id']}],
                                             'options':{'convert':legacy}})
            assert response.status_code==200,response.text
            job=response.json()[0]
            deadline=time.monotonic()+20
            while job['status'] not in ('complete','failed'):
                time.sleep(.1);assert time.monotonic()<deadline
                job=c.get('/api/jobs/'+job['id']).json()
            assert job['status']=='complete',job
            target=app.state.store.file(job['result']['files']['target'])
            assert target['path']==str(path.resolve())
            assert not list((tmp_path/'state'/'jobs'/job['id']).glob('playback*'))
        assert not list((tmp_path/'state'/'playback').glob('*'))


def test_other_audio_track_alignment_and_lossless_playback_with_origin_compensation(tmp_path):
    video=make_video(tmp_path/'video.mp4',3)
    source=tmp_path/'two tracks.mkv'
    ffmpeg('-i',video,'-itsoffset','-0.5','-f','lavfi','-i','sine=frequency=440:duration=3',
           '-f','lavfi','-i','sine=frequency=880:duration=3','-map','0:v','-map','1:a','-map','2:a',
           '-c:v','copy','-c:a','aac','-metadata:s:a:0','language=jpn','-metadata:s:a:1','language=eng',source)
    result=build(source,tmp_path,'remux',audio=1)
    assert result['timeline_verified'] and abs(result['timeline_offset'])>.3
    assert video_hash(source)==video_hash(result['path'])
    output=inspect_file(result['path'])
    assert len(output['audio_streams'])==1 and output['audio_streams'][0]['language']=='eng'
    app=create_app(tmp_path/'state',token='test',downloader=FakeDownload(source),picker=lambda kind:[source])
    with client_for(app) as c:
        c.post('/api/pipeline/pause')
        picked=c.post('/api/pick',json={'kind':'files'}).json()['files'][0]
        catalog=c.post('/api/resolve',json={'url':'BV1xx411c7mD'}).json()
        response=c.post('/api/jobs',json={'pairs':[{'source_key':catalog['items'][0]['key'],'file_id':picked['id']}],
                    'options':{'audio_stream':1,'convert':False}})
        assert response.status_code==200,response.text


def test_timeline_speed_change_rejected_even_with_same_packet_count(tmp_path):
    a=make_video(tmp_path/'original.mp4',3);b=tmp_path/'changed.mp4'
    ffmpeg('-itsscale','1.1','-i',a,'-c','copy',b)
    assert len(timestamps(a,'ffprobe'))==len(timestamps(b,'ffprobe'))
    with pytest.raises(ValueError,match='时间轴'):
        verify_timeline(a,b,'ffprobe','remux')


def test_preparation_failures_and_confirmation_do_not_schedule_alignment(tmp_path):
    path=make_video(tmp_path/'movie.mp4',2)
    config=AlignConfig(ffmpeg='/does-not-exist/ffmpeg')
    app=create_app(tmp_path/'state',token='test',config=config,picker=lambda kind:[path])
    with client_for(app) as c:
        item=import_video(c);url='/api/playback/items/'+item['id']+'/prepare'
        for strategy in ['remux','audio','transcode']:
            task=c.post(url,json={'strategy':strategy,'confirmed':strategy=='transcode'}).json()
            result=wait_result(c,task['id'])
            assert result['status']=='failed'
            assert '/does-not-exist' not in result['error']
        assert c.get('/api/status').json()['jobs']==[]
        assert c.get('/api/playback/items').json()['items'][0]['available']


def test_preparation_shutdown_and_stage_recovery(tmp_path):
    import os
    path=make_video(tmp_path/'movie.mp4',2)
    slow=tmp_path/'slow-ffmpeg'
    slow.write_text('#!/bin/sh\nsleep 60\n')
    slow.chmod(0o755)
    config=AlignConfig(ffmpeg=str(slow))
    def build_app():return create_app(tmp_path/'state',token='test',config=config,picker=lambda kind:[path])
    app=build_app()
    with client_for(app) as c:
        item=import_video(c);url='/api/playback/items/'+item['id']+'/prepare'
        task=c.post(url,json={'strategy':'remux'}).json()
        deadline=time.monotonic()+5
        while not app.state.preparer.processes:
            time.sleep(.02);assert time.monotonic()<deadline
        pid=next(iter(app.state.preparer.processes.values())).pid
        stage=app.state.store.directory/'playback'/task['id']
        (stage/'playback.partial.mp4').write_bytes(b'owned partial')
        assert c.post('/api/exit').status_code==200
        assert not (stage/'playback.partial.mp4').exists()
        assert not app.state.preparer.processes and not app.state.preparer.tasks
        with pytest.raises(ProcessLookupError):os.killpg(pid,0)
    with client_for(build_app()) as c:
        assert c.get('/api/playback/preparations/'+task['id']).json()['status']=='interrupted'
        restarted=c.post(url,json={'strategy':'remux'}).json()
        assert restarted['id']==task['id'] and restarted['status'] in ('queued','running')



def test_avi_can_remux_without_video_encoding(tmp_path):
    source=sample(tmp_path);avi=tmp_path/'raw.avi'
    ffmpeg('-i',source,'-map','0:v','-map','0:a','-c:v','libx264','-bf','0','-c:a','copy',avi)
    result=build(avi,tmp_path,'remux')
    assert result['timeline_verified'] and result['timestamps_checked']==72
    assert video_hash(avi)==video_hash(result['path'])

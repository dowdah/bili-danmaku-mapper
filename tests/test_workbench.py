import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from bili_danmaku_mapper import AlignConfig
from bili_danmaku_mapper.io import atomic_json
from bili_danmaku_mapper.workbench import bilibili as bili
from bili_danmaku_mapper.workbench.files import episode_number, inspect_file, suggest_pairs
from bili_danmaku_mapper.workbench.jobs import JobManager
from bili_danmaku_mapper.workbench.server import create_app
from bili_danmaku_mapper.workbench.state import Store
from tests.video_helpers import make_video


@pytest.mark.parametrize('url,kind,value,page',[
 ('BV1xx411c7mD','bv','BV1xx411c7mD',1),('av170001','av','170001',1),
 ('https://www.bilibili.com/video/BV1xx411c7mD?p=3&share_source=copy','bv','BV1xx411c7mD',3),
 ('https://www.bilibili.com/video/av170001/','av','170001',1),
 ('https://www.bilibili.com/bangumi/play/ep123','ep','123',1),
 ('https://www.bilibili.com/bangumi/play/ss456','ss','456',1),
 ('https://www.bilibili.com/bangumi/media/md789/','md','789',1),
 ('分享 https://b23.tv/abcd 好片','short',None,None)])
def test_links(url,kind,value,page):
    parsed=bili.parse_source(url)
    assert parsed['kind']==kind
    if value: assert (parsed['value'],parsed['page'])==(value,page)


@pytest.mark.parametrize('url',['https://evil.example/video/BV1xx411c7mD','https://www.bilibili.com.evil/video/BV1xx411c7mD',
 'https://user@www.bilibili.com/video/BV1xx411c7mD','https://www.bilibili.com:888/video/BV1xx411c7mD',
 'https://www.bilibili.com/video/BV1xx411c7mD?p=0','https://live.bilibili.com/1','https://www.bilibili.com/foo'])
def test_invalid_links(url):
    with pytest.raises(ValueError):bili.parse_source(url)


def test_pairing():
    assert episode_number('Show S01E03 1080p')==3
    assert episode_number('Show 第12集')==12
    assert episode_number('Show SP 01') is None
    assert episode_number('Show 01 02') is None
    sources=[{'key':'a','number':1},{'key':'b','number':3}]
    files=[{'id':'x','number':2},{'id':'y','number':4}]
    assert [p['file_id'] for p in suggest_pairs(sources,files,1)]==['x','y']
    assert suggest_pairs(sources,[*files,{'id':'z','number':2}],1)[0]['reason']=='ambiguous'
    assert all(p['file_id'] is None for p in suggest_pairs([{'key':'a','number':1},{'key':'b','number':1}],files,1))


def test_catalog_preserves_page_and_collection_pagination(tmp_path,monkeypatch):
    class FakeVideo:
        def __init__(self,bvid=None,aid=None,credential=None):self.bvid=bvid or 'BV1xx411c7mD'
        async def get_info(self):return {'bvid':self.bvid,'aid':1,'title':'Series','owner':{'mid':42},'ugc_season':{'id':7,'title':'Collection'}}
        async def get_pages(self):return [{'cid':1,'part':'One','duration':2},{'cid':2,'part':'Two','duration':2}] if self.bvid=='BV1xx411c7mD' else [{'cid':3,'part':'Third','duration':2}]
    pages=[]
    class FakeSeries:
        def __init__(self,**kwargs):pass
        async def get_videos(self,pn,ps):
            pages.append(pn)
            return {'archives':[{'bvid':'BV1xx411c7mD'}] if pn==1 else [{'bvid':'BV1xx411c7mD'},{'bvid':'BV1yy411c7mD'}], 'page':{'total':2}}
    monkeypatch.setattr(bili.video,'Video',FakeVideo);monkeypatch.setattr(bili.channel_series,'ChannelSeries',FakeSeries)
    service=bili.BiliService(bili.Auth(tmp_path/'cookies.json'),tmp_path,AlignConfig())
    result=asyncio.run(service.resolve('https://www.bilibili.com/video/BV1xx411c7mD?p=2'))
    assert pages==[1,2]
    assert len(result['items'])==3
    assert [s['cid'] for s in result['items'] if s['selected']]==[2]


def test_bangumi_sections_deduplicate_and_default(tmp_path,monkeypatch):
    class Fake:
        def __init__(self,**kwargs):self.kwargs=kwargs
        async def get_episode_list(self):
            one={'id':1,'title':'1','long_title':'Start','duration':2000,'cid':2}
            return {'main_section':{'episodes':[one,{'id':2,'title':'2','duration':2000}]},'section':[{'episodes':[one,{'id':3,'title':'PV','duration':1000}]}]}
        async def get_overview(self):return {}
        async def get_meta(self):return {'title':'Show'}
    monkeypatch.setattr(bili.bangumi,'Bangumi',Fake)
    service=bili.BiliService(bili.Auth(tmp_path/'cookie'),tmp_path,AlignConfig())
    ep=asyncio.run(service.resolve('ep2'));ss=asyncio.run(service.resolve('ss1'));md=asyncio.run(service.resolve('md1'))
    assert len(ep['items'])==3 and [s['epid'] for s in ep['items'] if s['selected']]==[2]
    assert [s['epid'] for s in ss['items'] if s['selected']]==[1,2]
    assert md['items']==ss['items']


class FakeDownload:
    def __init__(self,video,gate=None):self.video=video;self.calls=[];self.gate=gate
    async def resolve(self,url):return {'title':'Synthetic test','items':[{'key':'test:1','kind':'video','title':'Test episode 1','duration':4,'number':1,'selected':True}]}
    async def download(self,source,directory,quality,notify):
        self.calls.append(source['key'])
        notify('downloading',123,456)
        if source.get('fail'):raise ValueError('Controlled download failure')
        await asyncio.sleep(.02)
        raw=Path(directory)/'raw.json';atomic_json(raw,{'data':[{'time':1.2,'text':'本地合成测试','type':'scroll','color':'#ffffff'}]},overwrite=True)
        return {'source_video':str(self.video),'raw_danmaku':str(raw)}


async def until(predicate,timeout=10):
    start=time.monotonic()
    while not predicate():
        assert time.monotonic()-start<timeout,'timed out waiting for state'
        await asyncio.sleep(.02)


def test_pipeline_overlap_bound_pause_failure_and_restart(tmp_path):
    async def scenario():
        target=tmp_path/'movie.mp4';target.write_bytes(b'movie')
        store=Store(tmp_path/'db');fid=store.register_file(target,'selected')['id']
        download=FakeDownload(target);gate=asyncio.Event();entered=asyncio.Event();computed=[]
        async def compute(job,directory,notify):
            computed.append(job['id']);entered.set();await gate.wait()
            for name in ('mapped.json','mapped.report.json','mapped.review.json'):atomic_json(directory/name,{},overwrite=True)
            return {'target_video':str(target),'rates':{},'counts':{}}
        manager=JobManager(store,download,AlignConfig(),compute)
        opts={'quality':64,'mode':'strict'}
        jobs=[store.create_job({'key':str(i),'title':str(i),'fail':i==1},fid,opts) for i in range(7)]
        await manager.start();await entered.wait()
        await until(lambda:len(download.calls)>=4)
        # First computation is blocked while later downloads have completed.
        assert store.job(jobs[0]['id'])['status']=='computing'
        assert store.job(jobs[1]['id'])['status']=='failed'
        ready=[j for j in store.jobs() if j['status']=='waiting']
        assert len(ready)<=2
        calls=len(download.calls);await asyncio.sleep(.08);assert len(download.calls)==calls
        manager.pause(True);gate.set();await until(lambda:store.job(jobs[0]['id'])['status']=='complete')
        assert len(computed)==1
        manager.pause(False);await until(lambda:all(j['status'] in ('complete','failed') for j in store.jobs()))
        manager.retry(jobs[1]['id']);await until(lambda:store.job(jobs[1]['id'])['status']=='failed')
        job=store.create_job({'key':'later','title':'later'},fid,opts);manager.pause(True)
        await manager.close();store.close()
        reopened=Store(tmp_path/'db');manager2=JobManager(reopened,download,AlignConfig(),compute)
        assert reopened.job(jobs[0]['id'])['status']=='complete'
        await manager2.start();assert manager2.paused
        manager2.pause(False);await until(lambda:reopened.job(job['id'])['status']=='complete')
        await manager2.close();reopened.close()
    asyncio.run(scenario())


def test_service_auth_range_native_registration_and_worker(tmp_path):
    path=make_video(tmp_path/"episode 01 with 'quotes'.mp4",4)
    service=FakeDownload(path)
    app=create_app(tmp_path/'workbench',token='testing-secret',config=AlignConfig(cache_dir=str(tmp_path/'features')),
                   downloader=service,picker=lambda kind:[path])
    with TestClient(app,base_url='http://127.0.0.1:8000',raise_server_exceptions=False) as client:
        assert client.get('/api/status').status_code==401
        headers={'Authorization':'Bearer testing-secret'}
        assert client.post('/api/session',headers=headers).status_code==200
        assert client.get('/api/status',headers={'Origin':'http://evil.example'}).status_code==403
        assert client.get('/api/status',headers={'Host':'evil.example'}).status_code==403
        assert client.get('/api/files/missing').status_code==404
        picked=client.post('/api/pick',json={'kind':'files'}).json()['files'][0]
        assert 'path' not in picked and picked['number']==1
        response=client.get('/api/files/'+picked['id'],headers={'Range':'bytes=0-99'})
        assert response.status_code==206 and len(response.content)==100
        assert response.headers['content-range'].startswith('bytes 0-99/')
        resolved=client.post('/api/resolve',json={'url':'BV1xx411c7mD'}).json()
        created=client.post('/api/jobs',json={'pairs':[{'source_key':resolved['items'][0]['key'],'file_id':picked['id']}]}).json()[0]
        deadline=time.monotonic()+20
        while True:
            job=client.get('/api/jobs/'+created['id']).json()
            if job['status'] in ('complete','failed'):break
            assert time.monotonic()<deadline;time.sleep(.1)
        assert job['status']=='complete',job
        assert job['result']['counts']['reliable']==1
        assert 'checkpoint' not in job
        assert client.get('/api/files/'+job['result']['files']['report']).json()['counts']['total']==1
        bad=client.post('/api/revise/'+job['id'],json={'anchors':[{'source_time':99,'target_time':1}]})
        assert bad.status_code==400
        revision=client.post('/api/revise/'+job['id'],json={'anchors':[{'source_time':1,'target_time':1}]}).json()
        assert revision['parent_id']==job['id'] and revision['id']!=job['id']
        assert client.get('/api/jobs/'+job['id']).json()['status']=='complete'


@pytest.mark.parametrize('dash',[True,False])
def test_download_quality_segments_and_empty_comments(tmp_path,monkeypatch,dash):
    media=make_video(tmp_path/'part.mp4',1)
    data={'dash':{'video':[{'id':80,'codecs':'avc','baseUrl':'1080'}, {'id':64,'codecs':'hev','baseUrl':'hevc'},
                          {'id':32,'codecs':'avc1','baseUrl':'480'}],'audio':[]}} if dash else {'durl':[{'url':'one'},{'url':'two'}]}
    class Obj:
        async def get_download_url(self,**kwargs):return data
        async def get_danmakus(self,**kwargs):return []
    service=bili.BiliService(bili.Auth(tmp_path/'cookie'),tmp_path/'cache',AlignConfig())
    async def request(source,quality=64):return Obj()
    urls=[]
    async def download(url,destination,notify):urls.append(url);shutil.copyfile(media,destination)
    async def download_info(source,quality):return data
    monkeypatch.setattr(service,'_download_info',download_info)
    monkeypatch.setattr(service,'_request_object',request);monkeypatch.setattr(service,'_bytes',download)
    source={'key':'test','kind':'video','cid':1,'duration':2}
    result=asyncio.run(service.download(source,tmp_path/'job',64,lambda *args,**kwargs:None))
    assert urls==(['480'] if dash else ['one','two'])
    assert json.loads(Path(result['raw_danmaku']).read_text())['data']==[]
    assert inspect_file(result['source_video'])['duration']>=1
    # Interrupted metadata cache must be recomputed, not crash or accepted.
    next((tmp_path/'cache').glob('*/manifest.json')).write_text('broken')
    from types import SimpleNamespace
    async def colored_comments(self,**kwargs):
        return [SimpleNamespace(dm_time=1,text='top',crc32_id='test',color='a0ea',mode=5),
                SimpleNamespace(dm_time=.5,text='bottom',crc32_id='test',color='0',mode=4)]
    monkeypatch.setattr(Obj,'get_danmakus',colored_comments)
    retry=asyncio.run(service.download(source,tmp_path/'retry',64,lambda *args,**kwargs:None))
    comments=json.loads(Path(retry['raw_danmaku']).read_text())['data']
    assert [(d['color'],d['type']) for d in comments]==[('#000000','bottom'),('#00a0ea','top')]



def test_stopping_owned_worker_and_recovering_stage(tmp_path):
    async def scenario():
        path=tmp_path/'video.mp4';path.write_bytes(b'x');raw=tmp_path/'raw.json';raw.write_text('{}')
        store=Store(tmp_path/'db');fid=store.register_file(path,'selected')['id']
        ready=asyncio.Event()
        async def compute(job,directory,notify):ready.set();await asyncio.Event().wait()
        manager=JobManager(store,FakeDownload(path),AlignConfig(),compute)
        job=store.create_job({'key':'test','title':'test'},fid,{'quality':64,'mode':'strict'})
        store.update(job['id'],checkpoint={'source_video':str(path),'raw_danmaku':str(raw)},status='computing')
        await manager.start();await ready.wait();await manager.close()
        assert store.job(job['id'])['status']=='queued'
        assert manager.ready(store.job(job['id']))
        assert not manager.running
        store.close()
    asyncio.run(scenario())


def test_qr_states_and_expiry(tmp_path,monkeypatch):
    reply={'url':'https://www.bilibili.com/qr/mock','qrcode_key':'temporary-test-key'}
    class Api:
        def __init__(self,**kwargs):pass
        def update_params(self,**kwargs):return self
        @property
        async def result(self):return reply
    monkeypatch.setattr(bili,'Api',Api)
    async def scenario():
        auth=bili.Auth(tmp_path/'cookies')
        qr=await auth.generate_qr();assert qr['image'].startswith('data:image/png;base64,')
        reply.clear();reply.update(code=86101);assert (await auth.poll_qr())['state']=='waiting_scan'
        reply['code']=86090;assert (await auth.poll_qr())['state']=='waiting_confirm'
        reply['code']=86038;assert (await auth.poll_qr())['state']=='expired'
        assert not auth.cookies_path.exists()
    asyncio.run(scenario())


def test_reference_api_never_requests_8k(tmp_path,monkeypatch):
    requests=[]
    class Api:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def update_params(self,**kwargs):requests.append((self.kwargs,kwargs));return self
        @property
        async def result(self):return {}
    monkeypatch.setattr(bili,'Api',Api)
    async def scenario():
        service=bili.BiliService(bili.Auth(tmp_path/'cookie'),tmp_path,AlignConfig())
        await service._download_info({'kind':'video','bvid':'BV1xx411c7mD','cid':1,'aid':2},64)
        await service._download_info({'kind':'episode','epid':3},32)
    asyncio.run(scenario())
    assert [p['qn'] for _,p in requests]==[64,32]
    assert all(p['fnval']==16 and p['fourk']==0 for _,p in requests)
    assert requests[0][0]['wbi'] is True
    assert requests[1][1]['ep_id']==3


def test_mkv_admission_and_audio_selection(tmp_path):
    path=make_video(tmp_path/'episode01.mkv',2)
    app=create_app(tmp_path/'workbench',token='test',downloader=FakeDownload(path),picker=lambda kind:[path])
    with TestClient(app,base_url='http://127.0.0.1:8000',raise_server_exceptions=False) as c:
        c.headers['Authorization']='Bearer test'
        f=c.post('/api/pick',json={'kind':'files'}).json()['files'][0]
        c.post('/api/resolve',json={'url':'av170001'})
        pair={'pairs':[{'source_key':'test:1','file_id':f['id']}]}
        assert c.post('/api/jobs',json=pair).status_code==200
        pair['options']={'convert':True,'audio_stream':99}
        assert c.post('/api/jobs',json=pair).status_code==400
        pair['options']={'convert':True}
        job=c.post('/api/jobs',json=pair).json()[0]
        assert job['options']['convert'] is True


def test_state_file_capability_rejects_replaced_symlink(tmp_path):
    path=tmp_path/'selected.mp4';path.write_bytes(b'x')
    store=Store(tmp_path/'db');f=store.register_file(path,'selected');path.unlink();path.symlink_to('/etc/hosts')
    with pytest.raises(FileNotFoundError):store.file(f['id'])
    store.close()


def test_actual_worker_group_stops_ffmpeg_descendant(tmp_path):
    import os
    import shlex
    async def scenario():
        path=make_video(tmp_path/'source.mp4',4)
        marker=tmp_path/'decode-started'
        wrapper=tmp_path/'slow ffmpeg'
        wrapper.write_text('#!/bin/sh\nprintf started > '+shlex.quote(str(marker))+'\nsleep 30\nexec '+shlex.quote(shutil.which('ffmpeg'))+' "$@"\n')
        wrapper.chmod(0o755)
        store=Store(tmp_path/'state');fid=store.register_file(path,'selected')['id']
        manager=JobManager(store,FakeDownload(path),AlignConfig(ffmpeg=str(wrapper),cache_dir=str(tmp_path/'cache')))
        job=store.create_job({'key':'test','title':'test'},fid,{'quality':64,'mode':'strict'})
        await manager.start();await until(marker.exists)
        process=manager.processes[job['id']]
        await manager.stop(job['id'])
        assert process.returncode is not None
        assert store.job(job['id'])['status']=='stopped'
        assert not manager.processes
        await manager.close();store.close()
    asyncio.run(scenario())


def test_original_alignment_and_independent_playback_use_selected_audio(tmp_path):
    import hashlib
    import subprocess
    import numpy as np
    video=make_video(tmp_path/'silent.mp4',4)
    original=tmp_path/"original with 'quotes'.mkv"
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','lavfi','-i','sine=frequency=440:duration=4',
                    '-f','lavfi','-i','sine=frequency=880:duration=4','-map','0:v','-map','1:a','-map','2:a',
                    '-c:v','copy','-c:a','aac',str(original)],check=True)
    before=hashlib.sha256(original.read_bytes()).digest()
    app=create_app(tmp_path/'state',token='test',config=AlignConfig(cache_dir=str(tmp_path/'cache')),
                   downloader=FakeDownload(video),picker=lambda kind:[original])
    with TestClient(app,base_url='http://127.0.0.1:8000',raise_server_exceptions=False) as c:
        c.headers['Authorization']='Bearer test'
        f=c.post('/api/pick',json={'kind':'files'}).json()['files'][0]
        c.post('/api/resolve',json={'url':'av170001'})
        created=c.post('/api/jobs',json={'pairs':[{'source_key':'test:1','file_id':f['id']}],
                         'options':{'convert':True,'audio_stream':1}}).json()[0]
        deadline=time.monotonic()+20
        while True:
            job=c.get('/api/jobs/'+created['id']).json()
            if job['status'] in ('complete','failed'):break
            assert time.monotonic()<deadline;time.sleep(.05)
        assert job['status']=='complete',job
        target=Path(app.state.store.file(job['result']['files']['target'])['path'])
        assert target==original
        report=c.get('/api/files/'+job['result']['files']['report']).json()
        assert Path(report['alignment']['target'])==original
        assert report['alignment']['config']['target_audio_stream']==1
        from tests.test_playback_prepare import wait_result
        prepared=c.post('/api/playback/items/job:'+job['id']+'/prepare',json={'strategy':'remux','audio_stream':1}).json()
        result=wait_result(c,prepared['id'])
        assert result['status']=='complete',result
        target=Path(app.state.store.file(result['file_id'])['path'])
        assert target!=original and target.suffix=='.mp4'
        assert len(inspect_file(target)['audio_streams'])==1
        samples=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(target),'-t','1','-f','f32le',
                              '-ar','8000','-ac','1','pipe:1']),dtype='<f4')
        peak=np.argmax(np.abs(np.fft.rfft(samples)))*8000/len(samples)
        assert abs(peak-880)<3
        report=c.get('/api/files/'+job['result']['files']['report']).json()
        assert Path(report['alignment']['target'])==original
        assert report['alignment']['config']['target_audio_stream']==1
        assert app.state.store.file(job['result']['files']['target'])['path']==str(original)
    assert hashlib.sha256(original.read_bytes()).digest()==before


def test_large_media_range_reads_only_requested_tail(tmp_path):
    path=tmp_path/'large.mp4'
    size=1024**3
    with path.open('wb') as f:
        f.seek(size-4);f.write(b'tail')
    app=create_app(tmp_path/'state',token='test',downloader=FakeDownload(path))
    fid=app.state.store.register_file(path,'selected')['id']
    with TestClient(app,base_url='http://127.0.0.1:8000') as c:
        c.headers['Authorization']='Bearer test'
        response=c.get('/api/files/'+fid,headers={'Range':f'bytes={size-4}-{size-1}'})
        assert response.status_code==206 and response.content==b'tail'
        assert response.headers['content-length']=='4'
        assert response.headers['content-range']==f'bytes {size-4}-{size-1}/{size}'
        assert c.get('/api/files/'+fid,headers={'Range':f'bytes={size+1}-'}).status_code==416


@pytest.mark.parametrize('value,expected',[('a0ea','#00a0ea'),('#9843','#009843'),('ffffff','#ffffff'),(0,'#000000'),(255,'#0000ff'),('special','#ffffff'),('invalid','#ffffff')])
def test_danmaku_rgb_normalization(value,expected):
    assert bili.danmaku_color(value)==expected

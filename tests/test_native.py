import asyncio,json,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.workbench.state import Store
from bili_danmaku_mapper.workbench.playback import PlaybackLibrary
from bili_danmaku_mapper.workbench.native import NativeManager,native_items
from tests.video_helpers import make_video
from tests.test_playback import client_for
from bili_danmaku_mapper.workbench.server import create_app


def fixture(tmp):
    video=make_video(tmp/"原片 ' quotes.mp4",3)
    sub=tmp/'中文 ASS.ass';sub.write_text('[Script Info]\nScriptType: v4.00+\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,,{\\c&H00FFFF&}原样对白\\N第二行\n')
    dm=tmp/'弹幕.json';dm.write_text(json.dumps({'data':[{'time':0,'text':'彩色弹幕','type':'top','color':'#a0ea'}]}))
    store=Store(tmp/'state');library=PlaybackLibrary(store)
    v=store.register_file(video,'playback_video');d=store.register_file(dm,'danmaku');s=store.register_file(sub,'subtitle');item=library.import_item(v['id'],d['id'])
    return store,library,item,s


async def fake_media(self,s,source,directory):return source


class FakeIINA:
    def __init__(self,script_error=False):self.server=None;self.props={};self.quit=False;self.script_error=script_error;self.args=None
    async def launch(self,args):
        self.args=args;socket=next(x.split('=',1)[1] for x in args if x.startswith('--mpv-input-ipc-server='));path=args[-1]
        self.props={'path':path,'track-list':[{'id':1,'type':'video','ff-index':0}],'time-pos':1,'pause':True,'volume':100,'mute':False,'speed':1,'sub-delay':0,'sub-visibility':True,'audio-out-params':None,'video-params':{'w':160,'h':90},'video-sync':'display-vdrop','display-sync-active':False,'estimated-display-fps':None,'audio-speed-correction':1.0}
        async def handle(reader,writer):
            try:
                while line:=await reader.readline():
                    request=json.loads(line);cmd=request['command'];data=None;error='success'
                    if cmd[0]=='get_property':data=self.props.get(cmd[1])
                    elif cmd[0]=='set_property':self.props[cmd[1]]=cmd[2]
                    elif cmd[0]=='sub-add':self.props['track-list'].append({'id':3,'type':'sub','external-filename':cmd[1],'selected':True})
                    elif cmd[0]=='seek':self.props['time-pos']=cmd[1]
                    elif cmd[0]=='load-script' and self.script_error:error='loading failed'
                    elif cmd[0]=='script-message-to':
                        cfg=json.loads(Path(cmd[-1]).read_text());comments=json.loads(Path(cfg['comments']).read_text())['data'];self.props['user-data/bdm-native']={'token':cfg['token'],'count':len(comments),'heartbeat':__import__('time').monotonic(),'enabled':cfg['settings']['danmaku_enabled']}
                    elif cmd[0]=='quit':self.quit=True
                    writer.write((json.dumps({'request_id':request['request_id'],'error':error,'data':data})+'\n').encode());await writer.drain()
            finally:writer.close();await writer.wait_closed()
        self.server=await asyncio.start_unix_server(handle,socket)
        return SimpleNamespace(pid=os.getpid())
    async def close(self):
        if self.server:self.server.close();await self.server.wait_closed()


def test_statuses_preserve_duplicate_order_and_immutable_input():
    data={'data':[{'time':1,'text':'a'},{'time':1,'text':'b'}]};report={'entries':[{'target_time':1,'status':'estimated'},{'target_time':1,'status':'reliable'}]}
    result=native_items(data,report);assert [x['status'] for x in result['data']]==['estimated','reliable'];assert 'status' not in data['data'][0]


def test_native_requires_real_authorized_subtitle(tmp_path):
    store,l,item,sub=fixture(tmp_path);native=NativeManager(store,l,AlignConfig())
    async def run():
        with pytest.raises(ValueError,match='外挂'):await native.create(item['id'],None)
        with pytest.raises(ValueError,match='系统对话框'):await native.create(item['id'],item['danmaku_id'])
    asyncio.run(run());assert not native.sessions;store.close()


def test_ready_progress_ownership_controls_and_reconnect(tmp_path):
    store,l,item,sub=fixture(tmp_path);fake=FakeIINA();native=NativeManager(store,l,AlignConfig(),fake.launch);original=Path(store.file(sub['id'])['path']).read_bytes()
    async def run():
        with patch.object(native,'executable',return_value=Path('/IINA')),patch.object(NativeManager,'_start_media',fake_media):
            s=await native.create(item['id'],sub['id']);assert s['status']=='ready' and s['subtitle_ready'] and s['danmaku_ready'];assert s['comment_count']==1
            assert s['state']['preferred_engine']=='iina';assert s['render_sync']=={'mode':'display-vdrop','active':False,'paused':True,'display_fps':None,'audio_speed_correction':1.0};assert '--mpv-video-sync=display-vdrop' in fake.args;assert not any('display-resample' in arg or 'vf=' in arg for arg in fake.args);assert Path(store.file(sub['id'])['path']).read_bytes()==original
            assert l.save_state(item['id'],{'position':99})['position']!=99
            assert not any('ffmpeg' in arg for arg in fake.args)
            fake.props['pause']=False;fake.props['display-sync-active']=True;fake.props['estimated-display-fps']=60.0;fake.props['time-pos']=2.25;native.sessions[s['id']]['last_saved']=0;await native.sample(native.sessions[s['id']]);assert l.state(item['id'])['position']==2.25;assert native.public(native.sessions[s['id']])['render_sync']['active']
            await native.apply(item['id'],{**l.state(item['id']),'subtitle_offset':1.2,'danmaku_offset':-.5},{'subtitle_offset':1.2,'danmaku_offset':-.5});assert fake.props['sub-delay']==1.2
            fake.props['sub-visibility']=False;await native.sample(native.sessions[s['id']]);assert native.sessions[s['id']]['status']=='needs_attention'
            fake.props['sub-visibility']=True
            await native.close();assert not fake.quit,'exiting the workbench must keep IINA playing'
            fake.props['video-sync']='audio';second_library=PlaybackLibrary(store);second=NativeManager(store,second_library,AlignConfig());await second.start()
            assert fake.props['video-sync']=='display-vdrop';assert second_library.native_owners[item['id']]==s['id'];assert second.item_session(item['id'])['subtitle_ready']
            await second.end(s['id']);assert fake.quit;assert not second_library.native_owners;await second.close()
        await fake.close()
    asyncio.run(run());store.close()


def test_script_failure_never_reports_ready(tmp_path):
    store,l,item,sub=fixture(tmp_path);fake=FakeIINA(True);n=NativeManager(store,l,AlignConfig(),fake.launch)
    async def run():
        with patch.object(n,'executable',return_value=Path('/IINA')),patch.object(NativeManager,'_start_media',fake_media):
            with pytest.raises(ValueError,match='脚本'):await n.create(item['id'],sub['id'])
        assert all(s['status']=='failed' for s in n.sessions.values());assert fake.quit;assert not l.native_owners;await fake.close()
    asyncio.run(run());store.close()


def test_api_auth_missing_subtitle_unknown_ids_and_legacy_state(tmp_path):
    store,l,item,sub=fixture(tmp_path);store.close();app=create_app(tmp_path/'state',token='test',config=AlignConfig())
    with client_for(app) as c:
        assert c.get('/api/playback/native/capabilities',headers={'Authorization':''}).status_code==401
        assert c.post('/api/playback/native/sessions',json={'item_id':item['id']}).status_code==400
        assert c.post('/api/playback/native/sessions',json={'item_id':item['id'],'path':'/etc/hosts'}).status_code==422
        assert c.get('/api/playback/native/sessions/missing').status_code==404
        assert c.patch('/api/playback/items/'+item['id']+'/state',json={'preferred_engine':'shell'}).status_code==422
        assert c.patch('/api/playback/items/'+item['id']+'/state',json={'position':1}).status_code==200


def test_owned_range_reader_serves_original_bytes_and_survives_manager_close(tmp_path):
    import httpx
    store,l,item,sub=fixture(tmp_path);n=NativeManager(store,l,AlignConfig())
    async def run():
        key='range-test';directory=n.root/key;directory.mkdir();source=store.file(item['video_id'])['path']
        from bili_danmaku_mapper.cache import stamp
        s={'id':key,'identity':list(stamp(source)),'pid':os.getpid(),'status':'ready','item_id':item['id']}
        url=await n._start_media(s,source,directory);s['media_url']=url;n.save(s)
        original=Path(source).read_bytes()
        async with httpx.AsyncClient(trust_env=False) as client:
            response=await client.get(url,headers={'Range':'bytes=17-40'});assert response.status_code==206 and response.content==original[17:41]
            assert (await client.get(url.rsplit('/',2)[0]+'/wrong')).status_code==404
            assert (await client.get(url,headers={'Host':'evil.test'})).status_code==403
            await n.close()
            assert (await client.head(url)).status_code==200,'closing the workbench cannot stop active original-byte delivery'
        await n._stop_media(s)
    asyncio.run(run());store.close()

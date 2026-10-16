import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from bili_danmaku_mapper.workbench.playback import inspect_danmaku
from bili_danmaku_mapper.workbench.server import create_app
from tests.video_helpers import make_video


@pytest.fixture
def media(tmp_path):
    movie=make_video(tmp_path/"成品 with 'quotes'.mp4",3)
    comments=tmp_path/'成品弹幕.json'
    comments.write_text(json.dumps({'data':[{'time':1.2,'text':'hello','type':'scroll'}]}))
    return movie,comments


def app_for(tmp_path,media):
    movie,comments=media
    def pick(kind):return [comments if kind=='danmaku' else movie]
    return create_app(tmp_path/'state',token='test',picker=pick)


def client_for(app):
    return TestClient(app,base_url='http://127.0.0.1:8000',headers={'Authorization':'Bearer test'},raise_server_exceptions=False)


def import_video(client):
    picked=client.post('/api/pick',json={'kind':'playback_video'}).json()['files'][0]
    return client.post('/api/playback/items',json={'video_id':picked['id']}).json()


def test_manual_import_without_login_or_jobs_and_attach_later(tmp_path,media):
    app=app_for(tmp_path,media)
    with client_for(app) as c:
        item=import_video(c)
        assert item['kind']=='manual' and item['danmaku_id'] is None
        assert app.state.store.jobs()==[]
        saved=c.patch('/api/playback/items/'+item['id']+'/state',json={'position':1.5,'volume':.4,'playback_rate':1.5}).json()
        assert saved['position']==1.5
        picked=c.post('/api/pick',json={'kind':'danmaku'}).json()['files'][0]
        updated=c.post('/api/playback/items',json={'video_id':item['video_id'],'danmaku_id':picked['id'],'item_id':item['id']}).json()
        assert updated['id']==item['id'] and updated['state']['position']==1.5
        assert updated['state']['volume']==.4 and updated['state']['playback_rate']==1.5
        assert c.get('/api/files/'+updated['danmaku_id']).json()['data'][0]['text']=='hello'
        assert app.state.store.jobs()==[]
        assert 'path' not in json.dumps(c.get('/api/playback/items').json())
        assert c.post('/api/playback/items',json={'video_id':item['video_id'],'path':'/etc/hosts'}).status_code==422
        assert c.post('/api/playback/items',json={'video_id':'missing'}).status_code==404


def test_playback_state_survives_new_store_and_missing_files_are_reported(tmp_path,media):
    with client_for(app_for(tmp_path,media)) as c:
        item=import_video(c)
        patch={'position':1.7,'font_size':30,'opacity':.5,'danmaku_enabled':False,'show_estimated':False,'muted':True}
        assert c.patch('/api/playback/items/'+item['id']+'/state',json=patch).status_code==200
    with client_for(app_for(tmp_path,media)) as c:
        listing=c.get('/api/playback/items').json()
        assert listing['last_item_id']==item['id']
        assert all(listing['items'][0]['state'][k]==v for k,v in patch.items())
        media[0].rename(tmp_path/'moved.mp4')
        listing=c.get('/api/playback/items').json()
        assert listing['items'][0]['available'] is False
        assert c.get('/api/files/'+item['video_id']).status_code==400


@pytest.mark.parametrize('state',[{'position':-1},{'position':'NaN'},{'volume':1.1},{'opacity':-1},
                                    {'font_size':9},{'playback_rate':20},{'playing':True},{'path':'/etc/passwd'}])
def test_invalid_view_state_is_rejected(tmp_path,media,state):
    with client_for(app_for(tmp_path,media)) as c:
        item=import_video(c)
        assert c.patch('/api/playback/items/'+item['id']+'/state',json=state).status_code==422
        assert c.patch('/api/playback/items/missing/state',json={'position':1}).status_code==404


@pytest.mark.parametrize('document',[{},[],{'data':[{'time':-1,'text':'x'}]},
 {'data':[{'time':True,'text':'x'}]},{'data':[{'time':float('nan'),'text':'x'}]},
 {'data':[{'time':1,'text':7}]},{'data':[{'time':1,'text':'x','type':'html'}]}])
def test_invalid_danmaku_document(tmp_path,document):
    path=tmp_path/'comments.json';path.write_text(json.dumps(document))
    with pytest.raises(ValueError):inspect_danmaku(path)


def test_danmaku_picker_rejects_bad_json_allows_empty_and_cancel(tmp_path,media):
    path=media[1];app=app_for(tmp_path,media)
    with client_for(app) as c:
        path.write_text('broken')
        picked=c.post('/api/pick',json={'kind':'danmaku'}).json()
        assert not picked['files'] and picked['errors']
        path.write_text('{"data":[]}')
        picked=c.post('/api/pick',json={'kind':'danmaku'}).json()
        assert picked['files'][0]['count']==0
    app=create_app(tmp_path/'cancel',token='test',picker=lambda kind:[])
    with client_for(app) as c:
        assert c.post('/api/pick',json={'kind':'danmaku'}).json()=={'files':[],'errors':[]}


def test_finished_jobs_are_distinct_versions_use_final_file_and_never_change_task(tmp_path,media):
    app=app_for(tmp_path,media);store=app.state.store
    original=store.register_file(media[0],'selected',{'audio_streams':[]})
    target=store.register_file(media[0],'artifact')
    comments=store.register_file(media[1],'artifact')
    a=store.create_job({'key':'source','title':'Episode'},original['id'],{})
    b=store.create_job({'key':'source','title':'Episode'},original['id'],{},parent_id=a['id'])
    for job in (a,b):store.update(job['id'],status='complete',result={'files':{'target':target['id'],'danmaku':comments['id']}})
    with client_for(app) as c:
        listing=c.get('/api/playback/items').json()['items']
        assert {i['id'] for i in listing}=={'job:'+a['id'],'job:'+b['id']}
        assert all(i['video_id']==target['id'] for i in listing)
        c.patch('/api/playback/items/job:'+a['id']+'/state',json={'position':1.3})
        latest=c.get('/api/playback/items').json()['items']
        states={i['id']:i['state']['position'] for i in latest}
        assert states['job:'+a['id']]==1.3 and states['job:'+b['id']]==0
        assert store.job(a['id'])['status']=='complete' and len(store.jobs())==2
        assert c.get('/api/files/'+target['id'],headers={'Range':'bytes=10-39'}).status_code==206


def test_new_playback_endpoints_require_session(tmp_path):
    app=create_app(tmp_path/'state',token='secret',picker=lambda kind:[])
    with TestClient(app,base_url='http://127.0.0.1:8000') as c:
        assert c.get('/api/playback/items').status_code==401
        assert c.get('/api/playback/items/x/media').status_code==401
        assert c.get('/api/playback/items/x/prepared').status_code==401
        assert c.get('/api/playback/preparations/x').status_code==401
        assert c.post('/api/playback/items/x/prepare',json={'strategy':'remux'}).status_code==401
        assert c.post('/api/playback/items',json={'video_id':'x'}).status_code==401
        assert c.patch('/api/playback/items/x/state',json={'position':0}).status_code==401

import pytest
from bili_danmaku_mapper.workbench.subtitles import read_subtitles
from tests.test_playback import client_for, import_video, media


def read(tmp_path, content, suffix='.srt', encoding='utf-8'):
    path=tmp_path/('字幕 with quotes\''+suffix)
    path.write_bytes(content.encode(encoding))
    return read_subtitles(path)


@pytest.mark.parametrize('encoding',['utf-8-sig','utf-16','utf-32','gb18030'])
def test_srt_encodings_plain_text_and_overlaps(tmp_path,encoding):
    result=read(tmp_path,'2\r\n00:00:02,100 --> 00:00:04,200\r\n<b>中文</b><br>对白 &amp; text\r\n\r\n1\r\n00:00:01,000 --> 00:00:03,000\r\n<script>alert(1)</script>先出现',encoding=encoding)
    assert result['cues']==[{'start':1,'end':3,'text':'先出现'},{'start':2.1,'end':4.2,'text':'中文\n对白 & text'}]


def test_vtt_settings_notes_and_long_time(tmp_path):
    result=read(tmp_path,'WEBVTT\n\nNOTE ignored\ntext\n\nSTYLE\n::cue {color:red}\n\nfirst\n01:02.300 --> 01:04.000 align:start\n<c.red>字幕</c>\n\n02:03:04.000 --> 02:03:05.000\n末尾',suffix='.vtt')
    assert result['cues'][0]['start']==62.3
    assert result['cues'][1]['start']==7384
    assert result['cues'][0]['text']=='字幕'


@pytest.mark.parametrize('suffix',['.ass','.ssa'])
def test_ass_plain_dialogue_commas_and_drawing(tmp_path,suffix):
    result=read(tmp_path,r'''[Script Info]
Title: text
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,ignored
Dialogue: 0,0:00:01.20,0:00:03.50,Default,,0,0,0,,{\pos(1,2)\b1}对白, comma\N第二行\h空格
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,{\p1}m 0 0 l 100 100{\p0}正常对白
''',suffix)
    assert result['cues'][0]=={'start':1.2,'end':3.5,'text':'对白, comma\n第二行 空格'}
    assert result['cues'][1]['text']=='正常对白'
    assert '特效' in result['note']


@pytest.mark.parametrize('text,suffix',[('', '.srt'),('1\n00:00:02,000 --> 00:00:01,000\nx','.srt'),
 ('1\n00:61:00,000 --> 00:62:00,000\nx','.srt'),('1\n1 --> 2\nx','.srt'),
 ('WEBVTT\n\nmissing timing','.vtt'),('00:01.000 --> 00:02.000\nx','.vtt'),
 ('[Events]\nDialogue: broken','.ass'),('abc','.txt')])
def test_invalid_subtitles(tmp_path,text,suffix):
    with pytest.raises(ValueError):read(tmp_path,text,suffix)


def test_file_and_cue_limits(tmp_path,monkeypatch):
    import bili_danmaku_mapper.workbench.subtitles as subtitles
    monkeypatch.setattr(subtitles,'MAX_BYTES',10)
    with pytest.raises(ValueError,match='20 MiB'):read(tmp_path,'x'*11)
    monkeypatch.setattr(subtitles,'MAX_BYTES',1024)
    monkeypatch.setattr(subtitles,'MAX_CUES',1)
    with pytest.raises(ValueError,match='100000'):read(tmp_path,'1\n00:00:01,000 --> 00:00:02,000\nx\n\n2\n00:00:03,000 --> 00:00:04,000\ny')


def test_attach_subtitle_per_item_persist_missing_and_unload(tmp_path,media):
    subtitle=tmp_path/'字幕.srt';subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\n本地字幕')
    # Use a picker returning only the selected file, keeping arbitrary paths off APIs.
    from bili_danmaku_mapper.workbench.server import create_app
    def build():return create_app(tmp_path/'substate',token='test',picker=lambda kind:[subtitle if kind=='subtitle' else media[0]])
    with client_for(build()) as c:
        item=import_video(c);other=import_video(c)
        selected=c.post('/api/pick',json={'kind':'subtitle'}).json()['files'][0]
        assert selected['kind']=='subtitle' and selected['count']==1
        assert c.get('/api/playback/subtitles/'+selected['id']).json()['cues'][0]['text']=='本地字幕'
        url='/api/playback/items/'+item['id']+'/state'
        assert c.patch(url,json={'subtitle_id':selected['id'],'subtitle_offset':-5,'subtitle_enabled':False}).status_code==200
        assert c.patch(url,json={'subtitle_offset':5}).json()['subtitle_offset']==5
        for invalid in [-5.1,5.1,'NaN']:
            assert c.patch(url,json={'subtitle_offset':invalid}).status_code==422
        assert c.patch(url,json={'subtitle_id':item['video_id']}).status_code==400
        assert c.get('/api/playback/subtitles/'+item['video_id']).status_code==400
        assert c.get('/api/playback/subtitles/not-selected').status_code==404
    with client_for(build()) as c:
        items={i['id']:i for i in c.get('/api/playback/items').json()['items']}
        assert items[item['id']]['state']['subtitle_offset']==5
        assert items[other['id']]['state']['subtitle_id'] is None
        assert items[item['id']]['subtitle']['name']=='字幕.srt'
        assert 'path' not in str(items[item['id']])
        subtitle.unlink()
        items={i['id']:i for i in c.get('/api/playback/items').json()['items']}
        assert items[item['id']]['available'] and '丢失' in items[item['id']]['subtitle_warning']
        assert c.get('/api/playback/subtitles/'+selected['id']).status_code==400
        assert c.patch(url,json={'subtitle_id':None,'subtitle_offset':0}).status_code==200


def test_subtitles_and_vendored_assets_are_session_protected(tmp_path):
    from fastapi.testclient import TestClient
    from bili_danmaku_mapper.workbench.server import create_app
    app=create_app(tmp_path/'state',token='secret',picker=lambda kind:[])
    with TestClient(app,base_url='http://127.0.0.1:8000') as c:
        assert c.get('/api/playback/subtitles/x').status_code==401
    app=create_app(tmp_path/'assets',token='test',picker=lambda kind:[])
    with client_for(app) as c:
        assert c.get('/assets/subtitle-renderer.js').status_code==200
        assert c.get('/assets/vendor/nplayer-1.0.15/index.min.js').status_code==200
        assert 'MIT' in c.get('/assets/vendor/nplayer-1.0.15/LICENSE').text
        assert c.get('/assets/vendor/nplayer-1.0.15/provenance.json').json()['version']=='1.0.15'
        assert c.get('/assets/vendor/nplayer-1.0.15/other').status_code==404


def test_stale_state_cannot_overwrite_latest_subtitle_offset(tmp_path,media):
    from tests.test_playback import app_for
    with client_for(app_for(tmp_path,media)) as c:
        item=import_video(c);url='/api/playback/items/'+item['id']+'/state'
        assert c.patch(url,json={'subtitle_offset':2.3,'write_revision':200}).json()['subtitle_offset']==2.3
        assert c.patch(url,json={'subtitle_offset':0,'write_revision':100}).json()['subtitle_offset']==2.3
        assert c.patch(url,json={'subtitle_offset':-1,'write_revision':201}).json()['subtitle_offset']==-1
        assert 'write_revision' not in c.get('/api/playback/items').json()['items'][0]['state']

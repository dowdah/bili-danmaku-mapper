from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pytest

from bili_danmaku_mapper import AlignConfig, align_videos, map_danmaku
from bili_danmaku_mapper.cache import identity, session
from bili_danmaku_mapper.performance import collecting
from bili_danmaku_mapper.media import Features, VideoInfo, dense_windows, merge_windows, concatenate
from tests.video_helpers import make_video
from tests.audio_helpers import mux_sound


def test_identity_reuses_digest_and_invalidates_same_size_change(tmp_path, monkeypatch):
    from bili_danmaku_mapper import media
    p=tmp_path/'video';p.write_bytes(b'first')
    config=AlignConfig(cache_dir=str(tmp_path/'cache'))
    calls=[];original=media.content_digest
    monkeypatch.setattr(media,'content_digest',lambda path:(calls.append(path),original(path))[1])
    with collecting() as metrics,session():
        first=identity(p,config)
        assert identity(p,config)==first
    assert len(calls)==1 and metrics.counts['identity_memory_hits']==1
    with session():assert identity(p,config)==first
    assert len(calls)==1
    p.write_bytes(b'other')
    with session():assert identity(p,config)!=first
    assert len(calls)==2


def test_identity_change_during_task_is_rejected(tmp_path):
    p=tmp_path/'video';p.write_bytes(b'first')
    with pytest.raises(RuntimeError,match='changed'):
        with session():
            identity(p,AlignConfig(cache_dir=str(tmp_path/'cache')))
            p.write_bytes(b'other')


def test_merge_bounds_and_long_split():
    assert merge_windows([(2,5),(6,8),(20,270)],300)==[(2,8),(20,140),(140,260),(260,270)]
    assert merge_windows([(-1,2),(299,400)],300)==[(0,2),(299,300)]


def fake_features(info,start,end):
    times=np.arange(np.ceil(start*10)/10,end,.1)
    n=len(times)
    return Features(times,np.zeros((n,8),np.uint8),np.zeros((n,8),np.uint8),
                    np.zeros((n,24,32),np.uint8),np.ones(n,np.float32)*10,info)


def corrupt_deflate(path):
    import struct,zipfile
    with zipfile.ZipFile(path) as archive:entry=archive.infolist()[0]
    data=bytearray(path.read_bytes())
    names,extra=struct.unpack_from('<HH',data,entry.header_offset+26)
    # Reserved DEFLATE block type, inside the actual payload (including ZIP64 extra).
    data[entry.header_offset+30+names+extra]=255
    path.write_bytes(data)


@pytest.mark.parametrize('damage',['truncated','deflate'])
def test_dense_tiles_merge_reuse_repair_and_backend_partition(tmp_path,monkeypatch,damage):
    from bili_danmaku_mapper import media
    p=tmp_path/'video';p.write_bytes(b'x')
    info=VideoInfo(str(p),180,0,10)
    config=AlignConfig(cache_dir=str(tmp_path/'cache'))
    calls=[]
    def decode(info,config,rate,start,end):
        calls.append((start,end));return fake_features(info,start,end)
    monkeypatch.setattr(media,'_decode_retry',decode)
    with session():
        a=concatenate(list(dense_windows(info,[(1,80)],config)),info)
    assert calls==[(0.,90.75)]
    with session():b=concatenate(list(dense_windows(info,[(1,80)],config)),info)
    assert np.array_equal(a.times,b.times) and len(calls)==1
    broken=next((tmp_path/'cache/dense-v1').glob('*.npz'))
    if damage=='deflate':corrupt_deflate(broken)
    else:broken.write_bytes(b'broken')
    with session():list(dense_windows(info,[(1,80)],config))
    assert len(calls)==2
    with session():list(dense_windows(info,[(1,50)],replace(config,hwaccel='videotoolbox')))
    assert len(calls)==3


def test_alignment_cache_remap_and_anchor_invalidation(tmp_path,monkeypatch):
    import bili_danmaku_mapper.alignment as core
    video=make_video(tmp_path/'video.mp4',3)
    config=AlignConfig(cache_dir=str(tmp_path/'cache'),evidence_mode='visual')
    first=align_videos(video,video,config)
    def forbidden(*args,**kwargs):raise AssertionError('media decoding or matching on result-cache hit')
    monkeypatch.setattr(core,'_align_uncached',forbidden)
    warm=align_videos(video,video,config)
    assert warm.diagnostics['performance']['counters']['alignment_cache_hits']==1
    mapped=map_danmaku({'data':[{'time':1.2}]},warm)
    assert mapped.report['counts']['reliable']==1
    with pytest.raises(AssertionError):align_videos(video,video,config,[{'source_time':1.,'target_time':1.}])
    with pytest.raises(AssertionError):align_videos(video,video,replace(config,alignment_cache=False))
    with pytest.raises(AssertionError):align_videos(video,video,replace(config,crop_ratio=.7))


def test_merged_refinement_equivalence_and_warm_dense_cache(tmp_path):
    source=make_video(tmp_path/'source.mp4',5,timeline=lambda t:t+.7)
    target=make_video(tmp_path/'target.mp4',6,fps=30)
    config=AlignConfig(cache_dir=str(tmp_path/'cache'),evidence_mode='visual',alignment_cache=False)
    old=align_videos(source,target,replace(config,merge_refinement=False,dense_cache=False))
    new=align_videos(source,target,config)
    def points(a):return [(round(x.source_time,6),round(x.target_time,6)) for s in a.segments for x in s.anchors]
    assert points(old)==points(new)
    warm=align_videos(source,target,config)
    assert warm.diagnostics['performance']['counters'].get('ffmpeg_starts',0)==0
    assert points(warm)==points(new)


def test_audio_first_avoids_full_visual_pass(tmp_path):
    sv=make_video(tmp_path/'source-v.mp4',12,timeline=lambda t:t+.7)
    tv=make_video(tmp_path/'target-v.mp4',13,fps=30)
    source=mux_sound(sv,tmp_path/'source.mp4',12,lambda t:t+.7)
    target=mux_sound(tv,tmp_path/'target.mp4',13,lambda t:t)
    result=align_videos(source,target,AlignConfig(cache_dir=str(tmp_path/'cache'),hybrid_strategy='audio_first'))
    assert result.diagnostics['hybrid_strategy']=='audio_first'
    assert result.diagnostics['audio']['audio_anchors']>10
    events=result.diagnostics['performance']['decode_events']
    assert all(e.get('end',0)-e.get('start',0)<13 for e in events if 'end' in e and e.get('rate') != 0)
    mapped=map_danmaku({'data':[{'time':float(t)} for t in np.arange(1,11,.1)]},result)
    accepted=[r for r in mapped.report['entries'] if r['target_time'] is not None]
    assert len(accepted)>=95
    assert np.percentile([abs(r['target_time']-r['source_time']-.7) for r in accepted],95)<.2


def test_benchmark_yields_to_production_without_changing_jobs(tmp_path):
    from tests.optimization_benchmark import run_guarded
    import sqlite3,threading,time,sys
    db=tmp_path/'production.sqlite3'
    c=sqlite3.connect(db);c.execute('create table jobs(body text)');c.commit();c.close()
    def incoming():
        time.sleep(.4)
        c=sqlite3.connect(db);c.execute('insert into jobs values(?)',(json.dumps({'id':'new','status':'queued'}),));c.commit();c.close()
    thread=threading.Thread(target=incoming);thread.start()
    output=tmp_path/'guard.json'
    with pytest.raises(RuntimeError,match='production_queue_active'):
        run_guarded([sys.executable,'-c','import time;time.sleep(30)'],tmp_path,output,db,tmp_path)
    thread.join()
    assert json.loads(output.read_text())['stopped_reason']=='production_queue_active'
    c=sqlite3.connect(db)
    assert json.loads(c.execute('select body from jobs').fetchone()[0])=={'id':'new','status':'queued'}
    c.close()


def test_cached_dense_tiles_answer_sparse_windows_without_decode(tmp_path,monkeypatch):
    from bili_danmaku_mapper import media
    p=tmp_path/'video';p.write_bytes(b'x');info=VideoInfo(str(p),90,0,10)
    config=AlignConfig(cache_dir=str(tmp_path/'cache'))
    calls=[]
    def decode(info,config,rate,start,end):
        calls.append((start,end,rate));return fake_features(info,start,end)
    monkeypatch.setattr(media,'_decode_retry',decode)
    with session():list(dense_windows(info,[(1,50)],config))
    first=len(calls)
    with session():sparse=media.extract_features(p,config,rate=2,start=28.5,end=32.5,_info=info)
    assert len(calls)==first and len(sparse)>=7
    assert np.all(np.diff(sparse.times)>=.5-1e-7)


def test_sparse_refinement_does_not_decode_whole_tile(tmp_path,monkeypatch):
    from bili_danmaku_mapper import media
    p=tmp_path/'video';p.write_bytes(b'x');info=VideoInfo(str(p),180,0,10)
    config=AlignConfig(cache_dir=str(tmp_path/'cache'));calls=[]
    def decode(info,config,rate,start,end):
        calls.append((start,end));return fake_features(info,start,end)
    monkeypatch.setattr(media,'_decode_retry',decode)
    with session():first=list(dense_windows(info,[(35,36)],config))
    assert calls==[(34.25,36.75)]
    with session():warm=list(dense_windows(info,[(35,36)],config))
    assert len(calls)==1 and np.array_equal(first[0].times,warm[0].times)


def test_changed_media_is_not_a_hardware_retry(tmp_path,monkeypatch):
    from bili_danmaku_mapper import media
    from bili_danmaku_mapper.cache import InputChangedError
    info=VideoInfo(str(tmp_path/'video'),10,0,24);calls=[]
    def decode(*args):calls.append(1);raise InputChangedError('Input media changed')
    monkeypatch.setattr(media,'_decode',decode)
    with pytest.raises(InputChangedError):media._decode_retry(info,AlignConfig(hwaccel='videotoolbox'),2,0,1)
    assert len(calls)==1


def test_audio_change_does_not_publish_cache(tmp_path,monkeypatch):
    from bili_danmaku_mapper import audio
    from bili_danmaku_mapper.cache import InputChangedError
    p=tmp_path/'video';p.write_bytes(b'first');info=VideoInfo(str(p),10,0,24)
    def decode(*args):p.write_bytes(b'other');return None
    monkeypatch.setattr(audio,'_fingerprint_decode',decode)
    with pytest.raises(InputChangedError):audio.extract_audio(info,{'index':0,'ordinal':0},AlignConfig(cache_dir=str(tmp_path/'cache')))
    assert not list((tmp_path/'cache').glob('*.audio.npz'))


def test_isolated_audio_points_do_not_constrain_movie():
    from bili_danmaku_mapper.audio import supported_audio
    from bili_danmaku_mapper.models import Anchor
    points=[Anchor(5,10,evidence='audio'),Anchor(500,540,evidence='audio')]
    kept,seconds=supported_audio(points,1000,1100,AlignConfig())
    assert kept==[] and seconds==0


def test_normalized_audio_window_keeps_original_origin(tmp_path):
    from bili_danmaku_mapper.media import probe_video
    from bili_danmaku_mapper.audio import extract_audio,select_streams
    video=make_video(tmp_path/'sv.mp4',5)
    source=mux_sound(video,tmp_path/'source.mkv',5,offset=5)
    config=AlignConfig(cache_dir=str(tmp_path/'cache'))
    info=probe_video(source,config);stream,_,_=select_streams(source,source,config)
    features=extract_audio(info,stream,config,25/24,1,3)
    assert .08<features.frame_times[0]-25/24<.11
    assert features.frame_times[-1]<3*25/24
    assert np.all(np.diff(features.frame_times)>0)


@pytest.mark.parametrize('sped_up',[True,False])
def test_pal_rate_hypothesis_requires_actual_audio_evidence(tmp_path,sped_up):
    scale=25/24 if sped_up else 1.
    timeline=lambda t:t*scale+.7
    source_v=make_video(tmp_path/'sv.mp4',10,fps=25,timeline=timeline)
    target_v=make_video(tmp_path/'tv.mp4',12,fps=24)
    source=mux_sound(source_v,tmp_path/'s.mp4',10,timeline,rate=44100)
    target=mux_sound(target_v,tmp_path/'t.mp4',12,rate=48000)
    a=align_videos(source,target,AlignConfig(cache_dir=str(tmp_path/'cache')))
    normalization=a.diagnostics['audio']['normalization']
    if sped_up:
        assert normalization['status']=='verified_sequence'
        assert abs(normalization['scale']-25/24)<1e-8
    else:assert normalization['scale']==1.
    mapped=map_danmaku({'data':[{'time':float(t)} for t in np.arange(1,9,.1)]},a)
    accepted=[r for r in mapped.report['entries'] if r['target_time'] is not None]
    assert len(accepted)>=75
    assert np.percentile([abs(r['target_time']-timeline(r['source_time'])) for r in accepted],95)<.1


def test_audio_early_outlier_does_not_lock_future_matches():
    from bili_danmaku_mapper.audio import monotonic_audio_path
    from bili_danmaku_mapper.models import Anchor
    points=[Anchor(0,700,evidence='audio')]+[Anchor(float(t),float(t)+2,evidence='audio') for t in range(1,20)]
    path=monotonic_audio_path(points)
    assert len(path)==19 and path[0].source_time==1
    assert all(b.target_time>a.target_time for a,b in zip(path,path[1:]))


def test_audio_speed_search_configuration_precedence(tmp_path):
    env=tmp_path/'.env';env.write_text('AUDIO_SPEED_SEARCH=false\n')
    assert AlignConfig.resolve(env).audio_speed_search is False
    assert AlignConfig.resolve(env,audio_speed_search=True).audio_speed_search is True


def test_dense_tiles_answer_new_precision_windows_without_redecode(tmp_path,monkeypatch):
    from bili_danmaku_mapper import media
    p=tmp_path/'video';p.write_bytes(b'x');info=VideoInfo(str(p),90,0,10)
    config=AlignConfig(cache_dir=str(tmp_path/'cache'));calls=[]
    def decode(info,config,rate,start,end):
        calls.append((start,end));return fake_features(info,start,end)
    monkeypatch.setattr(media,'_decode_retry',decode)
    with session():list(dense_windows(info,[(1,50)],config))
    before=len(calls)
    with session():parts=list(dense_windows(info,[(35.1,36.2)],config))
    assert len(calls)==before and parts


def test_many_local_checks_share_one_target_coarse_decode(monkeypatch):
    from bili_danmaku_mapper import audio_first as flow
    from bili_danmaku_mapper import audio,alignment,media
    from bili_danmaku_mapper.models import Anchor
    times=np.arange(0,300,.5);rng=np.random.default_rng(42)
    hashes=rng.integers(0,256,(len(times),8),dtype=np.uint8)
    calls=[];backends=[]
    def info(path,config):return VideoInfo(str(path),300,0,24)
    def extract(path,config,start=0.,end=None,**kw):
        calls.append((str(path),start,end))
        backends.append((str(path),config.hwaccel))
        mask=(times>=start)&(times<(end or 300))
        return Features(times[mask],hashes[mask],hashes[mask],np.zeros((sum(mask),24,32),np.uint8),
                        np.ones(sum(mask),np.float32)*10,info(path,config))
    monkeypatch.setattr(flow,'probe_video',info)
    monkeypatch.setattr(flow,'extract_features',extract)
    monkeypatch.setattr(audio,'align_audio',lambda *a:([Anchor(float(t),float(t),.9,evidence='audio') for t in np.arange(.25,299.8,.5)],{}))
    monkeypatch.setattr(alignment,'refine_path',lambda path,s,t,c:[Anchor(float(s.times[i]),float(t.times[j]),q) for i,j,q in path])
    from bili_danmaku_mapper.progress import reporting
    with reporting(lambda event:json.dumps(event)):
        result=flow.align_audio_first('source','target',AlignConfig(cache_dir=None,hwaccel='videotoolbox',source_hwaccel='software'))
    target_calls=[c for c in calls if c[0]=='target']
    assert target_calls==[('target',0.,None)]
    assert result.diagnostics['shared_target_coarse'] is True
    assert all(backend is None if role=='source' else backend=='videotoolbox' for role,backend in backends)
    mapped=map_danmaku({'data':[{'time':120.2}]},result)
    assert mapped.report['entries'][0]['target_time']==pytest.approx(120.2)


def test_visual_flow_routes_each_input_backend(monkeypatch):
    from bili_danmaku_mapper import alignment
    calls=[]
    def extract(path,config,**kw):
        calls.append((path,config.hwaccel))
        return fake_features(VideoInfo(path,2,0,10),0,2)
    monkeypatch.setattr(alignment,'extract_features',extract)
    config=AlignConfig(evidence_mode='visual',refine=False,hwaccel='videotoolbox',source_hwaccel='software')
    alignment._align_uncached('source','target',config)
    assert calls==[('source',None),('target','videotoolbox')]


def test_precision_honors_target_backend_override(monkeypatch):
    from bili_danmaku_mapper import alignment,media
    info=VideoInfo('target',2,0,10);features=fake_features(info,0,2);calls=[]
    def dense(info,windows,config):
        calls.append(config.hwaccel);yield features
    monkeypatch.setattr(media,'dense_windows',dense)
    points=alignment.refine_path([(5,5,.9)],features,features,AlignConfig(target_hwaccel='videotoolbox'))
    assert calls==['videotoolbox'] and points[0].target_time==pytest.approx(.5)


def test_audio_first_requires_opt_in_until_real_film_acceptance():
    assert AlignConfig().hybrid_strategy == "visual_first"
    assert AlignConfig(hybrid_strategy="audio_first").hybrid_strategy == "audio_first"


def test_audio_deflate_cache_rebuild(tmp_path,monkeypatch):
    from bili_danmaku_mapper import audio
    video=tmp_path/'video';video.write_bytes(b'x')
    config=AlignConfig(cache_dir=str(tmp_path/'cache'));calls=[]
    features=audio.AudioFeatures(np.array([.1,.2]),np.ones((2,3),np.uint16),
        np.array([.1,.2]),np.ones(2,np.float32),np.ones((2,128),np.float32))
    monkeypatch.setattr(audio,'_fingerprint_decode',lambda *args:(calls.append(1),features)[1])
    info=VideoInfo(str(video),1,0,10);stream={'index':0,'ordinal':0}
    audio.extract_audio(info,stream,config)
    corrupt_deflate(next((tmp_path/'cache').glob('*.audio.npz')))
    repaired=audio.extract_audio(info,stream,config)
    assert len(calls)==2 and np.array_equal(repaired.keys,features.keys)


def test_identity_wrong_json_type_is_rebuilt(tmp_path):
    video=tmp_path/'video';video.write_bytes(b'content')
    config=AlignConfig(cache_dir=str(tmp_path/'cache'))
    original=identity(video,config)
    record=next((tmp_path/'cache/identity-v1').glob('*.json'));record.write_text('[]')
    assert identity(video,config)==original
    assert json.loads(record.read_text())['digest']==original

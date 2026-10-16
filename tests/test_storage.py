from types import SimpleNamespace
from pathlib import Path
import pytest
from bili_danmaku_mapper.workbench import storage
from bili_danmaku_mapper.workbench.playback_worker import build_variant, identity
from tests.video_helpers import make_video


def test_large_movie_rejected_before_directory_or_process_created(tmp_path,monkeypatch):
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=23*storage.GIB))
    with pytest.raises(ValueError,match='空间不足'):
        storage.check_space(tmp_path/'not-created',83*storage.GIB)
    assert not (tmp_path/'not-created').exists()
    assert storage.check_space(tmp_path,storage.GIB)['reserve_bytes']==2*storage.GIB


def test_runtime_floor_uses_destination_free_space(tmp_path,monkeypatch):
    seen=[]
    def available(path):
        seen.append(Path(path));return SimpleNamespace(free=storage.GIB)
    monkeypatch.setattr(storage.shutil,'disk_usage',available)
    with pytest.raises(ValueError,match='已停止生成'):
        storage.check_space(tmp_path/'new'/'nested',admission=False)
    assert seen==[tmp_path]


def test_runtime_guard_stops_owned_encoder_and_removes_partial(tmp_path,monkeypatch):
    import bili_danmaku_mapper.workbench.playback_worker as worker
    movie=make_video(tmp_path/'original.mp4',1)
    output=tmp_path/'copy'
    real_check=worker.check_space
    calls=[]
    def guard(path,*args,**kwargs):
        calls.append(kwargs.get('admission',True))
        if not kwargs.get('admission',True):raise ValueError('播放准备空间不足')
        return real_check(path,*args,**kwargs)
    monkeypatch.setattr(worker,'check_space',guard)
    with pytest.raises(ValueError,match='空间不足'):
        build_variant({'source':str(movie),'source_identity':identity(movie),'directory':str(output),
                       'strategy':'remux','audio_stream':0,'ffmpeg':'ffmpeg','ffprobe':'ffprobe'})
    assert calls[0] and False in calls
    assert not (output/'playback.partial.mp4').exists()
    assert not (output/'playback.mp4').exists()
    assert movie.exists()

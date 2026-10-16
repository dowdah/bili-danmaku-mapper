"""Isolated browser acceptance fixture. Never uses an account or Bilibili downloads."""
import asyncio
from pathlib import Path
import socket
import tempfile
import sys
import subprocess
import uvicorn

from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.io import atomic_json
from bili_danmaku_mapper.workbench.server import create_app
from tests.video_helpers import make_video


def main():
    base=Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='bdm-ui-'))
    base.mkdir(parents=True,exist_ok=True)
    videos=[(base/f'合成影片 {i:02d}.mp4') if (base/f'合成影片 {i:02d}.mp4').exists() else make_video(base/f'合成影片 {i:02d}.mp4',12,fps=24 if i==1 else 30) for i in (1,2)]
    playback_choices=[videos[1]]
    if len(sys.argv)>2 and sys.argv[2]=='containers':
        base_video=videos[1]
        mkv=base/'原片 第02集.mkv';avi=base/'需换封装.avi';unsupported=base/'需确认完整转码.mkv'
        def convert(output,codec):
            if not output.exists():
                subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(base_video),'-f','lavfi','-i','sine=frequency=660:duration=12','-map','0:v','-map','1:a','-c:v',codec,'-c:a','aac',str(output)],check=True)
        convert(mkv,'copy')
        if not avi.exists():
            subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(base_video),'-f','lavfi','-i','sine=frequency=660:duration=12','-map','0:v','-map','1:a','-c:v','libx264','-bf','0','-c:a','aac',str(avi)],check=True)
        convert(unsupported,'ffv1')
        videos[1]=mkv
        playback_choices=[mkv,avi,unsupported]
    movie_picks=0
    class Source:
        async def resolve(self,url):
            return {'title':'合成素材 · 浏览器验收','kind':'video','items':[
              {'key':f'synthetic:{i}','kind':'video','title':f'合成剧情 第{i}集','number':i,'duration':12,
               'selected':i==1,'focused':i==1} for i in (1,2)]}
        async def download(self,source,directory,quality,notify):
            notify('downloading',0,100)
            await asyncio.sleep(1)
            notify('downloading',100,100)
            path=directory/'raw.json'
            atomic_json(path,{'data':[{'time':t,'text':f'合成弹幕 {t:.1f}','type':'scroll','color':'#ffffff'} for t in (.7,1.2,2.2,3.2)]},overwrite=True)
            return {'source_video':str(videos[0]),'raw_danmaku':str(path)}
    manual_comments=base/'手动弹幕.json'
    atomic_json(manual_comments,{'data':[{'time':t,'text':f'手动成品弹幕 {t}','type':'scroll'} for t in (.7,1.2,2.2,5.2,7.2,9.2)]},overwrite=True)
    bad_comments=base/'无效弹幕.json';bad_comments.write_text('broken')
    empty_comments=base/'空弹幕.json';atomic_json(empty_comments,{'data':[]},overwrite=True)
    subtitles=base/'合成字幕.srt'
    subtitles.write_text('1\n00:00:01,000 --> 00:00:04,000\n这是外挂字幕 · 可提前或延后\n\n2\n00:00:05,000 --> 00:00:10,000\n字幕与弹幕共用影片时间轴\nSubtitle on the media clock')
    picks=0
    def picker(kind):
        nonlocal picks,movie_picks
        if kind=='subtitle':return [subtitles]
        if kind=='playback_video':
            result=playback_choices[min(movie_picks,len(playback_choices)-1)];movie_picks+=1;return [result]
        if kind=='danmaku':
            result=[manual_comments,bad_comments,empty_comments][min(picks,2)]
            picks+=1
            return [result]
        return videos
    server=None
    app=create_app(base/'state',token='browser-acceptance-only',config=AlignConfig(cache_dir=str(base/'features')),
                   downloader=Source(),picker=picker,exit_callback=lambda:setattr(server,'should_exit',True))
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
    server=uvicorn.Server(uvicorn.Config(app,access_log=False,log_level='warning'))
    print(f'http://127.0.0.1:{sock.getsockname()[1]}/#browser-acceptance-only',flush=True)
    server.run(sockets=[sock])


if __name__=='__main__':main()

"""Durable, bounded, owned playback workers, independent of the alignment queue."""
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from ..io import atomic_json
from .files import inspect_file
from .playback_worker import identity
from .storage import check_space


class PlaybackPreparer:
    def __init__(self, store, library, config):
        self.store, self.library, self.config = store, library, config
        self.tasks, self.processes = {}, {}
        self.slot = asyncio.Semaphore(1)
        self.closed = False
        with store.lock:
            store.db.execute('CREATE TABLE IF NOT EXISTS playback_preparations(id TEXT PRIMARY KEY, body TEXT NOT NULL)')
            for row in store.db.execute('SELECT id,body FROM playback_preparations').fetchall():
                body = json.loads(row['body'])
                if body['status'] in ('queued','running'):
                    body.update(status='interrupted', error='服务曾中断，重试时从本阶段重新准备')
                    store.db.execute('UPDATE playback_preparations SET body=? WHERE id=?',(json.dumps(body),row['id']))
            store.db.commit()

    def read(self, key):
        with self.store.lock:
            row = self.store.db.execute('SELECT body FROM playback_preparations WHERE id=?',(key,)).fetchone()
        if not row:
            raise KeyError('Unknown preparation')
        body = json.loads(row[0])
        if body['status'] == 'complete':
            try:
                source, output = self.store.file(body['source_id']), self.store.file(body['file_id'])
                if identity(source['path']) != body['source_identity'] or identity(output['path']) != body['output_identity']:
                    raise ValueError('Changed media')
            except (KeyError, OSError, ValueError):
                body.update(status='failed', error='原片或播放文件已变化，请重新准备')
        return body

    def write(self, key, **changes):
        with self.store.lock:
            row = self.store.db.execute('SELECT body FROM playback_preparations WHERE id=?',(key,)).fetchone()
            body = json.loads(row[0]) if row else {'id':key}
            body.update(changes)
            self.store.db.execute('INSERT OR REPLACE INTO playback_preparations VALUES(?,?)',(key,json.dumps(body,allow_nan=False)))
            self.store.db.commit()
        return body

    @staticmethod
    def public(body):
        names = ('id','status','strategy','audio_stream','started','file_id','error','timeline_verified','timeline_offset','max_video_error','timestamps_checked')
        return {name:body[name] for name in names if name in body}

    def cached(self, item_id, audio_stream=0):
        item = self.library.item(item_id)
        with self.store.lock:
            rows = self.store.db.execute('SELECT id,body FROM playback_preparations').fetchall()
        results = []
        for row in rows:
            body = json.loads(row['body'])
            if body.get('source_id') == item['video_id'] and body.get('audio_stream') == audio_stream:
                current = self.read(row['id'])
                if current['status'] == 'complete':
                    results.append(self.public(current))
        return results

    async def prepare(self, item_id, strategy, audio_stream=0, confirmed=False):
        if self.closed:
            raise ValueError('服务正在退出')
        if strategy not in ('remux','audio','transcode'):
            raise ValueError('播放准备方式无效')
        if strategy == 'transcode' and not confirmed:
            raise ValueError('完整转码需要在播放器中确认')
        item = self.library.item(item_id)
        source = self.store.file(item['video_id'])
        info = await asyncio.to_thread(inspect_file, source['path'], self.config.ffprobe)
        if self.closed:
            raise ValueError('服务正在退出')
        if audio_stream < 0 or audio_stream >= max(1,len(info['audio_streams'])):
            raise ValueError('播放音轨序号超出范围')
        stamp = identity(source['path'])
        key = hashlib.sha256(json.dumps([source['id'],stamp,audio_stream,strategy,'pts-verified-v2']).encode()).hexdigest()[:32]
        if key in self.tasks:
            return self.public(self.read(key))
        with contextlib.suppress(KeyError):
            cached = self.read(key)
            if cached['status'] == 'complete':
                return self.public(cached)
        check_space(self.store.directory, stamp[2])
        self.write(key,status='queued',strategy=strategy,audio_stream=audio_stream,source_id=source['id'],
                   source_identity=stamp,error=None,file_id=None,started=time.time())
        task = asyncio.create_task(self.run(key, source['path']))
        self.tasks[key] = task
        task.add_done_callback(lambda task:self.tasks.pop(key,None))
        return self.public(self.read(key))

    async def run(self, key, source):
        try:
            async with self.slot:
                body = self.write(key,status='running',started=time.time())
                directory = self.store.directory/'playback'/key
                directory.mkdir(parents=True,exist_ok=True)
                request = {**body,'source':source,'directory':str(directory),'ffmpeg':self.config.ffmpeg,
                           'ffprobe':self.config.ffprobe,'owner_pid':os.getpid()}
                atomic_json(directory/'request.json',request,overwrite=True,private=True)
                process = await asyncio.create_subprocess_exec(sys.executable,'-m','bili_danmaku_mapper.workbench.playback_worker',
                    str(directory/'request.json'),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,start_new_session=True)
                self.processes[key] = process
                stdout, _ = await process.communicate()
                response = json.loads(stdout)
                if process.returncode or 'result' not in response:
                    raise ValueError(response.get('error','播放准备失败'))
                result = response['result']
                file = self.store.register_file(result.pop('path'),'playback_variant')
                self.write(key,status='complete',file_id=file['id'],**result)
        except asyncio.CancelledError:
            self.write(key,status='interrupted',error='播放准备已中断；再次播放可重试')
            raise
        except Exception as error:
            self.write(key,status='failed',error=str(error) if isinstance(error,ValueError) else '播放准备失败，原片与弹幕保留')
        finally:
            process = self.processes.pop(key,None)
            if process:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid,signal.SIGTERM)
                if process.returncode is None:
                    try:
                        await asyncio.wait_for(process.wait(),3)
                    except asyncio.TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid,signal.SIGKILL)
                        await process.wait()
            # Interrupted stages are restarted, never byte-resumed. Remove only
            # this owned stage's unpublished media so a partial 4K copy cannot
            # strand tens of GiB after cancellation or a failed validation.
            if self.read(key)['status'] != 'complete':
                directory = self.store.directory/'playback'/key
                for name in ('playback.partial.mp4','playback.mp4'):
                    (directory/name).unlink(missing_ok=True)

    async def close(self):
        self.closed = True
        pending = list(self.tasks.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending,return_exceptions=True)

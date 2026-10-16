"""A bounded two-stage pipeline with durable stage checkpoints."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from ..io import atomic_json, read_json
from .bilibili import safe_error


class JobManager:
    def __init__(self, store, downloader, config, compute=None):
        self.store, self.downloader, self.config = store, downloader, config
        self.compute = compute or self.compute_process
        self.events = set()
        self.running = {}
        self.processes = {}
        self.loops = []
        self.paused = store.setting('paused', False)
        self.closed = False
        self.playback_busy = lambda: False
        self.wakeup = asyncio.Event()

    def publish(self, job):
        for queue in list(self.events):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(job)
        self.wakeup.set()

    def update(self, key, **values):
        job = self.store.update(key, **values)
        self.publish(job)
        return job

    def ready(self, job):
        checkpoint = job['checkpoint']
        return all(checkpoint.get(k) and Path(checkpoint[k]).is_file() for k in ('source_video', 'raw_danmaku'))

    async def start(self):
        for job in self.store.jobs():
            if job['status'] in ('downloading', 'waiting', 'computing'):
                self.update(job['id'], status='queued', phase='recovered', error=None)
        self.loops = [asyncio.create_task(self.slot('download')), asyncio.create_task(self.slot('compute'))]
        self.wakeup.set()

    async def slot(self, slot):
        while not self.closed:
            jobs = list(reversed(self.store.jobs()))
            chosen = None
            if not self.paused:
                waiting = sum(j['status'] in ('queued','waiting') and self.ready(j) for j in jobs)
                for job in jobs:
                    if job['status'] not in ('queued','waiting'):
                        continue
                    ready = self.ready(job)
                    if slot == 'compute' and self.playback_busy():
                        continue
                    if (slot == 'compute' and ready) or (slot == 'download' and not ready and waiting < 2):
                        chosen = job
                        break
            if chosen:
                task = asyncio.create_task(self.run(chosen, slot))
                self.running[chosen['id']] = task
                try:
                    await task
                except asyncio.CancelledError:
                    if self.closed:
                        return
                    if asyncio.current_task().cancelling():
                        raise
                finally:
                    self.running.pop(chosen['id'], None)
                continue
            self.wakeup.clear()
            try:
                await asyncio.wait_for(self.wakeup.wait(), 1)
            except asyncio.TimeoutError:
                pass

    async def run(self, job, slot):
        key = job['id']
        directory = self.store.directory/'jobs'/key
        directory.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self.update(key, status='downloading' if slot == 'download' else 'computing', phase=slot, started=started,
                    progress={}, error=None)
        last = 0
        def notify(phase, completed=None, total=None, **detail):
            nonlocal last
            now = time.monotonic()
            if completed is not None and now-last < .25:
                return
            last = now
            self.update(key, phase=phase, progress={'completed':completed, 'total':total, **detail})
        try:
            if slot == 'download':
                checkpoint = await self.downloader.download(job['source'], directory, job['options']['quality'], notify)
                self.update(key, status='waiting', phase='waiting', checkpoint=checkpoint, progress={})
            else:
                result = await self.compute(job, directory, notify)
                media = {'target':result.pop('target_video'), 'source':job['checkpoint']['source_video'],
                         'original':job['checkpoint']['raw_danmaku'], 'danmaku':str(directory/'mapped.json'),
                         'report':str(directory/'mapped.report.json'), 'review':str(directory/'mapped.review.json')}
                ids = {name:self.store.register_file(path, 'artifact')['id'] for name,path in media.items()}
                self.update(key, status='complete', phase='complete', result={**result, 'files':ids}, progress={})
        except asyncio.CancelledError:
            self.update(key, status='queued' if self.closed else 'stopped', phase='interrupted', progress={})
            raise
        except Exception as error:
            self.update(key, status='failed', phase='failed', error=safe_error(error) if slot == 'download' else '对齐失败，请检查原片、音轨和人工锚点后重试。', progress={})

    async def compute_process(self, job, directory, notify):
        target = self.store.file(job['target_id'])
        # Re-map through the content/config/anchor keyed alignment cache. Existing
        # result files alone cannot establish freshness after inputs change.
        config = {**self.config.to_dict(), 'target_audio_stream':job['options'].get('audio_stream'),
                  'coverage_recovery':job['options'].get('coverage_recovery', self.config.coverage_recovery)}
        request = {**job['checkpoint'], 'target':target['path'], 'directory':str(directory), 'config':config,
                   'owner_pid':os.getpid(), 'mode':job['options']['mode'], 'anchors':job['anchors']}
        atomic_json(directory/'work.json', request, overwrite=True, private=True)
        process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'bili_danmaku_mapper.workbench.worker',
                       str(directory/'work.json'), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                       start_new_session=True)
        self.processes[job['id']] = process
        result = None
        try:
            async for line in process.stdout:
                data = json.loads(line)
                event = data.pop('event')
                if event == 'progress':
                    notify(**data)
                elif event == 'checkpoint':
                    current = self.store.job(job['id'])['checkpoint']
                    self.update(job['id'], checkpoint={**current, 'playback':data['playback']})
                elif event == 'result':
                    result = data['result']
            code = await process.wait()
            if code or result is None:
                raise RuntimeError('Computation failed')
            return result
        finally:
            # Kill the owned group even if the leader exited, to reap FFmpeg descendants.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            self.processes.pop(job['id'], None)

    async def stop(self, key):
        job = self.store.job(key)
        if job['status'] == 'complete':
            raise ValueError('已完成结果保留，不能停止')
        task = self.running.get(key)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        else:
            self.update(key, status='stopped', phase='stopped')

    def retry(self, key):
        job = self.store.job(key)
        if job['status'] not in ('failed','stopped'):
            raise ValueError('仅失败或已停止的任务可重试')
        self.update(key, status='queued', phase='retry', error=None)

    def pause(self, value):
        self.paused = value
        self.store.set_setting('paused', value)
        self.wakeup.set()

    async def close(self):
        self.closed = True
        self.publish(None)
        self.wakeup.set()
        for task in list(self.running.values()):
            task.cancel()
        await asyncio.gather(*self.loops, return_exceptions=True)

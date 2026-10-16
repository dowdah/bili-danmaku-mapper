"""Idle-only real-media experiment. Never mutates the production store."""
import argparse, asyncio, json, os, resource, shutil, sqlite3, sys, time
from pathlib import Path
from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.io import atomic_json
from bili_danmaku_mapper.workbench.state import Store
from bili_danmaku_mapper.workbench.playback import PlaybackLibrary
from bili_danmaku_mapper.workbench.segments import SegmentManager

class Deferred(RuntimeError): pass

def require_idle(directory):
    directory=Path(directory);instance=directory/'instance.json'
    if instance.exists():
        try:
            state=json.loads(instance.read_text());os.kill(int(state['pid']),0)
        except ProcessLookupError: pass
        except (OSError,ValueError,KeyError): raise Deferred('不能确认工作台已退出，不启动影片实验')
        else: raise Deferred('工作台仍在运行；不操作当前观看，等待自然退出')
    database=directory/'state.sqlite3'
    if database.exists():
        db=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
        try:
            if any(json.loads(r[0])['status'] in ('queued','waiting','downloading','computing') for r in db.execute('SELECT body FROM jobs')):
                raise Deferred('生产队列有任务，影片实验暂缓')
        finally: db.close()

def bytes_owned(directory):
    total=0
    for parent,_,names in os.walk(directory):
        for name in names:
            try: total+=(Path(parent)/name).stat().st_size
            except FileNotFoundError: pass
    return total

async def benchmark(args):
    require_idle(args.production_dir)
    output=Path(args.output_dir).resolve();production=Path(args.production_dir).resolve()
    if output.is_relative_to(production) or production.is_relative_to(output):raise ValueError('实验必须使用独立目录')
    output.mkdir(parents=True,exist_ok=False)
    inputs=json.loads(Path(args.inputs).read_text());source=Path(inputs['target']).resolve(strict=True)
    store=Store(output/'state');library=PlaybackLibrary(store)
    file=store.register_file(source,'playback_video');item=library.import_item(file['id'])
    manager=SegmentManager(store,library,AlignConfig())
    results={'schema_version':1,'status':'running','video_encoding':'copy','rows':[],'browser_measurements':None,
             'real_alignment_accuracy_verified':False,'io_scope':'logical output bytes; physical I/O not measured'}
    initial=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    async def guard():
        while True:
            require_idle(args.production_dir)
            if shutil.disk_usage(output).free<5*1024**3:raise Deferred('可用空间低于 5 GiB，停止本实验')
            if bytes_owned(output)>4*1024**3:raise Deferred('本实验新增空间超出 4 GiB')
            await asyncio.sleep(.25)
    watchdog=asyncio.create_task(guard())
    async def guarded(awaitable):
        work=asyncio.ensure_future(awaitable)
        done,_=await asyncio.wait([work,watchdog],return_when=asyncio.FIRST_COMPLETED)
        if watchdog in done:
            work.cancel();await asyncio.gather(work,return_exceptions=True);await watchdog
        return await work
    try:
        session=await guarded(manager.create(item['id']));duration=session['duration']
        for label,position in [('begin',0),('middle',duration/2),('end',max(0,duration-31))]:
            for repetition in range(3):
                starts=manager.stats['ffmpeg_starts'];before=time.perf_counter()
                window=await guarded(manager.request(session['id'],position,len(results['rows'])))
                if window['id'] in manager.tasks:await guarded(asyncio.shield(manager.tasks[window['id']]))
                result=manager.public(session['id'],window['id'])
                row={'position':position,'label':label,'repetition':repetition,'cache':'window_missing' if manager.stats['ffmpeg_starts']>starts else 'window_reuse',
                     'request_seconds':time.perf_counter()-before,'first_part_seconds':result.get('first_part_seconds') if manager.stats['ffmpeg_starts']>starts else time.perf_counter()-before,
                     'ffmpeg_starts':manager.stats['ffmpeg_starts']-starts,'status':result['status'],'error':result.get('error'),
                     'parts':result['parts'],'cache_bytes':manager.used}
                results['rows'].append(row);atomic_json(output/'metrics.json',results,True)
                if result['status']!='complete':raise ValueError(result.get('error','影片实验失败'))
                if repetition and row['ffmpeg_starts']:raise ValueError('有效分段缓存未能复用')
        results['status']='backend_passed_browser_pending'
    except Deferred as e:results.update(status='deferred',reason=str(e))
    except Exception as e:results.update(status='failed',reason=str(e))
    finally:
        watchdog.cancel();await asyncio.gather(watchdog,return_exceptions=True);await manager.close();store.close()
        final=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        results['cpu_seconds']=final.ru_utime+final.ru_stime-initial.ru_utime-initial.ru_stime+child.ru_utime+child.ru_stime-children.ru_utime-children.ru_stime
        unit=1 if sys.platform=='darwin' else 1024
        results.update(main_peak_rss_bytes=final.ru_maxrss*unit,largest_child_peak_rss_bytes=child.ru_maxrss*unit,
                       memory_scope='main and largest child separately; not simultaneous process-tree peak',owned_bytes=bytes_owned(output))
        atomic_json(output/'metrics.json',results,True)
    print(json.dumps({'status':results['status'],'rows':len(results['rows'])},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--production-dir',required=True);p.add_argument('--inputs');p.add_argument('--output-dir');p.add_argument('--check-idle',action='store_true');a=p.parse_args()
    try:require_idle(a.production_dir)
    except Deferred as e:print(json.dumps({'status':'deferred','reason':str(e)},ensure_ascii=False));return 75
    if a.check_idle:print('{"status":"idle"}');return 0
    if not a.inputs or not a.output_dir:p.error('--inputs and --output-dir required')
    asyncio.run(benchmark(a));return 0
if __name__=='__main__':raise SystemExit(main())

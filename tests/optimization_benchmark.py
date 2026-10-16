"""Guarded A/B runner. Run --help; no production task is changed by this tool."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

ACTIVE = {'queued','waiting','downloading','computing'}


def production_busy(database):
    if not database or not Path(database).exists(): return False
    c=sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True,timeout=2)
    try:return any(json.loads(body).get('status') in ACTIVE for body, in c.execute('select body from jobs'))
    finally:c.close()


def owned_bytes(directory):
    total=0
    for folder,_,names in os.walk(directory):
        for name in names:
            try:total+=(Path(folder)/name).stat().st_size
            except FileNotFoundError:pass  # A temporary directory disappeared during sampling.
    return total


def run_guarded(command,cwd,output,database,budget_root):
    """Only kill the child group created here; database access stays read-only."""
    if production_busy(database):raise RuntimeError('production_queue_active')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<5*1024**3:raise RuntimeError('disk_free_below_5GiB')
    with output.with_suffix('.stdout').open('wb') as out,output.with_suffix('.stderr').open('wb') as err:
        temporary=Path(budget_root)/'tmp';temporary.mkdir(parents=True,exist_ok=True)
        p=subprocess.Popen(command,cwd=cwd,stdout=out,stderr=err,start_new_session=True,
                           env={**os.environ,'TMPDIR':str(temporary.resolve())})
        output.with_suffix('.pid').write_text(str(p.pid))
        started=time.monotonic();reason=None;peak=None;peak_owned=0;peak_temporary=0
        try:
            while p.poll() is None:
                used=owned_bytes(budget_root);peak_owned=max(peak_owned,used)
                peak_temporary=max(peak_temporary,owned_bytes(temporary))
                if production_busy(database):reason='production_queue_active'
                elif output.with_suffix('.stop').exists():reason='experiment_stop_requested'
                elif shutil.disk_usage(output.parent).free<5*1024**3:reason='disk_free_below_5GiB'
                elif used>4*1024**3:reason='experiment_budget_4GiB'
                # Numeric metadata only; no command lines or credentials are read.
                try:
                    rows=subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],stderr=subprocess.DEVNULL,text=True)
                    items=[tuple(map(int,line.split())) for line in rows.splitlines() if len(line.split())==3]
                    children={p.pid}
                    for _ in range(10):
                        found={pid for pid,parent,rss in items if parent in children}
                        if found<=children:break
                        children.update(found)
                    current=sum(rss*1024 for pid,parent,rss in items if pid in children)
                    peak=max(peak or 0,current)
                except (OSError,ValueError,subprocess.SubprocessError):pass
                if reason:break
                time.sleep(1)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            output.with_suffix('.pid').unlink(missing_ok=True)
        record={'returncode':p.returncode,'wall_seconds':time.monotonic()-started,'stopped_reason':reason,
                'process_tree_peak_rss_bytes_sampled':peak,'rss_sampling_seconds':1,
                'experiment_bytes':owned_bytes(budget_root),'experiment_peak_bytes_sampled':peak_owned,
                'temporary_peak_bytes_sampled':peak_temporary}
        output.write_text(json.dumps(record,indent=2))
    if reason or p.returncode:raise RuntimeError(reason or f'worker_exit_{p.returncode}; inspect {output.with_suffix(".stderr")}')
    return record




def code_digest(directory):
    import hashlib
    digest=hashlib.sha256();root=Path(directory)
    paths=sorted((root/'bili_danmaku_mapper').rglob('*.py'))+[root/'mapper_frame.py']
    for path in paths:
        if path.is_file():digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()

def instrument_baseline():
    """Observe the unchanged snapshot in memory; never change its decisions."""
    from collections import defaultdict
    import bili_danmaku_mapper.media as media
    import bili_danmaku_mapper.audio as audio
    import bili_danmaku_mapper.alignment as alignment
    seconds=defaultdict(float);counts=defaultdict(float);events=[]
    def timed(module,name,label,before=None):
        original=getattr(module,name)
        def wrapped(*args,**kwargs):
            if before:before(*args,**kwargs)
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:seconds[label]+=time.perf_counter()-start
        setattr(module,name,wrapped)
        return wrapped
    original=media.content_digest
    def digest(path):
        counts['digest_scans']+=1;counts['digest_logical_bytes']+=Path(path).stat().st_size
        start=time.perf_counter()
        try:return original(path)
        finally:seconds['identity']+=time.perf_counter()-start
    media.content_digest=audio.content_digest=digest
    timed(media,'probe_video','probe')
    decode=media._decode
    def decoding(info,config,rate,start,end,hwaccel):
        counts['ffmpeg_starts']+=1;begun=time.perf_counter()
        result=decode(info,config,rate,start,end,hwaccel)
        counts['decoded_selected_frames']+=len(result)
        counts['decoded_requested_seconds']+=(end if end is not None else info.duration)-start
        events.append({'start':start,'end':end if end is not None else info.duration,'rate':rate,
                       'actual_backend':'software' if not hwaccel else 'unverified',
                       'elapsed_seconds':time.perf_counter()-begun,'frames':len(result)})
        seconds['visual_decode_inclusive']+=time.perf_counter()-begun
        return result
    media._decode=decoding
    timed(media,'describe_frame','visual_descriptors')
    timed(alignment,'candidate_rows','candidate_search',lambda s,t,*a,**kw:counts.__setitem__('candidate_pairs',counts['candidate_pairs']+len(s)*len(t)))
    timed(alignment,'refine_path','refinement')
    timed(audio,'_fingerprint_decode','audio_decode_fingerprint_inclusive',lambda *a,**kw:counts.__setitem__('ffmpeg_starts',counts['ffmpeg_starts']+1))
    timed(audio,'audio_anchors','audio_candidate_search')
    return lambda:{'phase_seconds_inclusive':dict(seconds),'counters':dict(counts),'decode_events':events,
                   'io_scope':'logical application reads; not physical disk I/O',
                   'instrumentation':'in-memory observation of unchanged baseline snapshot'}

def worker(task):
    # The baseline is a working-tree snapshot, not Git HEAD (which predates upgrades).
    sys.path.insert(0,task['code'])
    from dataclasses import fields
    from bili_danmaku_mapper import AlignConfig,align_videos,map_danmaku
    options={k:v for k,v in task['config'].items() if k in {f.name for f in fields(AlignConfig)}}
    config=AlignConfig(**options)
    baseline_metrics=instrument_baseline() if not hasattr(config,'hybrid_strategy') else None
    begun=time.perf_counter();cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    from bili_danmaku_mapper.progress import reporting
    last=[0.,None]
    def progress(value):
        now=time.monotonic()
        if value.get('phase') != last[1] or now-last[0]>=5:
            print(json.dumps({'event':'progress',**value}),flush=True)
            last[:]=[now,value.get('phase')]
    with reporting(progress):
        alignment=align_videos(task['source'],task['target'],config,task.get('anchors'))
    if baseline_metrics:alignment.diagnostics['performance']=baseline_metrics()
    mapping=map_danmaku(json.loads(Path(task['danmaku']).read_text()),alignment)
    from bili_danmaku_mapper.io import save_mapping_outputs
    playback=Path(task['result']).with_name('playback-'+task['variant']+'.json')
    import hashlib
    data_digest=lambda data:hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()
    mapped_digest=data_digest(mapping.danmaku)
    if playback.exists() and data_digest(json.loads(playback.read_text()))!=mapped_digest:
        raise RuntimeError('Repeated alignment changed mapped danmaku; refusing to replace the earlier result')
    save_mapping_outputs(playback,mapping,alignment,True)
    cpu=resource.getrusage(resource.RUSAGE_CHILDREN);own=resource.getrusage(resource.RUSAGE_SELF)
    report={'variant':task['variant'],'cache_state':task['cache_state'],'elapsed_seconds':time.perf_counter()-begun,
            'counts':mapping.report['counts'],'rates':mapping.report['rates'],'diagnostics':alignment.diagnostics,
            'cpu_seconds_self_and_children':own.ru_utime+own.ru_stime+cpu.ru_utime+cpu.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,
            'cache_bytes':owned_bytes(config.cache_dir),'alignment':alignment.to_dict(),
            'code_digest':code_digest(task['code']),'playback_json':str(playback),'mapped_data_digest':mapped_digest}
    Path(task['result']).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('variant','cache_state','elapsed_seconds','rates')}),flush=True)


def full(args):
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise RuntimeError('Full benchmark requires a fresh empty output directory')
    inputs=json.loads(Path(args.inputs).read_text());records=[]
    import platform
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'ffmpeg':subprocess.check_output([inputs.get('config',{}).get('ffmpeg','ffmpeg'),'-version'],text=True).splitlines()[0],
                 'os_disk_cache_cleared':False,'full_runs_per_variant_and_state':1}
    for field,sysctl in [('cpu_model','machdep.cpu.brand_string'),('cpu_count','hw.ncpu'),('ram_bytes','hw.memsize')]:
        try:environment[field]=subprocess.check_output(['sysctl','-n',sysctl],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.SubprocessError):environment[field]=None
    (root/'environment.json').write_text(json.dumps(environment,indent=2))
    input_states={name:list((lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))(Path(inputs[name]).stat())) for name in ('source','target','danmaku')}
    (root/'input-states.json').write_text(json.dumps(input_states,indent=2))
    hardware_ok=False
    if args.hardware:
        from tests.hardware_compatibility import run
        hardware_ok=run(inputs,root/'hardware-compatibility.json')
    variants=[('optimized_software',str(Path(__file__).resolve().parents[1]),{'hwaccel':None}),
              ('baseline',str(Path(args.baseline_dir).resolve()),{})]
    if hardware_ok:
        hardware_options={'hwaccel':'videotoolbox'}
        if args.hardware_source_software:hardware_options['source_hwaccel']='software'
        variants.append(('optimized_videotoolbox',str(Path(__file__).resolve().parents[1]),hardware_options))
    expected=[{'variant':name,'cache_state':state} for name,code,extra in variants for state in ('cold','features_warm','alignment_reuse') if not (name=='baseline' and state=='alignment_reuse')]
    (root/'expected-runs.json').write_text(json.dumps(expected,indent=2))
    for name,code,extra in variants:
        cache=root/'caches'/name
        cache.mkdir(parents=True,exist_ok=True)
        # Refuse accidental "cold" reruns into existing caches; choose a fresh run directory.
        if any(cache.iterdir()):raise RuntimeError(f'Cold cache is not empty: {cache}')
        for state in ('cold','features_warm','alignment_reuse'):
            if name=='baseline' and state=='alignment_reuse':continue
            for role,expected in input_states.items():
                s=Path(inputs[role]).stat()
                if list((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))!=expected:raise RuntimeError('Benchmark input changed')
            config={**inputs['config'],'cache_dir':str(cache),**extra,'alignment_cache':state!='features_warm'}
            if name!='baseline':config['hybrid_strategy']='audio_first'
            task={**inputs,'code':code,'variant':name,'cache_state':state,'config':config,
                  'result':str(root/f'{name}-{state}.json')}
            request=root/f'{name}-{state}.request.json';request.write_text(json.dumps(task,indent=2))
            guard=run_guarded([sys.executable,str(Path(__file__).resolve()),'--worker',str(request)],code,
                              root/f'{name}-{state}.guard.json',args.workbench_db,root)
            for role,expected in input_states.items():
                s=Path(inputs[role]).stat()
                if list((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))!=expected:raise RuntimeError('Benchmark input changed')
            records.append({'variant':name,'cache_state':state,**guard,'result':task['result']})
            (root/'summary.json').write_text(json.dumps({'scope':'Full original files; OS disk cache not cleared',
                                                        'records':records},indent=2))
            print(name,state,'finished',flush=True)
            if args.report_experiment and args.report_output:
                from tests.optimization_report import generate
                generate(args.report_experiment,args.report_output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker');p.add_argument('--inputs');p.add_argument('--baseline-dir');p.add_argument('--output-dir');p.add_argument('--workbench-db',default=str(Path(__file__).resolve().parents[1]/'data/workbench/state.sqlite3'));p.add_argument('--report-experiment');p.add_argument('--report-output');p.add_argument('--guard-command',nargs=argparse.REMAINDER);p.add_argument('--hardware',action='store_true');p.add_argument('--hardware-source-software',action='store_true',help='Keep the reference video on software decoding in the hardware candidate')
    args=p.parse_args()
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    if args.worker:worker(json.loads(Path(args.worker).read_text()))
    elif args.guard_command:
        if not args.output_dir:p.error('--output-dir required for guard command')
        run_guarded(args.guard_command,Path.cwd(),Path(args.output_dir)/'guard.json',args.workbench_db,args.output_dir)
    else:
        if not all((args.inputs,args.baseline_dir,args.output_dir)):p.error('--inputs, --baseline-dir, --output-dir required')
        full(args)

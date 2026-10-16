"""Original-file visual component A/B. Self-alignment supplies exact PTS truth.
This is NOT accuracy evidence for the Bilibili/original-film pair.
"""
import argparse,json,subprocess,sys,time,shutil,signal
from pathlib import Path


def worker(task):
    sys.path.insert(0,task['code'])
    from dataclasses import fields
    import numpy as np
    from bili_danmaku_mapper import AlignConfig
    from bili_danmaku_mapper.media import Features,VideoInfo,extract_features
    from bili_danmaku_mapper.alignment import align_features
    from contextlib import nullcontext
    try:
        from bili_danmaku_mapper.performance import collecting
    except ImportError:collecting=nullcontext
    info=VideoInfo(**{k:v for k,v in task['info'].items() if k in {f.name for f in fields(VideoInfo)}})
    with np.load(task['source_features'],allow_pickle=False) as a:
        source=Features(**{k:a[k] for k in ('times','hashes','center_hashes','gray','quality')},info=info)
    config=AlignConfig(**{k:v for k,v in task['config'].items() if k in {f.name for f in fields(AlignConfig)}})
    records=[]
    for state in ('cold_features','warm_features'):
        started=time.perf_counter()
        with collecting() as metrics:
            target=extract_features(info.path,config,start=task['start'],end=task['end'],_info=info)
            result=align_features(source,target,config)
        errors=[abs(a.target_time-a.source_time) for s in result.segments for a in s.anchors]
        record={'variant':task['variant'],'repeat':task['repeat'],'window':task['window'],'cache_state':state,
                'elapsed_seconds':time.perf_counter()-started,'anchors':len(errors),'source_samples':len(source),
                'p95_self_alignment_error_seconds':float(np.percentile(errors,95)) if errors else None,
                'performance':metrics.to_dict() if metrics else None}
        records.append(record)
        print(json.dumps(record),flush=True)
    Path(task['result']).write_text(json.dumps(records,indent=2))


def run(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from dataclasses import asdict
    from bili_danmaku_mapper import AlignConfig
    from bili_danmaku_mapper.media import probe_video,extract_features,_save
    from bili_danmaku_mapper.cache import session,identity
    from tests.optimization_benchmark import run_guarded
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'metrics.json').exists():raise RuntimeError('Choose a fresh window benchmark output directory')
    inputs=json.loads(Path(args.inputs).read_text());target=inputs['target'];records=[]
    preparation=root/'preparation';config=AlignConfig(cache_dir=str(preparation),hwaccel='videotoolbox',evidence_mode='visual')
    begun=time.perf_counter()
    with session():
        info=probe_video(target,config)
        windows=[('head',30.),('middle',info.duration/2),('tail',info.duration-90.)]
        for name,start in windows:
            f=extract_features(target,config,start=start,end=start+60,_info=info)
            _save(root/f'{name}-source.npz',f)
    (root/'scope.json').write_text(json.dumps({'scope':'Original film self-alignment component benchmark; not Bilibili pair accuracy',
          'preparation_seconds_excluded_from_component_timings':time.perf_counter()-begun,'identity_preseeded_for_all_variants':True,
          'info':asdict(info)},indent=2))
    code=str(Path(__file__).resolve().parents[1])
    variants=[('baseline',args.baseline_dir,{}),('merged',code,{'merge_refinement':True,'dense_cache':False}),
              ('dense_cache',code,{'merge_refinement':True,'dense_cache':True})]
    for repeat in range(3):
        for name,start in windows:
            for variant,tree,extra in (variants if repeat%2==0 else list(reversed(variants))):
                directory=root/f'{name}-{repeat}-{variant}';directory.mkdir(exist_ok=True)
                cache=directory/'cache';cache.mkdir(exist_ok=True)
                if any(p.name not in ('identity-v1','probe-v1') for p in cache.iterdir()):
                    raise RuntimeError('Cold feature cache is not empty; choose a fresh output directory')
                # Component timings exclude identity preparation equally. Whole-file
                # benchmarks separately include cold hashing/probing end to end.
                for part in ('identity-v1','probe-v1'):
                    if (preparation/part).exists():shutil.copytree(preparation/part,cache/part,dirs_exist_ok=True)
                task={'code':str(Path(tree).resolve()),'info':asdict(info),'source_features':str(root/f'{name}-source.npz'),
                      'variant':variant,'repeat':repeat,'window':name,'start':start,'end':start+60,
                      'config':{'cache_dir':str(cache),'evidence_mode':'visual','alignment_cache':False,**extra},
                      'result':str(directory/'metrics.json')}
                request=directory/'request.json';request.write_text(json.dumps(task,indent=2))
                guard=run_guarded([sys.executable,str(Path(__file__).resolve()),'--worker',str(request)],tree,directory/'guard.json',args.workbench_db,root)
                for row in json.loads(Path(task['result']).read_text()):row['guard']=guard;records.append(row)
                (root/'metrics.json').write_text(json.dumps({'scope':'Original film self-alignment component benchmark; not Bilibili pair accuracy','records':records},indent=2))
                print(name,repeat,variant,'finished',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker');p.add_argument('--inputs');p.add_argument('--output-dir');p.add_argument('--baseline-dir');p.add_argument('--workbench-db',default=str(Path(__file__).resolve().parents[1]/'data/workbench/state.sqlite3'))
    a=p.parse_args()
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    if a.worker:worker(json.loads(Path(a.worker).read_text()))
    else:run(a)

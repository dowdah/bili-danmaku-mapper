"""Repeatable, guarded coverage ablations and human-review materials.

No real-film accuracy is inferred from agreement or model confidence. Use --help.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def read(path):return json.loads(Path(path).read_text())

def summarize(report):
    alignment=report.get('alignment',{})
    return {'counts':report['counts'],'rates':report['rates'],
        'unmapped_reasons':dict(Counter(e['reason'] for e in report['entries'] if e['status']=='unmapped')),
        'unmapped_categories':report.get('unmapped_categories'),
        'uniform_time_coverage':report.get('uniform_time_coverage'),
        'segments':len(alignment.get('segments',[])),
        'singletons':sum(len(s['anchors'])==1 for s in alignment.get('segments',[])),
        'real_accuracy_human_verified':False}


def compare(before,after):
    old={e['source_index']:e for e in before['entries']};new=after['entries'];counts=Counter();deltas=[]
    for e in new:
        b=old[e['source_index']];bt,nt=b['target_time'],e['target_time']
        if bt is None and nt is not None:counts['newly_accepted']+=1
        elif bt is not None and nt is None:counts['withdrawn']+=1
        elif bt is not None:
            counts['common']+=1;deltas.append(abs(bt-nt))
            if abs(bt-nt)>.2:counts['common_changed_gt_200ms']+=1
    import numpy as np
    return {**counts,'common_time_difference_p95_seconds':float(np.percentile(deltas,95)) if deltas else None,
            'comparison_is_not_ground_truth':True}


def audit_report(path):
    from bili_danmaku_mapper.models import AlignmentResult
    from bili_danmaku_mapper.mapping import map_time
    import copy,bisect
    report=read(path);a=AlignmentResult.from_dict(report['alignment']);out=summarize(report)
    idx=([s.source_start for s in a.segments],[[p.source_time for p in s.anchors] for s in a.segments])
    gap_bins=Counter()
    for e in report['entries']:
        if e['reason']!='gap_exceeds_strict_limit':continue
        i=bisect.bisect_right(idx[0],e['source_time'])-1;s=a.segments[i];j=bisect.bisect_left(idx[1][i],e['source_time'])
        gap=s.anchors[j].source_time-s.anchors[j-1].source_time
        gap_bins['1-1.05' if gap<=1.05 else '1.05-2' if gap<=2 else '2-3' if gap<=3 else '>3']+=1
    out['strict_gap_comment_bins']=dict(gap_bins)
    clone=copy.deepcopy(a);clone.diagnostics['unverified_intervals']=[x for x in a.diagnostics.get('unverified_intervals',[]) if x['reason']!='audio_visual_segment_conflict']
    recoverable=sum(e['target_time'] is None and map_time(e['source_time'],clone,_index=idx)[0] is not None for e in report['entries'])
    out['diagnostic_only_remove_segment_masks']={'new_eligible':recoverable,'safe_to_publish':False}
    return out


def seed_cache(source,destination):
    """Immutable feature payloads are linked; lock files are always independent."""
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError('Choose a fresh cache directory')
    destination.mkdir(parents=True)
    n=size=0
    for path in source.rglob('*'):
        if not path.is_file() or path.suffix=='.lock' or 'alignment-v1' in path.parts:continue
        target=destination/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.npz':os.link(path,target)
        elif path.suffix=='.json':shutil.copy2(path,target)
        else:continue
        n+=1;size+=path.stat().st_size
    return {'files':n,'logical_bytes':size,'npz_payloads_hardlinked':True,'source_payloads_never_modified':True}


def worker(task):
    from bili_danmaku_mapper import AlignConfig,map_danmaku,align_videos
    from bili_danmaku_mapper.models import AlignmentResult
    from bili_danmaku_mapper.coverage import fuse_verified
    from bili_danmaku_mapper.audio import align_audio
    from bili_danmaku_mapper.media import probe_video
    from bili_danmaku_mapper.cache import session
    from bili_danmaku_mapper.performance import collecting
    from bili_danmaku_mapper.progress import reporting
    from bili_danmaku_mapper.io import save_mapping_outputs
    from tests.optimization_benchmark import code_digest
    import resource,platform
    cpu_start=resource.getrusage(resource.RUSAGE_SELF);child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
    source,target=task['source'],task['target'];cfg=AlignConfig(**task['config']);begun=time.perf_counter();last=[0,None]
    def progress(v):
        now=time.monotonic()
        if v['phase']!=last[1] or now-last[0]>=5:
            print(json.dumps(v),flush=True);last[:]=[now,v['phase']]
    from bili_danmaku_mapper.evidence import observing_stages
    data=read(task['danmaku'])
    stage_summaries=[]
    def checkpoint(name,model):
        from bili_danmaku_mapper.io import atomic_json
        mapping=map_danmaku(data,model)
        folder=Path(task['result']).parent/'stages';folder.mkdir(exist_ok=True)
        document={**mapping.report,'alignment':model.to_dict()}
        atomic_json(folder/(name+'.report.json'),document,False)
        stage_summaries.append({'stage':name,**summarize(document)})
    with reporting(progress),observing_stages(checkpoint):
        if task.get('replay'):
            with collecting() as metrics,session():
                original=AlignmentResult.from_dict(read(task['baseline_report'])['alignment'])
                si,ti=probe_video(source,cfg),probe_video(target,cfg)
                from bili_danmaku_mapper.cache import identity,key
                from bili_danmaku_mapper.audio import AUDIO_VERSION
                from bili_danmaku_mapper.models import Anchor
                memo=Path(task['result']).parent.parent/'audio-anchors.json'
                signature=key([identity(source,cfg),identity(target,cfg),AUDIO_VERSION,cfg.source_audio_stream,cfg.target_audio_stream,cfg.audio_speed_search])
                stored=read(memo) if memo.exists() else {}
                if stored.get('signature')==signature:
                    audio=[Anchor(**a) for a in stored['anchors']];ad=stored['diagnostics']
                else:
                    audio,ad=align_audio(si,ti,cfg,[])
                    from bili_danmaku_mapper.io import atomic_json
                    atomic_json(memo,{'signature':signature,'anchors':[vars(a) for a in audio],'diagnostics':ad},True)
                visual=[a for s in original.segments for a in s.anchors if a.evidence=='visual']
                segments,intervals,diag=fuse_verified(visual,audio,[],cfg,si.duration,ti.duration)
                alignment=AlignmentResult(source,target,si.duration,ti.duration,segments,cfg.to_dict(),{
                    'unverified_intervals':intervals,'fusion':diag,'audio':ad,'real_accuracy_human_verified':False,
                    'replay_uses_previous_measured_visual_anchors':True})
            alignment.diagnostics['performance']=metrics.to_dict()
        else:alignment=align_videos(source,target,cfg,task.get('anchors'))
    mapping=map_danmaku(data,alignment)
    output=Path(task['result']);output.parent.mkdir(parents=True,exist_ok=True)
    save_mapping_outputs(output.with_suffix('.playback.json'),mapping,alignment,False)
    own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={"cpu_seconds_self_and_children":own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime-child_start.ru_utime-child_start.ru_stime,
        "peak_main_rss_bytes":own.ru_maxrss*(1 if platform.system()=="Darwin" else 1024),
        "rss_scope":"Main process only; process tree RSS depends on guard availability",**summarize({**mapping.report,'alignment':alignment.to_dict()}),
        'elapsed_seconds':time.perf_counter()-begun,'alignment':alignment.to_dict(),
        'stage_summaries':stage_summaries,'code_digest':code_digest(Path(__file__).resolve().parents[1]),'report':str(output.with_suffix('.playback.report.json'))}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ('elapsed_seconds','counts','rates','segments')}),flush=True)


def review(before,after,output):
    """Deterministic discovery/held-out split; all annotations start empty."""
    old={e['source_index']:e for e in before['entries']};groups={k:[] for k in ('newly_accepted','withdrawn','changed','common','still_unmapped')}
    for e in after['entries']:
        b=old[e['source_index']];bt,nt=b['target_time'],e['target_time']
        group='newly_accepted' if bt is None and nt is not None else 'withdrawn' if bt is not None and nt is None else 'changed' if bt is not None and abs(bt-nt)>.08 else 'common' if nt is not None else 'still_unmapped'
        key=f"{e['source_index']}:{e['source_time']}";split='validation' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2 else 'discovery'
        groups[group].append({'source_index':e['source_index'],'source_time':e['source_time'],'baseline_target_time':bt,
            'candidate_target_time':nt,'stratum':group,'split':split,'human_verified':False,
            'truth_target_time':None,'confirmed_no_correspondence':False,'reason':e['reason']})
    critical=[r for values in groups.values() for r in values if r['stratum']=='withdrawn' or
              (r['baseline_target_time'] is not None and r['candidate_target_time'] is not None and
               abs(r['baseline_target_time']-r['candidate_target_time'])>.2)]
    for row in critical:row['priority']='critical_change'
    rng=random.Random(20261015);picked=[]
    for name,values in groups.items():
        rng.shuffle(values);picked.extend(values[:60])
    picked=list({r['source_index']:r for r in picked+critical}.values())
    alignment=after.get('alignment',{});boundaries=[]
    for a,b in zip(alignment.get('segments',[]),alignment.get('segments',[])[1:]):
        boundaries.append({'source_start':a['source_end'],'source_end':b['source_start'],'human_verified':False})
    uniform=[]
    if before.get('alignment') and after.get('alignment'):
        import numpy as np
        from bili_danmaku_mapper.models import AlignmentResult
        from bili_danmaku_mapper.mapping import map_time
        ba=AlignmentResult.from_dict(before['alignment']);aa=AlignmentResult.from_dict(after['alignment'])
        uniform=[{'source_time':float(t),'baseline_target_time':map_time(float(t),ba)[0],
                  'candidate_target_time':map_time(float(t),aa)[0],'human_verified':False}
                 for t in np.linspace(.5,aa.source_duration-.5,30)]
    document={'uniform_records':uniform,'scope':'Real accuracy pending human annotation; predictions are not truth',
        'critical_records':len(critical),'stratum_counts':{k:len(v) for k,v in groups.items()},'records':sorted(picked,key=lambda e:e['source_time']),
        'boundaries':boundaries,'conflicts':alignment.get('diagnostics',{}).get('unverified_intervals',[])}
    Path(output).write_text(json.dumps(document,ensure_ascii=False,indent=2))


def evaluate_annotations(document,alignment,split='validation'):
    from bili_danmaku_mapper.mapping import map_time
    import math,numpy as np
    errors=[];wrong=accepted=positive=positive_accepted=checked=0
    for row in document.get('records',[]):
        if not row.get('human_verified') or row.get('split')!=split:continue
        stamp=row.get('source_time');expected=row.get('truth_target_time')
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):raise ValueError('Invalid annotated source time')
        if expected is None and not row.get('confirmed_no_correspondence'):raise ValueError('Unfinished human annotation')
        if expected is not None and (not isinstance(expected,(int,float)) or not math.isfinite(expected) or not 0<=expected<=alignment.target_duration):raise ValueError('Invalid annotated target time')
        value,status,reason=map_time(stamp,alignment);checked+=1;positive+=expected is not None
        if value is not None:
            accepted+=1
            if expected is None:wrong+=1
            else:
                error=abs(value-expected);errors.append(error);wrong+=error>.2;positive_accepted+=error<=.2
    return {'split':split,'human_checked':checked,'accepted_checked':accepted,'wrong_accepts':wrong,
        'precision_within_200ms':(accepted-wrong)/accepted if accepted else None,
        'p95_error_seconds':float(np.percentile(errors,95)) if errors else None,
        'labeled_comparable_coverage':positive_accepted/positive if positive else None,
        'scope':'Labeled samples only; does not establish whole-film accuracy',
        'status':'human_review_pending' if not checked else 'sampled_evaluation_only',
        'default_promotion_approved':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker');p.add_argument('--overwrite',action='store_true');p.add_argument('--annotations');p.add_argument('--alignment');p.add_argument('--budget-root');p.add_argument('--audit',nargs='+');p.add_argument('--output');p.add_argument('--inputs');p.add_argument('--baseline-report');p.add_argument('--seed-cache');p.add_argument('--output-dir');p.add_argument('--strategy',choices=['audio_first','visual_first'],default='visual_first');p.add_argument('--stage',choices=['fusion','visual','audio'],default='audio');p.add_argument('--replay',action='store_true');p.add_argument('--database',default='data/workbench/state.sqlite3');p.add_argument('--compare',nargs=2)
    args=p.parse_args()
    if args.worker:return worker(read(args.worker))
    if args.annotations:
        from bili_danmaku_mapper.models import AlignmentResult
        document=read(args.alignment);document=document.get('alignment',document)
        result=evaluate_annotations(read(args.annotations),AlignmentResult.from_dict(document))
        Path(args.output).write_text(json.dumps(result,indent=2));return
    if args.output and Path(args.output).exists() and not args.overwrite:
        raise FileExistsError('Output exists; use --overwrite or choose a new output')
    if args.compare and args.output and Path(args.output).with_suffix('.review.json').exists() and not args.overwrite:
        raise FileExistsError('Review exists; preserve annotations or choose a new output')
    if args.audit:
        out={str(path):audit_report(path) for path in args.audit};Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2));return
    if args.compare:
        before,after=map(read,args.compare);out={'baseline':summarize(before),'candidate':summarize(after),'agreement':compare(before,after)}
        Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2));review(before,after,Path(args.output).with_suffix('.review.json'));return
    if not args.inputs or not args.output_dir:p.error('--inputs and --output-dir required')
    from tests.optimization_benchmark import run_guarded
    root=Path(args.output_dir).resolve();root.mkdir(exist_ok=False,parents=True);inputs=read(args.inputs)
    info=seed_cache(args.seed_cache,root/'cache') if args.seed_cache else None
    task={**inputs,'baseline_report':str(Path(args.baseline_report).resolve()) if args.baseline_report else None,'replay':args.replay,'result':str(root/'result.json'),
          'config':{**inputs.get('config',{}),'cache_dir':str(root/'cache'),'coverage_recovery':True,
                    'recovery_stage':args.stage,'hybrid_strategy':args.strategy,'alignment_cache':False}}
    request=root/'request.json';request.write_text(json.dumps(task,indent=2));(root/'seed.json').write_text(json.dumps(info,indent=2))
    state={k:[getattr(Path(inputs[k]).stat(),x) for x in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')] for k in ('source','target','danmaku')}
    run_guarded([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(request)],Path(__file__).resolve().parents[1],root/'guard.json',str(Path(args.database).resolve()),Path(args.budget_root).resolve() if args.budget_root else root)
    after={k:[getattr(Path(inputs[k]).stat(),x) for x in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')] for k in state}
    (root/'input-invariants.json').write_text(json.dumps({'unchanged':state==after,'before':state,'after':after},indent=2))
    if state!=after:raise RuntimeError('Input changed during experiment')

if __name__=='__main__':main()

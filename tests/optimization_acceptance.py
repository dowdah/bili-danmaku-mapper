"""New/old hybrid accuracy on known timelines, independent from real-film agreement."""
from dataclasses import replace
from pathlib import Path
import argparse,json,subprocess,sys,tempfile
import numpy as np
from tests.video_helpers import make_video
from tests.audio_helpers import mux_sound


def run(root,baseline,selected=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise RuntimeError('Choose a fresh synthetic benchmark output directory')
    code=Path(__file__).resolve().parents[1];records=[]
    cases=('identity','head_cut','middle_cut','target_cut','volume','silence','origin','fully_obscured',
           'partial_obscured','unrelated_audio','delayed_audio','visual_speed','vfr','repeat','pal_speed','fps_conversion')
    if selected:
        if not set(selected)<=set(cases):raise ValueError("Unknown synthetic scenario")
        cases=tuple(selected)
    for name in cases:
        folder=root/name;folder.mkdir(exist_ok=True)
        sd=td=10.;st=tt=lambda t:t;expected=lambda t:t;gain=1.;mute=None;origin=0.;filters=None;variant=0;delay=0
        if name in ('head_cut','fully_obscured','partial_obscured','delayed_audio'):
            sd=9.3;st=lambda t:t+.7;expected=st
        if name=='middle_cut':sd=9.;st=lambda t:t if t<4 else t+1;expected=st
        if name=='target_cut':td=9.;tt=lambda t:t if t<4 else t+1;expected=lambda t:t if t<4 else None if t<5 else t-1
        if name=='volume':gain=.3
        if name=='silence':mute=(4,6)
        if name=='origin':origin=5
        if name=='fully_obscured':filters='drawbox=color=black:t=fill'
        if name=='partial_obscured':filters='drawbox=x=40:y=25:w=130:h=85:color=black:t=fill'
        if name=='unrelated_audio':variant=5
        if name=='delayed_audio':delay=.3
        if name=='visual_speed':sd=8.;st=lambda t:t*1.25;expected=st;variant=5
        if name=='vfr':filters="select='not(eq(mod(n,10),3)+eq(mod(n,10),4))'"
        if name=='repeat':sd=4.;td=8.;st=lambda t:t;tt=lambda t:t%4;expected=lambda t:None
        sfps,tfps=24,30
        if name in ('pal_speed','fps_conversion'):
            sfps,tfps=25,24;sd,td=10.,12.
            rate=25/24 if name=='pal_speed' else 1.
            st=lambda t:t*rate+.7;expected=st
        sv=make_video(folder/'sv.mp4',sd,fps=sfps,timeline=st,filters=filters,vfr=name=='vfr')
        tv=make_video(folder/'tv.mp4',td,fps=tfps,timeline=tt)
        ext='mkv' if origin else 'mp4'
        source=mux_sound(sv,folder/f'source.{ext}',sd,st,gain=gain,mute=mute,offset=origin,variant=variant,delay=delay)
        target=mux_sound(tv,folder/f'target.{ext}',td,tt,rate=48000,offset=origin)
        # Repeat the complete soundtrack too; the expected position is ambiguous.
        if name=='repeat':
            import wave
            # Reuse two identical encoded clips to retain truly repeated audio.
            concat=folder/'concat.txt';concat.write_text("file 'source.mp4'\nfile 'source.mp4'\n")
            target=folder/'repeat-target.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(target)],check=True)
        times=[float(t) for t in np.arange(.5,sd-.5,.1)]
        for label,tree in [('baseline',Path(baseline).resolve()),('optimized',code)]:
            task={'code':str(tree),'variant':label,'cache_state':'cold','source':str(source),'target':str(target),
                  'danmaku':str(folder/'danmaku.json'),'config':{'cache_dir':str(folder/label),'evidence_mode':'hybrid',
                  'hybrid_strategy':'audio_first','alignment_cache':False},'result':str(folder/f'{label}.json')}
            Path(task['danmaku']).write_text(json.dumps({'data':[{'time':t} for t in times]}))
            request=folder/f'{label}-request.json';request.write_text(json.dumps(task))
            subprocess.run([sys.executable,str(code/'tests/optimization_benchmark.py'),'--worker',str(request)],check=True,stdout=subprocess.DEVNULL)
            data=json.loads(Path(task['result']).read_text())
            # Evaluate a common uniform timestamp grid, rather than differing real danmaku density.
            sys.path.insert(0,str(code))
            from bili_danmaku_mapper import map_danmaku
            from bili_danmaku_mapper.models import AlignmentResult
            mapping=map_danmaku({'data':[{'time':t} for t in times]},AlignmentResult.from_dict(data['alignment']))
            errors=[];wrong=0;accepted=mapping.report['counts']['reliable'];comparable=sum(expected(t) is not None for t in times)
            for entry in mapping.report['entries']:
                if entry['target_time'] is None:continue
                truth=expected(entry['source_time'])
                if truth is None:wrong+=1
                else:
                    error=abs(entry['target_time']-truth);errors.append(error);wrong+=error>.2
            record={'scenario':name,'variant':label,'counts':mapping.report['counts'],'rates':mapping.report['rates'],
                    'precision_within_200ms':(accepted-wrong)/accepted if accepted else None,
                    'p95_error_seconds':float(np.percentile(errors,95)) if errors else None,
                    'correct_reliable_coverage':sum(e<=.2 for e in errors)/comparable if comparable else None,
                    'wrong_accepts':wrong,'elapsed_seconds':data['elapsed_seconds']}
            records.append(record)
            print(name,label,record['precision_within_200ms'],record['p95_error_seconds'],record['rates']['reliable'],flush=True)
        (root/'metrics.json').write_text(json.dumps({'scope':'Known synthetic timelines; not real-film accuracy','records':records},indent=2))
    failures=[]
    for name in cases:
        before,after=[r for r in records if r['scenario']==name]
        if name=='repeat':
            if after['counts']['reliable']:failures.append(name+': ambiguous output accepted')
        elif ((after['precision_within_200ms'] or 0)<.99 or (after['p95_error_seconds'] or 0)>.2
              or after['correct_reliable_coverage']<before['correct_reliable_coverage']-.02):failures.append(name)
    report={'scope':'Known synthetic timelines; not real-film accuracy','records':records,'failures':failures,'passed':not failures}
    (root/'metrics.json').write_text(json.dumps(report,indent=2))
    if failures:raise RuntimeError('Acceptance failed: '+', '.join(failures))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--baseline-dir',required=True);p.add_argument('--cases',nargs='+')
    a=p.parse_args();run(a.output_dir,a.baseline_dir,a.cases)

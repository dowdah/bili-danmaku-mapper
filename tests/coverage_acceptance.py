"""Known timelines on existing fixtures plus coverage-specific adversarial cases."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bili_danmaku_mapper import AlignConfig,align_videos,map_danmaku
from tests.video_helpers import make_video
from tests.audio_helpers import mux_sound,sound


def truth(name,t):
    if name in ('head_cut','fully_obscured','partial_obscured','delayed_audio'):return t+.7
    if name=='middle_cut':return t if t<4 else t+1
    if name=='target_cut':return t if t<4 else None if t<5 else t-1
    if name=='visual_speed':return t*1.25
    if name in ('pal_speed','multichannel_pal'):return t*25/24+.7
    if name=='fps_conversion':return t+.7
    if name=='repeat':return None
    return t


def assess(name,alignment,times):
    result=map_danmaku({'data':[{'time':float(t)} for t in times]},alignment)
    errors=[];wrong=0;comparable=sum(truth(name,t) is not None for t in times)
    for e in result.report['entries']:
        if e['target_time'] is None:continue
        expected=truth(name,e['source_time'])
        if expected is None:wrong+=1
        else:
            error=abs(e['target_time']-expected);errors.append(error);wrong+=error>.2
    count=result.report['counts']['reliable'];correct=sum(e<=.2 for e in errors)
    return {'counts':result.report['counts'],'precision_within_200ms':(count-wrong)/count if count else None,
            'p95_error_seconds':float(np.percentile(errors,95)) if errors else None,'wrong_accepts':wrong,
            'correct_reliable_coverage':correct/comparable if comparable else None,
            'uniform_time_coverage':result.report.get('uniform_time_coverage')}


def custom(root,name):
    if name=='source_multichannel_mix':
        base=root/'multichannel_mix'
        if not base.exists():custom(root,'multichannel_mix')
        return base/'target.mkv',base/'source.mp4',np.arange(.5,11.5,.1)
    folder=root/name;folder.mkdir();duration=12
    filters="drawbox=color=black:t=fill:enable='between(t,4,7)'" if name in ('multichannel_mix','multichannel_pal') else None
    timeline=(lambda t:4. if 4<=t<7 else t) if name=='long_static' else lambda t:t
    source_timeline=(lambda t:t*25/24+.7) if name=='multichannel_pal' else timeline
    sv=make_video(folder/'sv.mp4',duration,fps=25 if name in ('sampling_phase','multichannel_pal') else 24,timeline=source_timeline,filters=filters)
    target_duration=13.2 if name=='multichannel_pal' else duration
    tv=make_video(folder/'tv.mp4',target_duration,fps=24,timeline=timeline)
    if name not in ('multichannel_mix','multichannel_pal'):
        source=mux_sound(sv,folder/'source.mp4',duration,variant=5 if name=='sampling_phase' else 0)
        target=mux_sound(tv,folder/'target.mp4',duration)
    else:
        source=mux_sound(sv,folder/'source.mp4',duration,source_timeline)
        t=np.arange(int(target_duration*48000))/48000
        channels=np.column_stack([sound(t,1),sound(t,2),sound(t)*.25,sound(t,3),sound(t,4),sound(t,5)])
        wav=folder/'surround.wav'
        with wave.open(str(wav),'wb') as f:
            f.setnchannels(6);f.setsampwidth(2);f.setframerate(48000)
            f.writeframes((np.clip(channels,-1,1)*32767).astype('<i2').tobytes())
        target=folder/'target.mkv'
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(tv),'-channel_layout','5.1','-i',str(wav),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','flac',str(target)],check=True)
        wav.unlink()
    return source,target,np.arange(.5,duration-.5,.1)


def validation_failures(name,strategy,row,baseline=None):
    failures=[]
    if row['wrong_accepts'] or (row['p95_error_seconds'] or 0)>.2:
        failures.append(name+':'+strategy+': accuracy')
    if name!='repeat' and (row['precision_within_200ms'] or 0)<.99:
        failures.append(name+':'+strategy+': precision')
    if baseline and name!='repeat' and row['correct_reliable_coverage']<baseline['correct_reliable_coverage']-.02:
        failures.append(name+':'+strategy+': coverage regression')
    floors={'multichannel_mix':.95,'source_multichannel_mix':.95,'sampling_phase':.95,
            'long_static':.95,'multichannel_pal':.8}
    if name in floors and (row['correct_reliable_coverage'] or 0)<floors[name]:
        failures.append(name+':'+strategy+': recovery coverage floor')
    return failures


def run(fixtures,output,selected=None):
    fixtures=Path(fixtures);root=Path(output);root.mkdir(exist_ok=False,parents=True);records=[];failures=[]
    old=json.loads((fixtures/'metrics.json').read_text())['records']
    cases=list(dict.fromkeys(r['scenario'] for r in old))+['multichannel_mix','sampling_phase','long_static','multichannel_pal','source_multichannel_mix']
    if selected:cases=[n for n in cases if n in selected]
    for name in cases:
        if (fixtures/name/'optimized-request.json').exists():
            task=json.loads((fixtures/name/'optimized-request.json').read_text());source,target=task['source'],task['target']
            times=[e['time'] for e in json.loads(Path(task['danmaku']).read_text())['data']]
        else:source,target,times=custom(root,name)
        cfg=AlignConfig(cache_dir=str(root/(name+'-cache')),coverage_recovery=True,alignment_cache=False)
        digests=[]
        for strategy in ('visual_first','audio_first'):
            alignment=align_videos(source,target,replace(cfg,hybrid_strategy=strategy))
            row={'scenario':name,'strategy':strategy,**assess(name,alignment,times)};records.append(row)
            digests.append([[e.source_time,e.target_time] for s in alignment.segments for e in s.anchors])
            baseline=next((r for r in old if r['scenario']==name and r['variant']=='baseline'),None)
            failures.extend(validation_failures(name,strategy,row,baseline))
            (root/(name+'-'+strategy+'.alignment.json')).write_text(json.dumps(alignment.to_dict()))
            print(name,strategy,row['precision_within_200ms'],row['correct_reliable_coverage'],flush=True)
        if digests[0]!=digests[1]:failures.append(name+': strategy changed accepted anchors')
        (root/'metrics.json').write_text(json.dumps({'records':records,'failures':failures,'passed':not failures},indent=2))
    if failures:raise RuntimeError('Acceptance failed: '+', '.join(failures))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--cases',nargs='+');a=p.parse_args()
    run(a.fixtures,a.output_dir,a.cases)

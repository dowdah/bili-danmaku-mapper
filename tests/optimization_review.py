"""Export all boundaries/conflicts and a small local-only visual review sample."""
import argparse,html,json,re,subprocess,sys
from pathlib import Path
import numpy as np
from bili_danmaku_mapper.models import AlignmentResult
from bili_danmaku_mapper.mapping import map_time


def run(experiment,images=True):
    root=Path(experiment).resolve();inputs=json.loads((root/'inputs.json').read_text())
    full=root/'full';alignments={}
    for name in ('baseline','optimized_software','optimized_videotoolbox'):
        p=full/f'{name}-cold.json'
        if p.exists():alignments[name]=AlignmentResult.from_dict(json.loads(p.read_text())['alignment'])
    default=root/'validation/default-visual-first.json'
    if default.exists():
        alignments['default_visual_first']=AlignmentResult.from_dict(json.loads(default.read_text())['alignment'])
    if not alignments:raise RuntimeError('No completed alignment to review')
    primary='default_visual_first' if 'default_visual_first' in alignments else 'optimized_software' if 'optimized_software' in alignments else next(iter(alignments))
    selected=alignments[primary]
    positions={round(float(t),6):{'uniform'} for t in np.linspace(.5,selected.source_duration-.5,30)}
    def add(t,reason):
        t=round(max(0.,min(selected.source_duration,float(t))),6)
        positions.setdefault(t,set()).add(reason)
    for name,a in alignments.items():
        for left,right in zip(a.segments,a.segments[1:]):
            add(left.source_end,name+':segment_end');add(right.source_start,name+':segment_start')
        for interval in a.diagnostics.get('unverified_intervals',[]):
            add(interval['source_start'],name+':'+interval['reason']);add(interval['source_end'],name+':'+interval['reason'])
    from tests.optimization_report import compare_reports
    comparison=compare_reports(root,'baseline','optimized_software')
    if comparison and 'error' not in comparison:
        for entry in comparison['largest_differences']:add(entry['source_time'],'difference_gt_200ms')
    candidate_document=alignments.get('optimized_software',selected).to_dict()
    candidate_document['diagnostics']={**candidate_document['diagnostics'],'unverified_intervals':[]}
    candidate_alignment=AlignmentResult.from_dict(candidate_document)
    records=[]
    for stamp,reasons in sorted(positions.items()):
        mappings={}
        for name,a in alignments.items():
            target,status,reason=map_time(stamp,a)
            mappings[name]={'target_time':target,'status':status,'reason':reason}
        records.append({'source_time':stamp,'selection_reasons':sorted(reasons),'mappings':mappings,
                        'human_verified':False,'unverified_candidate_target_time':map_time(stamp,candidate_alignment)[0]
                        if 'optimized_software' in mappings and mappings['optimized_software']['target_time'] is None else None})
    out=root/'review';out.mkdir(exist_ok=True)
    result={'scope':'Automatic comparison; real-film correctness NOT human verified','primary_strategy':primary,'records':records,'images':[]}
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if images:
        uniform=[r for r in records if 'uniform' in r['selection_reasons']][::2]
        candidates=sorted((r for r in records if any(x!='uniform' for x in r['selection_reasons'])),
                          key=lambda r:('difference_gt_200ms' not in r['selection_reasons'],r['source_time']))
        boundaries=[]
        for row in candidates:
            if all(abs(row['source_time']-old['source_time'])>=3 for old in boundaries):boundaries.append(row)
            if len(boundaries)>=15:break
        picked=sorted({r['source_time']:r for r in uniform+boundaries}.values(),key=lambda r:r['source_time'])[:30]
        origin={}
        for role in ('source','target'):
            data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=start_time','-of','json',inputs[role]]))
            origin[role]=float(data.get('format',{}).get('start_time',0))
        for i,r in enumerate(picked):
            mapped=r['mappings'][primary]
            row={'source_time':r['source_time'],'target_time':mapped['target_time'],'target_strategy':primary,'files':{}}
            baseline=r['mappings'].get('baseline',{}).get('target_time')
            audio_first=r['mappings'].get('optimized_software',{}).get('target_time')
            if audio_first is not None and mapped['target_time'] is not None and abs(audio_first-mapped['target_time'])<1e-6:audio_first=None
            for role,stamp in (('source',r['source_time']),('target',mapped['target_time']),('baseline',baseline),
                               ('audio_first_target',audio_first),
                               ('unverified_candidate',r.get('unverified_candidate_target_time'))):
                if stamp is None:continue
                input_role='source' if role=='source' else 'target'
                dest=out/f'{i:03d}-{role}.png'
                p=subprocess.run(['ffmpeg','-hide_banner','-nostdin','-loglevel','info','-copyts','-ss',str(max(0.,stamp)),
                                  '-i',inputs[input_role],'-map','0:v:0','-an','-sn','-dn',
                                  '-frames:v','1','-vf','scale=640:-2,showinfo','-y',str(dest)],capture_output=True,text=True)
                if p.returncode:
                    row['files'][role]={'error':'preview_decode_failed'};continue
                match=re.search(r'\bn:\s*0\s+.*?\bpts_time:([^\s]+)',p.stderr)
                row['files'][role]={'path':dest.name,'requested_seconds':stamp,
                                    'actual_seconds':float(match[1])-origin[input_role] if match else None}
            result['images'].append(row)
            (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    cards=[]
    for row in result['images']:
        pictures=''.join(f'<figure><img loading="lazy" src="{html.escape(data["path"])}"><figcaption>{role} · requested {data["requested_seconds"]:.3f}s · actual {data["actual_seconds"]}</figcaption></figure>' for role,data in row['files'].items() if 'path' in data)
        cards.append(f'<article><h2>源 {row["source_time"]:.3f}s → {html.escape(row["target_strategy"])} {row["target_time"]}</h2><div>{pictures}</div></article>')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>映射待复核对照</title><style>body{background:#14181f;color:#eee;font:16px system-ui;margin:24px}article{padding:12px;border-bottom:1px solid #555}article>div{display:flex;flex-wrap:wrap}figure{margin:8px}img{max-width:640px;width:100%}</style><h1>自动生成，尚未经人工确认</h1><p>30个固定均匀位置与所有断点、冲突见 review.json；本页仅展示少量截图。截图为请求位置后的首帧，不代表匹配正确。unverified_candidate 是被冲突规则拦下的候选，仅供人工复核，不在正式弹幕输出中。</p>'+f'<p>target预览采用 {html.escape(primary)}；audio_first_target为音频优先正式输出，unverified_candidate只是假设候选。</p>'+''.join(cards))
    print('review entries',len(records),'image pairs',len(result['images']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--no-images',action='store_true')
    a=p.parse_args();run(a.experiment,not a.no_images)

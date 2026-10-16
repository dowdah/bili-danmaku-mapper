"""Check complete descriptors and resulting paths, beyond a pHash-only comparison."""
from dataclasses import replace
from pathlib import Path
import json,subprocess
import numpy as np
from bili_danmaku_mapper import AlignConfig
from bili_danmaku_mapper.media import VideoInfo,extract_features,FEATURE_FIELDS
from bili_danmaku_mapper.alignment import align_features
from bili_danmaku_mapper.performance import collecting


def _run_file(target,role,output,records):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
         'format=duration,start_time:stream=pix_fmt,avg_frame_rate','-of','json',target]))
    stream=data['streams'][0];num,den=stream['avg_frame_rate'].split('/')
    info=VideoInfo(target,float(data['format']['duration']),float(data['format'].get('start_time',0)),float(num)/float(den),stream['pix_fmt'])
    config=AlignConfig(cache_dir=None,evidence_mode='visual',refine=False,alignment_cache=False)
    for name,start in [('head',30.),('middle',info.duration/2),('tail',info.duration-90.)]:
        for rate,length in [(2,60),(0,6)]:
            length=min(length,max(.1,info.duration-.01));start=max(0.,min(start,info.duration-length))
            with collecting() as sw_metrics:
                sw=extract_features(target,config,rate=rate,start=start,end=start+length,use_cache=False,_info=info)
            with collecting() as hw_metrics:
                hw=extract_features(target,replace(config,hwaccel='videotoolbox'),rate=rate,start=start,end=start+length,use_cache=False,_info=info)
            same_pts=np.array_equal(sw.times,hw.times)
            differences={k:float(np.max(np.abs(getattr(sw,k).astype(float)-getattr(hw,k).astype(float))))
                         if getattr(sw,k).shape==getattr(hw,k).shape else None for k in FEATURE_FIELDS if k!='times'}
            a=align_features(sw,sw,config);b=align_features(sw,hw,config)
            points=lambda r:[(p.source_time,p.target_time) for s in r.segments for p in s.anchors]
            unchanged=points(a)==points(b)
            verified=any(e.get('actual_backend')=='videotoolbox' for e in hw_metrics.events) and not hw_metrics.counts.get('hardware_fallbacks',0)
            records.append({'role':role,'window':name,'rate':rate,'seconds':length,'pts_equal':same_pts,'max_feature_differences':differences,
                            'self_alignment_paths_equal':unchanged,'anchors':len(points(a)),
                            'actual_hardware_verified':verified,'passed':bool(same_pts and unchanged and verified)})
            print('hardware mapping compatibility',name,rate,records[-1]['passed'],flush=True)
            Path(output).write_text(json.dumps({'scope':'Backend consistency on original-file windows; not Bilibili-pair accuracy',
                                               'records':records,'complete':len(records)==12,'passed':len(records)==12 and all(r['passed'] for r in records)},indent=2))


def run(inputs,output):
    records=[]
    for role in ('source','target'):_run_file(inputs[role],role,output,records)
    return len(records)==12 and all(r['passed'] for r in records)

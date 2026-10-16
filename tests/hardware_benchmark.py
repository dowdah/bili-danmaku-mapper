"""Hardware component benchmark: run under optimization_benchmark.run_guarded.
Argument: an experiment directory containing inputs.json (source, target, config).
Writes hardware-metrics.json; no media copies. PTS/hash equality is not pair accuracy.
"""
import json,sys,time,subprocess
from pathlib import Path
from dataclasses import replace
import numpy as np
from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.media import VideoInfo, extract_features
from bili_danmaku_mapper.performance import collecting

root=Path(sys.argv[1]); inputs=json.loads((root/'inputs.json').read_text()); target=inputs['target']
data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','format=duration,start_time:stream=pix_fmt,avg_frame_rate','-of','json',target]))
stream=data['streams'][0];n,d=stream['avg_frame_rate'].split('/')
info=VideoInfo(target,float(data['format']['duration']),float(data['format'].get('start_time',0)),float(n)/float(d),stream['pix_fmt'])
config=AlignConfig(**{**inputs['config'],'cache_dir':None})
records=[]; references={}
for scenario in ('coarse','continuous_dense','frequent_seeks'):
 for repeat in range(3):
  for backend in ((None,'videotoolbox') if repeat%2==0 else ('videotoolbox',None)):
   windows=([(s,s+60) for s in (30,info.duration/2,info.duration-90)] if scenario=='coarse'
            else [(info.duration/2,info.duration/2+30)] if scenario=='continuous_dense'
            else [(info.duration/2+i*10,info.duration/2+i*10+2) for i in range(6)])
   begun=time.perf_counter();parts=[]
   with collecting() as metrics:
    for a,b in windows:parts.append(extract_features(target,replace(config,hwaccel=backend),rate=2 if scenario=='coarse' else 0,start=a,end=b,use_cache=False,_info=info))
   timing=time.perf_counter()-begun
   row={'scenario':scenario,'repeat':repeat,'backend_requested':backend or 'software','elapsed_seconds':timing,'performance':metrics.to_dict()}
   if backend is None:references[(scenario,repeat)]=parts
   # Sequence order alternates, so comparisons are calculated after each pair.
   records.append(row)
   np.savez_compressed(root/f'.hw-{scenario}-{repeat}-{backend or "software"}.npz',**{f'{i}_{k}':getattr(f,k) for i,f in enumerate(parts) for k in ('times','hashes')})
   (root/'hardware-metrics.json').write_text(json.dumps({'scope':'original-file windows; no full media copies','records':records},indent=2))
   print(scenario,repeat,backend or 'software',round(timing,3),flush=True)
  sw=np.load(root/f'.hw-{scenario}-{repeat}-software.npz');hw=np.load(root/f'.hw-{scenario}-{repeat}-videotoolbox.npz')
  for row in records[-2:]:
   row['pts_equal']=all(sw[f'{i}_times'].shape==hw[f'{i}_times'].shape and np.allclose(sw[f'{i}_times'],hw[f'{i}_times'],atol=1e-6) for i in range(len(windows)))
   row['hash_hamming_mean']=float(np.mean(np.concatenate([np.unpackbits(np.bitwise_xor(sw[f'{i}_hashes'],hw[f'{i}_hashes']),axis=1).sum(axis=1) for i in range(len(windows))]))) if row['pts_equal'] else None
  sw.close();hw.close()
  for suffix in ('software','videotoolbox'):(root/f'.hw-{scenario}-{repeat}-{suffix}.npz').unlink()
  (root/'hardware-metrics.json').write_text(json.dumps({'scope':'original-file windows; no full media copies','records':records},indent=2))

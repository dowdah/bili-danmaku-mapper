"""Audio-led scheduling: sparse cross-checks, local visual recovery, no extrapolation."""
from dataclasses import replace
import time
import numpy as np

from .config import ALGORITHM_VERSION
from .media import probe_video, extract_features
from .models import AlignmentResult, Anchor
from .performance import phase, count
from .progress import report


def align_audio_first(source, target, config, anchors=None):
    from .alignment import build_segments, validate_manual_anchors, align_features, _align_uncached, _informative
    from .audio import AudioUnavailable, align_audio, fuse_alignment
    started = time.perf_counter()
    si, ti = probe_video(source, config), probe_video(target, config)
    source_config, target_config = config.for_video("source"), config.for_video("target")
    manual = validate_manual_anchors(anchors, si.duration, ti.duration)
    report('audio_matching')
    try:
        with phase('audio_coarse_alignment'):
            audio, diagnostics = align_audio(si, ti, config, manual)
    except AudioUnavailable as error:
        audio, diagnostics = [], {'status':'unavailable', 'reason':str(error)}
    if not audio:
        report('visual_fallback', reason=diagnostics.get('reason', 'no_reliable_audio_match'))
        result = _align_uncached(source, target, replace(config, evidence_mode='visual'), anchors)
        result.config = config.to_dict()
        result.diagnostics.update(audio=diagnostics, hybrid_strategy='audio_first',
                                  fallback_reason=diagnostics.get('reason', 'no_reliable_audio_match'))
        return result
    segments = build_segments(audio, config)
    visual, forbidden = [], []
    local_config = replace(config, evidence_mode='visual', refine=False, alignment_cache=False)
    checked = set()
    requests=[]
    coarse_source=coarse_target=None
    refinement_jobs=[]
    def schedule(sa,sb,ta,tb,recovery=False):
        requests.append((sa,sb,ta,tb,recovery))

    def check(sa, sb, ta, tb, recovery=False):
        sa, sb, ta, tb = max(0., sa), min(si.duration, sb), max(0., ta), min(ti.duration, tb)
        if sb-sa < .4 or tb-ta < .4: return
        window_key = tuple(round(x, 6) for x in (sa,sb,ta,tb))
        if window_key in checked: return
        checked.add(window_key)
        report('visual_recovery' if recovery else 'visual_validation', completed=len(checked), unit='checks')
        from .media import subset
        sf=(subset(coarse_source,(coarse_source.times>=sa)&(coarse_source.times<sb)) if coarse_source is not None
            else extract_features(source,source_config,start=sa,end=sb,_info=si))
        tf=(subset(coarse_target,(coarse_target.times>=ta)&(coarse_target.times<tb)) if coarse_target is not None
            else extract_features(target,target_config,start=ta,end=tb,_info=ti))
        if not len(sf) or not len(tf): return
        if not np.any(_informative(sf)) or not np.any(_informative(tf)):
            count('visual_uninformative_windows'); return
        constrained = [m for m in (anchors or []) if sa <= m['source_time'] <= sb and ta <= m['target_time'] <= tb]
        with phase('visual_validation'):
            result = align_features(sf, tf, local_config, constrained)
        points = [a for seg in result.segments for a in seg.anchors if not a.manual]
        if points and not recovery:
            points=[min(points,key=lambda a:abs(a.source_time-(sa+sb)/2))]
        needs_refine = recovery
        for a in points:
            support = next((seg for seg in segments if seg.source_start <= a.source_time <= seg.source_end), None)
            if support:
                xs = [p.source_time for p in support.anchors]; ys = [p.target_time for p in support.anchors]
                expected = float(np.interp(a.source_time, xs, ys))
                if abs(expected-a.target_time) > .08: needs_refine = True
        deferred=bool(config.refine and points and needs_refine)
        if deferred:
            path=[(int(np.argmin(np.abs(sf.times-a.source_time))),
                   int(np.argmin(np.abs(tf.times-a.target_time))),a.confidence) for a in points]
            refinement_jobs.append((sf,tf,path))
            points=[]
        if not points and not recovery and not deferred:
            # Texture alone is not a contradiction (censoring can replace a moving
            # image). Record the failed cross-check; audio's own local verification
            # and ambiguity rejection still apply.
            count('visual_checks_without_match')
        visual.extend(points)
        count('visual_check_windows')

    # Stable segments: endpoints plus periodic context checks. Endpoints also
    # examine the two sides of every detected edit, never interpolate across it.
    for seg in segments:
        checks = list(np.arange(seg.source_start+5.,seg.source_end-1.5,10.))
        for t in checks:
            half=1.5
            a,b=max(seg.source_start,t-half),min(seg.source_end,t+half)
            if b-a < min(3.,seg.source_end-seg.source_start):
                a=max(seg.source_start,b-3.);b=min(seg.source_end,a+3.)
            predict=lambda x: float(np.interp(x,[p.source_time for p in seg.anchors],[p.target_time for p in seg.anchors]))
            schedule(a,b,predict(a)-2.,predict(b)+2.)
        # A boundary needs context on both sides to localize the actual edit;
        # restricting the window to the last audio anchor loses valid coverage.
        for s,t in ((seg.source_start,seg.target_start),(seg.source_end,seg.target_end)):
            schedule(s-1.5,s+1.5,t-3.5,t+3.5,recovery=True)

    # Recover unsupported intervals within reliable neighbors. A deleted interval
    # with no target span remains unmapped. Unknown heads/tails use their actual
    # available timeline bounds, rather than extrapolating an offset.
    supports = sorted(audio+manual,key=lambda a:a.source_time)
    bounds = [None]+supports+[None]
    recovery_windows=[]
    for left,right in zip(bounds,bounds[1:]):
        sa=left.source_time if left else 0.; sb=right.source_time if right else si.duration
        ta=left.target_time if left else 0.; tb=right.target_time if right else ti.duration
        if sb-sa > config.strict_gap+.1 and tb-ta > .4:
            recovery_windows.append((max(0.,sa-.75),min(si.duration,sb+.75),max(0.,ta-.75),min(ti.duration,tb+.75)))
    for window in recovery_windows:schedule(*window,recovery=True)
    # Consolidate overlapping short checks in both timelines. Keep long recovery
    # searches intact; never split them by an assumed 1:1 edit location.
    pending=[]
    for sa,sb,ta,tb,recovery in sorted(requests):
        sa,sb,ta,tb=max(0.,sa),min(si.duration,sb),max(0.,ta),min(ti.duration,tb)
        if (pending and sa<=pending[-1][1]+2 and ta<=pending[-1][3]+2
                and max(sb,pending[-1][1])-pending[-1][0]<=10
                and max(tb,pending[-1][3])-pending[-1][2]<=16):
            old=pending[-1];pending[-1]=(old[0],max(old[1],sb),min(old[2],ta),max(old[3],tb),old[4] or recovery)
        else:pending.append((sa,sb,ta,tb,recovery))
    # When sparse checks collectively cover much of a long movie, one shared
    # coarse decode is cheaper than repeated seeks. This also reuses existing
    # full-video features from an earlier interrupted task.
    from .media import merge_windows,dense_windows
    def coverage(windows,duration):return sum(b-a for a,b in merge_windows(windows,duration,gap=0))
    source_requested=float(coverage([(a,b) for a,b,_,_,_ in pending],si.duration))
    target_requested=float(coverage([(a,b) for _,_,a,b,_ in pending],ti.duration))
    share_source=bool(len(pending)>=20 and source_requested>=si.duration*.4)
    share_target=bool(len(pending)>=20 and target_requested>=ti.duration*.4)
    report('visual_plan',checks=len(pending),source_requested_seconds=source_requested,
           target_requested_seconds=target_requested,shared_source=share_source,shared_target=share_target)
    if share_source:
        report('source_features');coarse_source=extract_features(source,source_config,_info=si)
    if share_target:
        report('target_features');coarse_target=extract_features(target,target_config,_info=ti)
    for request in pending:check(*request)
    # Plan precision from actual coarse matches, rather than pre-decoding every
    # possible recovery region. Batch dense coverage once, then answer each job
    # from the same timestamped features.
    windows=[(max(0.,float(tf.times[j])-.6),min(ti.duration,float(tf.times[j])+.65))
             for sf,tf,path in refinement_jobs for i,j,_ in path]
    if windows and config.cache_dir and config.dense_cache:
        merged=merge_windows(windows,ti.duration)
        tiles={t for a,b in windows for t in range(int(a//30),int(max(a,b-1e-7)//30)+1)}
        tile_span=sum(min(30.,ti.duration-t*30) for t in tiles)
        if sum(b-a for a,b in merged)>=tile_span*.5:
            report('precision_prefetch')
            for block in dense_windows(ti,windows,target_config):pass
    from .alignment import refine_path
    for sf,tf,path in refinement_jobs:
        with phase('visual_local_precision'):visual.extend(refine_path(path,sf,tf,config))

    # Deduplicate overlap samples and honor exact manual correspondences globally.
    chosen={}
    for a in visual:
        if not all(abs(a.source_time-m.source_time)>.15 and
                   ((a.source_time<m.source_time and a.target_time<m.target_time) or
                    (a.source_time>m.source_time and a.target_time>m.target_time)) for m in manual):continue
        k=round(a.source_time,5)
        if k not in chosen or a.confidence>chosen[k].confidence:chosen[k]=a
    ordered=sorted(list(chosen.values())+manual,key=lambda a:a.source_time)
    monotonic=[]
    for a in ordered:
        if not monotonic or a.target_time>monotonic[-1].target_time:monotonic.append(a)
    result=AlignmentResult(si.path,ti.path,si.duration,ti.duration,build_segments(monotonic,config),config.to_dict(),
                           {'algorithm_version':ALGORITHM_VERSION,'hybrid_strategy':'audio_first',
                            'audio':diagnostics,'visual_check_windows':len(checked),
                            'visual_recovery_windows':len(recovery_windows),'planned_visual_checks':len(pending),
                            'shared_source_coarse':coarse_source is not None,'shared_target_coarse':coarse_target is not None,
                            'confidence_is_probability':False})
    result=fuse_alignment(result,audio,config)
    # Sparse checks must protect the entire affected segment, not just the few
    # checked frames, when a visual contradiction survives A/V-delay calibration.
    bias = result.diagnostics.get('audio_bias_seconds',0.)
    intervals = result.diagnostics.setdefault('unverified_intervals',[])
    for seg in segments:
        contradictions=[]
        xs=[p.source_time for p in seg.anchors];ys=[p.target_time for p in seg.anchors]
        for a in visual:
            if seg.source_start <= a.source_time <= seg.source_end:
                expected=float(np.interp(a.source_time,xs,ys))-bias
                if abs(expected-a.target_time)>.2:contradictions.append(a.source_time)
        if contradictions:
            intervals.append({'source_start':seg.source_start,'source_end':seg.source_end,
                              'reason':'audio_visual_segment_conflict'})
    result.diagnostics['elapsed_seconds']=time.perf_counter()-started
    result.validate()
    return result

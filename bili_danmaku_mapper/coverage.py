"""Experimental measured-evidence recovery. Predictions only constrain searches.

The strict interpolation rule is unchanged. Both scheduling strategies use this
same verifier; old default decisions remain available with coverage_recovery=False.
"""
from bisect import bisect_left
from dataclasses import replace
from collections import Counter
import math
import time
import numpy as np

from .models import Anchor, AlignmentResult, Segment
from .performance import count, phase
from .progress import report


def robust_segments(points, config):
    """Separate persistent changes from isolated timing outliers, without smoothing PTS.

    A model proposes an inlier set, never synthetic anchors. A short, unsupported
    model remains a singleton for exact-anchor mapping and cannot fill a hole.
    """
    points = sorted(points, key=lambda a: a.source_time)
    groups, pending = [], list(points)
    while pending:
        first = pending[0]
        seed = [a for a in pending[:40] if a.source_time-first.source_time <= 4]
        slopes = [1.0]
        for i, a in enumerate(seed):
            for b in seed[i+1:]:
                ds = b.source_time-a.source_time
                if .75 <= ds <= 4 and .5 <= (b.target_time-a.target_time)/ds <= 2:
                    slopes.append((b.target_time-a.target_time)/ds)
        best = None
        for slope in slopes:
            offsets = np.array([a.target_time-slope*a.source_time for a in seed])
            for offset in offsets:
                inside = np.abs(offsets-offset) <= .08+1e-7
                fit = [a for a, yes in zip(seed, inside) if yes]
                span = fit[-1].source_time-fit[0].source_time if fit else 0
                score = (sum((1e6 if a.manual else 1.)+a.confidence for a in fit), span, -abs(slope-1))
                if best is None or score > best[0]: best = (score, slope, float(offset), fit)
        _, slope, offset, fit = best
        if len(fit)>=3 and fit[-1].source_time-fit[0].source_time>=.75:
            xs=np.array([a.source_time for a in fit]);ys=np.array([a.target_time for a in fit])
            dx=xs-xs.mean();dy=ys-ys.mean()
            fitted=float(dx@dy/(dx@dx)) if dx@dx>1e-8 else slope
            fitted_offset=float(np.median(ys-fitted*xs))
            if .5<=fitted<=2 and np.max(np.abs(ys-fitted*xs-fitted_offset))<=.08+1e-7:
                slope,offset=fitted,fitted_offset
        if len(fit) < 3 or fit[-1].source_time-fit[0].source_time < .75:
            groups.append([pending.pop(0)]); continue
        # Do not let a later dominant run consume earlier unmatched anchors.
        start = pending.index(fit[0])
        groups.extend([[a] for a in pending[:start]])
        pending = pending[start:]
        run, skipped, cursor = [], [], 0
        for cursor, a in enumerate(pending):
            if run and a.source_time-run[-1].source_time > config.max_segment_gap: break
            if abs(a.target_time-(slope*a.source_time+offset)) <= .08+1e-7:
                run.append(a); skipped=[]
                if len(run)>=8 and len(run)%8==0:
                    recent=[v for v in run[-32:] if a.source_time-v.source_time<=8]
                    xs=np.array([v.source_time for v in recent]);ys=np.array([v.target_time for v in recent])
                    dx=xs-xs.mean();dy=ys-ys.mean()
                    fitted=float(dx@dy/(dx@dx)) if dx@dx>1e-8 else slope
                    off=float(np.median(ys-fitted*xs))
                    if .5<=fitted<=2 and np.max(np.abs(ys-fitted*xs-off))<=.08+1e-7:
                        slope,offset=fitted,off
            else:
                skipped.append(a)
                if a.manual or (len(skipped)>=3 and skipped[-1].source_time-skipped[0].source_time>=.75):
                    cursor -= len(skipped)-1; break
        else: cursor = len(pending)
        if not run:
            groups.append([pending.pop(0)]);continue
        groups.append(run)
        # Trailing outliers must be reconsidered as another possible scene.
        if cursor == len(pending) and skipped: cursor -= len(skipped)
        if cursor <= 0: cursor = 1
        pending = pending[cursor:]
    result=[]
    for group in groups:
        if not group:continue
        a,b=group[0],group[-1]
        slope=(b.target_time-a.target_time)/(b.source_time-a.source_time) if len(group)>1 else 1.
        result.append(Segment(a.source_time,b.source_time,a.target_time,b.target_time,slope,
                              a.target_time-slope*a.source_time,min(v.confidence for v in group),group))
    return result


def joint_path(visual, audio, manual=()):
    """Weighted monotone DP over both evidence types; manual anchors partition it."""
    automatic=[a for a in visual+audio if not a.manual and all(
        abs(a.source_time-m.source_time)>.15 and
        ((a.source_time<m.source_time and a.target_time<m.target_time) or
         (a.source_time>m.source_time and a.target_time>m.target_time)) for m in manual)]
    def solve(points):
        points=sorted(points,key=lambda a:(a.source_time,a.target_time,a.evidence))
        if not points:return []
        targets=sorted({a.target_time for a in points});tree=[None]*(len(targets)+1)
        score=[];parent=[];cursor=0;previous=[]
        def better(a,b):
            if a is None:return b
            if b is None:return a
            return a if score[a]>=score[b] else b
        for i,a in enumerate(points):
            # Independent PTS measurements close in source time are alternatives,
            # rather than adjacent anchors with an ill-conditioned slope.
            while cursor<i and points[cursor].source_time <= a.source_time-.16+1e-7:
                at=bisect_left(targets,points[cursor].target_time)+1
                while at<len(tree):tree[at]=better(tree[at],cursor);at+=at&-at
                cursor+=1
            at=bisect_left(targets,a.target_time);best=None
            while at:best=better(best,tree[at]);at-=at&-at
            reward=.5+a.confidence
            value=reward;pre=None
            if best is not None:value=score[best]+reward-.7;pre=best
            for k in range(max(0,i-45),i):
                b=points[k];ds=a.source_time-b.source_time;dt=a.target_time-b.target_time
                if ds<.16-1e-7 or dt<=0 or ds>5:continue
                ratio=dt/ds
                penalty=.04*ds + (.1*abs(math.log(ratio)) if .5<=ratio<=2 else .8)
                proposal=score[k]+reward-penalty
                if proposal>value:value,pre=proposal,k
            score.append(value);parent.append(pre)
        at=max(range(len(points)),key=lambda i:score[i]);path=[]
        while at is not None:path.append(points[at]);at=parent[at]
        return list(reversed(path))
    result=[];bounds=[None]+list(manual)+[None]
    for left,right in zip(bounds,bounds[1:]):
        result.extend(solve([a for a in automatic if (left is None or a.source_time>left.source_time)
                            and (right is None or a.source_time<right.source_time)]))
        if right is not None:result.append(right)
    return result


def _predict(segment, stamp):
    return float(np.interp(stamp,[a.source_time for a in segment.anchors],[a.target_time for a in segment.anchors]))


def _supported(segments):
    return [s for s in segments if len(s.anchors)>=6 and s.source_end-s.source_start>=2.5]


def fuse_verified(visual, audio, manual, config, source_duration, target_duration):
    """Only sequence-supported contradictions can quarantine a mapping interval."""
    visual=joint_path(visual,[],manual)
    vs=robust_segments(visual,config); aus=robust_segments(audio,config)
    adjusted=[];biases=[];forbidden=[];checks=[]
    supported_ids={id(s) for s in _supported(vs)}
    for left,right in zip(vs,vs[1:]):
        ds=right.source_start-left.source_end;dt=right.target_start-left.target_end
        if id(left) in supported_ids and id(right) in supported_ids and ds>2e-6 and abs(dt-ds*(left.slope+right.slope)/2)>.2:
            forbidden.append({'source_start':left.source_end+1e-6,'source_end':right.source_start-1e-6,
                              'reason':'edit_or_unverified_interval','category':'insufficient_evidence'})
    for seg in aus:
        overlap=[(v, _predict(seg,v.source_time)) for s in _supported(vs) for v in s.anchors
                 if seg.source_start<=v.source_time<=seg.source_end]
        differences=np.array([t-v.target_time for v,t in overlap]);bias=0.
        if len(overlap)>=6 and overlap[-1][0].source_time-overlap[0][0].source_time>=2.5:
            med=float(np.median(differences))
            if abs(med)<=.5 and np.percentile(np.abs(differences-med),90)<=.08: bias=med
        biases.append({'source_start':seg.source_start,'source_end':seg.source_end,'bias':bias})
        adjusted.extend(Anchor(a.source_time,a.target_time-bias,a.confidence,evidence='audio')
                        for a in seg.anchors if 0<=a.target_time-bias<=target_duration)
        if seg not in _supported(aus):continue
        agree=[v.source_time for v,t in overlap if abs(t-bias-v.target_time)<=.08]
        bad=[v.source_time for v,t in overlap if abs(t-bias-v.target_time)>.2]
        for stamp in bad:
            left=max((t for t in agree if t<stamp),default=seg.source_start)
            right=min((t for t in agree if t>stamp),default=seg.source_end)
            # No agreement on a side means the disputed extent remains unresolved,
            # rather than truncating uncertainty to an arbitrary small radius.
            forbidden.append({'source_start':left,'source_end':right,'reason':'audio_visual_conflict',
                              'category':'evidence_conflict'})
            checks.append(stamp)
    # Union masks, retaining precise brackets and no interval crossing by fit.
    merged=[]
    for value in sorted(forbidden,key=lambda x:x['source_start']):
        if merged and value['source_start']<=merged[-1]['source_end'] and value['reason']==merged[-1]['reason']:
            merged[-1]['source_end']=max(merged[-1]['source_end'],value['source_end'])
        else:merged.append(value.copy())
    measured=[a for s in vs for a in s.anchors]
    path=joint_path(measured,adjusted,manual)
    segments=robust_segments(path,config)
    return segments,merged,{'local_audio_biases':biases,'sequence_supported_conflicts':len(checks),
                           'joint_candidates':len(measured)+len(adjusted),'visual_sequence_outliers':len(visual)-len(measured),'joint_selected':sum(len(s.anchors) for s in segments)}


def _join_features(parts, info):
    from .media import Features
    fields=('times','hashes','center_hashes','gray','quality')
    arrays={k:np.concatenate([getattr(p,k) for p in parts]) for k in fields}
    if not len(arrays['times']):return Features(**arrays,info=info)
    order=np.argsort(arrays['times'],kind='stable');order=order[np.r_[True,np.diff(arrays['times'][order])>1e-6]]
    return Features(**{k:v[order] for k,v in arrays.items()},info=info)


def _align_recovered(source, target, config, anchors=None):
    from .alignment import align_features,validate_manual_anchors
    from .audio import align_audio,AudioUnavailable
    from .media import extract_features,probe_video,subset,dense_windows
    si,ti=probe_video(source,config),probe_video(target,config)
    manual=validate_manual_anchors(anchors,si.duration,ti.duration)
    local=replace(config,coverage_recovery=False,refine=config.refine,alignment_cache=False)
    audio=[];adiag={};source_features=target_features=None
    def audio_pass():
        nonlocal audio,adiag
        if config.evidence_mode!='hybrid':return
        report('audio_matching')
        try:audio,adiag=align_audio(si,ti,config,manual)
        except AudioUnavailable as error:adiag={'status':'unavailable','reason':str(error)}
        if adiag.get('normalization',{}).get('status')=='verified_sequence':
            from .evidence import _scales
            _scales.set((float(adiag['normalization']['scale']),))
    if config.hybrid_strategy=='audio_first':audio_pass()
    report('source_features');source_features=extract_features(source,config.for_video('source'),_info=si)
    report('target_features');target_features=extract_features(target,config.for_video('target'),_info=ti)
    # Full candidates provide a shared safety fallback when audio support is sparse.
    report('sequence_matching')
    if config.hybrid_strategy!='audio_first':audio_pass()
    base=align_features(source_features,target_features,config,anchors)
    visual=[a for s in base.segments for a in s.anchors if not a.manual]
    segments,forbidden,fdiag=fuse_verified(visual,audio,manual,config,si.duration,ti.duration)
    from .evidence import observe_stage
    observe_stage('joint_fusion',si,ti,segments,forbidden,config)
    attempted=set()
    audit={'rounds':[],'initial_visual_anchors':len(visual),'initial_segments':len(segments)}
    for round_number in range(0 if config.recovery_stage=="fusion" else 2):
        round_started=time.perf_counter()
        report('coverage_recovery',round=round_number+1)
        measured=[a for s in segments for a in s.anchors]
        windows=[]
        for s in segments:
            for a,b in zip(s.anchors,s.anchors[1:]):
                if config.strict_gap+1e-7 < b.source_time-a.source_time <=config.max_segment_gap:
                    windows.append((a,b,'short_evidence_gap'))
        for left,right in zip(segments,segments[1:]):
            a,b=left.anchors[-1],right.anchors[0];ds=b.source_time-a.source_time;dt=b.target_time-a.target_time
            if 0<ds<=60 and dt>0:
                reason='fragment_boundary' if abs(dt-ds*(left.slope+right.slope)/2)<=.16 else 'edit_boundary_search'
                windows.append((a,b,reason))
        # Contradictions get an actual local search, even if no short gap exists.
        for interval in forbidden:
            for stamp in np.arange(interval['source_start'],interval['source_end']+.01,6):
                before=[a for a in measured if a.source_time<=stamp-1.5]
                after=[a for a in measured if a.source_time>=stamp+1.5]
                if before and after:windows.append((before[-1],after[0],'conflict_check'))
        before_filter=len(windows)
        windows=[w for w in windows if (round(w[0].source_time,5),round(w[0].target_time,5),
            round(w[1].source_time,5),round(w[1].target_time,5),bool(round_number==1 and config.strict_gap<w[1].source_time-w[0].source_time<=3)) not in attempted]
        windows.sort(key=lambda w:(w[1].source_time-w[0].source_time,w[0].source_time))
        precise_source=[]
        if round_number==1:
            from .media import merge_windows
            requests=[(a.source_time,min(si.duration,b.source_time+.05)) for a,b,reason in windows
                      if config.strict_gap < b.source_time-a.source_time <=3]
            for sa,sb in merge_windows(requests,si.duration):
                precise_source.append(extract_features(source,config.for_video('source'),rate=4,
                    start=sa,end=sb,_info=si))
            if precise_source:precise_source=_join_features(precise_source,si)
            else:precise_source=None
        seen=set();added=[];stats=Counter()
        for left,right,reason in windows:
            key=(round(left.source_time,5),round(right.source_time,5))
            if key in seen:continue
            attempted.add((round(left.source_time,5),round(left.target_time,5),round(right.source_time,5),round(right.target_time,5),bool(round_number==1 and config.strict_gap<right.source_time-left.source_time<=3)))
            seen.add(key);sa=max(0.,left.source_time-1.5);sb=min(si.duration,right.source_time+1.5)
            ta=max(0.,left.target_time-2);tb=min(ti.duration,right.target_time+2)
            if tb<=ta or sb<=sa:continue
            sf=subset(source_features,(source_features.times>=sa)&(source_features.times<sb))
            tf=subset(target_features,(target_features.times>=ta)&(target_features.times<tb))
            controls=[{'source_time':left.source_time,'target_time':left.target_time},
                      {'source_time':right.source_time,'target_time':right.target_time}]
            if round_number==1 and config.strict_gap < right.source_time-left.source_time <=3:
                # Decode only new source samples inside the hole; context reuses
                # existing coarse PTS. Target precision is shared dense cache.
                if precise_source is not None:
                    sf_new=subset(precise_source,(precise_source.times>=left.source_time)&(precise_source.times<=right.source_time))
                    sf=_join_features([sf,sf_new],si)
            if len(sf)<3 or len(tf)<3:stats['insufficient_context']+=1;continue
            lc=replace(config,alignment_cache=False)
            result=align_features(sf,tf,lc,controls)
            candidates=[a for s in result.segments for a in s.anchors if not a.manual
                        and left.source_time<a.source_time<right.source_time]
            # Require a sequence agreeing with *both measured endpoints*. A
            # candidate alone is never allowed to bridge a suspected edit.
            supports=[a for s in result.segments for a in s.anchors]
            for seg in robust_segments(supports,config):
                both=(seg.source_start<=left.source_time+1e-6 and seg.source_end>=right.source_time-1e-6 and len(seg.anchors)>=3)
                one_side=(reason=='edit_boundary_search' and len(seg.anchors)>=6
                          and seg.source_end-seg.source_start>=2.5 and
                          (any(abs(a.source_time-left.source_time)<1e-6 for a in seg.anchors) or
                           any(abs(a.source_time-right.source_time)<1e-6 for a in seg.anchors)))
                if both or one_side:
                    inside=[a for a in candidates if a in seg.anchors]
                    added.extend(inside);stats['verified_new_anchors']+=len(inside)
            stats[reason]+=1
            if len(seen)%20==0:report('coverage_recovery',completed=len(seen),total=len(windows),round=round_number+1,unit='checks')
        count('coverage_windows',len(seen));count('coverage_new_anchors',stats['verified_new_anchors'])
        audit['rounds'].append(dict(stats,windows=len(seen),reused_decisions=before_filter-len(windows),elapsed_seconds=time.perf_counter()-round_started))
        if not added and round_number==1:break
        unique={(round(a.source_time,6),round(a.target_time,6)):a for a in visual+added}
        visual=list(unique.values())
        segments,forbidden,fdiag=fuse_verified(visual,audio,manual,config,si.duration,ti.duration)
        observe_stage('visual_retrieval' if round_number==0 else 'visual_dense',si,ti,segments,forbidden,config)
    if config.recovery_stage=='audio' and config.evidence_mode=='hybrid' and (audio or segments):
        with phase('audio_recovery'):
            additional,audit['audio_recovery']=rescue_audio(si,ti,segments,adiag,config,manual)
        if additional:
            unique={(round(a.source_time,6),round(a.target_time,6)):a for a in audio+additional}
            audio=list(unique.values())
            segments,forbidden,fdiag=fuse_verified(visual,audio,manual,config,si.duration,ti.duration)
    observe_stage('audio_profiles' if config.recovery_stage=='audio' else 'final_model',si,ti,segments,forbidden,config)
    result=AlignmentResult(si.path,ti.path,si.duration,ti.duration,segments,config.to_dict(),{
        **base.diagnostics,'coverage_recovery':audit,'audio':adiag,'hybrid_strategy':config.hybrid_strategy,
        'unverified_intervals':forbidden,'fusion':fdiag,'confidence_is_probability':False,
        'real_accuracy_human_verified':False})
    result.validate();return result


def align_recovered(source,target,config,anchors=None):
    from .evidence import auditing
    # Scale is acoustically verified within align_audio before it is used.
    with auditing() as audit:
        result=_align_recovered(source,target,config,anchors)
    result.diagnostics['candidate_audit']={**audit,'counts':dict(audit['counts']),
        'counts_are_stage_events_not_unique_frames':True,'examples_per_reason_limit':30}
    return result


def rescue_audio(source_info,target_info,segments,diagnostics,config,manual):
    """Probe alternate mixes in evidence holes; expand only verified profiles.

    All candidates must pass the original triplet verifier AND a two-second
    sequence check. Range predictions constrain retrieval; never create points.
    """
    from .audio import (AudioFeatures,extract_audio,select_streams,audio_anchors,_original_audio_points,
                        _verify_audio_position,supported_audio,AudioUnavailable)
    from .media import merge_windows
    report('audio_recovery')
    try:ss,ts,_=select_streams(source_info.path,target_info.path,config)
    except AudioUnavailable:return [],{'status':'unavailable'}
    if diagnostics.get('normalization',{}).get('status')=='verified_sequence':
        scale=float(diagnostics['normalization']['scale'])
    else:scale=1.
    scale_hypotheses=[scale]
    ratio=source_info.fps/target_info.fps if target_info.fps else 1.
    if scale==1 and config.audio_speed_search and any(abs(ratio-r)<.003 for r in (25/24,24/25)):
        scale_hypotheses.append(ratio)
    # Existing full mono features are reused for the first hypothesis.
    source_mono=extract_audio(source_info,ss,config,time_scale=scale)
    target_mono=extract_audio(target_info,ts,config)
    pairs=[]
    for segment in _supported(segments):
        for left,right in zip(segment.anchors,segment.anchors[1:]):
            if right.source_time-left.source_time>config.strict_gap+1e-7:
                pairs.append((left,right))
    def sub(f,lo,hi):
        keys=(f.times>=lo)&(f.times<=hi);frames=(f.frame_times>=lo)&(f.frame_times<=hi)
        return AudioFeatures(f.times[keys],f.keys[keys],f.frame_times[frames],f.rms[frames],f.spectra[frames])
    out=[];stats=Counter();profiles=Counter()
    # Merge probes by fixed source blocks, limiting repeated decode/feature work.
    blocks={}
    for left,right in pairs:
        block=int((left.source_time+right.source_time)/2//20)
        blocks.setdefault(block,[]).append((left,right))
    for number,(block,holes) in enumerate(sorted(blocks.items())):
        lo=max(0.,block*20.-2);hi=min(source_info.duration,(block+1)*20.+2)
        allanchors=[a for s in segments for a in s.anchors]
        before=[a for a in allanchors if a.source_time<=lo];after=[a for a in allanchors if a.source_time>=hi]
        if not allanchors:stats['unbounded_probe']+=1;continue
        left=before[-1] if before else allanchors[0]
        right=after[0] if after else allanchors[-1]
        lo=max(lo,left.source_time);hi=min(hi,right.source_time)
        if hi-lo<3:stats['unbounded_probe']+=1;continue
        ta=max(0.,left.target_time-1);tb=min(target_info.duration,right.target_time+1)
        if tb-ta<=2:continue
        # One seed subwindow is used to select a profile. Only a successful
        # profile is evaluated over the remainder of the block.
        seed=holes[0];mid=(seed[0].source_time+seed[1].source_time)/2
        seedlo=max(lo,mid-4);seedhi=min(hi,mid+4)
        profiles_to_try=[('mono','mono')]
        if ss.get('channels',1)>=2 and ts.get('channels',1)>=2:
            profiles_to_try += [('front','front'),('left','left'),('right','right')]
        has_center=lambda stream:stream.get('channels',1)>=3 and any(name in stream.get('channel_layout','') for name in ('5.1','7.1','3.0','5.0','6.1','7.0'))
        if has_center(ts):profiles_to_try += [('mono','center')]
        if has_center(ss):profiles_to_try += [('center','mono')]
        if has_center(ss) and has_center(ts):profiles_to_try += [('center','center')]
        for probe_scale,smix,tmix in [(v,a,b) for v in scale_hypotheses for a,b in profiles_to_try]:
            try:
                sf=source_mono if smix=='mono' and probe_scale==scale else extract_audio(source_info,ss,config,time_scale=probe_scale,start=lo,end=hi,mixdown=smix)
                tf=target_mono if tmix=='mono' else extract_audio(target_info,ts,config,start=ta,end=tb,mixdown=tmix)
                sf=sub(sf,lo*probe_scale,hi*probe_scale);tf=sub(tf,ta,tb)
                raw,_=audio_anchors(sf,tf,source_info.duration*probe_scale,target_info.duration,(),(seedlo*probe_scale,seedhi*probe_scale))
                verified=[]
                for a in raw:
                    if all(_verify_audio_position(sf,tf,a.source_time+d,a.target_time-a.source_time) for d in (-1.,-.5,0.,.5,1.)):
                        verified.append(a)
                seedpoints,support=supported_audio(_original_audio_points(verified,probe_scale),source_info.duration,target_info.duration,config)
                stats['profile_probes']+=1
                if support<2.5:continue
                raw,_=audio_anchors(sf,tf,source_info.duration*probe_scale,target_info.duration,(),(lo*probe_scale,hi*probe_scale))
                good=[a for a in raw if all(_verify_audio_position(sf,tf,a.source_time+d,a.target_time-a.source_time) for d in (-1.,-.5,0.,.5,1.))]
                candidates,support=supported_audio(_original_audio_points(good,probe_scale),source_info.duration,target_info.duration,config)
                candidates=[a for a in candidates if left.source_time<a.source_time<right.source_time
                            and left.target_time<a.target_time<right.target_time and all(
                                (a.source_time<m.source_time and a.target_time<m.target_time) or
                                (a.source_time>m.source_time and a.target_time>m.target_time) for m in manual)]
                out.extend(candidates);profiles[f'{smix}->{tmix}@{probe_scale:.9f}']+=len(candidates);stats['successful_profiles']+=1
                break
            except AudioUnavailable:stats['decode_failed']+=1
        if number%5==0:report('audio_recovery',completed=number+1,total=len(blocks),unit='checks')
    return out,{'counts':dict(stats),'profiles':dict(profiles),'verified_anchors':len(out),'probe_blocks':len(blocks)}

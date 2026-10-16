"""Bounded per-run candidate provenance, independent of matching decisions."""
from contextvars import ContextVar
from contextlib import contextmanager
from collections import Counter

_active=ContextVar('coverage_audit',default=None)
_scales=ContextVar('verified_context_scales',default=())

@contextmanager
def auditing(scales=()):
    data={'counts':Counter(),'examples':{},'calls':0,'primary_frames':[]}
    token=_active.set(data);scale_token=_scales.set(tuple(scales))
    try:yield data
    finally:_active.reset(token);_scales.reset(scale_token)

def scales():return _scales.get()

def rejected(reason,source_time=None,**detail):
    data=_active.get()
    if data is None:return
    data['counts'][reason]+=1
    if data['calls']==1 and source_time is not None:
        data.setdefault('_primary_reasons',{}).setdefault(round(float(source_time),6),set()).add(reason)
    examples=data['examples'].setdefault(reason,[])
    if source_time is not None and len(examples)<30:
        examples.append({'source_time':float(source_time),**detail})

def call():
    data=_active.get()
    if data is not None:data['calls']+=1


def primary_rows(source,target,rows,path,refined):
    data=_active.get()
    if data is None or data['primary_frames']:return
    reasons=data.pop('_primary_reasons',{})
    selected={i for i,j,c in path};times={round(a.source_time,6) for a in refined}
    data['primary_frames']=[{'source_time':float(source.times[i]),
        'candidates':[{'target_time':float(target.times[j]),'cost':float(c)} for j,c in row],
        'rejection_events':sorted(reasons.get(round(float(source.times[i]),6),set())),
        'path_selected':i in selected,'precision_retained':round(float(source.times[i]),6) in times}
        for i,row in enumerate(rows)]

_stage_observer=ContextVar('coverage_stage_observer',default=None)

@contextmanager
def observing_stages(callback):
    token=_stage_observer.set(callback)
    try:yield
    finally:_stage_observer.reset(token)

def observe_stage(name,source_info,target_info,segments,intervals,config):
    callback=_stage_observer.get()
    if callback is not None:
        from .models import AlignmentResult
        model=AlignmentResult(source_info.path,target_info.path,source_info.duration,target_info.duration,
                              segments,config.to_dict(),{'unverified_intervals':intervals})
        model.validate();callback(name,model)

from dataclasses import replace
import numpy as np
import pytest
from bili_danmaku_mapper.config import AlignConfig
from bili_danmaku_mapper.coverage import robust_segments,joint_path,fuse_verified
from bili_danmaku_mapper.models import Anchor,AlignmentResult
from bili_danmaku_mapper.mapping import map_time


def points(begin=0,end=12,offset=0,evidence='visual',step=.5):
    return [Anchor(float(t),float(t+offset),.9,evidence=evidence) for t in np.arange(begin,end,step)]


def result(visual,audio=()):
    c=AlignConfig(coverage_recovery=True)
    s,u,d=fuse_verified(visual,list(audio),[],c,20,30)
    return AlignmentResult('s','t',20,30,s,c.to_dict(),{'unverified_intervals':u,'fusion':d})


def test_recovery_explicit_opt_in():
    assert not AlignConfig().coverage_recovery
    assert AlignConfig(coverage_recovery=True).strict_gap==1
    with pytest.raises(ValueError):AlignConfig(coverage_recovery=True,algorithm='legacy')


def test_isolated_visual_outlier_cannot_quarantine_audio_run():
    visual=points();visual[10]=Anchor(5,5.8,.9)
    a=result(visual,points(evidence='audio'))
    assert not a.diagnostics['unverified_intervals']
    for t in (2.25,4.25,5.25,7.25,9.25):assert map_time(t,a)[0]==pytest.approx(t,abs=.08)


def test_persistent_contradiction_is_not_silently_accepted():
    a=result(points(offset=.8),points(evidence='audio'))
    assert a.diagnostics['unverified_intervals']
    assert map_time(5,a)[0] is None


def test_real_local_edit_not_bridged_by_robust_model():
    visual=points(0,4)+points(4,10,offset=1)
    a=result(visual)
    assert map_time(3.75,a)[0] is None
    assert map_time(5.25,a)[0]==pytest.approx(6.25)


def test_missing_source_interval_stays_unmapped():
    a=result(points(0,3)+points(5,10,offset=-2))
    for t in (3.1,4,4.9):assert map_time(t,a)[0] is None


def test_no_relaxation_of_one_second_gap():
    a=result([Anchor(t,t,.9) for t in (0,.52,1.56,2.08,2.6,3.12)])
    assert map_time(1,a)[2]=='gap_exceeds_strict_limit'


def test_joint_path_is_global_and_preserves_manual():
    manual=[Anchor(4,5,1,True,'manual')]
    path=joint_path(points(0,10,offset=1)+[Anchor(1,19,.95)],points(0,10,offset=1,evidence='audio'),manual)
    assert manual[0] in path
    assert all(b.source_time>a.source_time and b.target_time>a.target_time for a,b in zip(path,path[1:]))
    assert all(a.target_time<19 for a in path)


def test_constant_av_delay_is_calibrated_from_sequences():
    a=result(points(),points(offset=.3,evidence='audio'))
    assert not a.diagnostics['unverified_intervals']
    assert any(abs(r['bias']-.3)<.01 for r in a.diagnostics['fusion']['local_audio_biases'])
    assert map_time(4.25,a)[0]==pytest.approx(4.25,abs=.08)


def test_manual_anchor_retained_after_robust_fit():
    manual=Anchor(4,5,1,True,'manual')
    c=AlignConfig(coverage_recovery=True)
    ss=robust_segments(points(0,4,offset=1)+[manual]+points(4.5,10,offset=1),c)
    assert manual in [v for s in ss for v in s.anchors]


def test_quantized_non_unit_speed_does_not_make_periodic_false_boundaries():
    source=np.arange(0,30,.52)
    measured=[Anchor(float(s),round((s*25/24+.7)*24)/24,.9) for s in source]
    segments=robust_segments(measured,AlignConfig(coverage_recovery=True))
    assert len(segments)==1
    assert len(segments[0].anchors)==len(measured)


def test_conflicting_middle_keeps_verified_outer_regions():
    visual=points(0,4)+points(4,8,offset=.8)+points(8,12)
    a=result(visual,points(evidence='audio'))
    assert map_time(1,a)[0]==pytest.approx(1,abs=.08)
    assert map_time(5,a)[0] is None
    assert map_time(10.5,a)[0]==pytest.approx(10.5,abs=.08)


def test_audit_does_not_change_default_decisions():
    from tests.test_alignment import features
    from bili_danmaku_mapper.alignment import align_features
    from bili_danmaku_mapper.evidence import auditing
    f=features(list(range(12)));c=AlignConfig(refine=False)
    before=align_features(f,f,c).to_dict()
    with auditing() as audit:after=align_features(f,f,c).to_dict()
    assert before==after
    assert len(audit['primary_frames'])==12


def test_no_evidence_does_not_create_anchors():
    assert joint_path([],[])==[]
    a=result([],[])
    assert map_time(1,a)[0] is None


def test_candidate_cache_signature_tracks_recovery_parameters():
    c=AlignConfig()
    assert c.feature_signature()==replace(c,coverage_recovery=True,recovery_stage='fusion').feature_signature()
    assert c.to_dict()!=replace(c,coverage_recovery=True).to_dict()


def test_cli_and_environment_recovery_precedence(tmp_path):
    from bili_danmaku_mapper.cli import parser,_config
    env=tmp_path/'settings';env.write_text('COVERAGE_RECOVERY=true\nRECOVERY_STAGE=fusion\n')
    args=parser().parse_args(['--env-file',str(env),'align','--source','s','--target','t','--output','out','--recovery-stage','visual'])
    c=_config(args)
    assert c.coverage_recovery and c.recovery_stage=='visual'


def test_empty_annotations_never_claim_real_accuracy():
    from tests.coverage_benchmark import review
    import tempfile,pathlib,json
    report={'entries':[{'source_index':0,'source_time':1,'target_time':None,'status':'unmapped','reason':'no_verified_segment'}]}
    with tempfile.TemporaryDirectory() as directory:
        dest=pathlib.Path(directory)/'review.json';review(report,report,dest)
        row=json.loads(dest.read_text())['records'][0]
        assert not row['human_verified'] and row['truth_target_time'] is None


def test_audio_cache_profiles_are_separate_and_long_verifier_rejects_silence(tmp_path,monkeypatch):
    from bili_danmaku_mapper import audio
    from bili_danmaku_mapper.media import VideoInfo
    movie=tmp_path/'a';movie.write_bytes(b'a')
    c=AlignConfig(cache_dir=str(tmp_path/'cache'));info=VideoInfo(str(movie),4,0,24)
    calls=[];wave=np.sin(np.arange(44100*4)/44100*2*np.pi*440).astype(np.float32)
    monkeypatch.setattr(audio,'_fingerprint_decode',lambda *args:(calls.append(args[-1] if len(args)>3 else 'mono') or audio.fingerprint(wave[::4])))
    stream={'index':1,'ordinal':0}
    audio.extract_audio(info,stream,c)
    audio.extract_audio(info,stream,c,mixdown='center')
    audio.extract_audio(info,stream,c,mixdown='center')
    assert calls==['mono','center']
    f=audio.fingerprint(np.zeros(11025*4))
    assert not audio._verify_audio_position(f,f,2,0)


def test_audio_anchor_cache_rejects_corruption(tmp_path,monkeypatch):
    from bili_danmaku_mapper import audio
    from bili_danmaku_mapper.media import VideoInfo
    source=tmp_path/'s';target=tmp_path/'t';source.write_bytes(b's');target.write_bytes(b't')
    si=VideoInfo(str(source),4,0,24);ti=VideoInfo(str(target),4,0,24);c=AlignConfig(coverage_recovery=True,cache_dir=str(tmp_path/'cache'))
    stream={'index':0,'ordinal':0};monkeypatch.setattr(audio,'select_streams',lambda *a:(stream,stream,'selected'))
    calls=[];monkeypatch.setattr(audio,'_align_audio',lambda *a:(calls.append(1) or (points(0,3,evidence='audio'),{'status':'matched'})))
    audio.align_audio(si,ti,c);audio.align_audio(si,ti,c)
    assert len(calls)==1
    cached=next((tmp_path/'cache'/'audio-anchors-v1').glob('*.json'));cached.write_text('[]')
    audio.align_audio(si,ti,c)
    assert len(calls)==2


def test_sparse_acoustic_run_does_not_fit_a_zero_span_window():
    anchors=[Anchor(t,t,.9,evidence='audio') for t in (0,.5,1,1.5,2,2.5,3,12,12.5,13)]
    result=robust_segments(anchors,AlignConfig(coverage_recovery=True))
    assert sum(len(s.anchors) for s in result)==len(anchors)


def test_annotation_evaluator_keeps_pending_and_scores_false_acceptance():
    from tests.coverage_benchmark import evaluate_annotations
    a=result(points())
    pending=evaluate_annotations({'records':[]},a)
    assert pending['precision_within_200ms'] is None
    rows=[{'source_time':2,'split':'validation','human_verified':True,'truth_target_time':2},
          {'source_time':3,'split':'validation','human_verified':True,'truth_target_time':None,'confirmed_no_correspondence':True}]
    checked=evaluate_annotations({'records':rows},a)
    assert checked['wrong_accepts']==1 and checked['precision_within_200ms']==.5
    assert not checked['default_promotion_approved']


def test_review_includes_all_withdrawn_and_large_changes(tmp_path):
    import json
    from tests.coverage_benchmark import review
    before={'entries':[{'source_index':i,'source_time':i,'target_time':i,'reason':'verified_anchor'} for i in range(200)]}
    after={'entries':[{'source_index':i,'source_time':i,'target_time':None if i<105 else i+.3 if i<121 else i,'reason':'test'} for i in range(200)]}
    output=tmp_path/'review.json';review(before,after,output);d=json.loads(output.read_text())
    assert d['critical_records']==121
    assert set(range(121)) <= {r['source_index'] for r in d['records']}
    assert all(not r['human_verified'] for r in d['records'])


def test_default_result_cache_signature_remains_compatible(tmp_path,monkeypatch):
    from bili_danmaku_mapper import alignment,media,cache
    from bili_danmaku_mapper.models import AlignmentResult
    info=media.VideoInfo('source',20,0,24)
    monkeypatch.setattr(media,'probe_video',lambda *a:info)
    monkeypatch.setattr(cache,'identity',lambda path,c:'identity:'+str(path))
    def compute(s,t,c,anchors):
        return AlignmentResult(s,t,20,20,robust_segments(points(),c),c.to_dict())
    monkeypatch.setattr(alignment,'_align_uncached',compute)
    c=AlignConfig(evidence_mode='visual',cache_dir=str(tmp_path/'cache'))
    old_options={k:v for k,v in c.to_dict().items() if k not in ('cache_dir','alignment_cache','coverage_recovery','recovery_stage')}
    expected=cache.key({'version':'sequence-4','source':'identity:source','target':'identity:target',
                        'config':old_options,'streams':None,'anchors':[]})
    a=alignment.align_videos('source','target',c)
    assert a.diagnostics['input_signature']==expected
    b=alignment.align_videos('source','target',replace(c,coverage_recovery=True))
    assert b.diagnostics['input_signature']!=expected


def test_known_timeline_gate_rejects_no_acceptance_and_tiny_coverage():
    from tests.coverage_acceptance import validation_failures
    row={'wrong_accepts':0,'p95_error_seconds':None,'precision_within_200ms':None,'correct_reliable_coverage':0}
    assert validation_failures('multichannel_mix','visual_first',row)
    row.update(p95_error_seconds=0,precision_within_200ms=1,correct_reliable_coverage=.01)
    assert validation_failures('multichannel_mix','visual_first',row)

"""Generate a report from actual finished records; missing stages stay pending."""
import argparse,json,statistics,shutil
from datetime import datetime,timezone
from pathlib import Path


def load(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def hardware_summary(records):
    summary=[]
    for scene in sorted({r['scenario'] for r in records}):
        group=[r for r in records if r['scenario']==scene]
        sw=[r['elapsed_seconds'] for r in group if r['backend_requested']=='software']
        hw=[r['elapsed_seconds'] for r in group if r['backend_requested']=='videotoolbox']
        verified=all(r.get('pts_equal') and not r['performance']['counters'].get('hardware_fallbacks',0) for r in group)
        if sw and hw:
            a,b=statistics.median(sw),statistics.median(hw)
            summary.append({'scenario':scene,'software_seconds_median':a,'videotoolbox_seconds_median':b,
                'software_seconds_range':[min(sw),max(sw)],'videotoolbox_seconds_range':[min(hw),max(hw)],
                'speedup':a/b,'reduction':1-b/a,'pts_equal_and_no_fallback':verified,
                'max_mean_hash_hamming':max((r.get('hash_hamming_mean') or 0) for r in group),
                'three_repeats':len(sw)==len(hw)==3})
    return summary



def compare_reports(root,left,right,*,right_directory='full'):
    a=load(root/'full'/f'playback-{left}.report.json')
    b=load(root/right_directory/f'playback-{right}.report.json')
    if not a or not b:return None
    if len(a['entries'])!=len(b['entries']):return {'error':'source_entries_differ'}
    differences=[];new_only=old_only=neither=0
    from collections import Counter
    reasons=Counter();evidence=Counter()
    for old,new in zip(a['entries'],b['entries']):
        if old['source_index']!=new['source_index'] or old['source_time']!=new['source_time']:
            return {'error':'source_entries_differ'}
        x,y=old['target_time'],new['target_time']
        if new['status']=='unmapped':reasons[new['reason']]+=1
        else:evidence['+'.join(new.get('evidence') or ['unknown'])]+=1
        if x is not None and y is not None:
            differences.append({'source_index':new['source_index'],'source_time':new['source_time'],
                'left_target_time':x,'right_target_time':y,'difference_seconds':y-x})
        elif y is not None:new_only+=1
        elif x is not None:old_only+=1
        else:neither+=1
    errors=sorted(abs(r['difference_seconds']) for r in differences)
    def quantile(p):
        if not errors:return None
        at=(len(errors)-1)*p;lo=int(at);hi=min(len(errors)-1,lo+1)
        return errors[lo]+(at-lo)*(errors[hi]-errors[lo])
    return {'scope':'Mapping agreement without ground truth; NOT accuracy','left':left,'right':right,
        'common_mapped':len(errors),'new_only':new_only,'old_only':old_only,'both_unmapped':neither,
        'agreement_within_200ms':sum(e<=.2 for e in errors)/len(errors) if errors else None,
        'difference_p50_seconds':quantile(.5),'difference_p95_seconds':quantile(.95),
        'maximum_difference_seconds':max(errors) if errors else None,
        'right_unmapped_reasons':dict(reasons),'right_evidence_counts':dict(evidence),
        'largest_differences':sorted((r for r in differences if abs(r['difference_seconds'])>.2),
                                   key=lambda r:abs(r['difference_seconds']),reverse=True)[:100]}

def generate(experiment,output):
    root=Path(experiment).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    hardware=load(root/'hardware-metrics.json',{}).get('records',[])
    synthetic=load(root/'synthetic-final5/metrics.json')
    if synthetic is None:synthetic=load(root/'synthetic-final4/metrics.json',{})
    windows=load(root/'windows-final/metrics.json',{}).get('records',[])
    full=[]
    for p in sorted((root/'full').glob('*.json')):
        if any(t in p.name for t in ('.guard.','.request.','summary')):continue
        r=load(p,{})
        if 'variant' in r and 'alignment' in r:
            guard=load(p.with_name(p.stem+'.guard.json'),{})
            full.append({k:r[k] for k in ('variant','cache_state','elapsed_seconds','counts','rates','cpu_seconds_self_and_children','cache_bytes','diagnostics','code_digest','playback_json','mapped_data_digest') if k in r}|{'guard':guard,'configuration':r['alignment'].get('config',{})})
    target_hardware_summary=hardware_summary(hardware)
    source_hardware_summary=hardware_summary(load(root/'source-hardware/hardware-metrics.json',{}).get('records',[]))
    window_summary=[]
    for variant,state,window in sorted({(r['variant'],r['cache_state'],r['window']) for r in windows}):
        group=[r for r in windows if (r['variant'],r['cache_state'],r['window'])==(variant,state,window)]
        values=[r['elapsed_seconds'] for r in group]
        window_summary.append({'variant':variant,'cache_state':state,'window':window,'repeats':len(group),
            'seconds_median':statistics.median(values),'seconds_range':[min(values),max(values)],
            'anchors_median':statistics.median(r['anchors'] for r in group),'source_samples':group[0]['source_samples']})
    metrics={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),
             'hardware_components':target_hardware_summary,'source_hardware_components':source_hardware_summary,
             'hardware_mapping_compatibility':load(root/'full/hardware-compatibility.json'),'synthetic':synthetic,
             'window_components':windows,'window_component_summary':window_summary,
             'full_file_runs':full,'environment':load(root/'full/environment.json'),'real_film_accuracy':'pending_human_annotation',
             'audio_preflight':{'scope':'Cached-audio diagnostic; excluded from full-file timed runs',
               'diagnostics':load(root/'real-audio-preflight.json',{}).get('diagnostics')},
             'physical_disk_io':'not measured; digest byte counters are logical reads',
             'mapping_agreement':{'baseline_to_software':compare_reports(root,'baseline','optimized_software'),
                 'software_to_hardware':compare_reports(root,'optimized_software','optimized_videotoolbox'),
                 'baseline_to_default':compare_reports(root,'baseline','default_visual_first',right_directory='validation')},
             'full_file_comparison_complete':all(any(r['variant']==entry['variant'] and r['cache_state']==entry['cache_state'] for r in full)
               for entry in load(root/'full/expected-runs.json',[
                   {'variant':v,'cache_state':state} for v in ('baseline','optimized_software')
                   for state in ('cold','features_warm','alignment_reuse') if not (v=='baseline' and state=='alignment_reuse')]))}
    from bili_danmaku_mapper import AlignConfig
    metrics['release_policy']={'default_hybrid_strategy':AlignConfig().hybrid_strategy,
        'audio_first_real_film_acceptance':'not_passed; lower reliable coverage, no human ground truth',
        'hardware_default_enabled':False}
    metrics['hardware_selection_check']=load(root/'hardware-selection-check.json')
    metrics['full_cached_backend_consistency']=load(root/'backend-feature-consistency.json')
    metrics['timed_source_snapshots']={name:str(root/directory) for name,directory in (
        ('baseline','baseline'),('optimized_software','pre-hardware-select'),('optimized_videotoolbox','timed-hardware-source'))}
    metrics['post_timing_repairs']=['Rebuild corrupt DEFLATE feature/audio caches','Rebuild non-object identity JSON records']
    metrics['feature_cache_import']=load(root/'feature-cache-import.json')
    metrics['aborted_hardware_trial']=load(root/'aborted-hardware-before-select/optimized_videotoolbox-cold.guard.json')
    validation=load(root/'validation/software-after-backend-dispatch.json',{})
    if validation:
        metrics['software_output_validation_after_backend_dispatch']={k:validation[k] for k in (
            'elapsed_seconds','rates','mapped_data_digest','code_digest','diagnostics')}
    default_validation=load(root/'validation/default-visual-first.json',{})
    if default_validation:
        metrics['default_strategy_warm_validation']={k:default_validation[k] for k in (
            'elapsed_seconds','rates','mapped_data_digest','code_digest','diagnostics','playback_json')}

    old=next((r for r in full if r['variant']=='baseline' and r['cache_state']=='cold'),None)
    new=next((r for r in full if r['variant']=='optimized_software' and r['cache_state']=='cold'),None)
    if old and new:
        a=old['diagnostics'].get('performance',{}).get('counters',{})
        b=new['diagnostics'].get('performance',{}).get('counters',{})
        phases=new['diagnostics'].get('performance',{}).get('phase_seconds_inclusive',{})
        metrics['cold_software_comparison']={
            'elapsed_reduction':1-new['elapsed_seconds']/old['elapsed_seconds'],
            'reliable_coverage_change_percentage_points':100*(new['rates']['reliable']-old['rates']['reliable']),
            'counter_ratios_new_over_old':{k:b.get(k,0)/a[k] for k in (
                'candidate_pairs','ffmpeg_starts','decoded_selected_frames',
                'decoded_requested_seconds','digest_logical_bytes') if a.get(k)},
            'visual_decode_inclusive_seconds':phases.get('visual_decode_inclusive'),
            'visual_decode_inclusive_fraction_of_elapsed':phases.get('visual_decode_inclusive',0)/new['elapsed_seconds'],
            'coverage_scope':'All original danmaku; not annotated corresponding-region coverage or accuracy'}
    hw=next((r for r in full if r['variant']=='optimized_videotoolbox' and r['cache_state']=='cold'),None)
    if old and new and hw:
        metrics['cold_hardware_comparison']={
            'reduction_vs_baseline':1-hw['elapsed_seconds']/old['elapsed_seconds'],
            'speedup_vs_same_strategy_software':new['elapsed_seconds']/hw['elapsed_seconds'],
            'source_backend':'software','target_backend':'videotoolbox',
            'hardware_fallbacks':hw['diagnostics']['performance']['counters'].get('hardware_fallbacks',0),
            'mapped_data_equal_to_software':hw['mapped_data_digest']==new['mapped_data_digest']}
    from tests.optimization_benchmark import owned_bytes
    guards=[load(path,{}) for path in root.rglob('*.guard.json')]
    peak=max((g.get('experiment_peak_bytes_sampled',0) for g in guards),default=0)
    imported=metrics.get('feature_cache_import') or {}
    metrics['space_accounting']={'scope':'Logical file lengths; not allocated blocks or physical I/O',
        'experiment_peak_bytes_sampled':peak,'final_experiment_bytes':owned_bytes(root),
        'promoted_feature_bytes_added':imported.get('added_bytes',0),
        'free_disk_bytes_at_report':shutil.disk_usage(root).free,'budget_bytes':4*1024**3}

    (output/'optimization-metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    lines=['# 对齐优化验收记录','',f"生成时间：{metrics['generated_utc']}",'',
           '原任务不被本实验暂停、恢复或覆盖；实验只使用已有本地素材，不下载、不生成整片视频副本。',
           '', '## 硬件组件实测','',
           '每个输入、每个后端、每个场景交替运行三次。coarse为片头/中段/结尾各60秒的合计；continuous_dense为中段连续30秒；frequent_seeks为六次2秒精定位。',
           '', '| 输入及场景 | 软件中位秒（范围） | VideoToolbox中位秒（范围） | 倍数 | PTS及后端验证 |',
           '|---|---:|---:|---:|---|']
    for role,rows in [('4K原片',target_hardware_summary),('720P参考',source_hardware_summary)]:
        for r in rows:
            a,b=r['software_seconds_range'],r['videotoolbox_seconds_range']
            lines.append(f"| {role} {r['scenario']} | {r['software_seconds_median']:.3f}（{a[0]:.3f}–{a[1]:.3f}） | {r['videotoolbox_seconds_median']:.3f}（{b[0]:.3f}–{b[1]:.3f}） | {r['speedup']:.2f} | {'通过' if r['pts_equal_and_no_fallback'] and r['three_repeats'] else '未通过'} |")
    lines+=['','这些是原片窗口的解码与特征计算结果，不代表整片音画对齐提速。沙箱内 IOSurface 初始化失败的实验已作废；有效数据来自可访问硬件的本机环境。',
            '', '倍数为软件耗时除以硬解耗时，小于1表示硬解反而更慢。',
            '', '720P参考视频短窗口硬解反而更慢，因此最终硬件候选按输入显式设置：参考视频软件解码、4K原片VideoToolbox。两边硬解的中止试验单独归档，不计作整片耗时。硬件筛帧顺序调整保持PTS和特征一致，不能据此宣称该顺序本身带来显著提速。',
            '', '## 合成准确性','']
    consistency=metrics.get('full_cached_backend_consistency')
    if consistency:
        rows=list(consistency['coarse'].values())+consistency['dense']
        same=all(r['pts_equal'] and r['shapes_equal'] and not any(r.get(k) for k in (
            'max_hash_hamming','max_center_hamming','max_gray_difference','max_quality_difference')) for r in rows)
        lines[-2:-2]=['',f"整片缓存后端检查：源粗帧 {consistency['coarse']['source']['software_frames']}、目标粗帧 {consistency['coarse']['target']['software_frames']}，以及 {len(consistency['dense'])} 个精细分块，PTS与全部描述符{'一致' if same else '存在差异，见JSON'}；精细分块计数含边缘重叠。这是后端一致性，不是真实正确率。", ""]
    if synthetic.get('records'):
        lines+=['| 场景 | 新版精度（200ms内） | 新版误差P95 ms | 可靠覆盖 旧/新 |','|---|---:|---:|---:|']
        for scene in dict.fromkeys(r['scenario'] for r in synthetic['records']):
            old,new=[r for r in synthetic['records'] if r['scenario']==scene]
            precision='不适用（歧义拒绝）' if new['precision_within_200ms'] is None else f"{new['precision_within_200ms']*100:.2f}%"
            error='—' if new['p95_error_seconds'] is None else f"{new['p95_error_seconds']*1000:.2f}"
            lines.append(f"| {scene} | {precision} | {error} | {old['rates']['reliable']*100:.2f}% / {new['rates']['reliable']*100:.2f}% |")
        lines+=['',f"合成验收：{'通过' if synthetic.get('passed') else '未通过'}。覆盖率为全部固定测试时间点中的可靠比例；有删除区间的可对应覆盖率另见 JSON。重复片段旧版接受的结果无唯一真值，新版要求拒绝。"]
    else:lines+=['待完成。']
    lines+=['','## 原片自身窗口组件对照','',
            '使用片头、中段、结尾各60秒，自身PTS作为时间真值；此项不证明B站版本与原片对应正确。文件身份及源特征准备时间单列，不计入组件耗时。',
            '', '| 方案 | 缓存状态 | 已完成窗口次数 | 中位秒 |','|---|---|---:|---:|']
    for variant in sorted({r['variant'] for r in windows}):
        for state in ('cold_features','warm_features'):
            group=[r for r in windows if r['variant']==variant and r['cache_state']==state]
            if group:lines.append(f"| {variant} | {state} | {len(group)}/9 | {statistics.median(r['elapsed_seconds'] for r in group):.3f} |")
    lines+=['','分窗口统计，每行均为三次重复；锚点数量不是弹幕可靠覆盖率。',
            '', '| 窗口 | 方案 | 缓存状态 | 中位秒 | 范围秒 | 锚点/源采样 |','|---|---|---|---:|---:|---:|']
    for row in window_summary:
        a,b=row['seconds_range']
        lines.append(f"| {row['window']} | {row['variant']} | {row['cache_state']} | {row['seconds_median']:.3f} | {a:.3f}–{b:.3f} | {row['anchors_median']:.0f}/{row['source_samples']} |")
    lines+=['','窗口统计口径：baseline 的局部粗特征不缓存；merged 缓存局部粗特征，精定位仍重新解码；dense_cache 同时缓存粗细特征。比较冷组评估合并后的组合开销，比较两种新版暖组评估精细缓存收益；不能将旧版暖组到 merged 暖组的全部降幅归因于区间合并。', '', '## 整片性能','',
            '整片每个方案、每种缓存状态运行一次；片段组件每组运行三次。整片的小幅耗时差异不表示统计显著提速。',
            '', '| 方案 | 缓存状态 | 耗时秒 | 实际视觉后端 | 可靠/估算/未映射 |','|---|---|---:|---|---|']
    for r in full:
        events=r['diagnostics'].get('performance',{}).get('decode_events',[])
        backends=', '.join(sorted({e.get('actual_backend','未验证') for e in events})) or '无视觉解码'
        rates=r['rates'];lines.append(f"| {r['variant']} | {r['cache_state']} | {r['elapsed_seconds']:.3f} | {backends} | {rates['reliable']*100:.2f}% / {rates['estimated']*100:.2f}% / {rates['unmapped']*100:.2f}% |")
    if any(r['variant']=='optimized_software' and r['cache_state']=='cold' for r in full):
        mapping=load(root/'full/playback-optimized_software.report.json',{})
        if mapping.get('entries'):
            from collections import Counter
            reasons=Counter(e['reason'] for e in mapping['entries'] if e['status']=='unmapped')
            lines+=['','新版软件严格模式未映射原因：'+ '；'.join(f'{k} {v}条' for k,v in reasons.most_common())+'。']
            if mapping['rates']['reliable']<.5:
                lines+=['','**这部真实影片的自动结果尚不足以作为完整可靠成品。高精度合成结果和缓存提速不能替代这部分验收；应重点复核音画冲突区间。**']
    cold=[r for r in full if r['cache_state']=='cold']
    if cold:
        lines+=['','| 方案 | 候选帧对比较数 | FFmpeg媒体进程数 | 请求视觉解码秒数 | 完整摘要逻辑读取 GiB |',
                '|---|---:|---:|---:|---:|']
        for r in cold:
            c=r['diagnostics'].get('performance',{}).get('counters',{})
            lines.append(f"| {r['variant']} | {c.get('candidate_pairs',0):.0f} | {c.get('ffmpeg_starts',0):.0f} | {c.get('decoded_requested_seconds',0):.2f} | {c.get('digest_logical_bytes',0)/1024**3:.3f} |")
        lines+=['','| 首次处理方案 | 进程树CPU秒 | 采样峰值RSS MiB | 特征及结果缓存 MiB | 采样临时空间峰值 MiB |',
                '|---|---:|---:|---:|---:|']
        for r in cold:
            guard=r.get('guard',{})
            mib=lambda x:'未测得' if x is None else f'{x/1024**2:.1f}'
            lines.append(f"| {r['variant']} | {r.get('cpu_seconds_self_and_children',0):.1f} | {mib(guard.get('process_tree_peak_rss_bytes_sampled'))} | {mib(r.get('cache_bytes'))} | {mib(guard.get('temporary_peak_bytes_sampled'))} |")
        lines+=['','整片总耗时体现组合优化。音频引导的收益通过候选数量、视觉请求范围及阶段耗时说明；请求秒数也受到窗口合并影响，不能单独换算成音频提速倍数。硬件倍数、缓存倍数与整片倍数不能相乘。']
    comparison=metrics.get('cold_software_comparison')
    if comparison:
        ratios=comparison['counter_ratios_new_over_old']
        lines+=['',f"软件首次处理耗时降低 {comparison['elapsed_reduction']*100:.2f}%；全部原弹幕的可靠覆盖率变化 {comparison['reliable_coverage_change_percentage_points']:+.2f} 个百分点。真实可对应区间尚未人工标注，此值不能替代该区间的覆盖率验收。",
            '',f"瓶颈仍在视觉解码链路：新版解码及特征计算包含计时约占总耗时 {comparison['visual_decode_inclusive_fraction_of_elapsed']*100:.1f}%。FFmpeg启动数降至旧版的 {ratios.get('ffmpeg_starts',0)*100:.1f}%，但实际选取帧数变为 {ratios.get('decoded_selected_frames',0)*100:.1f}%，请求解码时长变为 {ratios.get('decoded_requested_seconds',0)*100:.1f}%。这部影片音频支持稀疏，仍触发整片粗解码及大量精定位；分块和边缘扩展还会增加解码量，减少进程启动并不等于同等幅度提速。"]
    if not metrics['full_file_comparison_complete']:lines+=['','**整片受控对比尚未完成，不能宣称达到整片提速30%的目标。**']
    else:
        old=next(r for r in full if r['variant']=='baseline' and r['cache_state']=='cold')
        new=next(r for r in full if r['variant']=='optimized_software' and r['cache_state']=='cold')
        reduction=1-new['elapsed_seconds']/old['elapsed_seconds']
        lines+=['',f"整片首次处理软件方案耗时降低 {reduction*100:.2f}%，{'达到' if reduction>=.3 else '未达到'}30%目标。"]
    hardware_comparison=metrics.get('cold_hardware_comparison')
    if hardware_comparison:
        lines+=['',f"同一音频优先流程下，分输入硬件候选相对软件组快 {hardware_comparison['speedup_vs_same_strategy_software']:.2f} 倍；相对旧版首次处理耗时降低 {hardware_comparison['reduction_vs_baseline']*100:.2f}%。实际为参考片软件、原片VideoToolbox，硬件回退 {hardware_comparison['hardware_fallbacks']:.0f} 次，输出{'完全一致' if hardware_comparison['mapped_data_equal_to_software'] else '存在差异'}。硬件候选达到30%耗时目标，不代表默认软件流程已达到该目标或真实准确率已通过。"]
    lines+=['','## 限制与真实影片复核','',
            f"发布默认调度为 `{metrics['release_policy']['default_hybrid_strategy']}`，全局默认软件解码。整片表中的新版软件和硬件候选均显式启用了 `audio_first`；这部影片覆盖率下降，音频优先暂不作为默认，不能把该表说成默认流程的整片验收。缓存与解码调度优化对两种策略均有效。", '',
            '真实影片准确率等待人工标注。自动一致性、覆盖率和自对齐误差不能替代正确率。复核清单包含30个固定均匀位置及所有检测到的断点、冲突；图片仅提供少量示例。',
            '', f"[复核页面]({root/'review/review.html'}) · [机器可读复核清单]({root/'review/review.json'})。共30组对照截图；所有位置的 human_verified 仍为 false。", 
            '', '应用冷缓存不代表物理磁盘冷缓存；未清理系统缓存。CPU计时与按秒采样的进程树内存见JSON；更短的内存峰值可能漏采。阶段计时存在嵌套，不能直接相加。',
            '', '计时源码已分别冻结：旧版baseline、新版软件pre-hardware-select、硬件timed-hardware-source。最终版另修补损坏压缩缓存和错误身份JSON的恢复，正常输入的匹配流程未改变；后端拆分后的软件整片暖缓存输出也已单独对照。具体摘要见JSON。',
            '', '诊断中的字节数是逻辑读取量，未测物理磁盘I/O。默认仍使用软件解码，硬件需显式选择。']
    for name,comparison in metrics['mapping_agreement'].items():
        if comparison and 'error' not in comparison:
            pct=comparison['agreement_within_200ms']
            agreement='无共同映射' if pct is None else f"{pct*100:.2f}%"
            lines+=['',f"{name}：共同映射 {comparison['common_mapped']} 条，其中差异不超过200ms为 {agreement}；绝对时间差P95为 {comparison['difference_p95_seconds']} 秒。右侧方案新增映射 {comparison['new_only']} 条，左侧有而右侧未映射 {comparison['old_only']} 条。这是映射一致性，不是正确率。"]
    if default_validation:
        counters=default_validation['diagnostics']['performance']['counters']
        lines+=['',f"默认视觉优先混合流程只进行暖特征验证：{default_validation['elapsed_seconds']:.3f} 秒，可靠覆盖 {default_validation['rates']['reliable']*100:.2f}%，FFmpeg启动 {counters.get('ffmpeg_starts',0):.0f} 次。此项不作为默认流程首次处理耗时。", '',
            f"默认流程验证弹幕：[{Path(default_validation['playback_json']).name}]({default_validation['playback_json']})。原任务结果未修改。"]
    imported=metrics.get('feature_cache_import')
    if imported:
        lines+=['',f"有效特征缓存复用：状态 {imported['status']}，向默认缓存目录新增 {imported['added_files']} 个缺失文件（{imported['added_bytes']/1024**2:.1f} MiB），{imported['existing_unchanged']} 个已有条目保持原样。未导入实验对齐结果，未替换任务产物。服务未重启，尚未在服务内重试任务。"]
    lines+=['','缓存及临时空间按文件长度统计，稀疏测试文件可能大于实际磁盘占用；未采集物理I/O。']
    space=metrics['space_accounting']
    lines+=['',f"实验目录采样峰值 {space['experiment_peak_bytes_sampled']/1024**3:.2f} GiB；最终实验目录与新增共享特征合计 {(space['final_experiment_bytes']+space['promoted_feature_bytes_added'])/1024**3:.2f} GiB。预算4 GiB；报告生成时可用空间 {space['free_disk_bytes_at_report']/1024**3:.2f} GiB。进程内存统计为RSS，不代表额外内核/驱动内存。"]
    for row in full:
        if row['variant']=='optimized_software' and row['cache_state']=='cold' and row.get('playback_json'):
            lines+=['',f"新版软件流程的映射弹幕另存于 [{Path(row['playback_json']).name}]({row['playback_json']})，可在播放器手动导入；未覆盖或登记为原任务结果。"]
    (output/'optimization-acceptance.md').write_text('\n'.join(lines)+'\n')
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();generate(a.experiment,a.output_dir)

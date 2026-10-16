"""Local-only review exporter. Images and annotations are never uploaded."""
import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def export(review_path,alignment_path,images=0):
    review_path=Path(review_path);data=json.loads(review_path.read_text());raw=json.loads(Path(alignment_path).read_text());alignment=raw.get('alignment',raw)
    images_data=[];folder=review_path.with_suffix('');folder.mkdir(exist_ok=True)
    if images:
        origin={}
        for role in ('source','target'):
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=start_time','-of','json',alignment[role]]))
            origin[role]=float(probe.get('format',{}).get('start_time',0))
        selected=[]
        for stratum in ('newly_accepted','changed','withdrawn','still_unmapped','common'):
            candidates=sorted((r for r in data['records'] if r['stratum']==stratum),key=lambda r:(r.get('priority')!='critical_change',r['source_time']))
            for row in candidates:
                if all(abs(row['source_time']-r['source_time'])>30 for r in selected):selected.append(row);break
        for row in selected[:images]:
            item={'source_index':row['source_index'],'files':{}}
            for role,input_role,stamp in [('source','source',row['source_time']),('target','target',row.get('candidate_target_time')),('baseline_target','target',row.get('baseline_target_time') if row.get('candidate_target_time')!=row.get('baseline_target_time') else None)]:
                if stamp is None:continue
                dest=folder/(str(row['source_index'])+'-'+role+'.png')
                if dest.exists():raise FileExistsError(dest)
                p=subprocess.run(['ffmpeg','-v','info','-nostdin','-copyts','-ss',str(stamp),'-i',alignment[input_role],'-map','0:v:0','-an','-sn','-dn','-frames:v','1','-vf','scale=640:-2,showinfo',str(dest)],capture_output=True,text=True)
                if p.returncode:item['files'][role]={'error':'decode_failed'};continue
                match=re.search(r'\bn:\s*0\s+.*?\bpts_time:([^\s]+)',p.stderr)
                item['files'][role]={'file':folder.name+'/'+dest.name,'requested_seconds':stamp,
                                     'actual_seconds':float(match[1])-origin[input_role] if match else None}
            images_data.append(item)
    data['preview_images']=images_data
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>覆盖率恢复复核</title><style>
body{font:15px system-ui;margin:24px;background:#151a23;color:#e6edf3}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #394354;text-align:left}input{width:100px}button,select{padding:8px;margin:8px}img{max-width:46%;margin:1%}.validation{color:#9bd5ff}.pending{color:#ffd48d}summary{cursor:pointer}
</style><h1>覆盖率恢复 · 人工复核</h1><p>新增接受仍是候选结果。请结合连续音画确认位置与编辑边界，单张截图不能证明 200 ms 精度。记录分为 discovery 和 validation；请只使用 discovery 调参，最后用 validation 验收。</p>
<p>时间均为播放器秒数。确认前请在双视频复核播放器中定位；所有标注只保存在下载的本地 JSON 中。</p><select id="filter"><option value="">全部类别</option></select><button id="download">导出人工标注 JSON</button><label>导入已有标注 <input id="load" type="file" accept=".json"></label><p id="status"></p><div id="images"></div><table><thead><tr><th>源时间</th><th>旧目标 / 新目标</th><th>类别 / 分组</th><th>正确目标秒数</th><th>确认无对应</th><th>人工确认</th></tr></thead><tbody id="rows"></tbody></table>
<script id="data" type="application/json">PAYLOAD</script><script>
let data=JSON.parse(document.getElementById('data').textContent);const el=id=>document.getElementById(id);
function render(){el('rows').replaceChildren();for(const r of data.records){if(el('filter').value&&r.stratum!==el('filter').value)continue;const tr=document.createElement('tr');tr.className=r.split;for(const text of [r.source_time.toFixed(3),`${r.baseline_target_time??'—'} / ${r.candidate_target_time??'—'}`,`${r.stratum} / ${r.split}`]){const td=document.createElement('td');td.textContent=text;tr.append(td);}const target=document.createElement('input');target.type='number';target.min='0';target.step='.001';target.value=r.truth_target_time??'';target.oninput=()=>{r.truth_target_time=target.value===''?null:Number(target.value);r.confirmed_no_correspondence=false;absent.checked=false;r.human_verified=false;checked.checked=false;};const absent=document.createElement('input');absent.type='checkbox';absent.checked=r.confirmed_no_correspondence;absent.onchange=()=>{r.confirmed_no_correspondence=absent.checked;r.human_verified=false;checked.checked=false;if(absent.checked){r.truth_target_time=null;target.value='';}};const checked=document.createElement('input');checked.type='checkbox';checked.checked=r.human_verified;checked.onchange=()=>{if(checked.checked&&r.truth_target_time===null&&!r.confirmed_no_correspondence){checked.checked=false;el('status').textContent='请填写正确目标时间，或确认没有对应片段。';return;}r.human_verified=checked.checked;};for(const node of [target,absent,checked]){const td=document.createElement('td');td.append(node);tr.append(td);}el('rows').append(tr);}el('status').textContent=`已确认 ${data.records.filter(r=>r.human_verified).length} / ${data.records.length}，尚未确认的行不参与准确性计算。`;}
for(const name of [...new Set(data.records.map(r=>r.stratum))]){const o=document.createElement('option');o.value=name;o.textContent=name;el('filter').append(o);}el('filter').onchange=render;
el('download').onclick=()=>{const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='coverage-annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
el('load').onchange=async()=>{try{const imported=JSON.parse(await el('load').files[0].text());const map=new Map(imported.records.map(r=>[r.source_index,r]));for(const r of data.records){const old=map.get(r.source_index);if(old&&old.source_time===r.source_time){r.truth_target_time=old.truth_target_time;r.confirmed_no_correspondence=old.confirmed_no_correspondence;r.human_verified=old.human_verified;}}render();}catch(e){el('status').textContent='无法读取标注 JSON。';}};
for(const item of data.preview_images||[]){const block=document.createElement('details'),summary=document.createElement('summary');summary.textContent='预览：弹幕 '+item.source_index;block.append(summary);for(const [role,value]of Object.entries(item.files)){if(!value.file)continue;const img=document.createElement('img');img.src=value.file;img.alt=`${role} 实际 PTS ${value.actual_seconds}`;block.append(img);}el('images').append(block);}render();
</script></html>'''
    dest=review_path.with_suffix('.html');dest.write_text(template.replace('PAYLOAD',payload))
    review_path.write_text(json.dumps(data,ensure_ascii=False,indent=2));return dest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--alignment',required=True);p.add_argument('--images',type=int,default=0);a=p.parse_args();print(export(a.review,a.alignment,a.images))

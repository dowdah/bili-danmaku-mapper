/* Optional workbench integration. Standalone player remains file based. */
(async()=>{
 const key=new URLSearchParams(location.search).get('job');if(!key)return;
 const ask=async text=>{const dialog=document.createElement('dialog'),p=document.createElement('p'),yes=document.createElement('button'),no=document.createElement('button');p.textContent=text;yes.textContent='进入复核';no.textContent='取消';dialog.append(p,yes,no);document.body.append(dialog);return new Promise(resolve=>{dialog.addEventListener('close',()=>{const approved=dialog.returnValue==='yes';dialog.remove();resolve(approved);},{once:true});yes.onclick=()=>dialog.close('yes');no.onclick=()=>dialog.close('no');dialog.showModal();});};
 const request=async(path,body)=>{const response=await fetch('/api/'+path,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:'请求失败');return value;};
 try{
  if(!await ask('高级映射复核会展示对照影片、未观看剧情和弹幕。日常观影无需进入。确认继续？')){message('已取消复核。请返回工作台继续观看。');return;}
  const job=await request('jobs/'+key);if(job.status!=='complete')throw Error('本集尚未处理完成，请返回工作台查看');
  document.querySelector('.files').hidden=true;document.querySelector('header p').textContent=job.source.title+' · 对照剧情，检查映射，直接保存人工修正。';
  const file=name=>'/api/files/'+job.result.files[name];
  const [dms,raw,data]=await Promise.all(['danmaku','original','report'].map(async name=>{const response=await fetch(file(name));if(!response.ok)throw Error('任务产物缺失');return response.json();}));
  mapped=comments(dms);original=comments(raw);alignment=data.alignment;report=data;anchors=job.anchors||[];
  sourceVideo.src=file('source');targetVideo.src=file('target');sourceVideo.load();targetVideo.load();refreshItems();renderReview();renderAnchors();
  const save=document.createElement('button');save.textContent='保存锚点并重新对齐';save.onclick=async()=>{try{validateAnchors(anchors);save.disabled=true;const revision=await request('revise/'+key,{anchors});message('新结果版本已开始计算，旧结果保留。返回工作台查看进度。');save.textContent='已创建新版本';}catch(e){message(e.message,true);save.disabled=false;}};document.querySelector('.toolbar').append(save);
  const repair=document.createElement('a');repair.href='/?tab=player&item='+encodeURIComponent('job:'+key);repair.textContent='在正片播放器中修复播放';document.querySelector('.toolbar').append(repair);
  byId('addAnchor').onclick=()=>{try{const next=[...anchors,{source_time:Number(sourceVideo.currentTime.toFixed(6)),target_time:Number(targetVideo.currentTime.toFixed(6))}].sort((a,b)=>a.source_time-b.source_time);validateAnchors(next);anchors=next;renderAnchors();message('锚点已记录，点击「保存锚点并重新对齐」应用。');}catch(e){message(e.message,true);}};
  const back=document.createElement('a');back.href='/';back.textContent='← 返回工作台';document.querySelector('header').prepend(back);document.querySelector('footer').textContent='处理完成不表示全部弹幕可靠。不确定弹幕保留在待复核结果中；人工修改会生成新版本。';message('影片、弹幕和报告已自动加载。');
 }catch(e){message(e.message,true);}
})();

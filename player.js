/* Native HTML video + local danmaku renderer; no CDN or account access. */
"use strict";
const byId = id => document.getElementById(id);
const sourceVideo = byId("sourceVideo"), targetVideo = byId("targetVideo");
let alignment = null, report = null, mapped = [], original = [], anchors = [], urls = {};
const renderer = new DanmakuRenderer(targetVideo, byId("danmakuLayer"), {fontSize:20});
const reasonNames = {no_verified_segment:"无可靠片段",edit_or_unverified_interval:"编辑边界或未确认区间",gap_exceeds_strict_limit:"超过严格模式插值间隔",long_gap_interpolation:"较长间隙估算",outside_source_duration:"超出源视频范围",audio_visual_conflict:"音频与画面对应关系冲突"};
function message(text, error=false){byId("message").textContent=text;byId("message").classList.toggle("error",error);}
function resetComments(){renderer.reset();}
function refreshItems(){renderer.setItems(mapped, report);}
function targetTime(sourceTime){
  if(!alignment)return null;
  for(const segment of alignment.segments){
    if(sourceTime<segment.source_start||sourceTime>segment.source_end)continue;
    const a=segment.anchors;
    for(let i=0;i<a.length;i++){
      if(Math.abs(sourceTime-a[i].source_time)<1e-6)return a[i].target_time;
      if(i&&sourceTime<a[i].source_time){const before=a[i-1],after=a[i];return before.target_time+(sourceTime-before.source_time)/(after.source_time-before.source_time)*(after.target_time-before.target_time);}
    }
  }
  return null;
}
function jump(){const time=targetTime(sourceVideo.currentTime);if(time===null){targetVideo.pause();message("当前源时间没有已确认的映射，目标视频已暂停。");return false;}targetVideo.currentTime=time;return true;}
let previousSegment = null;
function tick(){
  byId("sourceTime").textContent=`${sourceVideo.currentTime.toFixed(3)} s`;byId("targetTime").textContent=`${targetVideo.currentTime.toFixed(3)} s`;
  if(byId("sync").checked&&!sourceVideo.paused){
    const t=targetTime(sourceVideo.currentTime);
    const segment=alignment?.segments.find(s=>sourceVideo.currentTime>=s.source_start&&sourceVideo.currentTime<=s.source_end);
    if(t===null){targetVideo.pause();previousSegment=null;}
    else {if(segment!==previousSegment||Math.abs(targetVideo.currentTime-t)>.12)targetVideo.currentTime=t;
      targetVideo.playbackRate=Math.min(4,Math.max(.25,sourceVideo.playbackRate*segment.slope));
      if(targetVideo.paused)targetVideo.play().catch(()=>{});previousSegment=segment;}
  }
  requestAnimationFrame(tick);
}

function validateAnchors(values){
  if(!Array.isArray(values))throw Error("锚点必须是 JSON 数组");
  let s=-1,t=-1;
  for(const a of values){if(!Number.isFinite(a.source_time)||!Number.isFinite(a.target_time)||a.source_time<0||a.target_time<0||a.source_time<=s||a.target_time<=t)throw Error("锚点必须是非负秒数，并在两边共同递增");
    if((Number.isFinite(sourceVideo.duration)&&a.source_time>sourceVideo.duration)||(Number.isFinite(targetVideo.duration)&&a.target_time>targetVideo.duration))throw Error("锚点超出视频时长");s=a.source_time;t=a.target_time;}
}
function renderAnchors(){
  byId("anchors").replaceChildren();anchors.forEach((a,index)=>{const li=document.createElement("li");li.textContent=`${a.source_time.toFixed(3)} s → ${a.target_time.toFixed(3)} s`;const del=document.createElement("button");del.textContent="删除";del.onclick=()=>{anchors.splice(index,1);renderAnchors();};li.append(del);byId("anchors").append(li);});
}
function renderReview(){
  byId("uncertain").replaceChildren();
  const entries=report?.entries?.filter(e=>e.status!=="reliable")||[];
  byId("summary").textContent=report?.counts?`共 ${report.counts.total} 条：可靠 ${report.counts.reliable}，估算 ${report.counts.estimated}，未映射 ${report.counts.unmapped}`:"已加载对齐结果；弹幕报告包含逐条复核信息。";
  const pageSize=200;let rendered=0;
  function appendPage(){for(const entry of entries.slice(rendered,rendered+pageSize)){const button=document.createElement("button");const text=original[entry.source_index]?.text||"";button.textContent=`${entry.source_time.toFixed(3)} s · ${reasonNames[entry.reason]||entry.reason} ${text}`;button.onclick=()=>{sourceVideo.currentTime=entry.source_time;if(entry.target_time!==null)targetVideo.currentTime=entry.target_time;else jump();};byId("uncertain").append(button);}rendered+=pageSize;}
  appendPage();if(entries.length>pageSize){const more=document.createElement("button");more.textContent="加载更多";more.onclick=()=>{more.remove();appendPage();if(rendered<entries.length)byId("uncertain").append(more);};byId("uncertain").append(more);}
}
async function readFile(input, callback){const file=input.files[0];if(!file)return;try{callback(JSON.parse(await file.text()));message(`已加载 ${file.name}`);}catch(error){message(error.message,true);}}
for(const [input,video]of [["sourceInput",sourceVideo],["videoInput",targetVideo]])byId(input).onchange=()=>{const file=byId(input).files[0];if(!file)return;if(urls[input])URL.revokeObjectURL(urls[input]);urls[input]=URL.createObjectURL(file);video.src=urls[input];video.load();message(`已加载 ${file.name}`);};
function comments(data){return DanmakuRenderer.parse(data);}
byId("danmakuInput").onchange=()=>readFile(byId("danmakuInput"),data=>{mapped=comments(data);refreshItems();});
byId("originalInput").onchange=()=>readFile(byId("originalInput"),data=>{original=comments(data);if(report)renderReview();});
byId("reportInput").onchange=()=>readFile(byId("reportInput"),data=>{const candidate=data.alignment||data;if(candidate.schema_version!==1||!Array.isArray(candidate.segments))throw Error("不支持的报告版本");
  let lastS=-1,lastT=-1;for(const s of candidate.segments){if(!Array.isArray(s.anchors)||!s.anchors.length||!Number.isFinite(s.slope)||s.slope<=0)throw Error("报告片段无效");for(const a of s.anchors){if(!Number.isFinite(a.source_time)||!Number.isFinite(a.target_time)||a.source_time<=lastS||a.target_time<=lastT)throw Error("报告锚点顺序无效");lastS=a.source_time;lastT=a.target_time;}}
  if(data.entries&&!Array.isArray(data.entries))throw Error("报告 entries 必须为数组");alignment=candidate;report=data.entries?data:null;refreshItems();renderReview();});
byId("anchorsInput").onchange=()=>readFile(byId("anchorsInput"),values=>{validateAnchors(values);anchors=values.map(a=>({source_time:a.source_time,target_time:a.target_time}));renderAnchors();});
byId("jump").onclick=jump;
byId("addAnchor").onclick=()=>{try{if(!sourceVideo.src||!targetVideo.src)throw Error("请先加载两个视频");const next=[...anchors,{source_time:Number(sourceVideo.currentTime.toFixed(6)),target_time:Number(targetVideo.currentTime.toFixed(6))}].sort((a,b)=>a.source_time-b.source_time);validateAnchors(next);anchors=next;renderAnchors();message("锚点已记录，导出后可重新计算映射。");}catch(error){message(error.message,true);}};
byId("exportAnchors").onclick=()=>{try{validateAnchors(anchors);const url=URL.createObjectURL(new Blob([JSON.stringify(anchors,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="anchors.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){message(error.message,true);}};
byId("sync").onchange=()=>{if(byId("sync").checked){if(!alignment){byId("sync").checked=false;message("请先加载映射报告。",true);return;}jump();}else {targetVideo.pause();targetVideo.playbackRate=1;}};
byId("showEstimated").onchange=()=>renderer.configure({showEstimated:byId("showEstimated").checked});
sourceVideo.addEventListener("seeking",()=>{if(byId("sync").checked)jump();});
sourceVideo.addEventListener("pause",()=>{if(byId("sync").checked)targetVideo.pause();});
targetVideo.addEventListener("seeking",resetComments);
for(const video of [sourceVideo,targetVideo])video.addEventListener("error",()=>message("浏览器无法播放这个视频，请转换为 H.264/AAC MP4。",true));
window.addEventListener("beforeunload",()=>{for(const url of Object.values(urls))URL.revokeObjectURL(url);});
requestAnimationFrame(tick);

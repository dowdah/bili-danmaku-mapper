import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{id,value:'',checked:true,disabled:false,hidden:false,children:[],style:{},dataset:{},listeners:new Map(),classList:{toggle(){}},textContent:'',append(node){this.children.push(node);},replaceChildren(){this.children=[];},setAttribute(k,v){this[k]=v;},addEventListener(k,fn){this.listeners.set(k,fn);},emit(k){this.listeners.get(k)?.();}});return elements.get(id);}
for(const [id,value] of [['watchFont','24'],['watchOpacity','1'],['watchRate','1'],['watchVolume','1']])element(id).value=value;
const video=element('watchVideo');Object.assign(video,{currentTime:0,duration:20,volume:1,muted:false,playbackRate:1,paused:true,ended:false,plays:0,
 pause(){const changed=!this.paused;this.paused=true;if(changed)this.emit('pause');},play(){this.plays++;this.paused=false;this.emit('play');return Promise.resolve();},load(){this.emit('loadedmetadata');}});
const defaults={position:0,volume:1,muted:false,playback_rate:1,danmaku_enabled:true,font_size:24,opacity:1,show_estimated:true,subtitle_id:null,subtitle_enabled:true,subtitle_offset:0};
let replyAudioCopy=false,replyHasAudio=true;
let listing=['a','b'].map((id,i)=>({id:'manual:'+id,kind:'manual',video_id:id,danmaku_id:id+'json',title:id,available:true,state:{...defaults,position:i?3:1.2,playback_rate:1.5}}));
const saves=[];let delayed=null;let entered=null;let lastItem=null;let preparationReplies=[],preparationCalls=[],allowTranscode=false,pendingPrepare=null;
async function fetchMock(path,options={}){
 if(path.includes('/prepared?'))return {ok:true,json:async()=>[]};
 if(path.endsWith('/media'))return {ok:true,json:async()=>({audio_streams:[{ordinal:0,codec:'aac',language:'jpn'},{ordinal:1,codec:'aac',language:'eng'}]})};
 if(path==='/api/playback/sessions'){if(pendingPrepare)await pendingPrepare;preparationCalls.push(JSON.parse(options.body));return {ok:true,json:async()=>({id:'session',mime:replyHasAudio?'video/mp4; codecs="avc1.640028,mp4a.40.2"':'video/mp4; codecs="avc1.640028"',duration:20,audio_copy:replyAudioCopy})};}
 if(path.endsWith('/prepare')){preparationCalls.push(JSON.parse(options.body));if(pendingPrepare)await pendingPrepare;const result=preparationReplies.shift();assert.ok(result,'unexpected preparation');return {ok:true,json:async()=>result};}
 if(path.endsWith('/state')){saves.push({path,body:JSON.parse(options.body)});return {ok:true,json:async()=>saves.at(-1).body};}
 if(path==='/api/playback/items')return {ok:true,json:async()=>({items:structuredClone(listing),last_item_id:lastItem})};
 if(path.endsWith('/subtitles/invalid'))return {ok:false,json:async()=>({detail:'无效字幕'})};
 if(path.includes('/subtitles/'))return {ok:true,json:async()=>({cues:[{start:1,end:3,text:'subtitle'}],format:'SRT',note:''})};
 if(path.endsWith('missing'))return {ok:false,json:async()=>({})};
 if(delayed&&path.endsWith('ajson')){entered();await delayed;}
 return {ok:true,json:async()=>({data:[{time:1,text:path}]})};
}
class Renderer{constructor(){this.data=[];}configure(s){this.settings=s;}setItems(v){this.data=v;}static parse(v){return v.data;}}
const document={getElementById:element,createElement:tag=>element('created-'+Math.random()),fullscreenElement:null,async exitFullscreen(){this.fullscreenElement=null;}};
element('watchStage').requestFullscreen=async()=>{document.fullscreenElement=element('watchStage');};
class Player{constructor(options){this.opts=options;this.video=options.video;this.el=element('nplayer');this.webFullscreen={toggle:()=>{this.full=!this.full;}};}mount(){}getControlItem(){}getSettingItem(){return this.opts.settings[0];}}
class Captions{constructor(){this.cues=[];this.offset=0;}setCues(cues){this.cues=cues;}configure({enabled,offset}){this.enabled=enabled;this.offset=Math.round(Math.max(-5,Math.min(5,Number(offset)||0))*10)/10;}}
class Transport{constructor(video){this.video=video;this.session=null;}async close(){this.session=null;}async open(s,pos){this.session=s;video.src='blob:session';video.currentTime=pos;video.load();}}
const context=vm.createContext({SegmentPlayer:Transport,document,window:{addEventListener(){}},NPlayer:{Player},SubtitleRenderer:Captions,DanmakuRenderer:Renderer,fetch:fetchMock,confirmAction:async()=>allowTranscode,setTimeout:fn=>setTimeout(fn,0),setInterval:()=>1,clearInterval(){},URL,Date,console});
vm.runInContext(fs.readFileSync(new URL('../web/watch.js',import.meta.url),'utf8'),context);
const p=vm.runInContext('finishedPlayer',context);
await p.refresh();await p.open('manual:a');assert.equal(video.currentTime,1.2);assert.equal(video.paused,true);assert.equal(video.playbackRate,1.5);assert.equal(p.speedSetting.value,1.5);assert.equal(p.player.video,video);assert.ok(p.player.el.children.includes(element('watchSubtitles')));
video.currentTime=8;video.paused=false;await p.pauseAndSave();assert.equal(video.paused,true);
await p.open('manual:b');assert.equal(video.currentTime,3);
await p.open('manual:a');assert.equal(video.currentTime,8,'switching back must use the just-saved in-memory position');
assert.ok(saves.some(s=>s.body.position===8));assert.equal(video.plays,0,'loading and restoring must never autoplay');
video.paused=false;const beforeSrc=video.src;
listing.push({id:'job:ready',kind:'job',title:'new ready task',video_id:'c',danmaku_id:'cjson',available:true,state:{...defaults}});
await p.refresh();assert.equal(p.current.id,'manual:a');assert.equal(video.src,beforeSrc);assert.equal(video.currentTime,8);assert.equal(video.paused,false,'a completion refresh must not interrupt watching');
assert.ok(p.player.el.children.includes(element('watchDanmaku')),'overlays share NPlayer fullscreen root');
p.subtitleId='srt1';p.subtitles.setCues([{start:1,end:3,text:'subtitle'}]);element('subtitleOffset').value='-2.3';p.configureSubtitle();await p.persist();assert.equal(saves.at(-1).body.subtitle_offset,-2.3);assert.equal(saves.at(-1).body.subtitle_id,'srt1');
await p.open('manual:b');assert.equal(p.subtitles.cues.length,0,'switch clears previous captions');assert.equal(p.subtitleId,null);
await p.open('manual:a');assert.equal(p.subtitles.offset,-2.3);assert.equal(p.subtitleId,'srt1');assert.equal(p.subtitles.cues.length,1,'return restores caption id and offset');
p.pick=async()=>({id:'srt2',name:'第二字幕.srt'});await p.importSubtitle();assert.equal(p.subtitleId,'srt2');assert.equal(p.subtitles.offset,0);const previousSubtitles=p.subtitles.cues;p.pick=async()=>({id:'invalid'});await assert.rejects(p.importSubtitle());assert.equal(p.subtitleId,'srt2');assert.equal(p.subtitles.cues,previousSubtitles,'invalid replacement preserves attached subtitles');
element('subtitleOffset').value='-2.3';p.configureSubtitle();p.adjustSubtitle(0.1);assert.equal(p.subtitles.offset,-2.2);p.removeSubtitle();assert.equal(p.subtitleId,null);assert.equal(p.subtitles.cues.length,0);assert.equal(p.subtitles.offset,0);
await p.open('manual:b');let release;
delayed=new Promise(resolve=>release=resolve);const ready=new Promise(resolve=>entered=resolve);
const slow=p.open('manual:a');await ready;await p.open('manual:b');release();await slow;
assert.equal(p.current.id,'manual:b','late media requests must not replace the latest selection');
// Playback errors use bounded sessions, never the whole-file preparation API.
video.currentTime=7;const overlay=p.renderer.data;
await p.repairPlayback();assert.equal(video.src,'blob:session');assert.equal(p.mediaClock.currentTime,7);assert.equal(p.mediaStage,'segments');assert.equal(p.renderer.data,overlay);
assert.ok(preparationCalls.every(call=>!call.strategy),'watching must not schedule a remux or transcode copy');
video.error={code:4};await p.mediaError();assert.match(element('playbackStatus').textContent,/更换支持/);
p.retryOriginal();assert.equal(video.src,'/api/files/b');assert.equal(p.mediaStage,'original');
const repairMethod=p.repairPlayback;let missingVideoRepairs=0;p.repairPlayback=async()=>{missingVideoRepairs++;};p.mediaInfo.video_codec='ffv1';video.videoWidth=0;video.videoHeight=0;video.emit('loadeddata');await Promise.resolve();assert.equal(missingVideoRepairs,1,'audio-only decoding must not silently count as a playable movie');p.repairPlayback=repairMethod;video.videoWidth=256;video.videoHeight=144;
const count=preparationCalls.length;video.error={code:2};await p.mediaError();assert.equal(preparationCalls.length,count,'network errors must not trigger conversion');
let finishPrepare;pendingPrepare=new Promise(resolve=>finishPrepare=resolve);preparationReplies.push({status:'complete',timeline_verified:true,file_id:'late-copy'});
const late=p.repairPlayback();await new Promise(resolve=>setTimeout(resolve,0));await p.open('manual:a');const newSrc=video.src;finishPrepare();await late;pendingPrepare=null;assert.equal(video.src,newSrc,'late preparation must not replace another movie');assert.equal(p.current.id,'manual:a');
await p.open('manual:b');video.error=null;
const previousComments=p.renderer.data;
p.items.push({id:'manual:missing',kind:'manual',title:'missing',available:true,video_id:'bad',danmaku_id:'missing',state:{...defaults}});
await assert.rejects(p.open('manual:missing'));assert.equal(p.current.id,'manual:b');assert.equal(p.renderer.data,previousComments,'failed media loads must preserve the previous comments');
listing.push({id:'manual:missing',kind:'manual',title:'missing',available:true,video_id:'bad',danmaku_id:'missing',state:{...defaults}});
lastItem='manual:missing';p.sequence=0;await p.init();assert.match(element('watchMessage').textContent,/无法恢复上次观看/);
video.emit('error');assert.match(element('watchMessage').textContent,/无法读取或解码/);assert.equal(p.loaded,false);
await p.pauseAndSave();p.stop();assert.equal(video.paused,true);
element('danmakuOffset').value='1.2';p.configure();assert.equal(p.renderer.settings.offset,1.2);p.adjustDanmaku(.1);assert.equal(Number(element('danmakuOffset').value),1.3);p.adjustDanmaku(0,true);assert.equal(Number(element('danmakuOffset').value),0);
console.log('Finished viewer: restore, switch persistence, completion isolation, fullscreen target and load races passed');

// Native nonzero origins and MSE windows share the normalized media clock.
p.stopped=false;p.mediaInfo.origin=1.5;p.loadPlayback('b','original',0);assert.equal(video.currentTime,1.5);assert.equal(p.player.currentTime,0);assert.equal(p.player.duration,18.5);p.player.currentTime=4;assert.equal(video.currentTime,5.5);assert.equal(p.mediaClock.currentTime,4);

// Audio preflight: unsupported DTS must switch without an error or a manual click.
const Viewer=p.constructor;
video.canPlayType=type=>type.includes('mp4a.40.2')?'probably':'';
assert.equal(Viewer.audioSupport(video,{codec:'dts'}),'unsupported');
assert.equal(Viewer.audioSupport(video,{codec:'aac'}),'supported');
assert.equal(Viewer.audioSupport(video,{codec:'unknown'}),'unknown');
assert.equal(Viewer.audioSupport(video,null),'unknown');
p.stopped=false;p.autoAudioBypass=false;p.restoreAudio=false;p.tried=new Set();p.mediaInfo={origin:0,audio_streams:[{ordinal:0,codec:'dts'}]};element('watchAudio').value='0';
const autoBefore=preparationCalls.length;video.error=null;p.loadPlayback('b','original',4);
await new Promise(resolve=>setTimeout(resolve,0));await p.saveChain;
assert.equal(preparationCalls.length,autoBefore+1);assert.equal(preparationCalls.at(-1).audio_copy,false);
assert.equal(p.mediaStage,'segments');assert.equal(video.src,'blob:session');assert.equal(video.currentTime,4);assert.equal(video.paused,true,'automatic fallback cannot autoplay');
video.emit('loadedmetadata');await new Promise(resolve=>setTimeout(resolve,0));assert.equal(preparationCalls.length,autoBefore+1,'segmented metadata cannot recursively trigger repair');
p.retryOriginal();await new Promise(resolve=>setTimeout(resolve,0));assert.equal(video.src,'/api/files/b','explicit retry can bypass conservative detection');assert.equal(preparationCalls.length,autoBefore+1);
p.autoAudioBypass=false;p.tried=new Set();p.mediaInfo.audio_streams[0].codec='aac';p.loadPlayback('b','original',4);await new Promise(resolve=>setTimeout(resolve,0));assert.equal(video.src,'/api/files/b');assert.equal(preparationCalls.length,autoBefore+1,'supported audio stays direct');
p.mediaInfo.audio_streams[0].codec='unknown';p.loadPlayback('b','original',4);await new Promise(resolve=>setTimeout(resolve,0));assert.equal(preparationCalls.length,autoBefore+1,'missing capability knowledge must not force conversion');
p.tried=new Set();p.mediaInfo.audio_streams[0].codec='dts';p.loadPlayback('b','original',4);p.sequence++;await new Promise(resolve=>setTimeout(resolve,0));assert.equal(preparationCalls.length,autoBefore+1,'stale audio preflight cannot affect a new selection');
console.log('Automatic audio: DTS fallback, AAC direct, unknown safety, explicit override and stale-selection isolation passed');

// Quality loss remains visible independently of the frequently refreshed status.
p.autoAudioBypass=true;p.mediaInfo.audio_streams=[{ordinal:0,codec:'dts'}];p.loadPlayback('b','original',4);await p.repairPlayback(true);
assert.equal(element('audioQualityNotice').hidden,false);assert.match(element('audioQualityNotice').textContent,/AAC 256 kb\/s 双声道（有损）/);assert.match(element('audioQualityNotice').textContent,/多声道环绕声/);
const qualityText=element('audioQualityNotice').textContent;p.playbackMessage('已缓存下一片段');video.emit('loadeddata');assert.equal(element('audioQualityNotice').textContent,qualityText,'progress and loadeddata cannot overwrite the quality warning');
p.retryOriginal();assert.equal(element('audioQualityNotice').hidden,true,'direct original has no conversion warning');
replyAudioCopy=true;await p.repairPlayback(true);assert.equal(element('audioQualityNotice').hidden,true,'copied AAC must not be called lossy re-encoding');
replyAudioCopy=false;replyHasAudio=false;p.mediaInfo.audio_streams=[];p.loadPlayback('b','original',4);await p.repairPlayback(true);assert.equal(element('audioQualityNotice').hidden,true,'video-only sessions have no audio quality loss');
p.loadPlayback('b','audio',4,0);assert.equal(element('audioQualityNotice').hidden,false);assert.doesNotMatch(element('audioQualityNotice').textContent,/256|双声道/,'old prepared audio files do not promise unverified encoding parameters');
p.loadPlayback('b','remux',4,0);assert.equal(element('audioQualityNotice').hidden,true);
p.loadPlayback('b','transcode',4,0);assert.equal(element('audioQualityNotice').hidden,false);assert.doesNotMatch(element('audioQualityNotice').textContent,/视频未重新编码/);
p.stop();assert.equal(element('audioQualityNotice').hidden,true);
console.log('Audio quality notice: converted, copied, absent audio, cached variants and persistent visibility passed');

// Native playback is lazy, requires an external subtitle, and owns the clock.
p.stopped=false;p.current={id:'manual:b',video_id:'b',state:{...defaults,position:5,subtitle_offset:0,danmaku_offset:0,volume:.5,muted:true}};p.subtitleId=null;p.nativeSession=null;p.nativeTimer=null;p.mediaInfo.audio_streams=[{ordinal:0,codec:'aac'}];
const requestBefore=p.request;let nativeCreates=0,pickedSubtitles=0;
p.importSubtitle=async()=>{pickedSubtitles++;p.subtitleId='native-ass';};
p.request=async(path,body,method)=>{
 if(path==='native/capabilities')return {available:true};
 if(path==='native/sessions'){nativeCreates++;assert.equal(body.subtitle_id,'native-ass');return {id:'native-1',status:'ready',subtitle_name:'原样.ass',comment_count:3,render_sync:{mode:'display-vdrop',active:true,paused:false,audio_speed_correction:1},source_audio:{profile:'DTS-HD MA',channel_layout:'5.1',sample_rate:48000},output_audio:{channels:'stereo',samplerate:48000},state:{...p.current.state,subtitle_id:'native-ass',preferred_engine:'iina'}};}
 if(path==='native/sessions/native-1'&&method==='DELETE')return {};
 if(path==='items')return {items:[{id:'manual:b',state:{...p.current.state,position:7.5,preferred_engine:'web'}}]};
 return requestBefore.call(p,path,body,method);
};
video.currentTime=5;await p.openNative();assert.equal(pickedSubtitles,1);assert.equal(nativeCreates,1);assert.equal(element('watchStage').hidden,true);assert.equal(video.paused,true);assert.match(element('nativeStatus').textContent,/5.1/);assert.match(element('subtitleInfo').textContent,/原样加载/);
assert.match(element('nativeStatus').textContent,/弹幕显示同步已启用/);
p.adoptNative({...p.nativeSession,render_sync:{mode:'display-vdrop',active:false,paused:true}});assert.match(element('nativeStatus').textContent,/播放后确认/);
p.adoptNative({...p.nativeSession,render_sync:{mode:'display-vdrop',active:false,paused:false}});assert.match(element('nativeStatus').textContent,/当前显示同步未启用/);
video.currentTime=77;assert.equal(p.snapshot().position,5,'the hidden webpage clock cannot overwrite IINA position');
p.removeSubtitle();assert.equal(p.subtitleId,'native-ass','native mode must keep the required subtitle');
await p.returnFromNative();assert.equal(p.nativeSession,null);assert.equal(element('watchStage').hidden,false);assert.equal(video.currentTime,7.5);assert.equal(video.paused,true);assert.equal(video.src,'/api/files/b');
p.request=requestBefore;
console.log('Native viewer: required subtitle, original-style message, clock ownership and paused web handoff passed');

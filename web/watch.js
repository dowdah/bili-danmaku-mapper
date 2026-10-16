/* Finished-media viewer. Playback preparation is independent of alignment. */
'use strict';
var FinishedPlayer = class {
  constructor() {
    this.byId=id=>document.getElementById(id);
    this.video=this.byId('watchVideo'); this.current=null; this.items=[]; this.sequence=0;
    this.loaded=false; this.restoring=false; this.stopped=false; this.saveChain=Promise.resolve(); this.refreshChain=Promise.resolve(); this.lastSave=0;this.writeRevision=0;
    // 1.0.15 seeks to 0.3s on canplay. Disable that workaround on desktop so
    // restoring/rewinding to zero keeps the exact media clock used by overlays.
    this.video.addEventListener('canplay',()=>{this.player.opts.autoSeekTime=-1;},true);
    // NPlayer measures its settings panel on mount; hidden tabs have zero size.
    const panel=this.byId('playerPanel'),wasHidden=panel.hidden;panel.hidden=false;
    this.speedSetting={id:'speed',html:'播放速度',type:'select',value:1,options:[0.25,0.5,0.75,1,1.25,1.5,2,3,4].map(value=>({value,html:value===1?'正常':value+'×'})),change(value,player){player.playbackRate=value;}};
    this.player=new NPlayer.Player({video:this.video,container:this.byId('watchStage'),i18n:'zh-cn',themeColor:'#93c9ad',posterEnable:false,autoSeekTime:-1,settings:[this.speedSetting],contextMenus:['version'],videoProps:{preload:'metadata',playsinline:'true'}});
    this.player.mount();panel.hidden=wasHidden;
    for(const [id,label] of [['play','播放'],['volume','静音'],['settings','播放设置'],['web-fullscreen','网页全屏'],['fullscreen','全屏']]){const control=this.player.getControlItem(id)?.el;if(!control)continue;control.setAttribute('role','button');control.setAttribute('aria-label',label);control.tabIndex=0;control.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();control.click();}});}
    for(const id of ['watchDanmaku','watchSubtitles','watchEmpty'])this.player.el.append(this.byId(id));
    this.clockOffset=0;const viewer=this;this.mediaClock={get currentTime(){return viewer.video.currentTime-viewer.clockOffset;},get paused(){return viewer.video.paused;},get ended(){return viewer.video.ended;},addEventListener(...args){viewer.video.addEventListener(...args);},removeEventListener(...args){viewer.video.removeEventListener(...args);}};
    Object.defineProperties(this.player,{
      currentTime:{get:()=>Math.max(0,this.mediaClock.currentTime),set:value=>{if(Number.isFinite(value))this.video.currentTime=Math.max(0,Math.min(this.video.duration,value+this.clockOffset));}},
      duration:{get:()=>Math.max(0,(this.video.duration||0)-this.clockOffset)},
      buffered:{get:()=>{const ranges=this.video.buffered,offset=this.clockOffset;return {get length(){return ranges?.length||0;},start:i=>Math.max(0,ranges.start(i)-offset),end:i=>Math.max(0,ranges.end(i)-offset)};}}
    });
    this.renderer=new DanmakuRenderer(this.mediaClock,this.byId('watchDanmaku'));
    this.subtitles=new SubtitleRenderer(this.mediaClock,this.byId('watchSubtitles'));
    this.transport=new SegmentPlayer(this.video,(...args)=>this.request(...args),(text,error)=>this.playbackMessage(text,error));
    this.subtitleId=null;this.mediaStage='original';this.tried=new Set();this.autoAudioBypass=false;this.repairBusy=false;
    this.byId('repairPlayback').onclick=()=>this.repairPlayback(true).catch(e=>this.playbackMessage(e.message,true));
    this.nativeSession=null;this.nativeBusy=false;
    this.byId('nativeOpen').onclick=()=>this.openNative().catch(e=>this.message(e.message,true));
    this.byId('nativeReturn').onclick=()=>this.returnFromNative().catch(e=>this.message(e.message,true));
    this.byId('retryOriginal').onclick=()=>this.retryOriginal();
    this.byId('continueWatch').onclick=()=>this.video.play().catch(e=>this.message(e.message,true));
    this.byId('danmakuOffset').oninput=()=>{this.configure();this.persist();};
    this.byId('danmakuEarlier').onclick=()=>this.adjustDanmaku(-.1);
    this.byId('danmakuLater').onclick=()=>this.adjustDanmaku(.1);
    this.byId('danmakuReset').onclick=()=>this.adjustDanmaku(0,true);
    this.byId('feedback').onclick=()=>this.feedback().catch(e=>this.message(e.message,true));
    this.byId('watchAudio').onchange=()=>{if(this.nativeSession){this.persist();return;}this.persist();this.repairPlayback(true).catch(e=>this.playbackMessage(e.message,true));};
    this.byId('importSubtitle').onclick=()=>this.importSubtitle().catch(e=>this.subtitleMessage(e.message,true));
    this.byId('removeSubtitle').onclick=()=>this.removeSubtitle();
    this.byId('subtitleEnabled').oninput=()=>{this.configureSubtitle();this.persist();};
    this.byId('subtitleOffset').oninput=()=>{this.configureSubtitle();this.persist();};
    this.byId('subtitleEarlier').onclick=()=>this.adjustSubtitle(-0.1);
    this.byId('subtitleLater').onclick=()=>this.adjustSubtitle(0.1);
    this.byId('subtitleReset').onclick=()=>this.adjustSubtitle(0,true);
    this.byId('importMovie').onclick=()=>this.importMovie().catch(e=>this.message(e.message,true));
    this.byId('importComments').onclick=()=>this.importComments().catch(e=>this.message(e.message,true));
    this.byId('refreshLibrary').onclick=()=>this.refresh().catch(e=>this.message(e.message,true));
    for(const id of ['watchComments','watchFont','watchOpacity','watchEstimated'])this.byId(id).oninput=()=>{this.configure();this.persist();};
    this.video.addEventListener('loadedmetadata',()=>this.metadata());
    for(const event of ['pause','ended','volumechange','ratechange','seeked'])this.video.addEventListener(event,()=>this.persist());
    this.video.addEventListener('ratechange',()=>this.syncSpeed());
    for(const event of ['play','pause'])this.video.addEventListener(event,()=>this.player.getControlItem('play')?.el?.setAttribute('aria-label',this.video.paused?'播放':'暂停'));
    this.video.addEventListener('timeupdate',()=>{if(Date.now()-this.lastSave>5000)this.persist();});
    this.video.addEventListener('error',()=>this.mediaError());
    this.video.addEventListener('loadeddata',()=>{if(this.nativeSession||this.mediaStage==='native_pending')return;this.recordPlaybackReady();this.playbackReady=this.video.videoWidth>0;this.byId('continueWatch').disabled=Boolean(this.nativeSession)||this.nativeBusy||!this.playbackReady;if(this.playbackReady)this.playbackMessage(this.mediaStage==='original'?'原片画面已就绪，声音与持续播放请在观看时确认。':'正在按需播放：视频未重新编码，弹幕和字幕沿用原片时间。');if(this.video.videoWidth===0&&this.video.videoHeight===0&&this.mediaInfo?.video_codec)this.mediaError(true);});
    this.video.addEventListener('loadstart',()=>{const metric=this.playMetrics;if(!metric)return;if(this.frameRequest!=null)this.video.cancelVideoFrameCallback?.(this.frameRequest);if(this.video.requestVideoFrameCallback)this.frameRequest=this.video.requestVideoFrameCallback(()=>{if(this.playMetrics===metric){metric.first_presented_ms=Date.now()-metric.started_ms;this.publishMetrics();}});});
    this.video.addEventListener('waiting',()=>{if(this.playMetrics){this.playMetrics.waiting_events++;this.publishMetrics();}});
    this.video.addEventListener('seeking',()=>{this.seekStarted=Date.now();});
    this.video.addEventListener('seeked',()=>{if(this.playMetrics&&this.seekStarted&&this.video.readyState>=2){this.playMetrics.last_seek_ready_ms=Date.now()-this.seekStarted;this.publishMetrics();}});
    window.addEventListener('pagehide',()=>{this.persist(true);this.transport.close();});
  }
  async request(path,body,method,keepalive=false){const response=await fetch('/api/playback/'+path,{method:method||(body===undefined?'GET':'POST'),headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),keepalive});const value=await response.json();if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:'播放器请求失败');return value;}
  beginPlaybackMetrics(transport){
    if(this.frameRequest!=null)this.video.cancelVideoFrameCallback?.(this.frameRequest);
    const metric=this.playMetrics={transport,started_ms:Date.now(),first_ready_ms:null,first_presented_ms:null,last_seek_ready_ms:null,waiting_events:0};

    this.publishMetrics();
  }
  recordPlaybackReady(){if(this.playMetrics&&this.video.videoWidth>0&&this.playMetrics.first_ready_ms==null){this.playMetrics.first_ready_ms=Date.now()-this.playMetrics.started_ms;this.publishMetrics();}}
  publishMetrics(){if(this.video.dataset&&this.playMetrics)this.video.dataset.playbackMetrics=JSON.stringify(this.playMetrics);}
  message(text,error=false){this.byId('watchMessage').textContent=text;this.byId('watchMessage').classList.toggle('error',error);}
  static audioSupport(video,stream){
    if(!stream||typeof video.canPlayType!=='function')return 'unknown';
    const types={aac:['audio/mp4; codecs="mp4a.40.2"','audio/aac'],mp3:['audio/mpeg'],
      ac3:['audio/mp4; codecs="ac-3"','audio/ac3'],eac3:['audio/mp4; codecs="ec-3"','audio/eac3'],
      dts:['audio/mp4; codecs="dtsc"','audio/mp4; codecs="dtsh"','audio/mp4; codecs="dtsl"','audio/vnd.dts'],
      opus:['audio/webm; codecs="opus"','audio/ogg; codecs="opus"'],vorbis:['audio/webm; codecs="vorbis"','audio/ogg; codecs="vorbis"'],
      flac:['audio/flac'],alac:['audio/mp4; codecs="alac"']};
    const probes=types[stream.codec];if(!probes)return 'unknown';
    try{return probes.some(type=>['maybe','probably'].includes(video.canPlayType(type)))?'supported':'unsupported';}catch{return 'unknown';}
  }
  scheduleAudioCompatibility(){
    if(!this.current||this.stopped||this.nativeSession||this.nativeBusy||this.mediaStage!=='original'||this.repairBusy||this.autoAudioBypass)return;
    const ordinal=Number(this.byId('watchAudio').value)||0,stream=this.mediaInfo?.audio_streams?.[ordinal];
    const unsupported=FinishedPlayer.audioSupport(this.video,stream)==='unsupported';
    if(!unsupported&&!this.restoreAudio)return;
    const key='audio:'+ordinal;if(this.tried.has(key))return;this.tried.add(key);this.restoreAudio=false;
    const seq=this.sequence;
    this.playbackMessage(unsupported?'浏览器不支持所选音轨，正在自动准备兼容音频；视频保持原编码。':'正在恢复所选音轨；视频保持原编码。');
    Promise.resolve().then(()=>{if(seq===this.sequence&&!this.stopped&&this.mediaStage==='original')return this.repairPlayback();}).catch(e=>{if(seq===this.sequence)this.playbackMessage(e.message,true);});
  }
  static time(value){if(!Number.isFinite(value)||value<0)return '00:00';const seconds=Math.floor(value);const h=Math.floor(seconds/3600),m=Math.floor(seconds%3600/60),s=seconds%60;return (h?h+':':'')+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');}
  async init(){try{const result=await this.refresh();if(!this.sequence&&result.last_item_id&&this.items.some(i=>i.id===result.last_item_id&&i.available))await this.open(result.last_item_id);}catch(error){this.message('无法恢复上次观看：'+error.message,true);}}
  refresh(){this.refreshChain=this.refreshChain.catch(()=>{}).then(async()=>{const result=await this.request('items');this.items=result.items;this.renderLibrary();return result;});return this.refreshChain;}
  renderLibrary(){const list=this.byId('watchItems');list.replaceChildren();for(const item of this.items){const b=document.createElement('button');b.className='watch-item';b.type='button';b.textContent=item.title;b.setAttribute('aria-current',String(item.id===this.current?.id));b.disabled=!item.available;const note=document.createElement('small');note.textContent=!item.available?item.error:(item.kind==='job'?'已完成任务'+(item.parent_id?' · 可选修订版本':''):'手动导入')+' · '+(item.danmaku_id?'带弹幕':'仅影片')+(item.state.position>0?' · 上次 '+FinishedPlayer.time(item.state.position):'');b.append(note);b.onclick=()=>this.open(item.id).catch(e=>this.message(e.message,true));list.append(b);}if(!this.items.length){const p=document.createElement('p');p.className='empty';p.textContent='完成任务后会出现在这里，也可单独导入影片。';list.append(p);}}
  snapshot(){return {position:this.nativeSession?this.current.state.position:(this.loaded?Math.max(0,this.mediaClock.currentTime):this.current.state.position),volume:this.nativeSession?this.current.state.volume:this.video.volume,muted:this.nativeSession?this.current.state.muted:this.video.muted,playback_rate:this.nativeSession?this.current.state.playback_rate:this.video.playbackRate,danmaku_enabled:this.byId('watchComments').checked,font_size:Number(this.byId('watchFont').value),opacity:Number(this.byId('watchOpacity').value),show_estimated:this.byId('watchEstimated').checked,danmaku_offset:Number(this.byId('danmakuOffset').value)||0,subtitle_id:this.subtitleId,subtitle_enabled:this.byId('subtitleEnabled').checked,subtitle_offset:this.subtitles.offset,playback_audio_stream:Number(this.byId('watchAudio').value)||0};}
  persist(keepalive=false){if(!this.current||this.restoring||this.stopped)return this.saveChain;const id=this.current.id,state=this.snapshot();this.current.state=state;const listed=this.items.find(item=>item.id===id);if(listed)listed.state={...state};this.lastSave=Date.now();this.writeRevision=Math.max(Date.now()*1000,this.writeRevision+1);const payload={...state,write_revision:this.writeRevision};const send=()=>this.request('items/'+encodeURIComponent(id)+'/state',payload,'PATCH',keepalive);if(keepalive){send().catch(()=>{});return this.saveChain;}this.saveChain=this.saveChain.catch(()=>{}).then(send).catch(()=>{if(!this.stopped)this.message('观看进度暂未保存，请检查本地服务连接。',true);});return this.saveChain;}
  async pauseAndSave(){this.video.pause();await this.persist();}
  stop(){clearInterval(this.nativeTimer);this.setAudioQualityNotice();this.transport.close();this.stopped=true;this.video.pause();this.player.el.inert=true;for(const id of ['repairPlayback','retryOriginal','watchAudio','importMovie','importComments','refreshLibrary','importSubtitle','removeSubtitle','subtitleEarlier','subtitleLater','subtitleReset','subtitleOffset','subtitleEnabled','danmakuOffset','danmakuEarlier','danmakuLater','danmakuReset','feedback','feedbackKind','continueWatch','nativeOpen','nativeReturn'])this.byId(id).disabled=true;this.byId('watchItems').inert=true;this.message('服务已停止。可以关闭此网页。');}
  configure(){this.renderer.configure({enabled:this.byId('watchComments').checked,fontSize:Number(this.byId('watchFont').value),opacity:Number(this.byId('watchOpacity').value),showEstimated:this.byId('watchEstimated').checked,offset:Number(this.byId('danmakuOffset').value)||0});this.byId('danmakuOffsetValue').textContent=(Number(this.byId('danmakuOffset').value)>=0?'+':'')+Number(this.byId('danmakuOffset').value||0).toFixed(1)+' 秒';this.byId('watchFontValue').textContent=this.byId('watchFont').value;this.byId('watchOpacityValue').textContent=Math.round(Number(this.byId('watchOpacity').value)*100)+'%';}
  async open(key){const seq=++this.sequence;this.opening=true;try{await this.pauseAndSave();if(seq!==this.sequence)return;const item=this.items.find(i=>i.id===key);if(!item?.available)throw Error(item?.error||'成品不存在，请刷新列表');if(this.current?.id===key)return;this.message('正在载入影片与弹幕…');const file=async id=>{const response=await fetch('/api/files/'+id);if(!response.ok)throw Error('影片或弹幕文件已移动或无法读取，请重新选择');return response.json();};let data={data:[]},report=null,warning='';if(item.danmaku_id)data=await file(item.danmaku_id);const comments=DanmakuRenderer.parse(data);if(item.report_id){try{report=await file(item.report_id);}catch{warning='报告无法读取，估算标记可能不完整。';}}let captions=null,captionWarning='';if(item.state.subtitle_id){try{captions=await this.request('subtitles/'+encodeURIComponent(item.state.subtitle_id));}catch(error){captionWarning='外挂字幕无法读取：'+error.message+'。请重新选择，影片仍可播放。';}}let mediaInfo=null;try{mediaInfo=await this.request('items/'+encodeURIComponent(key)+'/media');}catch{}if(seq!==this.sequence)return;this.renderer.setItems([]);this.restoring=true;this.loaded=false;this.current={...item,state:{...item.state}};const state=this.current.state;this.mediaInfo=mediaInfo;this.mediaStage='original';this.tried=new Set();this.autoAudioBypass=false;this.repairBusy=false;this.configureAudio(state.playback_audio_stream||0);this.restoreAudio=Number(this.byId('watchAudio').value)>0;this.subtitleId=state.subtitle_id||null;this.subtitles.setCues(captions?.cues||[]);this.byId('subtitleEnabled').checked=state.subtitle_enabled!==false;this.byId('subtitleOffset').value=String(state.subtitle_offset||0);this.byId('importSubtitle').disabled=false;this.configureSubtitle();this.subtitleMessage(captionWarning||(captions?this.captionInfo(item.subtitle?.name||'外挂字幕',captions):'尚未挂载字幕。支持 SRT、VTT、ASS/SSA 对白文本。'),Boolean(captionWarning));this.byId('watchTitle').textContent=item.title;this.byId('watchEmpty').hidden=true;this.byId('importComments').disabled=false;this.byId('watchComments').checked=state.danmaku_enabled;this.byId('watchFont').value=state.font_size;this.byId('watchOpacity').value=state.opacity;this.byId('watchEstimated').checked=state.show_estimated;this.byId('danmakuOffset').value=String(state.danmaku_offset||0);this.byId('continueWatch').hidden=false;this.byId('continueWatch').disabled=true;this.byId('continueWatch').textContent=state.position>0?'继续观看':'开始观看';this.video.volume=state.volume;this.video.muted=state.muted;this.video.defaultPlaybackRate=state.playback_rate;this.video.playbackRate=state.playback_rate;this.renderer.setItems(comments,report);this.configure();this.byId('watchCommentInfo').textContent=comments.length?`已加载 ${comments.length} 条弹幕。`+(report?'估算弹幕可单独隐藏。':'此 JSON 未附带映射报告；仅筛选其中显式标记的估算弹幕。'):'没有弹幕，可直接观看影片或另行加载 JSON。';this.nativeSession=null;clearInterval(this.nativeTimer);this.nativeTimer=null;this.byId('watchStage').hidden=false;this.byId('nativeReturn').hidden=true;this.byId('nativeStatus').hidden=true;this.byId('nativeOpen').disabled=false;if(!await this.restoreNative(seq)){if(state.preferred_engine==='iina'){this.mediaStage='native_pending';this.loaded=false;this.playbackReady=false;this.transport.close();this.video.removeAttribute?.('src');this.video.load();this.byId('watchStage').hidden=true;this.byId('nativeReturn').hidden=false;this.byId('nativeStatus').hidden=false;this.byId('nativeStatus').textContent='上次使用 IINA 原音质观看。点击原音质观看继续，或返回网页播放。';}else this.loadPlayback(item.video_id,'original',state.position);}this.renderLibrary();this.message(warning||'已载入，点击播放继续。',Boolean(warning));}finally{if(seq===this.sequence)this.opening=false;}}
  metadata(){if(this.nativeSession||this.mediaStage==='native_pending')return;if(!this.current||!Number.isFinite(this.video.duration))return;this.video.volume=this.current.state.volume;this.video.muted=this.current.state.muted;this.video.playbackRate=this.current.state.playback_rate;this.syncSpeed();this.loaded=true;this.video.currentTime=Math.max(0,Math.min(this.current.state.position+this.clockOffset,this.video.duration));this.video.pause();this.restoring=false;this.persist();this.scheduleAudioCompatibility();}
  syncSpeed(){const setting=this.speedSetting;if(!setting)return;setting.value=this.video.playbackRate;if(setting._selectedEl)setting._selectedEl.textContent=this.video.playbackRate===1?'正常':this.video.playbackRate+'×';}
  configureAudio(ordinal=0){const select=this.byId('watchAudio');select.replaceChildren();const streams=this.mediaInfo?.audio_streams||[];for(const stream of streams){const option=document.createElement('option');option.value=String(stream.ordinal);option.textContent=`第 ${stream.ordinal+1} 条 · ${stream.codec} · ${stream.language}`;select.append(option);}if(!streams.length){const option=document.createElement('option');option.value='0';option.textContent=this.mediaInfo?'无音轨':'默认音轨';select.append(option);}select.value=String(ordinal<streams.length?ordinal:0);select.disabled=streams.length<2;this.repairControls();}
  playbackMessage(text,error=false){this.byId('playbackStatus').textContent=text;this.byId('playbackStatus').classList.toggle('error',error);}
  setAudioQualityNotice(mode=null){
    const notice=this.byId('audioQualityNotice');
    const text=mode==='segments'?'音质提醒：当前音轨已按需转换为 AAC 256 kb/s 双声道（有损），音质可能降低；多声道环绕声会混合为双声道。视频未重新编码，原片不变。':
      mode==='audio'?'音质提醒：当前播放文件的音轨已转换为 AAC（有损），音质可能降低。视频未重新编码，原片不变。':
      mode==='transcode'?'音质提醒：当前使用已转码的播放文件，音频为 AAC（有损），音质可能降低。原片不变。':'';
    notice.textContent=text;notice.hidden=!text;
  }
  repairControls(){for(const id of ['repairPlayback','retryOriginal'])this.byId(id).disabled=!this.current||Boolean(this.nativeSession)||this.repairBusy||this.stopped;this.byId('watchAudio').disabled=this.repairBusy||this.stopped||(this.mediaInfo?.audio_streams?.length||0)<2;}
  loadPlayback(fileId,stage,position,offset=(stage==='original'?Number(this.mediaInfo?.origin)||0:0)){this.transport.close();this.beginPlaybackMetrics(stage);this.byId('continueWatch').disabled=true;this.restoring=true;this.video.pause();this.loaded=false;this.playbackReady=false;this.current.state.position=position;this.clockOffset=offset;this.mediaStage=stage;this.setAudioQualityNotice(stage==='audio'||stage==='transcode'?stage:null);this.activeFileId=fileId;this.renderer.reset?.();this.subtitles.render?.();this.playbackMessage(stage==='original'?'正在直接读取原片；不会因为 MKV 自动生成副本。':({remux:'已使用无损换封装文件，视频与音频均未重新编码。',audio:'已使用兼容音频文件，视频保持原编码。',transcode:'已使用经确认生成的 H.264/AAC 播放文件。'}[stage])+' 已核对视频时间轴，弹幕和字幕继续使用原片时间。');this.repairControls();this.video.src='/api/files/'+fileId;this.video.load();}
  mediaError(missingVideo=false){if(!this.current||this.stopped||this.nativeSession||this.mediaStage==='native_pending')return;const code=missingVideo?4:this.video.error?.code;this.current.state.position=this.loaded&&this.playbackReady?Math.max(0,this.mediaClock.currentTime):this.current.state.position;this.loaded=false;this.restoring=false;this.message('浏览器无法读取或解码这部影片。原片和弹幕保留。',true);if(code===3||code===4){return this.repairPlayback().catch(e=>this.playbackMessage(e.message,true));}else this.playbackMessage('读取失败，请检查文件和本地服务后重试原片；也可手动修复播放。',true);}
  async repairPlayback(reset=false){
    if(!this.current||this.stopped||this.nativeSession||this.nativeBusy||this.repairBusy)return;
    const seq=this.sequence,position=this.loaded?Math.max(0,this.mediaClock.currentTime):this.current.state.position;
    const forceAudio=this.mediaStage==='segments'&&this.transport.session?.audio_copy;
    if(this.mediaStage==='segments'&&!reset&&!forceAudio){this.playbackMessage('此浏览器无法解码原片视频，请更换支持该编码的浏览器；不会转码视频或生成整片副本。',true);await this.transport.close();return;}
    this.restoreAudio=false;this.repairBusy=true;this.repairControls();await this.pauseAndSave();
    try{
      const audio=Number(this.byId('watchAudio').value)||0;
      const cached=await this.request('items/'+encodeURIComponent(this.current.id)+'/prepared?audio_stream='+audio);
      if(seq!==this.sequence||this.stopped)return;
      const unsupported=FinishedPlayer.audioSupport(this.video,this.mediaInfo?.audio_streams?.[audio])==='unsupported';
      const existing=cached.find(item=>item.timeline_verified&&!(unsupported&&item.strategy==='remux'));
      if(existing){this.loadPlayback(existing.file_id,existing.strategy,position,existing.timeline_offset||0);return;}
      const copyAudio=!forceAudio&&this.mediaInfo?.audio_streams?.[audio]?.codec==='aac'&&Boolean(this.video.canPlayType?.('audio/mp4; codecs="mp4a.40.2"'));
      this.beginPlaybackMetrics('segments');
      const session=await this.request('sessions',{item_id:this.current.id,audio_stream:audio,audio_copy:copyAudio});
      if(seq!==this.sequence||this.stopped||this.nativeSession||this.nativeBusy){await this.request('sessions/'+session.id,undefined,'DELETE');return;}
      this.restoring=true;this.loaded=false;this.playbackReady=false;this.current.state.position=position;this.clockOffset=0;this.mediaStage='segments';
      this.playbackMessage('正在准备当前位置的播放片段；视频保持原编码。');
      await this.transport.open(session,position);
      if(seq===this.sequence&&!this.stopped&&!this.nativeSession&&!this.nativeBusy)this.setAudioQualityNotice(!session.audio_copy&&session.mime?.includes('mp4a.40.2')?'segments':null);
    }catch(error){if(seq===this.sequence&&!this.stopped){this.restoring=false;this.setAudioQualityNotice();this.playbackMessage(error.message,true);}}
    finally{if(seq===this.sequence){this.repairBusy=false;this.repairControls();}}
  }
  retryOriginal(){if(!this.current||this.repairBusy)return;this.tried=new Set();this.autoAudioBypass=true;this.restoreAudio=false;this.byId('watchAudio').value='0';this.loadPlayback(this.current.video_id,'original',this.loaded?Math.max(0,this.mediaClock.currentTime):this.current.state.position);}
  adjustDanmaku(delta,reset=false){if(this.opening)return;this.byId('danmakuOffset').value=String(reset?0:Math.round(Math.max(-5,Math.min(5,(Number(this.byId('danmakuOffset').value)||0)+delta))*10)/10);this.configure();this.persist();}
  async feedback(){if(this.opening)throw Error('影片正在载入，请稍候');if(!this.current)throw Error('请先选择影片');await this.request('items/'+encodeURIComponent(this.current.id)+'/feedback',{position:this.snapshot().position,kind:this.byId('feedbackKind').value});this.message('已记录当前时间；观看和映射保持不变。');}
  async restoreNative(seq=this.sequence){
    try{const result=await this.request('native/items/'+encodeURIComponent(this.current.id));this.nativeSuggestion=result.subtitle_suggestion;
      if(seq!==this.sequence)return true;
      if(result.session&&['ready','needs_attention','starting'].includes(result.session.status)){this.adoptNative(result.session);return true;}
    }catch{}
    return false;
  }
  adoptNative(session){
    this.nativeSession=session;if(session.state)this.current.state={...this.current.state,...session.state};
    this.video.pause();this.transport.close();this.setAudioQualityNotice();this.playbackMessage('IINA 直接播放原片，原音轨保留。');this.byId('watchStage').hidden=true;
    this.subtitleMessage((session.subtitle_name||'外挂字幕')+' · IINA 原样加载，保留 ASS 样式、定位与换行；字幕偏移与弹幕独立。');
    this.byId('continueWatch').disabled=true;this.byId('nativeReturn').hidden=false;this.byId('nativeOpen').disabled=true;
    const source=session.source_audio;const output=session.output_audio;
    const sourceText=source?`${source.profile||source.codec_name} · ${source.channel_layout||source.channels+' 声道'} · ${source.sample_rate||'?'} Hz`:'无音轨';
    const outputText=output?`${output.channels||'?'} · ${output['samplerate']||output['sample-rate']||'?'} Hz`:'开始播放后确认';
    const sync=session.render_sync;const syncText=sync?.mode==='display-vdrop'?(sync.active?'弹幕显示同步已启用，音频速度保持原值。':sync.paused?'弹幕显示同步已配置，播放后确认启用。':'当前显示同步未启用，弹幕可能随影片帧率更新。'):'弹幕显示同步状态待确认。';
    const text=this.byId('nativeStatus');text.hidden=false;text.textContent=`IINA 原音质播放：${session.status==='ready'?'影片、中文外挂、弹幕均已就绪':'需要处理：'+(session.error||'正在准备')}\n字幕：${session.subtitle_name||'未就绪'} · 弹幕 ${session.comment_count||0} 条\n源音轨：${sourceText} → 播放器输出：${outputText}\n${syncText}\n不新增有损音频转码；耳机或电脑扬声器的双声道输出不代表真实 5.1。Ctrl+Shift+D 可独立开关弹幕。`;
    this.byId('watchComments').checked=this.current.state.danmaku_enabled;this.byId('subtitleEnabled').checked=this.current.state.subtitle_enabled;this.byId('subtitleOffset').value=String(this.current.state.subtitle_offset||0);this.byId('danmakuOffset').value=String(this.current.state.danmaku_offset||0);this.byId('watchAudio').value=String(this.current.state.playback_audio_stream||0);this.configure();this.configureSubtitle();
    this.repairControls();
    if(!this.nativeTimer){const seq=this.sequence;this.nativeTimer=setInterval(async()=>{
      if(!this.nativeSession||seq!==this.sequence)return;
      try{const value=await this.request('native/sessions/'+this.nativeSession.id);if(seq!==this.sequence)return;
        if(['closed','failed'].includes(value.status)){clearInterval(this.nativeTimer);this.nativeTimer=null;this.byId('nativeStatus').textContent=value.error||'IINA 已关闭，可返回网页继续';this.nativeSession=value;return;}
        this.nativeSession=value;if(value.state)this.current.state={...this.current.state,...value.state};this.adoptNative(value);
      }catch(e){this.byId('nativeStatus').textContent='原生会话状态暂不可用：'+e.message;}
    },2000);}
  }
  async openNative(){
    if(!this.current||this.nativeBusy||this.stopped)return;
    this.nativeBusy=true;this.byId('nativeOpen').disabled=true;this.byId('continueWatch').disabled=true;
    try{await this.pauseAndSave();const seq=this.sequence;
      const capabilities=await this.request('native/capabilities');if(!capabilities.available)throw Error('未找到 IINA，网页播放仍可使用');
      if(!this.subtitleId){this.subtitleMessage('原音质观看必须加载中文外挂。请选择与本片对应的 ASS 或 SRT；同目录可能已有 .zh.ass。'+(this.nativeSuggestion?' 建议选择：'+this.nativeSuggestion:''));await this.importSubtitle();}
      if(!this.subtitleId)throw Error('尚未选择中文外挂，未启动无字幕播放');
      if(seq!==this.sequence)return;
      const session=await this.request('native/sessions',{item_id:this.current.id,subtitle_id:this.subtitleId});
      if(seq!==this.sequence){await this.request('native/sessions/'+session.id,undefined,'DELETE');return;}this.adoptNative(session);this.message('已在 IINA 打开，恢复位置并保持暂停。请在 IINA 点击播放。');
    }finally{this.nativeBusy=false;this.byId('nativeOpen').disabled=Boolean(this.nativeSession)||this.stopped;this.byId('continueWatch').disabled=Boolean(this.nativeSession)||this.nativeBusy||!this.playbackReady||this.stopped;}
  }
  async returnFromNative(){
    if(!this.current)return;const seq=this.sequence;
    if(this.nativeSession)await this.request('native/sessions/'+this.nativeSession.id,undefined,'DELETE');
    if(seq!==this.sequence)return;clearInterval(this.nativeTimer);this.nativeTimer=null;this.nativeSession=null;
    await this.request('items/'+encodeURIComponent(this.current.id)+'/state',{preferred_engine:'web'},'PATCH');
    const result=await this.request('items');const item=result.items.find(i=>i.id===this.current.id);if(item)this.current.state=item.state;
    this.byId('watchStage').hidden=false;this.byId('nativeReturn').hidden=true;this.byId('nativeStatus').hidden=true;this.byId('nativeOpen').disabled=false;this.subtitleMessage('网页字幕按对白文本显示；ASS 原样样式可使用 IINA。');
    this.autoAudioBypass=false;this.tried=new Set();this.loadPlayback(this.current.video_id,'original',this.current.state.position);
  }
  subtitleMessage(text,error=false){this.byId('subtitleInfo').textContent=text;this.byId('subtitleInfo').classList.toggle('error',error);}
  captionInfo(name,data){return `${name} · ${data.format} · ${data.cues.length} 条。`+(data.note||'')+' 正数延后、负数提前；设置按影片保存。';}
  configureSubtitle(){this.subtitles.configure({enabled:this.byId('subtitleEnabled').checked,offset:this.byId('subtitleOffset').value});this.byId('subtitleOffset').value=String(this.subtitles.offset);this.byId('subtitleOffsetValue').textContent=(this.subtitles.offset>=0?'+':'')+this.subtitles.offset.toFixed(1)+' 秒';const loaded=Boolean(this.subtitleId&&this.subtitles.cues.length);for(const id of ['subtitleEnabled','subtitleOffset','subtitleEarlier','subtitleLater','subtitleReset'])this.byId(id).disabled=!loaded;this.byId('removeSubtitle').disabled=!this.subtitleId;this.player.el.classList.toggle('-has-subtitle',loaded&&this.subtitles.enabled);}
  adjustSubtitle(delta,reset=false){this.byId('subtitleOffset').value=reset?'0':String(this.subtitles.offset+delta);this.configureSubtitle();this.persist();}
  removeSubtitle(){if(this.nativeSession){this.subtitleMessage('原音质观看必须保留外挂字幕；可选择新的字幕替换。',true);return;}this.subtitleId=null;this.subtitles.setCues([]);this.byId('subtitleOffset').value='0';this.byId('subtitleEnabled').checked=true;this.configureSubtitle();this.subtitleMessage('已卸载字幕，原文件保留。');this.persist();}
  async importSubtitle(){const active=this.current;if(!active)throw Error('请先选择影片');this.byId('importSubtitle').disabled=true;try{const file=await this.pick('subtitle');if(!file)return;const data=await this.request('subtitles/'+encodeURIComponent(file.id));if(this.current?.id!==active.id||this.stopped)throw Error('影片已切换，请为当前影片重新选择字幕');this.subtitleId=file.id;this.current.subtitle={id:file.id,name:file.name};this.subtitles.setCues(data.cues);this.byId('subtitleOffset').value='0';this.byId('subtitleEnabled').checked=true;this.configureSubtitle();this.subtitleMessage(this.captionInfo(file.name,data));await this.persist();}finally{this.byId('importSubtitle').disabled=!this.current||this.stopped;}}
  async pick(kind){const response=await fetch('/api/pick',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind})});const result=await response.json();if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:'文件选择失败');if(result.errors?.length)throw Error(result.errors.map(e=>e.reason).join('；'));return result.files[0]||null;}
  async importMovie(){this.byId('importMovie').disabled=true;try{await this.pauseAndSave();const file=await this.pick('playback_video');if(!file)return;const item=await this.request('items',{video_id:file.id});await this.refresh();await this.open(item.id);}finally{this.byId('importMovie').disabled=false;}}
  async importComments(){const active=this.current;if(!active)throw Error('请先选择影片');this.byId('importComments').disabled=true;try{await this.pauseAndSave();const file=await this.pick('danmaku');if(!file)return;if(this.current?.id!==active.id)throw Error('影片已切换，请为当前影片重新选择弹幕');await this.pauseAndSave();const state=this.snapshot();const item=await this.request('items',{video_id:active.video_id,danmaku_id:file.id,item_id:active.kind==='manual'?active.id:null});await this.request('items/'+encodeURIComponent(item.id)+'/state',state,'PATCH');this.current=null;await this.refresh();await this.open(item.id);}finally{this.byId('importComments').disabled=!this.current;}}
};
var finishedPlayer = new FinishedPlayer();

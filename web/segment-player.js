/* MSE transport only. NPlayer, subtitles and danmaku keep the original clock. */
'use strict';
var SegmentPlayer = class {
  constructor(video,request,status){this.video=video;this.request=request;this.status=status;this.session=null;this.epoch=0;this.generation=0;this.chain=Promise.resolve();this.lastSeek=-1;this.lastHeartbeat=0;
    this.seek=()=>{if(!this.session)return;const t=this.video.currentTime;if(Math.abs(t-this.lastSeek)<.15)return;this.lastSeek=t;this.generation++;this.pump(t).catch(e=>this.fail(e));};
    this.play=()=>this.pump(this.video.currentTime).catch(e=>this.fail(e));
    this.pause=()=>{if(this.session&&!this.starting)this.request('sessions/'+this.session.id+'/pause',{}).catch(()=>{});};
    video.addEventListener('seeking',this.seek);video.addEventListener('play',this.play);video.addEventListener('pause',this.pause);
  }
  async open(session,position){
    await this.close();
    if(typeof MediaSource==='undefined'||!MediaSource.isTypeSupported(session.mime)){
      await this.request('sessions/'+session.id,undefined,'DELETE');
      throw Error('此浏览器不支持原视频编码的分段播放，请更换支持该编码的浏览器。不会转码视频或生成整片副本。');
    }
    this.session=session;this.appended=[];this.starting=true;this.lastSeek=position;const epoch=this.epoch;
    this.media=new MediaSource();this.url=URL.createObjectURL(this.media);
    const opened=new Promise((resolve,reject)=>{this.media.addEventListener('sourceopen',resolve,{once:true});this.media.addEventListener('error',()=>reject(Error('浏览器无法建立分段播放缓冲')),{once:true});});
    this.video.src=this.url;this.video.load();await opened;
    if(epoch!==this.epoch)return;
    this.buffer=this.media.addSourceBuffer(session.mime);this.media.duration=session.duration;
    this.timer=setInterval(()=>{
      if(!this.session)return;
      if(Date.now()-this.lastHeartbeat>10000){this.lastHeartbeat=Date.now();this.request('sessions/'+this.session.id+'/heartbeat',{}).catch(e=>this.fail(e));}
      if(!this.video.paused&&this.bufferEnd()-this.video.currentTime<2)this.pump(this.video.currentTime).catch(e=>this.fail(e));
    },500);
    await this.pump(position);this.starting=false;
  }
  finishIfComplete(state,seen,session){
    const last=state.parts.at(-1);
    if(state.status==='complete'&&last&&last.end>=session.duration-.05&&seen.has(last.name)&&this.media?.readyState==='open')this.media.endOfStream();
  }
  bufferEnd(){const ranges=this.video.buffered;if(!ranges)return 0;for(let i=0;i<ranges.length;i++)if(ranges.start(i)<=this.video.currentTime+.15&&ranges.end(i)>this.video.currentTime)return ranges.end(i);return 0;}
  fail(error){if(!this.session)return;this.status(error.message,true);this.close().catch(()=>{});}
  operation(action){const epoch=this.epoch;this.chain=this.chain.catch(()=>{}).then(()=>new Promise((resolve,reject)=>{if(epoch!==this.epoch)return resolve();
    const buffer=this.buffer;if(!buffer)return resolve();
    const done=()=>{cleanup();resolve();},bad=()=>{cleanup();reject(Error('浏览器无法解码播放片段，请更换支持原片的浏览器'));};
    const cleanup=()=>{buffer.removeEventListener('updateend',done);buffer.removeEventListener('error',bad);buffer.removeEventListener('abort',done);};
    buffer.addEventListener('abort',done,{once:true});buffer.addEventListener('updateend',done,{once:true});buffer.addEventListener('error',bad,{once:true});
    try{action(buffer);if(!buffer.updating){cleanup();resolve();}}catch(e){cleanup();reject(e);}
  }));return this.chain;}
  covered(part){if(this.appended.some(p=>p.start<=part.start+.002&&p.end>=part.end-.002))return true;const ranges=this.video.buffered;for(let i=0;i<(ranges?.length||0);i++)if(ranges.start(i)<=part.start+.002&&ranges.end(i)>=part.end-.002)return true;return false;}
  async prune(){
    // Remove only up to an original random-access boundary. Keeping eight seconds
    // is already over 100 MiB for UHD remuxes; preserve just the current GOP.
    const time=this.video.currentTime,part=this.appended.filter(p=>p.start<=time-.15&&p.end>=time-.15).at(-1);
    if(part&&part.start>0){await this.operation(buffer=>buffer.remove(0,part.start));this.appended=this.appended.filter(p=>p.end>part.start);}
  }
  async append(data,part,session){
    await this.prune();
    await this.operation(buffer=>{buffer.timestampOffset=part.offset;buffer.appendWindowStart=0;buffer.appendWindowEnd=session.duration+.1;});
    // Incremental parsing also avoids presenting a whole high-bitrate GOP as one
    // enormous append. Eviction runs before every new fragment.
    for(let at=0;at<data.byteLength;at+=1024*1024){const chunk=data.slice(at,at+1024*1024);await this.operation(buffer=>buffer.appendBuffer(chunk));}
    this.appended.push(part);
  }
  async pump(position){
    if(!this.session)return;
    const epoch=this.epoch,generation=this.generation,session=this.session;
    if(this.running===generation)return;
    this.running=generation;
    try{
      let requestAt=position;
      const bufferedEnd=this.bufferEnd();if(bufferedEnd-position>=2||bufferedEnd>=session.duration-.05)return;
      if(bufferedEnd>position)requestAt=Math.max(position,bufferedEnd-.05);
      const window=await this.request('sessions/'+session.id+'/windows',{position:requestAt,generation});
      const base='sessions/'+session.id+'/windows/'+window.id;
      const seen=new Set();let state=window;
      while(epoch===this.epoch&&generation===this.generation){
        if(state.status==='failed')throw Error(state.error);
        for(const part of state.parts){
          if(seen.has(part.name)||part.end<=position-.1)continue;
          if(this.covered(part)){seen.add(part.name);continue;}
          if(epoch!==this.epoch||generation!==this.generation)return;
          const response=await fetch('/api/playback/'+base+'/parts/'+encodeURIComponent(part.name));
          if(!response.ok)throw Error('播放片段已失效，请重试当前位置');
          const data=await response.arrayBuffer();
          if(epoch!==this.epoch||generation!==this.generation)return;
          await this.append(data,part,session);
          seen.add(part.name);this.finishIfComplete(state,seen,session);
          await this.prune();
          if(this.bufferEnd()>position+.1){this.status('正在按需播放：视频未重新编码，弹幕和字幕沿用原片时间。');
            if((this.video.paused||this.bufferEnd()-this.video.currentTime>=2)&&!(state.parts.at(-1)?.end>=session.duration-.05&&state.status==='running'&&this.video.paused)){if(this.video.paused)await this.request('sessions/'+session.id+'/pause',{});return;}}
        }
        this.finishIfComplete(state,seen,session);
        if(['complete','cancelled'].includes(state.status))break;
        await new Promise(resolve=>setTimeout(resolve,100));
        if(epoch!==this.epoch||generation!==this.generation)return;
        state=await this.request(base);
      }
    }finally{if(this.running===generation)this.running=null;}
  }
  async close(){
    this.epoch++;this.generation++;if(this.buffer?.updating){try{this.buffer.abort();}catch{}}this.chain=Promise.resolve();clearInterval(this.timer);const session=this.session;this.session=null;this.buffer=null;this.running=null;
    if(this.url){URL.revokeObjectURL(this.url);this.url=null;}
    if(session)await this.request('sessions/'+session.id,undefined,'DELETE').catch(()=>{});
  }
  destroy(){this.close();this.video.removeEventListener('seeking',this.seek);this.video.removeEventListener('play',this.play);this.video.removeEventListener('pause',this.pause);}
};

/* Plain-text captions on the video clock; positive offset delays the dialogue. */
'use strict';
var SubtitleRenderer = class {
  constructor(video, layer) {
    this.video=video;this.layer=layer;this.cues=[];this.tree=[];this.offset=0;this.enabled=true;this.frame=null;this.last='';
    for(const event of ['timeupdate','seeking','seeked','loadedmetadata','ratechange','pause','ended'])video.addEventListener(event,()=>this.render());
    video.addEventListener('play',()=>this.tick());
  }
  setCues(cues) {
    this.cues=cues.map(cue=>({...cue})).sort((a,b)=>a.start-b.start||a.end-b.end);
    this.tree=[];
    const build=(node,left,right)=>{if(left===right)return this.tree[node]=this.cues[left].end;const mid=(left+right)>>1;return this.tree[node]=Math.max(build(node*2,left,mid),build(node*2+1,mid+1,right));};
    if(this.cues.length)build(1,0,this.cues.length-1);
    this.render();
  }
  configure({enabled=this.enabled,offset=this.offset}={}) {
    this.enabled=Boolean(enabled);this.offset=Math.round(Math.max(-5,Math.min(5,Number(offset)||0))*10)/10;this.render();
  }
  render() {
    const found=[],time=this.video.currentTime-this.offset;
    const visit=(node,left,right)=>{if(found.length>=20||this.tree[node]<=time||this.cues[left].start>time)return;if(left===right){found.push(this.cues[left].text);return;}const mid=(left+right)>>1;visit(node*2,left,mid);visit(node*2+1,mid+1,right);};
    if(this.enabled&&this.cues.length)visit(1,0,this.cues.length-1);
    const text=[...new Set(found)].join('\n');
    if(text!==this.last){this.layer.textContent=text;this.last=text;}
    this.layer.hidden=!text;
  }
  tick() {
    if(this.frame!==null)cancelAnimationFrame(this.frame);
    this.render();this.frame=null;
    if(!this.video.paused&&!this.video.ended)this.frame=requestAnimationFrame(()=>this.tick());
  }
};

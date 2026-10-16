import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
let media,blocked=null,entered=null,allowed=true,appends=0,deleted=0,paused=0,windows=0;
const ranges=[];
class Buffer extends EventTarget{constructor(){super();this.updating=false;}appendBuffer(){appends++;this.updating=true;queueMicrotask(()=>{ranges.splice(0,ranges.length,[video.currentTime,video.currentTime+4]);this.updating=false;this.dispatchEvent(new Event('updateend'));});}remove(){this.updating=true;queueMicrotask(()=>{this.updating=false;this.dispatchEvent(new Event('updateend'));});}abort(){this.updating=false;this.dispatchEvent(new Event('abort'));}}
class Source extends EventTarget{constructor(){super();media=this;this.readyState='open';}endOfStream(){this.readyState='ended';}static isTypeSupported(){return allowed;}addSourceBuffer(){return this.buffer=new Buffer();}}
const video=new EventTarget();Object.assign(video,{currentTime:1,paused:true,src:'original',buffered:{get length(){return ranges.length;},start:i=>ranges[i][0],end:i=>ranges[i][1]},load(){queueMicrotask(()=>media.dispatchEvent(new Event('sourceopen')));}});
const requests=[];
const request=async(path,body,method)=>{requests.push({path,body,method});if(method==='DELETE'){deleted++;return {};}if(path.endsWith('/pause')){paused++;return {};}
 if(path.endsWith('/windows')){windows++;return {id:'window'+windows,status:'complete',parts:[{name:'part-0000.mp4',start:0,end:video.currentTime+4,offset:0}]};}
 return {};};
const context=vm.createContext({video,request,status(){},MediaSource:Source,URL:{createObjectURL(){return 'blob:test';},revokeObjectURL(){}},fetch:async()=>{if(blocked){const pending=blocked;blocked=null;entered();await pending;}return {ok:true,arrayBuffer:async()=>new ArrayBuffer(8)};},Event,EventTarget,setInterval:()=>1,clearInterval(){},setTimeout,Date,console});
vm.runInContext(fs.readFileSync(new URL('../web/segment-player.js',import.meta.url),'utf8'),context);
const p=vm.runInContext('new SegmentPlayer(video,request,status)',context);
await p.open({id:'a',mime:'video/mp4',duration:100},1);assert.equal(appends,1);assert.equal(paused,1,'paused preload must stop owned prefetch');assert.equal(media.duration,100);await p.close();assert.equal(deleted,1);
allowed=false;video.src='original';await assert.rejects(()=>p.open({id:'unsupported',mime:'video/mp4',duration:100},1));assert.equal(video.src,'original','unsupported codecs must not replace direct media');assert.equal(deleted,2);allowed=true;
// A completed old fetch cannot append into the next session.
ranges.length=0;let release;blocked=new Promise(resolve=>release=resolve);const ready=new Promise(resolve=>entered=resolve);
video.currentTime=31;const old=p.open({id:'old',mime:'video/mp4',duration:100},31);await ready;await p.close();ranges.length=0;video.currentTime=51;
await p.open({id:'new',mime:'video/mp4',duration:100},51);const count=appends;release();await old;assert.equal(appends,count);assert.equal(p.session.id,'new');
await p.close();ranges.length=0;video.currentTime=1;await p.open({id:'eof',mime:'video/mp4',duration:5},1);assert.equal(media.readyState,'ended','EOF must be signalled even when the complete film was preloaded while paused');await p.close();assert.ok(requests.every(r=>!r.path.endsWith('/prepare')));console.log('Segment transport: capability gate, pause, global duration, stale fetch isolation and teardown passed');

// A high-bitrate remux must keep only the current GOP behind the playhead.
p.appended=[{start:0,end:4},{start:4,end:8},{start:8,end:12}];video.currentTime=9;let removed;const operation=p.operation;p.operation=async action=>action({remove:(start,end)=>removed=[start,end]});await p.prune();assert.deepEqual(removed,[0,8]);assert.equal(p.appended.length,1);p.operation=operation;assert.equal(p.covered({start:8,end:12}),true);assert.equal(p.covered({start:8,end:12.1}),false,'a partially covered tail cannot be skipped');
console.log('UHD buffering: current GOP eviction and complete-fragment deduplication passed');

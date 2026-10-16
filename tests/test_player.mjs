// Pure DOM mocks exercise local player logic without browser automation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, {
    id, currentTime: 0, duration: 10, paused: true, checked: false, children: [],
    classList: {toggle(){}, add(){}}, style: {setProperty(){}}, dataset: {},
    addEventListener(){}, replaceChildren(){this.children=[]}, append(child){this.children.push(child)},
    remove(){}, pause(){this.paused=true}, play(){this.paused=false;return Promise.resolve()},
  });
  return elements.get(id);
};
const context = vm.createContext({document: {getElementById: element, createElement: tag=>element(`new-${Math.random()}`)},
  window: {addEventListener(){}}, requestAnimationFrame(){}, URL, Blob, setTimeout, console});
vm.runInContext(fs.readFileSync(new URL('../danmaku-renderer.js', import.meta.url), 'utf8'), context);
vm.runInContext(fs.readFileSync(new URL('../player.js', import.meta.url), 'utf8'), context);
vm.runInContext(`alignment={segments:[{source_start:1,source_end:2,slope:1,anchors:[{source_time:1,target_time:3},{source_time:2,target_time:4}]},{source_start:5,source_end:6,slope:1,anchors:[{source_time:5,target_time:6},{source_time:6,target_time:7}]}]}`,context);
const run = expression => vm.runInContext(expression,context);
assert.equal(run('targetTime(1.5)'), 3.5);
assert.equal(run('targetTime(3)'), null);
assert.equal(run('targetTime(0)'), null);
assert.equal(run('targetTime(7)'), null);
assert.throws(()=>run('validateAnchors([{source_time:1,target_time:3},{source_time:2,target_time:2}])'));
assert.throws(()=>run('validateAnchors([{source_time:11,target_time:3}])'));
assert.throws(()=>run('comments({data:[{time:null}]})'));
assert.throws(()=>run('comments({data:[{time:-1}]})'));
run('validateAnchors([{source_time:0,target_time:0},{source_time:2,target_time:3}])');
run('sourceVideo.currentTime=3;targetVideo.paused=false;jump()');
assert.equal(element('targetVideo').paused,true);
assert.match(element('message').textContent,/没有已确认/);
console.log('Player timeline, validation and failure-pause tests passed');

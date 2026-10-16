/* Shared renderer: positions and lifetimes follow media time, including rate changes. */
'use strict';
var DanmakuRenderer = (() => {
  class Renderer {
    static color(value) {
      if(Number.isInteger(value)&&value>=0&&value<=0xffffff)return '#'+value.toString(16).padStart(6,'0');
      if(typeof value==='string'&&/^#[0-9a-f]{1,6}$/i.test(value))return '#'+value.slice(1).padStart(6,'0');
      return '#ffffff';
    }
    static parse(data) {
      if (!data || !Array.isArray(data.data) || data.data.some(item => !item || !Number.isFinite(item.time) || item.time < 0)) {
        throw Error('弹幕必须包含带有效 time 的 data 数组');
      }
      return data.data;
    }
    static withStatuses(items, report) {
      const states = new Map();
      for (const entry of report?.entries || []) {
        if (!Number.isFinite(entry.target_time)) continue;
        const key = entry.target_time.toFixed(6);
        if (!states.has(key)) states.set(key, {values: [], index: 0});
        states.get(key).values.push(entry.status);
      }
      return items.map(item => {
        const group = states.get(item.time.toFixed(6));
        return {...item, status: group?.values[group.index++] || item.status || 'reliable'};
      });
    }
    constructor(video, layer, settings = {}) {
      this.video = video; this.layer = layer; this.items = []; this.active = [];
      this.settings = {enabled: true, fontSize: 24, opacity: 1, showEstimated: true, offset: 0, ...settings};
      this.lastTime = null; this.cursor = 0; this.row = 0;
      this.onSeek = () => this.reset();
      video.addEventListener('seeking', this.onSeek);
      const loop = () => {this.tick(); this.frame = requestAnimationFrame(loop);};
      this.frame = requestAnimationFrame(loop);
    }
    setItems(items, report = null) {
      this.items = Renderer.withStatuses(items, report).sort((a,b) => a.time-b.time);
      this.reset();
    }
    configure(values) {Object.assign(this.settings, values); this.layer.style.opacity = String(this.settings.opacity); this.reset();}
    clear() {for (const item of this.active) item.node.remove(); this.active = [];}
    reset() {this.clear(); this.lastTime = null; this.row = 0;}
    bound(time, inclusive = false) {
      let low = 0, high = this.items.length;
      while (low < high) {const mid = (low+high) >>> 1; if (this.items[mid].time < time || (inclusive && this.items[mid].time === time)) low=mid+1; else high=mid;}
      return low;
    }
    duration(item) {return item.type === 'top' || item.type === 'bottom' ? 5 : 10;}
    show(item, time) {
      if ((!this.settings.showEstimated && item.status === 'estimated') || time-item.time >= this.duration(item)) return;
      const node = document.createElement('span'); node.className = 'danmaku-comment'+(item.status==='estimated'?' estimated':'');
      node.textContent = item.text || '';
      node.style.color = Renderer.color(item.color);
      node.style.fontSize = `${this.settings.fontSize}px`;
      const fixed = item.type === 'top' || item.type === 'bottom';
      const lanes = fixed ? 3 : Math.max(1, Math.min(12, Math.floor(this.layer.clientHeight/(this.settings.fontSize*1.3))-1));
      node.style[fixed && item.type==='bottom' ? 'bottom' : 'top'] = `${8+(this.row++%lanes)*this.settings.fontSize*1.3}px`;
      if (fixed) {node.style.left='50%'; node.style.transform='translateX(-50%)';}
      else node.style.left='0';
      this.layer.append(node); this.active.push({item,node,fixed});
    }
    tick() {
      const time = this.video.currentTime - (Number(this.settings.offset)||0);
      if (!this.settings.enabled) {this.clear(); this.lastTime=null; return;}
      if (!Number.isFinite(time) || !this.layer.clientWidth) return;
      const end = this.bound(time, true);
      if (this.lastTime === null || time < this.lastTime-.05 || time > this.lastTime+.5) {
        this.clear(); this.cursor=this.bound(Math.max(0,time-10));
      }
      // Bound work and DOM size even for many comments at one timestamp.
      this.cursor=Math.max(this.cursor,end-100);
      while (this.cursor < end) {this.show(this.items[this.cursor++],time);}
      const survivors=[];
      for (const current of this.active) {
        const age = time-current.item.time;
        if (age < 0 || age >= this.duration(current.item)) {current.node.remove(); continue;}
        if (!current.fixed) current.node.style.transform=`translateX(${this.layer.clientWidth-(this.layer.clientWidth+current.node.offsetWidth)*age/10}px)`;
        survivors.push(current);
      }
      this.active=survivors;
      while (this.active.length > 100) this.active.shift().node.remove();
      this.lastTime=time;
    }
    destroy() {cancelAnimationFrame(this.frame); this.video.removeEventListener('seeking',this.onSeek); this.clear();}
  }
  return Renderer;
})();

-- Private per-player OSD. Dialogue stays on mpv's ordinary subtitle track.
local mp = require 'mp'
local utils = require 'mp.utils'
local overlay = mp.create_osd_overlay('ass-events')
local items, settings, nonce, media, loaded_path = {}, {}, '', nil, nil
local enabled, dimensions, layout, previous_data = true, '', {}, nil
local clock = {position=nil, wall=0, raw=nil}
local timer, deadline, interval = nil, nil, 1/60
local stats = {ticks=0, updates=0, interpolated=0, clock_resets=0, loads=0, layouts=0, draw_seconds=0}
local started = mp.get_time()
local function number(name, default) return mp.get_property_number(name, default) end
local function frozen()
  return mp.get_property_native('pause',true) or mp.get_property_native('paused-for-cache',false)
    or mp.get_property_native('seeking',false) or mp.get_property_native('core-idle',false)
end
local function reset_clock()
  clock.position=nil;clock.raw=nil;stats.clock_resets=stats.clock_resets+1
end
local function media_time(now)
  local raw=number('time-pos',0)
  local speed=number('speed',1)
  local quantum=1/math.max(12,number('container-fps',24))
  if frozen() or not clock.position or speed~=clock.speed or raw<(clock.raw or raw)-.0001
    or math.abs(raw-clock.position)>.25+quantum then
    clock.position=raw
  else
    -- Follow wall time between quantized video PTS updates, without letting a
    -- stalled clock move more than one source frame ahead. Never walk backward
    -- except after an explicit seek/discontinuity or pause synchronization.
    local predicted=clock.position+math.max(0,now-clock.wall)*speed
    if raw>predicted+quantum then predicted=raw-quantum end
    predicted=math.min(predicted,raw+quantum)
    clock.position=math.max(clock.position,predicted)
    if math.abs(clock.position-raw)>.0001 then stats.interpolated=stats.interpolated+1 end
  end
  clock.wall=now;clock.raw=raw;clock.speed=speed
  return clock.position-(tonumber(settings.danmaku_offset) or 0)
end
local function read_json(path)
  local f=io.open(path,'rb');if not f then error('Cannot read native configuration') end
  local data=f:read(24*1024*1024);f:close()
  local value=utils.parse_json(data);if not value then error('Invalid native configuration') end
  return value
end
local function escape(text)
  return tostring(text or ''):gsub('\\','\\\239\187\191'):gsub('{','\\{'):gsub('}','\\}'):gsub('[\r\n]',' ')
end
local function color(value)
  if type(value)=='number' and value>=0 and value<=16777215 then value=string.format('#%06x',value) end
  if type(value)~='string' or not value:match('^#%x+$') or #value>7 then value='#ffffff' end
  value=string.rep('0',7-#value)..value:sub(2)
  return value:sub(5,6)..value:sub(3,4)..value:sub(1,2)
end
local function publish()
  mp.set_property_native('user-data/bdm-native',{token=nonce,count=#items,enabled=enabled,heartbeat=mp.get_time(),
    animation={ticks=stats.ticks,updates=stats.updates,interpolated=stats.interpolated,
      clock_resets=stats.clock_resets,comment_loads=stats.loads,layout_builds=stats.layouts,
      draw_seconds=stats.draw_seconds,elapsed=mp.get_time()-started,target_fps=1/interval}})
end
local function clear()
  if previous_data~='' then overlay:remove();previous_data='' end
end
local function draw()
  local begin=mp.get_time();stats.ticks=stats.ticks+1
  local t=media_time(begin)
  local w,h=mp.get_osd_size();if not w or not h or w<=0 or h<=0 then return end
  if not enabled or (media and mp.get_property('path')~=media) then clear();return end
  local size=tostring(w)..':'..tostring(h)
  if dimensions~=size then dimensions=size;layout={} end
  local width,height=1920,1920*h/w
  -- Retain the existing per-film size choice and subtitle exclusion area.
  local font=(tonumber(settings.font_size) or 24)*2.4
  local lanes=math.max(1,math.floor((height*.75-20)/(font*1.3)))
  local first,high=1,#items+1
  while first<high do local mid=math.floor((first+high)/2);if items[mid].time<t-10 then first=mid+1 else high=mid end end
  local active={}
  for i=first,#items do
    local item=items[i];if item.time>t then break end
    if t-item.time<item.duration and (settings.show_estimated or item.status~='estimated') then
      active[#active+1]=i
    end
  end
  local out={}
  for a=math.max(1,#active-99),#active do
    local i=active[a];local item=items[i];local cached=layout[i]
    if not cached then
      local row=(i-1)%(item.fixed and 3 or math.min(12,lanes))
      local anchor=item.fixed and (item.type=='bottom' and 2 or 8) or 7
      local y=item.fixed and item.type=='bottom' and height*.75-10-row*font*1.3 or 10+row*font*1.3
      local opacity=math.max(0,math.min(1,tonumber(settings.opacity) or 1))*(item.status=='estimated' and .65 or 1)
      cached={y=y,text_width=math.max(font,item.characters*font),prefix=string.format('{\\an%d\\fs%.2f\\bord2\\shad1\\1c&H%s&\\3c&H000000&\\alpha&H%02X&',anchor,font,item.bgr,math.floor((1-opacity)*255+.5)),text=item.escaped}
      layout[i]=cached;stats.layouts=stats.layouts+1
    end
    local x=item.fixed and width/2 or width-(width+cached.text_width)*(t-item.time)/10
    out[#out+1]=cached.prefix..string.format('\\pos(%.2f,%.2f)}',x,cached.y)..cached.text
  end
  local data=table.concat(out,'\n')
  if data~=previous_data or overlay.res_x~=width or overlay.res_y~=height then
    overlay.res_x=width;overlay.res_y=height;overlay.data=data;overlay:update();previous_data=data
    stats.updates=stats.updates+1
  end
  -- Expired layout entries cannot accumulate for a full feature film.
  for i in pairs(layout) do if i<first then layout[i]=nil end end
  stats.draw_seconds=stats.draw_seconds+mp.get_time()-begin
end
local function target_interval()
  local hz=number('display-fps',60)
  if not hz or hz~=hz or hz<30 then hz=60 end
  return 1/math.max(30,math.min(120,hz))
end
local function stop_timer()
  if timer then timer:kill();timer=nil end
  deadline=nil
end
local schedule
local function tick()
  timer=nil
  if frozen() or not enabled then deadline=nil;draw();return end
  draw()
  if not schedule then return end
  local now=mp.get_time()
  deadline=(deadline or now)+interval
  if deadline<now then deadline=now+interval end
  schedule()
end
schedule=function()
  if frozen() or not enabled then stop_timer();return end
  local now=mp.get_time();deadline=deadline or now+interval
  timer=mp.add_timeout(math.max(.001,deadline-now),tick)
end
local function refresh(reset)
  stop_timer();interval=target_interval()
  if reset then reset_clock() end
  draw();schedule()
end
mp.register_script_message('configure',function(path)
  local ok,err=pcall(function()
    local cfg=read_json(path)
    if cfg.comments~=loaded_path or cfg.token~=nonce then
      local input=read_json(cfg.comments).data or {};items={}
      for i,item in ipairs(input) do
        local fixed=item.type=='top' or item.type=='bottom'
        items[i]={time=item.time,text=item.text,type=item.type,status=item.status,order=i,fixed=fixed,
          duration=fixed and 5 or 10,bgr=color(item.color),escaped=escape(item.text),
          characters=select(2,tostring(item.text):gsub('[^\128-\193]',''))}
      end
      table.sort(items,function(a,b)if a.time==b.time then return a.order<b.order end;return a.time<b.time end)
      loaded_path=cfg.comments;stats.loads=stats.loads+1;reset_clock()
    end
    nonce=cfg.token;media=cfg.media;settings=cfg.settings or {};enabled=settings.danmaku_enabled~=false
    layout={};refresh(false);publish()
  end)
  if not ok then stop_timer();mp.set_property_native('user-data/bdm-native',{token=nonce,error=tostring(err)});clear() end
end)
mp.add_key_binding('Ctrl+Shift+d','bdm-toggle',function()enabled=not enabled;refresh(true);publish()end)
for _,name in ipairs({'pause','paused-for-cache','seeking','core-idle','speed'}) do
  mp.observe_property(name,'native',function()refresh(true)end)
end
mp.observe_property('display-fps','number',function()refresh(false)end)
mp.observe_property('osd-dimensions','native',function()refresh(false)end)
mp.observe_property('path','string',function()refresh(true)end)
mp.register_event('seek',function()refresh(true)end)
mp.register_event('playback-restart',function()refresh(true)end)
mp.register_event('shutdown',function()stop_timer();overlay:remove()end)
mp.add_periodic_timer(1,function()if nonce~='' then publish() end end)

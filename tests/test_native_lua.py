from pathlib import Path
import shutil,subprocess
import pytest

# Real Lua with independent, quantized video PTS and a deterministic wall clock.
HARNESS=r'''
local script,config,comments=arg[1],arg[2],arg[3]
local now,base,pos=0,0,1
local messages,keys,observed,events,props,timers={},{},{},{},{},{}
props.pause=true;props.speed=1;props['container-fps']=24;props['display-fps']=60;props.path='video'
local frames,reads={},0
local config_token='test'
local settings={danmaku_enabled=true,danmaku_offset=0,font_size=24,opacity=1,show_estimated=false}
local data={{time=0,text='red',type='top',color='#ff0000'},{time=0,text='blue',type='bottom',color='#a0ea'},
 {time=0,text='hidden',status='estimated'},{time=0,text='{\\pos(1,1)}',type='scroll'}}
local function raw()
 if props.pause or props['paused-for-cache'] or props.seeking or props.stall then return pos end
 return math.floor((pos+(now-base)*props.speed)*props['container-fps']+1e-6)/props['container-fps']
end
local function timer(delay,fn,period)
 local t={due=now+delay,fn=fn,period=period,live=true}
 function t:kill()self.live=false;self.cancelled=true end
 function t:stop()self.live=false;self.cancelled=true end
 timers[#timers+1]=t;return t
end
local function advance(seconds)
 local target=now+seconds;local count=0
 while true do
  local next=nil
  for _,t in ipairs(timers) do if t.live and t.due<=target+1e-9 and (not next or t.due<next.due) then next=t end end
  if not next then break end
  now=next.due;next.live=false;next.fn();count=count+1;assert(count<20000)
  if next.period and not next.cancelled then next.live=true;next.due=now+next.period end
 end
 now=target
end
local function set(name,value)
 if name=='pause' or name=='speed' or name=='paused-for-cache' then pos=raw();base=now end
 props[name]=value
 if observed[name] then observed[name](name,value) end
end
local function seek(value)
 pos=value;base=now
 if events.seek then events.seek() end
 if events['playback-restart'] then events['playback-restart']() end
end
local overlay={remove=function(self)self.data='' end,update=function(self)frames[#frames+1]={wall=now,data=self.data,raw=raw()} end}
local mp={get_time=function()return now end,create_osd_overlay=function()return overlay end,
 get_property_number=function(k,d)if k=='time-pos' then return raw() end;return props[k] or d end,
 get_property_native=function(k,d)if props[k]~=nil then return props[k] end;return d end,
 get_property=function(k)return props[k] end,get_osd_size=function()return 1920,1080 end,
 set_property_native=function(k,v)props[k]=v end,register_script_message=function(k,f)messages[k]=f end,
 add_key_binding=function(k,n,f)keys[k]=f end,observe_property=function(k,t,f)observed[k]=f end,
 register_event=function(k,f)events[k]=f end,add_timeout=function(n,f)return timer(n,f)end,
 add_periodic_timer=function(n,f)return timer(n,f,n)end}
local utils={parse_json=function(s)if s=='CONFIG' then return {token=config_token,comments=comments,media='video',settings=settings} end
 reads=reads+1;return {data=data} end,format_json=function()return ''end}
package.preload['mp']=function()return mp end;package.preload['mp.utils']=function()return utils end;package.preload['mp.assdraw']=function()return {}end
assert(loadfile(script))();messages.configure(config)
'''


def run_lua(tmp_path,body,script=None):
    lua=shutil.which('lua')
    if not lua:pytest.skip('optional Lua interpreter unavailable')
    config=tmp_path/'config';config.write_text('CONFIG');comments=tmp_path/'comments';comments.write_text('COMMENTS')
    harness=tmp_path/'test.lua';harness.write_text(HARNESS+body)
    script=script or Path(__file__).parents[1]/'bili_danmaku_mapper/workbench/native_danmaku.lua'
    r=subprocess.run([lua,str(harness),str(script),str(config),str(comments)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    return r.stdout


def test_lua_overlay_clock_colors_reservation_and_safe_text(tmp_path):
    run_lua(tmp_path,r'''
assert(props['user-data/bdm-native'].count==4);assert(overlay.data:find('1c&H0000ff&',1,true));assert(overlay.data:find('1c&Heaa000&',1,true))
assert(not overlay.data:find('hidden',1,true));assert(overlay.data:find('\\{',1,true))
local before=overlay.data;local updates=#frames;advance(1);assert(before==overlay.data and updates==#frames,'paused animation must be idle')
settings.danmaku_offset=2;messages.configure(config);assert(overlay.data=='','positive delay cannot show future comments')
settings.danmaku_offset=0;seek(11);messages.configure(config);assert(overlay.data=='','expired comments clear after seek')
seek(1);messages.configure(config);keys['Ctrl+Shift+d']();assert(overlay.data=='');assert(not props['user-data/bdm-native'].enabled)
keys['Ctrl+Shift+d']();assert(overlay.data~='');assert(overlay.res_y==1080)
settings.font_size=12;messages.configure(config);assert(overlay.data:find('\\fs28.80',1,true));assert(reads==1)
settings.opacity=.5;messages.configure(config);assert(overlay.data:find('\\alpha&H80&',1,true));assert(reads==1)
props.path='another movie';observed.path();assert(overlay.data=='','never draw on another movie')
''')


@pytest.mark.parametrize('fps,hz,speed',[(24,60,1),(30,60,1),(24,120,1.5),(24,60,2)])
def test_lua_smooth_clock_quantized_pts_speed_and_bounded_error(tmp_path,fps,hz,speed):
    output=run_lua(tmp_path,f'''
props['container-fps']={fps};props['display-fps']={hz};props.speed={speed};observed['display-fps']();
data={{{{time=0,text='scroll',type='scroll'}}}};config_token='new-set'
-- A fresh token requests a new immutable comment set.
messages.configure(config)
frames={{}};set('pause',false);advance(1)
assert(#frames>={hz}*.85,'animation must not be limited to source FPS')
local previous=nil;local max_error=0
for _,frame in ipairs(frames) do
 local x=tonumber(frame.data:match('\\\\pos%(([%d%.%-]+),'))
 if x then
  if previous then assert(x<=previous+.01,'scroll cannot move backward') end;previous=x
  local predicted=(1920-x)/(1920+6*57.6)*10
  local real=1+frame.wall*{speed}
  max_error=math.max(max_error,math.abs(predicted-real))
 end
end
assert(max_error<=1/{fps}+.002,'clock exceeded one source frame: '..max_error)
local n=#frames;set('pause',true);local held=overlay.data;advance(1);assert(held==overlay.data,'paused text cannot drift')
set('pause',false);advance(.2);set('paused-for-cache',true);held=overlay.data;advance(1);assert(held==overlay.data,'buffering must freeze animation')
set('paused-for-cache',false);set('speed',.5);advance(.3);seek(1);local after=overlay.data;advance(.1);assert(after~=overlay.data,'seek must restart clock')
pos=raw();base=now;props.stall=true;advance(.2);held=overlay.data;advance(.5);assert(held==overlay.data,'stalled source cannot drift')
print('frames='..n..' max_error='..max_error)
''')
    assert 'frames=' in output


def test_lua_settings_keep_comments_cached_and_disabled_idle(tmp_path):
    run_lua(tmp_path,r'''
set('pause',false);advance(.2)
for size=12,24 do settings.font_size=size;messages.configure(config) end
assert(reads==1,'settings must not reread all comments')
local counters=props['user-data/bdm-native'].animation
assert(counters.comment_loads==1 and counters.interpolated>0)
keys['Ctrl+Shift+d']();local updates=#frames;advance(2);assert(#frames==updates,'disabled overlay must stop animating')
''')

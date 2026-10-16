#!/usr/bin/env python3
"""Standard-library bootstrap: no project imports before dependencies exist."""
import argparse
import fcntl
import hmac
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
os.environ['PATH'] = str(ROOT/'tools'/'ffmpeg'/'bin')+os.pathsep+os.environ.get('PATH','')
RUNTIME = ROOT/'data'/'workbench'
STATE = RUNTIME/'instance.json'
REQUIRED = ('fastapi','uvicorn','httpx','bilibili_api','numpy','scipy','imagehash','PIL','dotenv','qrcode','xmltodict')


def write_state(port, token, mode):
    temporary = STATE.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
    with os.fdopen(fd,'w') as f:
        json.dump({'pid':os.getpid(),'port':port,'token':token,'mode':mode},f)
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,STATE)


def request_alive(state):
    try:
        request=urllib.request.Request(f"http://127.0.0.1:{state['port']}/api/status", headers={'Authorization':'Bearer '+state['token']})
        with urllib.request.urlopen(request,timeout=1) as response:
            return response.status==200
    except Exception:
        return False


def supported_python(executable):
    try:
        return subprocess.run([str(executable),'-c','import sys;sys.exit(sys.version_info < (3,11))'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10).returncode==0
    except (OSError,subprocess.SubprocessError):
        return False


def dependencies():
    missing=[]
    for name in REQUIRED:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError,ValueError):
            missing.append(name)
    return missing


def start():
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(RUNTIME/'launcher.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            state=json.loads(STATE.read_text())
        except (OSError,ValueError):
            state=None
        if state and request_alive(state):
            webbrowser.open(f"http://127.0.0.1:{state['port']}/#{state['token']}")
            return
        token=secrets.token_urlsafe(32)
        python=ROOT/'.venv'/'bin'/'python'
        executable=str(python) if python.exists() and supported_python(python) else sys.executable
        log=RUNTIME/'service.log'
        fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as output:
            process=subprocess.Popen([executable,str(ROOT/'launcher.py'),'--serve','--token',token],cwd=ROOT,
                stdout=output,stderr=output,start_new_session=True)
        for _ in range(200):
            try:
                state=json.loads(STATE.read_text())
                if state['pid']==process.pid and request_alive(state):
                    webbrowser.open(f"http://127.0.0.1:{state['port']}/#{token}")
                    return
            except (OSError,ValueError):
                pass
            if process.poll() is not None:
                raise RuntimeError('服务启动失败，请查看 data/workbench/service.log')
            time.sleep(.1)
        raise RuntimeError('服务启动超时，请查看 data/workbench/service.log')


def bootstrap(sock,token,missing):
    state={'missing':missing,'installing':False,'error':None}
    port=sock.getsockname()[1]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization','').removeprefix('Bearer '),token)
        def send(self,status,value,kind='application/json'):
            data=value.encode() if isinstance(value,str) else json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Referrer-Policy','no-referrer');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/':
                self.send(200,BOOTSTRAP_HTML,'text/html; charset=utf-8')
            elif self.path=='/api/status' and self.authorized():
                if state['missing']==['FFmpeg'] and shutil.which('ffmpeg') and shutil.which('ffprobe'):
                    state['missing']=[]
                    threading.Thread(target=server.shutdown,daemon=True).start()
                self.send(200,state)
            else:self.send(401,{'detail':'请通过启动脚本打开'})
        def do_POST(self):
            if not self.authorized() or self.headers.get('Origin')!=f'http://127.0.0.1:{port}':
                return self.send(403,{})
            if self.path=='/api/install' and state['missing'] and not state['installing']:
                state['installing']=True
                threading.Thread(target=install,daemon=True).start()
                self.send(200,{'ok':True})
            elif self.path=='/api/exit':
                self.send(200,{'ok':True});threading.Thread(target=server.shutdown,daemon=True).start()
            else:self.send(400,{'detail':'请先完成安装指引'})
    def install():
        python=ROOT/'.venv'/'bin'/'python'
        try:
            if python.exists() and not supported_python(python):
                state["error"]="现有 .venv 的 Python 版本过旧。请将它改名备份，再用 Python 3.11+ 重新启动。"
                state["installing"]=False
                return
            if not python.exists():
                subprocess.run([sys.executable,'-m','venv',str(ROOT/'.venv')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements.txt')],cwd=ROOT,
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            state['missing']=[]
        except (OSError,subprocess.SubprocessError):
            state['error']='安装失败，请检查网络后重试。也可在项目目录运行 .venv/bin/python -m pip install -r requirements.txt。'
            state['installing']=False
            return
        state['installing']=False
        server.shutdown()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler,bind_and_activate=False)
    server.socket.close();server.socket=sock;server.server_address=sock.getsockname()
    server.serve_forever(poll_interval=.2)
    server.server_close()
    if not state['missing'] and not state['error']:
        python=ROOT/'.venv'/'bin'/'python'
        if not python.exists():
            python=Path(sys.executable)
        os.execv(str(python),[str(python),str(ROOT/'launcher.py'),'--serve','--token',token,'--port',str(port)])


BOOTSTRAP_HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>弹幕工作台 · 首次启动</title><style>body{font:16px -apple-system,sans-serif;background:#f3f5f0;color:#17292b;max-width:650px;margin:10vh auto;padding:30px}button{padding:12px 24px;background:#126a59;color:white;border:0;border-radius:8px;cursor:pointer}p{line-height:1.8}a{color:#126a59}</style><h1>准备本地工作台</h1><p id="status">正在检查环境…</p><button id="install" hidden>确认安装项目依赖</button><p>安装只写入本项目的 .venv。需要 Python 3.11+ 与 FFmpeg，首次安装需要网络。</p><details><summary>图形安装步骤</summary><p>Python：打开下方官网，下载 macOS 安装包，双击并按安装向导完成。</p><p>FFmpeg：打开下方官网的 macOS 下载链接，下载并解压 ffmpeg 与 ffprobe 两个可执行文件。在项目中建立 tools/ffmpeg/bin 文件夹，将两个文件放入，再双击启动脚本。已安装 Homebrew 的用户也可通过 brew install ffmpeg 安装。</p></details><p><a href="https://www.python.org/downloads/macos/" target="_blank" rel="noreferrer">Python macOS 安装页</a> · <a href="https://ffmpeg.org/download.html" target="_blank" rel="noreferrer">FFmpeg 安装页</a></p><button id="exit">停止并退出</button><script>const token=location.hash.slice(1);history.replaceState(null,'','/');const status=document.getElementById('status');const install=document.getElementById('install');const request=(p,post=false)=>fetch('/api/'+p,{method:post?'POST':'GET',headers:{Authorization:'Bearer '+token}}).then(r=>{if(!r.ok)throw Error('会话失效，请重新双击启动');return r.json()});install.onclick=()=>{install.disabled=true;request('install',true).catch(e=>status.textContent=e.message)};document.getElementById('exit').onclick=()=>request('exit',true).then(()=>{clearInterval(timer);status.textContent='服务已停止'});const timer=setInterval(()=>request('status').then(s=>{if(!('missing' in s)){location.href='/#'+token;return;}status.textContent=s.error||(s.installing?'正在安装，请稍候…':s.missing.length?'缺少：'+s.missing.join('、'):'正在进入工作台…');install.hidden=!s.missing.some(x=>x!=='FFmpeg');install.disabled=s.installing;}).catch(e=>status.textContent=e.message),1500);</script></html>'''


def serve(token,port=0):
    RUNTIME.mkdir(parents=True,exist_ok=True,mode=0o700)
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('127.0.0.1',port));sock.listen(128);sock.set_inheritable(False)
    port=sock.getsockname()[1]
    missing=dependencies()
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        missing.append('FFmpeg')
    try:
        write_state(port,token,'bootstrap' if missing else 'workbench')
        if missing:
            bootstrap(sock,token,missing)
        else:
            import uvicorn
            from bili_danmaku_mapper.workbench.server import create_app
            server=None
            app=create_app(token=token,exit_callback=lambda:setattr(server,'should_exit',True))
            server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,access_log=False,log_level='warning',timeout_graceful_shutdown=5))
            server.run(sockets=[sock])
    finally:
        sock.close()
        try:
            if json.loads(STATE.read_text())['pid']==os.getpid():
                STATE.unlink()
        except (OSError,ValueError):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');parser.add_argument('--token');parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args()
    if args.serve:serve(args.token or secrets.token_urlsafe(32),args.port)
    else:start()

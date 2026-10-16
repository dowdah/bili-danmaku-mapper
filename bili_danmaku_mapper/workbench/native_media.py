"""Private original-byte Range reader; outlives the workbench, not its player."""
import asyncio,json,os,secrets,socket,sys,time
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import FileResponse,JSONResponse
from ..cache import stamp
from ..io import atomic_json


def run(config_path):
    path=Path(config_path);config=json.loads(path.read_text());source=Path(config['source']);token=config['token'];server=None
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128);port=sock.getsockname()[1]
    @asynccontextmanager
    async def lifecycle(app):
        async def watch():
            started=time.monotonic()
            while True:
                try:
                    record=json.loads((path.parent/'session.json').read_text());pid=record.get('pid')
                    if pid:os.kill(pid,0)
                    elif time.monotonic()-started>30:break
                    if record.get('status') in ('closed','failed'):break
                except ProcessLookupError:break
                except (OSError,ValueError):
                    if time.monotonic()-started>30:break
                await asyncio.sleep(2)
            server.should_exit=True
        task=asyncio.create_task(watch())
        try:yield
        finally:task.cancel();await asyncio.gather(task,return_exceptions=True)
    app=FastAPI(lifespan=lifecycle,docs_url=None,redoc_url=None,openapi_url=None)
    @app.middleware('http')
    async def guard(request,call_next):
        if request.headers.get('host')!=f'127.0.0.1:{port}' or request.headers.get('origin') not in (None,f'http://127.0.0.1:{port}'):
            return JSONResponse({'detail':'Invalid local host'},403)
        if request.url.path not in ('/'+token+'/media','/'+token+'/stop'):
            return JSONResponse({'detail':'Not found'},404)
        response=await call_next(request);response.headers['Cache-Control']='no-store';response.headers['Referrer-Policy']='no-referrer';return response
    @app.api_route('/'+token+'/media',methods=['GET','HEAD'])
    async def media():
        if list(stamp(source))[:4]!=config['identity'][:4]:return JSONResponse({'detail':'Original media changed'},409)
        return FileResponse(source,media_type='application/octet-stream',filename=source.name,content_disposition_type='inline')
    @app.post('/'+token+'/stop')
    async def stop():
        asyncio.get_running_loop().call_later(.1,lambda:setattr(server,'should_exit',True));return {'stopped':True}
    atomic_json(path.parent/'media-ready.json',{'port':port,'pid':os.getpid()},True)
    server=uvicorn.Server(uvicorn.Config(app,access_log=False,log_level='error'));server.run(sockets=[sock])


if __name__=='__main__':run(sys.argv[1])

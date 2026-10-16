"""Socket acceptance, separate from the offline pytest suite.

Run: .venv/bin/python -m tests.verify_service_exit
An SSE reader remains connected until the service actively closes it.
"""
import subprocess
import sys
import urllib.request


def main():
    process=subprocess.Popen([sys.executable,'-m','tests.workbench_preview'],stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL,text=True)
    try:
        address=process.stdout.readline().strip().split('#')[0]
        headers={'Authorization':'Bearer browser-acceptance-only'}
        stream=urllib.request.urlopen(urllib.request.Request(address+'api/events',headers=headers),timeout=8)
        assert stream.readline().decode().startswith('event: snapshot')
        request=urllib.request.Request(address+'api/exit',data=b'{}',headers=headers)
        with urllib.request.urlopen(request,timeout=8) as result:
            assert result.status==200
        tail=stream.read().decode()
        assert 'event: shutdown' in tail
        stream.close()
        assert process.wait(timeout=8)==0
        print('Connected SSE received shutdown; owned service process exited cleanly')
    finally:
        if process.poll() is None:
            process.kill();process.wait()


if __name__=='__main__':main()

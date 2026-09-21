"""Own Java's stdin and stop/reap it on controller EOF, exit, or SIGTERM."""
import os, selectors, signal, subprocess, sys, time

def main():
    child=subprocess.Popen(sys.argv[1:],stdin=subprocess.PIPE,stdout=sys.stdout,stderr=sys.stderr,bufsize=0)
    stopping=False
    def request_stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    selector=selectors.DefaultSelector();selector.register(sys.stdin,selectors.EVENT_READ)
    try:
        while child.poll() is None and not stopping:
            if selector.select(.2):
                data=os.read(sys.stdin.fileno(),65536)
                if not data:break
                child.stdin.write(data);child.stdin.flush()
        if child.poll() is None:
            try:child.stdin.write(b'save-all flush\nstop\n');child.stdin.flush()
            except BrokenPipeError:pass
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=4)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=4)
        return child.returncode or 0
    finally:
        selector.close()
        if child.poll() is None:child.kill();child.wait()
if __name__=='__main__':sys.exit(main())

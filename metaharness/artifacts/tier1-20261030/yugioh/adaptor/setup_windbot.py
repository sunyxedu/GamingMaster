"""Build the pinned WindBot library and pipe bridge. No gameplay downloads."""
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parent

def build():
    if not shutil.which('mcs') or not shutil.which('mono'):
        raise SystemExit('Install the free Mono toolchain: sudo apt-get install mono-devel')
    out=ROOT/'runtime/windbot';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'runtime/vendor/windbot/ExecutorBase'
    files=sorted(str(p) for p in source.rglob('*.cs') if not {'obj','bin'}&set(p.relative_to(source).parts))
    subprocess.run(['mcs','-target:library','-optimize+','-langversion:7','-r:System.Data',
        '-r:System.Runtime.Serialization','-r:Mono.Data.Sqlite',f'-out:{out}/ExecutorBase.dll',*files],check=True,cwd=ROOT)
    subprocess.run(['mcs','-optimize+','-langversion:7','-r:System.Web.Extensions',
        f'-r:{out}/ExecutorBase.dll',f'-out:{out}/WindbotBridge.exe','WindbotBridge.cs','PoolExecutor.cs'],check=True,cwd=ROOT)
if __name__=='__main__':build()

"""Rebuild staged inputs offline; --download acquires only the frozen revisions."""
import argparse,hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPOS={
 'windbot':('https://github.com/ProjectIgnis/windbot','bffe6b62679c8b2fafea8f59740e03a132517da4'),
 'core':('https://github.com/edo9300/ygopro-core','6864373c24d7313c8e451df17877c12c4f713c46'),
 'scripts':('https://github.com/ProjectIgnis/CardScripts','2ddefd0115a122f8cd3f494b6affbcbc27b4f974'),
 'cdb':('https://github.com/ProjectIgnis/BabelCDB','2e6f985a5537d9b16ec7396787b5279d94af419d'),
 'edopro':('https://github.com/edo9300/edopro','d0f111d6f3e0d4ee529b88a869ae0a971954bae3')}
def run(*cmd):subprocess.run(cmd,cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
 for name,(url,commit) in REPOS.items():
  target=ROOT/'runtime/vendor'/name
  if not target.exists():
   if not args.download:raise SystemExit(f'Missing {target}; restore the staged bundle or use --download during setup')
   target.mkdir(parents=True);run('git','-C',str(target),'init');run('git','-C',str(target),'remote','add','origin',url)
   run('git','-C',str(target),'fetch','--depth','1','origin',commit);run('git','-C',str(target),'checkout','--detach','FETCH_HEAD')
  actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
  if actual!=commit:raise SystemExit(f'{name}: refusing unpinned revision {actual}')
 if not (ROOT/'runtime/vendor/core/lua/src/lua.h').exists():
  if not args.download:raise SystemExit('Pinned Lua submodule is missing')
  run('git','-C','runtime/vendor/core','submodule','update','--init','--depth','1')
 lua=subprocess.check_output(['git','-C',str(ROOT/'runtime/vendor/core/lua/src'),'rev-parse','HEAD'],text=True).strip()
 if lua!='6e22fedb74cf0c9b6656e9fce8b7331db847c605':raise SystemExit('Unpinned Lua revision')
 strings=ROOT/'runtime/strings.conf'
 if not strings.exists():
  if not args.download:raise SystemExit('Missing pinned strings.conf')
  url='https://raw.githubusercontent.com/ProjectIgnis/Distribution/54a6e2395c532648ff762540e9615319fac4f51b/config/strings.conf'
  urllib.request.urlretrieve(url,strings)
 # Reject changed pinned data even when rebuilding the platform-specific binaries.
 lock=json.loads((ROOT/'assets.lock.json').read_text())
 for name,expected in lock['files'].items():
  if name.startswith('runtime/vendor/') or name=='runtime/strings.conf':
   if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Pinned input changed: '+name)
 try:import mesonbuild
 except ImportError:raise SystemExit('Install requirements.txt into the evaluator venv (meson is build-only).')
 if not (ROOT/'runtime/build/build.ninja').exists():
  run(sys.executable,'-m','mesonbuild.mesonmain','setup','runtime/build','runtime/vendor/core','--buildtype=release','-Ddefault_library=shared')
 run('ninja','-C','runtime/build','-j','4')
 run('g++','-std=c++17','-O2','-shared','-fPIC','-Iruntime/vendor/core','bridge.cpp','-Lruntime/build','-locgcore','-Wl,-rpath,$ORIGIN/build','-o','runtime/libbridge.so')
 run(sys.executable,'setup_windbot.py')
 run(sys.executable,'prepare_assets.py')
 print('Offline native build ready. No downloader is used by reset/step.')
if __name__=='__main__':main()

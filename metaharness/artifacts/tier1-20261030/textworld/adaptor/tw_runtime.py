"""Bounded workers and versioned, integrity-checked local artifact storage."""
from __future__ import annotations
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import select
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime'
SUITE = 'generic-glulx-v1'
PROFILES = (
    dict(name='easy', rooms=3, objects=5, quest_length=2, cap=100),
    dict(name='medium', rooms=5, objects=10, quest_length=5, cap=200),
    dict(name='hard', rooms=10, objects=20, quest_length=10, cap=400),
)
GENERATION_TIMEOUT_S = 120
RPC_TIMEOUT_S = 20


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def configuration():
    import textworld
    if textworld.__version__ != '1.6.2':
        raise RuntimeError('This suite requires textworld==1.6.2')
    package = Path(textworld.__file__).resolve().parent
    # Include generator, grammar, logic, compiler and interpreter assets.
    files = sorted(p for p in package.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix != '.pyc')
    aggregate = hashlib.sha256()
    for path in files:
        aggregate.update(str(path.relative_to(package)).encode())
        aggregate.update(bytes.fromhex(digest(path)))
    return dict(suite=SUITE, textworld='1.6.2', python=platform.python_version(),
                packages={n: importlib.metadata.version(n) for n in
                          ('numpy', 'networkx', 'tatsu', 'hashids', 'more-itertools', 'cffi')},
                textworld_assets_sha256=aggregate.hexdigest(),
                format='ulx', parallel_quests=1, quest_breadth=1,
                generation_timeout_s=GENERATION_TIMEOUT_S,
                information_policy='text-and-admissible-v1', profiles=[dict(p) for p in PROFILES],
                pythonhashseed=0, inform_debug=False)


def worker_env(directory):
    env = os.environ.copy()
    # Freeze implicit compiler switches, runtime seed, and temporary-file location.
    env.pop('INFORM_HOME', None)
    env.pop('TEXTWORLD_I6_DEBUG', None)
    env.pop('TEXTWORLD_DEBUG', None)
    env['PYTHONHASHSEED'] = '0'
    temporary = directory / 'tmp'
    temporary.mkdir(exist_ok=True)
    env['TMPDIR'] = str(temporary)
    env['TEMP'] = str(temporary)
    env['TMP'] = str(temporary)
    return env


def stop_process(process):
    # Worker, compiler children and interpreter all belong to this process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def resolve(seed):
    config = configuration()
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    folder = RUNTIME / 'games' / SUITE / key / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = folder / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for name, expected in manifest['artifacts'].items():
                if digest(folder / name) != expected:
                    raise RuntimeError(f'Cached artifact checksum mismatch: {name}')
            if manifest['config'] != json.loads(json.dumps(config)) or manifest['root_seed'] != seed:
                raise RuntimeError('Cache manifest configuration mismatch')
            return folder / manifest['story'], manifest
        failures = []
        for retry in range(5):
            attempt = folder / f'attempt-{retry}'
            attempt.mkdir(exist_ok=True)
            job = dict(generation_seed=(seed + 1_000_003 * retry) % 2**32,
                       profile=PROFILES[seed % 3])
            (attempt / 'config.json').write_text(json.dumps(job))
            accepted = attempt / 'accepted.json'
            accepted.unlink(missing_ok=True)
            started = time.monotonic()
            with (attempt / 'generation.log').open('wb') as log:
                proc = subprocess.Popen([sys.executable, str(ROOT / 'tw_backend.py'),
                                         'generate', str(attempt)],
                                        cwd=attempt, env=worker_env(attempt),
                                        stdout=log, stderr=log, start_new_session=True)
                try:
                    code = proc.wait(timeout=GENERATION_TIMEOUT_S)
                    if code != 0 or not accepted.exists():
                        raise RuntimeError(f'Generator exited {code}; see {attempt / "generation.log"}')
                except Exception as exc:
                    failures.append(dict(retry=retry, generation_seed=job['generation_seed'],
                                         reason=f'{type(exc).__name__}: {exc}'))
                    (folder / 'failures.json').write_text(json.dumps(failures, indent=2))
                    continue
                finally:
                    stop_process(proc)
            artifacts = {str(p.relative_to(folder)): digest(p) for p in
                         (attempt / 'story.ulx', attempt / 'story.json', attempt / 'story.ni')}
            manifest = dict(config=config, root_seed=seed, retry=retry,
                            generation_seed=job['generation_seed'], profile=job['profile'],
                            story=str((attempt / 'story.ulx').relative_to(folder)),
                            artifacts=artifacts, acceptance=json.loads(accepted.read_text()),
                            failures=failures, generation_seconds=time.monotonic() - started)
            pending = folder / 'manifest.pending'
            pending.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            pending.replace(manifest_path)
            return folder / manifest['story'], manifest
        raise RuntimeError(f'All five deterministic generation attempts failed; see {folder}')


class Interpreter:
    def __init__(self, path):
        sessions = RUNTIME / 'sessions'
        sessions.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='episode-', dir=sessions))
        self.log = (self.directory / 'interpreter.log').open('wb')
        self.process = None
        self.buffer = b''
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(ROOT / 'tw_backend.py'), 'serve', str(path)],
                cwd=self.directory, env=worker_env(self.directory),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                start_new_session=True)
        except Exception:
            self.log.close()
            raise

    def request(self, op, **kwargs):
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError('TextWorld worker is not running')
        process.stdin.write((json.dumps(dict(op=op, **kwargs)) + '\n').encode())
        process.stdin.flush()
        deadline = time.monotonic() + RPC_TIMEOUT_S
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f'TextWorld {op} exceeded {RPC_TIMEOUT_S}s')
            data = os.read(process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f'TextWorld worker exited; see {self.directory / "interpreter.log"}')
            self.buffer += data
            if len(self.buffer) > 200_000:
                raise ValueError('TextWorld observation exceeds 200 kB')
        line, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(line)

    def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            if process.poll() is None:
                try:
                    process.stdin.write(b'{"op":"close"}\n')
                    process.stdin.flush()
                    process.wait(timeout=2)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    pass
        finally:
            stop_process(process)
            process.stdin.close()
            process.stdout.close()
            self.log.close()

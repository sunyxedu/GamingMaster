#!/usr/bin/env python3
"""Install pinned runtime artifacts. Run with Python -I; downloaded data is isolated."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runtime'
VERSION = '1.21.1'
SERVER_SHA1 = '59353fb40c36d304f2035d51e7d6e6baa98dc05c'
SERVER_SHA256 = 'e3bc55693e93cda0188f2e60aea28113fc647c5e85a15fa3d1b347349231b4bb'


def download(url: str, filename: str) -> Path:
    """Every downloaded file gets a new empty directory; never execute there."""
    destination = Path(tempfile.mkdtemp(prefix='download-', dir=RUN / 'downloads')) / filename
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '20',
                    '--max-time', '180', '--output', str(destination), url], check=True, cwd=ROOT)
    return destination


def checked_jar(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open('rb') as stream:
        sha1 = hashlib.file_digest(stream, 'sha1').hexdigest()
    with path.open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    return sha1 == SERVER_SHA1 and sha256 == SERVER_SHA256


def main() -> None:
    (RUN / 'downloads').mkdir(parents=True, exist_ok=True)
    (RUN / 'tmp').mkdir(exist_ok=True)
    java = os.environ.get('JAVA_BIN') or shutil.which('java')
    if not java:
        raise RuntimeError('Java 21 is required; set JAVA_BIN to its executable')
    java = str(Path(shutil.which(java) or java).resolve())
    check = subprocess.run([java, '-version'], capture_output=True, text=True, check=True, cwd=ROOT)
    if 'version "21.' not in check.stderr:
        raise RuntimeError('JAVA_BIN must select Java 21')
    node = os.environ.get('NODE_BIN', 'node')
    node_version = subprocess.check_output([node, '--version'], text=True, cwd=ROOT).strip()
    if int(node_version.lstrip('v').split('.')[0]) < 22:
        raise RuntimeError('Pinned Mineflayer needs Node 22 or newer; Node 24 is tested')
    existing = {}
    if (RUN / 'artifacts.json').exists():
        existing = json.loads((RUN / 'artifacts.json').read_text())
    supplied = os.environ.get('MC_SERVER_JAR') or existing.get('server_jar')
    jar = Path(supplied) if supplied else None
    if jar is not None:
        jar = (jar if jar.is_absolute() else ROOT / jar).resolve()
    if jar is not None and not jar.is_relative_to(RUN):
        raise ValueError('MC_SERVER_JAR must be inside adaptor/runtime')
    if jar is None or not checked_jar(jar):
        if os.environ.get('MC_SERVER_JAR'):
            raise RuntimeError('MC_SERVER_JAR is not the pinned official 1.21.1 JAR')
        manifest_path = download('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json', 'manifest.json')
        manifest = json.loads(manifest_path.read_text())
        version_url = next(v['url'] for v in manifest['versions'] if v['id'] == VERSION)
        metadata_path = download(version_url, 'version.json')
        metadata = json.loads(metadata_path.read_text())
        server = metadata['downloads']['server']
        if server['sha1'] != SERVER_SHA1:
            raise RuntimeError('Official metadata disagrees with the pinned checksum')
        jar = download(server['url'], f'server-{VERSION}.jar')
        if not checked_jar(jar):
            raise RuntimeError('Downloaded server checksum mismatch')
    env = dict(os.environ, TMPDIR=str(RUN / 'tmp'))
    subprocess.run(['npm', 'ci', '--prefix', str(ROOT / 'bridge'), '--cache', str(RUN / 'npm-cache'),
                    '--ignore-scripts', '--no-audit', '--no-fund'], check=True, cwd=ROOT, env=env)
    result = {'java': java, 'server_jar': str(jar.relative_to(ROOT)), 'version': VERSION,
              'server_sha1': SERVER_SHA1, 'server_sha256': SERVER_SHA256,
              'java_version': check.stderr, 'node_version': node_version}
    (RUN / 'artifacts.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('Setup complete. The requested local server setup writes eula=true; Minecraft EULA applies:')
    print('https://www.minecraft.net/en-us/eula')


if __name__ == '__main__':
    main()

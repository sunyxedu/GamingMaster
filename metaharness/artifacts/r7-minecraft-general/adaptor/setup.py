#!/usr/bin/env python3
"""Download owned runtime artifacts, verify hashes, and install locked Node dependencies."""
import hashlib, json, os, pathlib, platform, subprocess, tarfile, urllib.request
ROOT = pathlib.Path(__file__).resolve().parent
RUN = pathlib.Path(os.environ.get('MC_RUN_DIR', ROOT / 'runtime')).resolve()

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl', '-L', '--fail', '--retry', '3', '--connect-timeout', '20', '-o', str(path), url], check=True)
    return path

def metadata(url, path):
    fetch(url, path)
    return json.loads(path.read_text())

def main():
    dl = RUN / 'downloads'
    manifest = metadata('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json', dl/'version-manifest.json')
    url = next(v['url'] for v in manifest['versions'] if v['id'] == '1.21.1')
    version = metadata(url, dl/'1.21.1.json')
    server = version['downloads']['server']
    jar = pathlib.Path(os.environ.get('MC_SERVER_JAR', dl/'server-1.21.1.jar')).resolve()
    if not jar.exists(): fetch(server['url'], jar)
    if hashlib.sha1(jar.read_bytes()).hexdigest() != server['sha1']:
        raise RuntimeError('Minecraft server SHA-1 does not match official metadata')
    java = os.environ.get('JAVA_BIN')
    java_meta = None
    if not java:
        matches = list((RUN/'jdk-21').glob('**/bin/java'))
        if not matches:
            arch = {'arm64':'aarch64', 'aarch64':'aarch64', 'x86_64':'x64'}[platform.machine()]
            system = {'Darwin':'mac', 'Linux':'linux'}[platform.system()]
            java_meta = metadata(f'https://api.adoptium.net/v3/assets/latest/21/hotspot?architecture={arch}&image_type=jdk&os={system}', dl/'java-metadata.json')[0]
            pkg = java_meta['binary']['package']
            archive = dl/pkg['name']
            if not archive.exists(): fetch(pkg['link'], archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != pkg['checksum']:
                raise RuntimeError('Java SHA-256 mismatch')
            (RUN/'jdk-21').mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tf: tf.extractall(RUN/'jdk-21', filter='data')
            matches = list((RUN/'jdk-21').glob('**/bin/java'))
        java = str(matches[0])
    check = subprocess.run([java, '-version'], capture_output=True, text=True, check=True)
    if 'version "21.' not in check.stderr: raise RuntimeError('JAVA_BIN must be Java 21')
    subprocess.run(['npm', 'ci' if (ROOT/'bridge/package-lock.json').exists() else 'install', '--prefix', str(ROOT/'bridge'), '--cache', str(RUN/'npm-cache'), '--ignore-scripts', '--no-audit', '--no-fund'], check=True)
    result = {'java':str(pathlib.Path(java).resolve()), 'server_jar':str(jar), 'version':'1.21.1', 'server_sha1':server['sha1'], 'server_sha256':hashlib.sha256(jar.read_bytes()).hexdigest(), 'java_version':check.stderr}
    (RUN/'artifacts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()

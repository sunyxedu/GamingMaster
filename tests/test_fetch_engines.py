"""scripts/engines/fetch_engines.sh: syntax, CLI and host detection (no network, no installs).

The script is sourced with a simulated host (OS / ARCH / CPU flags) to check which
builds it would pick — e.g. for the AMD EPYC 9355 (Zen 5, AVX-512 VNNI) labelling nodes.
"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "engines", "fetch_engines.sh")
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not installed")

# /proc/cpuinfo flags of an EPYC 9355 (abridged to the ones the script looks at)
EPYC_9355 = ("fpu sse sse2 ssse3 sse4_1 sse4_2 popcnt avx f16c fma bmi1 avx2 bmi2 avx512f avx512dq "
             "avx512ifma avx512cd avx512bw avx512vl avx512_bf16 avx512vbmi avx512_vbmi2 gfni vaes "
             "vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq avx_vnni")
OLD_X86 = "fpu sse sse2 ssse3 sse4_1 sse4_2 popcnt"


def _host(snippet, tmp_path, os_="Linux", arch="x86_64", flags=""):
    cmd = f'source "{SCRIPT}"; OS={os_}; ARCH={arch}; FLAGS=" {flags} "; {snippet}'
    env = {**os.environ, "GM_ENGINES": str(tmp_path)}
    env.pop("KATAGO_BACKEND", None)
    env.pop("STOCKFISH_ARCH", None)
    env.pop("YANEURAOU_TARGET_CPU", None)
    r = subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_syntax_and_usage(tmp_path):
    subprocess.run([BASH, "-n", SCRIPT], check=True)
    r = subprocess.run([BASH, SCRIPT], capture_output=True, text=True, env={**os.environ, "GM_ENGINES": str(tmp_path)})
    assert r.returncode == 2 and "fetch_engines.sh stockfish" in r.stdout
    r = subprocess.run([BASH, SCRIPT, "nonsense"], capture_output=True, text=True,
                       env={**os.environ, "GM_ENGINES": str(tmp_path)})
    assert r.returncode != 0 and "unknown engine" in r.stderr
    r = subprocess.run([BASH, SCRIPT, "paths"], capture_output=True, text=True, check=True,
                       env={**os.environ, "GM_ENGINES": str(tmp_path / "empty")})
    assert "stockfish: (not installed)" in r.stdout and "katago: (not installed)" in r.stdout


def test_epyc_9355_picks(tmp_path):
    cands = _host("stockfish_candidates sf_19", tmp_path, flags=EPYC_9355)
    assert cands[0] == "stockfish-linux-x86-64-universal.tar.gz"
    assert cands[1:4] == ["stockfish-ubuntu-x86-64-avx512icl.tar", "stockfish-ubuntu-x86-64-vnni512.tar",
                          "stockfish-ubuntu-x86-64-avx512.tar"]
    assert cands[-1] == "stockfish-ubuntu-x86-64.tar"
    assert _host("yaneuraou_target_cpu", tmp_path, flags=EPYC_9355) == ["AVX512VNNI"]
    if shutil.which("nvidia-smi") is None:
        assert _host("katago_backend", tmp_path, flags=EPYC_9355) == ["eigenavx2"]


def test_old_x86_and_macos_picks(tmp_path):
    cands = _host("stockfish_candidates sf_17.1", tmp_path, flags=OLD_X86)
    assert cands == ["stockfish-linux-x86-64-universal.tar.gz", "stockfish-ubuntu-x86-64-sse41-popcnt.tar",
                     "stockfish-ubuntu-x86-64.tar"]
    assert _host("yaneuraou_target_cpu", tmp_path, flags=OLD_X86) == ["SSE42"]
    if shutil.which("nvidia-smi") is None:
        assert _host("katago_backend", tmp_path, flags=OLD_X86) == ["eigen"]
    mac = _host("stockfish_candidates sf_19; yaneuraou_target_cpu", tmp_path, os_="Darwin", arch="arm64")
    assert mac == ["stockfish-macos-universal.tar.gz", "stockfish-macos-m1-apple-silicon.tar", "APPLEM1"]
    forced = _host("STOCKFISH_ARCH=x86-64-avx2; stockfish_x86_arches", tmp_path, flags=EPYC_9355)
    assert forced == ["avx2"]


def test_egaroucid_build_flags(tmp_path):
    """AVX-512 + AMD tuning on the EPYC 9355 nodes; generic builds on ARM and pre-AVX2 CPUs."""
    assert _host("VENDOR=AuthenticAMD; egaroucid_defs", tmp_path, flags=EPYC_9355) == ["-DHAS_AVX512", "-DHAS_AMD_PROCESSOR"]
    assert _host("VENDOR=GenuineIntel; egaroucid_defs", tmp_path, flags=EPYC_9355) == ["-DHAS_AVX512"]
    assert _host("VENDOR=GenuineIntel; egaroucid_defs", tmp_path, flags="sse4_2 popcnt avx avx2 bmi2") == []
    assert _host("VENDOR=; egaroucid_defs", tmp_path, flags=OLD_X86) == ["-DHAS_NO_AVX2"]
    assert _host("egaroucid_defs; egaroucid_cpu_flags", tmp_path, os_="Darwin", arch="arm64") == [
        "-DHAS_NO_AVX2", "-DHAS_ARM_PROCESSOR", "-mcpu=native"]
    assert _host("egaroucid_cpu_flags", tmp_path, flags=EPYC_9355) == ["-march=native", "-mtune=native"]

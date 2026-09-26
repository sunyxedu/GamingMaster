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


def _host(snippet, tmp_path, os_="Linux", arch="x86_64", flags="", vendor=""):
    """Source the script with a SIMULATED host: OS, ARCH, CPU flags AND vendor are all
    pinned, so nothing of the machine running the test leaks in (the script reads the real
    /proc/cpuinfo when sourced: AuthenticAMD on GitHub's runners, nothing on macOS)."""
    cmd = (f'source "{SCRIPT}"; OS={os_}; ARCH={arch}; FLAGS=" {flags} "; VENDOR="{vendor}"; '
           f'{snippet}')
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
    assert _host("VENDOR=AuthenticAMD; egaroucid_flavor", tmp_path, flags=EPYC_9355) == ["x86_64-avx512-amd"]
    assert _host("egaroucid_flavor", tmp_path, os_="Darwin", arch="arm64") == ["arm64-generic"]
    # an ARM host never gets the AMD tag, whatever the vendor string says
    assert _host("egaroucid_flavor", tmp_path, os_="Linux", arch="aarch64",
                 vendor="AuthenticAMD") == ["aarch64-generic"]
    assert _host("egaroucid_flavor", tmp_path, flags=EPYC_9355, vendor="AuthenticAMD") == ["x86_64-avx512-amd"]
    assert _host("egaroucid_flavor", tmp_path, flags=EPYC_9355, vendor="GenuineIntel") == ["x86_64-avx512"]


def _bash(snippet, tmp_path, check=True):
    r = subprocess.run([BASH, "-c", f'source "{SCRIPT}"; {snippet}'], capture_output=True, text=True,
                       env={**os.environ, "GM_ENGINES": str(tmp_path)})
    if check:
        assert r.returncode == 0, r.stderr
    return r


def test_egaroucid_source_patches(tmp_path):
    """The labelling patches: heap transposition table + a fixed RNG seed (without the
    seed the table's hash codes, and so the searches, differ between processes)."""
    eng = tmp_path / "src" / "src" / "engine"
    eng.mkdir(parents=True)
    (eng / "setting.hpp").write_text("// x\n#define TT_USE_STACK true\n#define OTHER true\n")
    (eng / "common.hpp").write_text("inline uint64_t tim();\nstd::mt19937 raw_myrandom(tim());\nint y;\n")
    r = _bash(f'egaroucid_patch "{tmp_path / "src"}"; echo "$EGAROUCID_SEED"', tmp_path)
    assert "warning" not in r.stderr, r.stderr
    seed = r.stdout.split()[-1]
    assert seed.isdigit()
    assert (eng / "setting.hpp").read_text() == "// x\n#define TT_USE_STACK false\n#define OTHER true\n"
    assert (eng / "common.hpp").read_text() == f"inline uint64_t tim();\nstd::mt19937 raw_myrandom({seed});\nint y;\n"
    # a source the patch does not match is reported, not silently accepted
    (eng / "common.hpp").write_text("std::mt19937 raw_myrandom(std::random_device{}());\n")
    assert "RNG seed patch did not apply" in _bash(f'egaroucid_patch "{tmp_path / "src"}"', tmp_path).stderr


def test_egaroucid_probe_parses_hint_rows(tmp_path):
    """The determinism probe keeps depth, move, score and node count of each hint row."""
    fake = tmp_path / "fake-egaroucid"
    fake.write_text(
        "#!/bin/sh\ncat >/dev/null\n"
        "echo '|          Level|          Depth|           Move|          Score|           Time|          Nodes|            NPS|'\n"
        "echo '|             11|         11@74%|             d6|            +13|  000:00:00.013|         111651|        8588538|'\n"
        "echo '[ERROR] something else on stdout'\n")
    fake.chmod(0o755)
    rows = _bash(f'egaroucid_probe "{fake}"', tmp_path).stdout.splitlines()
    assert [r.split() for r in rows] == [["Depth", "Move", "Score", "Nodes"], ["11@74%", "d6", "+13", "111651"]]


def test_katago_backend_and_tag(tmp_path):
    """KataGo release picks: CUDA when an NVIDIA GPU plus cuDNN 9 / cuBLAS 12 are loadable
    without root (LD_LIBRARY_PATH, e.g. `module load` or pip wheels), else the static CPU
    builds; the CPU / OpenCL / TensorRT assets live on v1.18.1 (v1.18.2 is CUDA-only)."""
    fake_bin, libs = tmp_path / "bin", tmp_path / "libs"
    fake_bin.mkdir()
    libs.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'GPU 0: Fake GPU'\n")
    smi.chmod(0o755)
    path = f"{fake_bin}:{os.environ.get('PATH', '')}"
    # LDCONFIG=true: an empty ldconfig cache, whatever the test host has installed
    probe = f'PATH="{path}"; LD_LIBRARY_PATH="{libs}"; LDCONFIG=true; katago_backend'
    assert _host(probe, tmp_path, flags=EPYC_9355) == ["eigenavx2"], "GPU without cuDNN → CPU build"
    (libs / "libcudnn.so.9").write_text("")
    (libs / "libcublas.so.12").write_text("")
    assert _host(probe, tmp_path, flags=EPYC_9355) == ["cuda12.8-cudnn9.8.0"]
    # one pinned release for every Linux backend (the engine version is part of meta.teacher)
    assert _host("katago_tag eigenavx2; katago_tag opencl; katago_tag trt10.9.0-cuda12.8; katago_tag cuda12.8-cudnn9.8.0",
                 tmp_path) == ["v1.18.1"] * 4
    assert _host("KATAGO_TAG=v1.17.1; katago_tag cuda12.8-cudnn9.8.0", tmp_path) == ["v1.17.1"]
    assert _host("KATAGO_BACKEND=trt10.9.0-cuda12.8; katago_backend", tmp_path) == ["trt10.9.0-cuda12.8"]


def test_katago_libs_found_in_a_large_ldconfig_cache(tmp_path):
    """System-wide CUDA libraries are found through the ldconfig cache even when the cache
    is large and the match comes early (`ldconfig -p | grep -q` under `set -o pipefail`
    reports SIGPIPE as "not found"); $LDCONFIG names an ldconfig that is not on PATH (the
    script also tries /sbin and /usr/sbin, where Debian keeps it)."""
    fake_bin = tmp_path / "sbin"
    fake_bin.mkdir()
    ldconfig = fake_bin / "ldconfig"
    ldconfig.write_text(
        "#!/bin/sh\n"
        "echo '\tlibcudnn.so.9 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudnn.so.9'\n"
        "echo '\tlibcublas.so.12 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcublas.so.12'\n"
        "i=0; while [ $i -lt 20000 ]; do echo \"\tlibfiller$i.so.1 (libc6,x86-64) => /usr/lib/libfiller$i.so.1\"; i=$((i+1)); done\n")
    ldconfig.chmod(0o755)
    smi = fake_bin / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'GPU 0: Fake GPU'\n")
    smi.chmod(0o755)
    path = f"{fake_bin}:{os.environ.get('PATH', '')}"
    probe = f'PATH="{path}"; LD_LIBRARY_PATH=""; katago_backend'
    assert _host(probe, tmp_path, flags=EPYC_9355) == ["cuda12.8-cudnn9.8.0"]
    # an ldconfig outside PATH, named by $LDCONFIG; LDCONFIG=true = an empty cache
    probe = f'PATH="{tmp_path / "nothing"}:/usr/bin:/bin"; LD_LIBRARY_PATH=""; LDCONFIG="{ldconfig}"; have_lib libcublas.so.12 && echo yes'
    assert _host(probe, tmp_path, flags=EPYC_9355) == ["yes"]
    probe = f'LD_LIBRARY_PATH=""; LDCONFIG=true; have_lib libcublas.so.12 || echo no'
    assert _host(probe, tmp_path, flags=EPYC_9355) == ["no"]

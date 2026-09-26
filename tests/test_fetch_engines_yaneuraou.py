"""scripts/engines/fetch_engines.sh, YaneuraOu parts (no network, no build): the build
target per host, the source patches applied before building, and the isready smoke test.

The script is sourced with a simulated host, like tests/test_fetch_engines.py.
"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "engines", "fetch_engines.sh")
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not installed")

# the two v9.40 snippets yaneuraou_patch rewrites (verbatim, incl. the UTF-8 comment)
SHM_H = """            if (fallback_backend.get_status() == SystemWideSharedConstantAllocationStatus::LocalMemory)
            {
                sync_cout << "info string shared memory: fallback to local memory";
                if (error_message)
                    sync_cout << " (" << *error_message << ")";
                sync_cout << sync_endl;
            }
"""
ENGINE_CPP = """    auto th = std::thread([&] {
        thread_started = true;
        int count = 0;
        while (!thread_end)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (++count >= 50 /* 5秒 */)
            {
                count = 0;
                sync_cout << sync_endl;  // 改行を送信する。
            }
        }
    });
    while (!thread_started)
        Tools::sleep(100);
"""


def _sh(snippet, tmp_path, env=None, check=True):
    e = {**os.environ, "GM_ENGINES": str(tmp_path / "engines"), **(env or {})}
    e.pop("YANEURAOU_TARGET_CPU", None)
    r = subprocess.run([BASH, "-c", f'source "{SCRIPT}"; {snippet}'], capture_output=True, text=True, env=e)
    if check:
        assert r.returncode == 0, r.stderr
    return r


def test_target_cpu_per_host(tmp_path):
    def target(os_, arch, flags):
        return _sh(f'OS={os_}; ARCH={arch}; FLAGS=" {flags} "; yaneuraou_target_cpu', tmp_path).stdout.strip()

    assert target("Linux", "x86_64", "avx2 bmi2 avx512f avx512bw avx512_vnni") == "AVX512VNNI"
    assert target("Linux", "x86_64", "sse4_2 popcnt") == "SSE42"
    assert target("Darwin", "arm64", "") == "APPLEM1"
    # Linux aarch64: the Makefile's GRAVITON2 (armv8.2-a+fp16+rcpc+dotprod+crypto) or portable
    assert target("Linux", "aarch64", "fp asimd aes asimdhp asimddp lrcpc") == "GRAVITON2"
    assert target("Linux", "aarch64", "fp asimd") == "OTHER"


def test_source_patches(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "shm.h").write_text(SHM_H, encoding="utf-8")
    (src / "engine.cpp").write_text(ENGINE_CPP, encoding="utf-8")
    for _ in range(2):  # idempotent
        _sh(f'yaneuraou_patch "{src}"', tmp_path)
        shm = (src / "shm.h").read_text(encoding="utf-8")
        # one sync_cout (output lock) per message: the nested ones deadlocked in isready
        assert shm.count("sync_cout") == 1
        assert 'std::cout << " (" << *error_message << ")";' in shm and "std::cout << sync_endl;" in shm
        eng = (src / "engine.cpp").read_text(encoding="utf-8")
        assert "milliseconds(1));" in eng and "++count >= 5000" in eng and "Tools::sleep(1);" in eng
        assert "milliseconds(100)" not in eng and "Tools::sleep(100)" not in eng
    _sh(f'yaneuraou_patch "{tmp_path / "missing"}"', tmp_path)  # other trees: a no-op


def _fake_engine(tmp_path, answers_ready):
    exe = tmp_path / ("ok-engine" if answers_ready else "hung-engine")
    ready = 'echo readyok' if answers_ready else 'sleep 30'
    exe.write_text(f"""#!/bin/sh
while read -r line; do
    case "$line" in
        usi) echo "id name FakeYO 1.0"; echo usiok ;;
        isready) {ready} ;;
        quit) exit 0 ;;
    esac
done
""")
    exe.chmod(0o755)
    return exe


def test_smoke_test_needs_readyok(tmp_path):
    ok = _sh(f'yaneuraou_smoke "{_fake_engine(tmp_path, True)}"', tmp_path, {"YANEURAOU_SMOKE_TIMEOUT": "10"})
    assert ok.stdout.strip() == "FakeYO 1.0"
    hung = _sh(f'yaneuraou_smoke "{_fake_engine(tmp_path, False)}"', tmp_path, {"YANEURAOU_SMOKE_TIMEOUT": "2"},
               check=False)
    assert hung.returncode != 0 and "no readyok within 2 s" in hung.stderr

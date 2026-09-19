"""Entrypoints still work after moving them out of the repository root."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING_MODULES = ("train_il", "train_rl", "train_il_game", "train_rl_game", "eval_il")


@pytest.mark.parametrize("module", TRAINING_MODULES)
def test_training_module_help(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", f"training.{module}", "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_shell_syntax():
    for directory in (ROOT / "scripts", ROOT / "data_processing"):
        for script in sorted(directory.rglob("*.sh")):
            result = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == 0, f"{script}: {result.stderr}"


@pytest.fixture
def cluster(tmp_path):
    """A checkout with spaces and command stubs; never contact a scheduler."""
    checkout = tmp_path / "Gaming Master"
    shutil.copytree(ROOT / "scripts/slurm", checkout / "scripts/slurm")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.jsonl"
    stub = f"#!{sys.executable}\n" + '''import json
import os
from pathlib import Path
import sys

command = Path(sys.argv[0]).name
with open(os.environ["COMMAND_LOG"], "a") as output:
    output.write(json.dumps({
        "command": command, "args": sys.argv[1:], "cwd": os.getcwd(),
        "root": os.environ.get("GAMINGMASTER_ROOT"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "gpus": os.environ.get("GPUS_PER_NODE"),
    }) + "\\n")
if command == "sbatch":
    print("12345")
elif command == "scontrol":
    print("gpu001")
'''
    for command in ("sbatch", "scontrol", "srun"):
        path = bin_dir / command
        path.write_text(stub)
        path.chmod(0o755)
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("SLURM_") and key != "GAMINGMASTER_ROOT"
    }
    environment.update({
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "COMMAND_LOG": str(log),
        "PYTHONPATH": "/existing/python/path",
        "USER": "layout-test",
        "SLURM_JOB_NODELIST": "gpu001",
        "SLURM_JOB_ID": "12345",
    })
    return checkout.resolve(), environment, log


@pytest.mark.parametrize("helper,args,job,count", [
    ("data/collect_data.sh", [], "data/prepare_data_job.sh", 5),
    ("il/submit_phase1.sh", ["3", "4"], "il/train_il_phase1_job.sh", 1),
])
def test_submission_from_outside_checkout(cluster, tmp_path, helper, args, job, count):
    checkout, environment, log = cluster
    result = subprocess.run(
        ["bash", str(checkout / "scripts/slurm" / helper), *args],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == count
    for call in calls:
        assert call["command"] == "sbatch"
        assert call["cwd"] == call["root"] == str(checkout)
        assert call["pythonpath"] == f"{checkout}:/existing/python/path"
        assert str(checkout / "scripts/slurm" / job) in call["args"]
        assert f"--chdir={checkout}" in call["args"]
        assert "--export=ALL" in call["args"]
    if helper.startswith("data/"):
        assert calls[1]["args"][-1] == "Average Joe"
    else:
        assert calls[0]["gpus"] == "4"
        assert "--nodes=3" in calls[0]["args"]


@pytest.mark.parametrize("root_source", ["submit_root", "submit_subdir", "explicit"])
def test_spooled_job_finds_checkout(cluster, tmp_path, root_source):
    checkout, environment, log = cluster
    spool = tmp_path / "slurm_script"
    shutil.copyfile(checkout / "scripts/slurm/il/train_job.sh", spool)
    if root_source == "explicit":
        environment["GAMINGMASTER_ROOT"] = str(checkout)
        environment["SLURM_SUBMIT_DIR"] = str(tmp_path)  # explicit root wins
    else:
        submit_dir = checkout if root_source == "submit_root" else checkout / "scripts/slurm/il"
        environment["SLURM_SUBMIT_DIR"] = str(submit_dir)
    result = subprocess.run(
        ["bash", str(spool)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    launch, = [call for call in calls if call["command"] == "srun"]
    assert launch["cwd"] == launch["root"] == str(checkout)
    assert launch["pythonpath"] == f"{checkout}:/existing/python/path"
    position = launch["args"].index("--module")
    assert launch["args"][position + 1] == "training.train_il"


def test_spooled_job_without_checkout_fails_before_launch(cluster, tmp_path):
    checkout, environment, log = cluster
    spool = tmp_path / "slurm_script"
    shutil.copyfile(checkout / "scripts/slurm/il/train_job.sh", spool)
    environment["SLURM_SUBMIT_DIR"] = str(tmp_path)
    result = subprocess.run(
        ["bash", str(spool)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "Cannot find GamingMaster" in result.stderr
    assert not log.exists()

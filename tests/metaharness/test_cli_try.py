"""`metaharness try`: a proposer plays a harness on a training seed of its archive before the evaluation."""
from __future__ import annotations

import json

import pytest

from metaharness.cli import main
from metaharness.core.archive import Archive


def test_try_plays_a_training_seed_with_the_archive_settings(tmp_runs, capsys):
    rc = main(["loop", "--tier", "2", "--game", "snake", "--system1", "heuristic", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "1", "--seeds", "0,1", "--heldout", "100",
               "--name", "t", "--runs-dir", str(tmp_runs)])
    assert rc == 0
    a = Archive(tmp_runs / "t")
    jev_only = a.candidates()[0]
    prompt = (a.candidates()[-1].dir / "prompt.md").read_text()
    assert "-m metaharness try " in prompt and "Only\ntraining seeds are accepted" in prompt
    capsys.readouterr()
    assert main(["try", str(jev_only.dir / "harness.py"), "--seed", "1", "--minutes", "1"]) == 0
    out = capsys.readouterr().out
    assert "seed 1: score=" in out and "trace: " in out
    tries = sorted((jev_only.dir / "tries").iterdir())
    assert len(tries) == 1 and tries[0].name.startswith("seed1-")
    e = json.loads((tries[0] / "eval.json").read_text())
    assert e["n_episodes"] == 1 and e["system2_calls"] == 0 and e["system1_calls"] > 0   # Astra off, Jev on
    assert jev_only.eval()["n_episodes"] == 2                       # the candidate's own evaluation is untouched
    with pytest.raises(SystemExit, match="not a training seed"):
        main(["try", str(jev_only.dir / "harness.py"), "--seed", "100"])

"""End-to-end offline loop: seeds -> scripted proposer -> evaluate -> leaderboard -> held-out."""
from __future__ import annotations

import json

from metaharness.cli import main
from metaharness.core.archive import Archive


def test_tier1_offline_loop(tmp_runs):
    rc = main(["loop", "--tier", "1", "--game", "rulepuzzle", "--system1", "none", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "1", "--seeds", "0", "--heldout", "",
               "--s2-budget", "5", "--name", "t1", "--runs-dir", str(tmp_runs)])
    assert rc == 0
    a = Archive(tmp_runs / "t1")
    cands = a.candidates()
    assert len(cands) == 2
    e = cands[1].eval()
    assert e["fatal"] is None and e["system2_calls"] == 6      # 5 allowed + the call that tripped the budget
    assert e["errors"] == []                                   # reference harness handles the budget cleanly


def test_proposer_text_mode_extracts_fenced_block(tmp_runs):
    from metaharness.core.proposer import propose
    from metaharness.models.mock import ScriptedSystem2
    a = Archive(tmp_runs / "p")
    m = ScriptedSystem2(replies=["Thoughts...\n```python\ndef build(ctx):\n    return None\n```\nHypothesis: X"])
    c = propose(m, a, contract_doc="C", task_md="T", agentic=False)
    assert c.harness_path.read_text().startswith("def build")
    assert "Hypothesis: X" in (c.dir / "proposal.md").read_text()
    assert "candidates/0000" in (c.dir / "prompt.md").read_text()

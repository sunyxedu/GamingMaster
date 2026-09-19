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
    assert e["fatal"] is None and e["system2_calls"] == 5      # rejected requests are not backend calls
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


def test_completed_artifacts_survive_proposer_timeout(tmp_path):
    from metaharness.core.proposer import propose
    from metaharness.models.mock import ScriptedSystem2

    class TimedOut(ScriptedSystem2):
        def run(self, prompt, *, cwd, **kwargs):
            d = sorted((cwd / 'candidates').iterdir())[-1]
            (d / 'harness.py').write_text('def build(ctx):\n    return None\n')
            (d / 'proposal.md').write_text('Completed written proposal before CLI timeout.')
            raise TimeoutError('CLI deadline')

    a = Archive(tmp_path / 'archive')
    c = propose(TimedOut(), a, contract_doc='contract', task_md='task')
    assert c.harness_path.exists()
    assert json.loads((c.dir / 'proposal_error.json').read_text())['status'] == 'timed_out'
    # This does not mark it as evaluated or eligible; only the evaluator can do that.
    assert c.eval() is None and a.best() is None

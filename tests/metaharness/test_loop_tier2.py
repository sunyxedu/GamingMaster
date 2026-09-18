"""End-to-end offline tier-2 loop on the virtual clock."""
from __future__ import annotations

import json

from metaharness.cli import main
from metaharness.core.archive import Archive


def test_tier2_offline_loop(tmp_runs):
    rc = main(["loop", "--tier", "2", "--game", "snake", "--system1", "heuristic", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "1", "--seeds", "0,1", "--heldout", "100",
               "--name", "t2", "--runs-dir", str(tmp_runs)])
    assert rc == 0
    a = Archive(tmp_runs / "t2")
    cands = a.candidates()
    assert [c.dir.name for c in cands] == ["0000-tier2_jev_only", "0001-tier2_astra_only", "0002"]
    for c in cands:
        e = c.eval()
        assert e and e["fatal"] is None and e["n_episodes"] == 2
    astra_only = cands[1].eval()
    assert astra_only["stale_fraction"] == 1.0            # 2 s think vs 0.25 s tick: always stale
    assert cands[2].eval()["system2_calls"] > 0 and cands[2].eval()["system1_calls"] > 0
    assert (a.root / "leaderboard.md").exists() and (a.root / "task.md").exists()
    assert (a.root / "candidates" / "0002" / "proposal.md").exists()
    best = a.best()
    assert best is not None and (best.dir / "heldout" / "eval.json").exists()
    trace = (cands[2].dir / "trace_seed0.jsonl").read_text().splitlines()
    kinds = {json.loads(l).get("type") for l in trace}
    assert {"episode_start", "tick", "model_call", "episode_end"} <= kinds



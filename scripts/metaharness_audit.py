"""Reproducible live Astra/Jev audit. Search and held-out evaluation are separate.

python -m scripts.metaharness_audit search --tier 2 --root /tmp/mh-tier2-audit
python -m scripts.metaharness_audit test --tier 2 --root /tmp/mh-tier2-audit
All requests use the configured Astra/Jev profiles; no mocked latency.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from metaharness.config import REPO_ROOT, load_dotenv
from metaharness.core.archive import Archive
from metaharness.core.contract import Budget
from metaharness.core.loop import LoopConfig, MetaHarnessLoop
from metaharness.core.runner import evaluate_harness
from metaharness.games.registry import make_game
from metaharness.models import make_system1, make_system2


def setup(tier):
    if tier == 2:
        from metaharness.tiers import tier2 as task
        game, kw = "snake", {"tick_s": 0.5, "max_ticks": 120}
        prior = "astra-tier2-snake-heldout-lanerunner/0004/harness.py"
        budget = Budget(max_system2_calls=4, max_system1_calls=240, max_wall_s=240)
    else:
        from metaharness.tiers import tier1 as task
        game, kw = "rulepuzzle", {"n_levels": 2, "budget_per_level": 50}
        prior = "astra-tier1-rulepuzzle-heldout-arc3/0002/harness.py"
        budget = Budget(max_system2_calls=12, max_system1_calls=0, max_wall_s=900)
    return task, game, kw, REPO_ROOT / "metaharness/artifacts" / prior, budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["search", "test"])
    p.add_argument("--tier", type=int, choices=[1, 2], required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=2)
    a = p.parse_args()
    load_dotenv()
    task, game, kw, prior, budget = setup(a.tier)
    archive = Archive(a.root.resolve())
    s2 = make_system2("astra", reasoning_effort="high")
    s1 = make_system1("jev") if a.tier == 2 else None
    if a.phase == "search":
        if not archive.candidates():
            for baseline in task.baseline_paths():
                archive.add_seed(baseline, label=baseline.stem)
            archive.add_seed(prior, label="previous_astra_winner",
                             note="Frozen prior Astra-generated winner. Re-evaluated under the new protocol.\n")
        extra = (
            "This is a COLD-START experiment: every episode starts without a cached policy. "
            "At tier 2 Astra must compile/adapt while the world keeps ticking. Jev must have a "
            "meaningful measurable role. Use public observations and rules only, and log when "
            "each model's output actually changes control. Design a generic discovery/compilation "
            "procedure, not a hand-written game solver. Prefer a compact bridge that Astra can "
            "produce quickly. Join all outstanding futures even if the game ends. "
            "Do not inspect files outside this experiment archive or run your own model calls. "
            "Do not import game implementations. Do not inspect or test held-out games."
        )
        loop = MetaHarnessLoop(
            archive=archive, game_factory=lambda **k: make_game(game, **k), game_kwargs=kw,
            task_md=task.task_md(game, budget, "wall", kw), contract_doc=task.contract_doc(),
            proposer=make_system2("astra", reasoning_effort="high"), system2=s2, system1=s1,
            config=LoopConfig(n_iterations=a.iterations, seeds_train=[0, 1, 2] if a.tier == 2 else [0, 1],
                              seeds_heldout=[], budget=budget, clock_mode="wall",
                              harness_timeout_s=budget.max_wall_s, proposer_timeout_s=600,
                              memory_mode="episode"), proposer_extra=extra)
        provenance = {"git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "command": "search", "tier": a.tier}
        with (archive.root / "provenance.jsonl").open("a") as fh:
            fh.write(json.dumps(provenance) + "\n")
        best = loop.run()
        if best is None:
            raise SystemExit("No eligible candidate")
        print("SELECTED", best.dir.name, flush=True)
    else:
        best = archive.best()
        if best is None:
            raise SystemExit("Search must finish before testing")
        # Freeze before exposing any test outcomes; never optimize from these results.
        selected = [c for c in archive.candidates()
                    if c.dir.name.endswith("previous_astra_winner") or c.idx == best.idx]
        selected += [c for c in archive.candidates()
                     if c.dir.name.endswith("tier2_jev_only") or c.dir.name.endswith("tier1_astra_naive")]
        cases = [("heldout-cold", game, kw, [700, 701, 702] if a.tier == 2 else [700, 701], "episode")]
        if a.tier == 2:
            cases += [("heldout-shared", game, kw, [700, 701, 702], "shared"),
                      ("heldout-lanerunner", "lanerunner", {"tick_s": .5, "max_ticks": 120},
                       [700, 701, 702], "episode")]
        else:
            cases += [("heldout-arc3-ls20", "arc3", {"game_id": "ls20", "max_actions": 150},
                       [700], "episode"),
                      ("heldout-arc3-cd82", "arc3", {"game_id": "cd82", "max_actions": 150},
                       [700], "episode")]
        freeze = {"selected": best.dir.name, "candidates": [c.dir.name for c in selected],
                  "cases": cases, "selection": "training scores only"}
        freeze_path = archive.root / "test_manifest.json"
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != json.loads(json.dumps(freeze)):
            raise SystemExit("Frozen test configuration changed")
        freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
        for label, name, game_kw, seeds, memory_mode in cases:
            for cand in selected:
                out = cand.dir / label
                if (out / "eval.json").exists():
                    continue
                result = evaluate_harness(cand.harness_path, lambda **k: make_game(name, **k), seeds,
                                          system2=s2, system1=s1, out_dir=out, clock_mode="wall",
                                          game_kwargs=game_kw, budget_per_episode=budget,
                                          harness_timeout_s=budget.max_wall_s, memory_mode=memory_mode)
                print(label, cand.dir.name, result.mean_score,
                      result.fatal, [e.error for e in result.episodes if e.error], flush=True)
                archive.write_leaderboard()
                if result.fatal:
                    raise SystemExit(result.fatal)
        if a.tier == 2:
            for label, slow, fast in (("heldout-no-astra", None, s1),
                                      ("heldout-no-jev", s2, None)):
                out = best.dir / label
                if (out / "eval.json").exists():
                    continue
                result = evaluate_harness(best.harness_path, lambda **k: make_game(game, **k),
                                          [700, 701, 702], system2=slow, system1=fast,
                                          out_dir=out, clock_mode="wall", game_kwargs=kw,
                                          budget_per_episode=budget, harness_timeout_s=budget.max_wall_s,
                                          memory_mode="episode")
                print(label, best.dir.name, result.mean_score, result.fatal,
                      [e.error for e in result.episodes if e.error], flush=True)
                archive.write_leaderboard()
    print(archive.write_leaderboard(), flush=True)


if __name__ == "__main__":
    main()

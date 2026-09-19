"""Filesystem archive of harness candidates — the proposer's whole memory.

    runs/<experiment>/
      task.md                   what the game is, the contract, the budget
      leaderboard.md            auto-regenerated after every evaluation
      candidates/0000-baseline/ harness.py  proposal.md  eval.json  trace_seed*.jsonl
      candidates/0001/          ...

Following Meta-Harness (arXiv 2603.28052) the proposer gets *unrestricted*
read access to this directory rather than a compressed summary.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Candidate:
    idx: int
    dir: Path

    @property
    def harness_path(self) -> Path:
        return self.dir / "harness.py"

    @property
    def eval_path(self) -> Path:
        return self.dir / "eval.json"

    def eval(self) -> Optional[dict[str, Any]]:
        if self.eval_path.exists():
            return json.loads(self.eval_path.read_text())
        return None

    @property
    def score(self) -> Optional[float]:
        e = self.eval()
        return None if not e else e.get("mean_score")


class Archive:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.cand_dir = self.root / "candidates"
        self.cand_dir.mkdir(parents=True, exist_ok=True)

    # -- candidates -----------------------------------------------------
    def candidates(self) -> list[Candidate]:
        out = []
        for d in sorted(self.cand_dir.iterdir()):
            if d.is_dir() and d.name[:4].isdigit():
                out.append(Candidate(int(d.name[:4]), d))
        return out

    def next_index(self) -> int:
        cs = self.candidates()
        return (cs[-1].idx + 1) if cs else 0

    def new_candidate(self, label: str = "") -> Candidate:
        idx = self.next_index()
        name = f"{idx:04d}" + (f"-{label}" if label else "")
        d = self.cand_dir / name
        d.mkdir(parents=True, exist_ok=False)
        return Candidate(idx, d)

    def add_seed(self, harness_file: Path, label: str = "baseline", note: str = "") -> Candidate:
        c = self.new_candidate(label)
        shutil.copy(harness_file, c.harness_path)
        c.dir.joinpath("proposal.md").write_text(note or f"Seed candidate copied from {harness_file}.\n")
        return c

    def best(self) -> Optional[Candidate]:
        def eligible(c):
            e = c.eval() or {}
            score = e.get("mean_score")
            expected = e.get("metadata", {}).get("seeds")
            return (score is not None and math.isfinite(score) and not e.get("fatal")
                    and not e.get("errors") and e.get("n_episodes", 0) > 0
                    and (expected is None or len(expected) == e.get("n_episodes")))
        scored = [c for c in self.candidates() if eligible(c)]
        if not scored:
            return None
        def key(c: Candidate):
            e = c.eval() or {}
            # Turn games reward action efficiency; real-time games reward
            # survival. Fewer elapsed ticks must not reward an early death.
            realtime = (e.get("metadata", {}).get("game_kind") == "realtime"
                        or e.get("stale_fraction") is not None)
            steps = e.get("mean_steps") or 0.0
            return (e.get("mean_score", 0.0), -(e.get("stale_fraction") or 0.0), -e.get("system2_calls", 0),
                    steps if realtime else -steps)
        return max(scored, key=key)

    # -- docs -----------------------------------------------------------
    def write_task(self, text: str) -> None:
        (self.root / "task.md").write_text(text)

    def write_leaderboard(self) -> str:
        rows = ["| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | status |",
                "|---|-----------|-----------:|---------:|------------:|---------:|---------:|--------|"]
        for c in self.candidates():
            e = c.eval()
            if e is None:
                rows.append(f"| {c.idx:04d} | {c.dir.name} | – | – | – | – | – | not evaluated |")
                continue
            status = "fatal: " + e["fatal"][:60].replace("\n", " ") if e.get("fatal") else (
                f"{len(e.get('errors', []))} episode errors" if e.get("errors") else "ok")
            ms = e.get("mean_score")
            sf = e.get("stale_fraction")
            rows.append(
                f"| {c.idx:04d} | {c.dir.name} | {ms if ms is None else f'{ms:.3f}'} | {e.get('n_episodes', 0)} | "
                f"{'–' if sf is None else f'{100*sf:.0f}%'} | {e.get('system1_calls', 0)} | {e.get('system2_calls', 0)} | {status} |")
        best = self.best()
        text = "# Leaderboard\n\n" + "\n".join(rows) + "\n\n" + (
            f"Best so far: `{best.dir.name}` (mean score {best.score:.3f})\n" if best else "No scored candidate yet.\n")
        held = []
        for c in self.candidates():
            for d in sorted(c.dir.glob("heldout*")):
                ej = d / "eval.json"
                if ej.exists():
                    e = json.loads(ej.read_text())
                    ms = e.get("mean_score")
                    held.append(f"- `{c.dir.name}` / {d.name} ({e.get('game')}): mean score "
                                f"{'–' if ms is None else f'{ms:.3f}'} over {e.get('n_episodes', 0)} episodes"
                                + (f", fatal: {e['fatal'][:80]}" if e.get("fatal") else "")
                                + (f", {len(e['errors'])} episode errors" if e.get("errors") else ""))
        if held:
            text += "\nHeld-out evaluations (not used for selection):\n" + "\n".join(held) + "\n"
        (self.root / "leaderboard.md").write_text(text)
        return text

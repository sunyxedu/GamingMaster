"""Filesystem archive of harness candidates — the proposer's whole memory.

    runs/<experiment>/
      task.md                   what the games are, the contract, the budget
      leaderboard.md            auto-regenerated after every evaluation
      notes.md                  cross-iteration lab notebook (proposer appends)
      candidates/0000-baseline/ harness.py  proposal.md  eval.json  trace_seed*.jsonl
      candidates/0001/          ...           (multi-game: game-<name>/eval.json + traces,
                                               eval.json = combined summary)

Following Meta-Harness (arXiv 2603.28052) the proposer gets *unrestricted*
read access to this directory rather than a compressed summary.

Selection is *generality-aware*: with several training games, each game's mean
score is min-max normalised across the archive (0 = worst candidate on that
game, 1 = best), and a candidate is ranked by the mean normalised score, then
by its worst game, so a harness that wins one game and dies on another does
not beat a harness that is decent everywhere.
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

    def game_evals(self) -> dict[str, dict[str, Any]]:
        """Per-game evaluation summaries: ``{game: eval}`` (one entry for single-game archives)."""
        e = self.eval()
        if not e:
            return {}
        if isinstance(e.get("per_game"), dict):
            return dict(e["per_game"])
        return {str(e.get("game", "game")): e}

    @property
    def score(self) -> Optional[float]:
        e = self.eval()
        return None if not e else e.get("mean_score")


def combine_evals(per_game: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Combined summary of several single-game eval dicts (what gets written to eval.json)."""
    games = list(per_game)
    means = [g.get("mean_score") for g in per_game.values()]
    finite = [m for m in means if m is not None and math.isfinite(m)]
    stale_tot = stale_num = 0
    acts = acts_s1 = acts_s2 = 0
    for g in per_game.values():
        for ep in g.get("episodes", []):
            stale_num += ep.get("stale_ticks", 0)
            stale_tot += ep.get("stale_ticks", 0) + ep.get("decided_ticks", 0)
            acts += ep.get("acts", 0)
            acts_s1 += ep.get("acts_after_s1", 0)
            acts_s2 += ep.get("acts_after_s2", 0)
    fatal = [f"{name}: {g['fatal']}" for name, g in per_game.items() if g.get("fatal")]
    return {
        "game": "+".join(games), "games": games,
        "fatal": "; ".join(fatal) if fatal else None,
        "mean_score": (sum(finite) / len(finite)) if finite and len(finite) == len(means) else None,
        "stale_fraction": (stale_num / stale_tot) if stale_tot else None,
        "n_episodes": sum(g.get("n_episodes", 0) for g in per_game.values()),
        "system1_calls": sum(g.get("system1_calls", 0) for g in per_game.values()),
        "system2_calls": sum(g.get("system2_calls", 0) for g in per_game.values()),
        "failed_s1": sum(g.get("failed_s1", 0) for g in per_game.values()),
        "failed_s2": sum(g.get("failed_s2", 0) for g in per_game.values()),
        "mean_steps": (sum(g["mean_steps"] for g in per_game.values()) / len(per_game)
                       if per_game and all(g.get("mean_steps") is not None for g in per_game.values()) else None),
        "timeouts_s1": sum(g.get("timeouts_s1", 0) for g in per_game.values()),
        "timeouts_s2": sum(g.get("timeouts_s2", 0) for g in per_game.values()),
        "s1_informed_fraction": (acts_s1 / acts) if acts else None,
        "s2_informed_fraction": (acts_s2 / acts) if acts else None,
        "errors": [f"{name}: {err}" for name, g in per_game.items() for err in g.get("errors", [])],
        "per_game": {name: {k: v for k, v in g.items() if k != "episodes"} | {
            "episodes": [{k: ep.get(k) for k in ("seed", "score", "steps", "stale_ticks", "decided_ticks",
                                                  "system1_calls", "system2_calls", "acts", "acts_after_s1",
                                                  "acts_after_s2", "error", "ended_by", "wall_s", "latency_s1",
                                                  "latency_s2", "calls_s2", "info") if k in ep}
                         for ep in g.get("episodes", [])]}
            for name, g in per_game.items()},
        "metadata": {"seeds": seeds, "games": games, "expected_episodes": len(seeds) * len(games),
                     "game_kind": next((g.get("metadata", {}).get("game_kind") for g in per_game.values()), None)},
    }


def latency_summary(e: dict[str, Any]) -> dict[str, Optional[float]]:
    """Median over episodes of the per-episode p50 / max model latency (seconds)."""
    import statistics
    out: dict[str, Optional[float]] = {}
    games = e["per_game"].values() if isinstance(e.get("per_game"), dict) else [e]
    for kind in ("s1", "s2"):
        p50s, maxs = [], []
        for g in games:
            for ep in g.get("episodes", []):
                st = ep.get(f"latency_{kind}") or {}
                if st.get("n"):
                    p50s.append(float(st["p50"]))
                    maxs.append(float(st["max"]))
        out[f"{kind}_p50"] = statistics.median(p50s) if p50s else None
        out[f"{kind}_max"] = max(maxs) if maxs else None
    return out


def _last_line(text: str, limit: int = 160) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    line = lines[-1] if lines else str(text).strip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


def ineligibility(e: Optional[dict[str, Any]]) -> Optional[str]:
    """Why an evaluation cannot be selected, or None. Fatal, partial and erroneous evaluations
    cannot win; a budget or deadline end is not an error (protocol 3)."""
    if not e:
        return "not evaluated"
    if e.get("fatal"):
        return "fatal: " + _last_line(e["fatal"])
    score = e.get("mean_score")
    if score is None or not math.isfinite(score):
        return "no finite mean score"
    if e.get("errors"):
        return f"{len(e['errors'])} episode error(s), first: " + _last_line(e["errors"][0])
    meta = e.get("metadata", {})
    expected = meta.get("expected_episodes")
    if expected is None and meta.get("seeds") is not None:
        expected = len(meta["seeds"])
    n = e.get("n_episodes", 0)
    if n <= 0:
        return "no episodes"
    if expected is not None and expected != n:
        return f"partial: {n} of {expected} episodes"
    return None


def _eligible(e: dict[str, Any]) -> bool:
    return ineligibility(e) is None


def _episodes(e: dict[str, Any]) -> list[dict[str, Any]]:
    games = e["per_game"].values() if isinstance(e.get("per_game"), dict) else [e]
    return [ep for g in games for ep in g.get("episodes", [])]


def latency_fit(calls: list[tuple[float, int]]) -> Optional[tuple[float, float, float, int]]:
    """Least-squares System-2 latency ~ a + b * output_tokens: (a, b, r, n), or None."""
    n = len(calls)
    if n < 5:
        return None
    xs = [float(t) for _, t in calls]
    ys = [float(lat) for lat, _ in calls]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b, b * math.sqrt(sxx / syy), n


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

    # -- selection ------------------------------------------------------
    def games(self) -> list[str]:
        seen: list[str] = []
        for c in self.candidates():
            e = c.eval() or {}
            if not e.get("n_episodes") and not isinstance(e.get("per_game"), dict):
                continue           # a load error / empty proposal names no real game
            for g in c.game_evals():
                if g not in seen:
                    seen.append(g)
        return seen

    def _norm_table(self) -> dict[str, tuple[float, float]]:
        """Per game: (min, max) of mean score over eligible candidates."""
        table: dict[str, list[float]] = {}
        for c in self.candidates():
            e = c.eval() or {}
            if not _eligible(e):
                continue
            for g, ge in c.game_evals().items():
                m = ge.get("mean_score")
                if m is not None and math.isfinite(m):
                    table.setdefault(g, []).append(m)
        return {g: (min(v), max(v)) for g, v in table.items()}

    def normalized(self, c: Candidate, table: Optional[dict[str, tuple[float, float]]] = None) -> dict[str, float]:
        table = self._norm_table() if table is None else table
        out: dict[str, float] = {}
        for g, ge in c.game_evals().items():
            m = ge.get("mean_score")
            if m is None or g not in table:
                continue
            lo, hi = table[g]
            out[g] = ((m - lo) / (hi - lo)) if hi > lo else 0.0
        return out

    def selection_score(self, c: Candidate, table: Optional[dict[str, tuple[float, float]]] = None
                        ) -> tuple[float, float]:
        """(mean normalised score over games, worst game) — higher is better."""
        n = self.normalized(c, table)
        if not n:
            return (float("-inf"), float("-inf"))
        return (sum(n.values()) / len(n), min(n.values()))

    def best(self) -> Optional[Candidate]:
        scored = [c for c in self.candidates() if _eligible(c.eval() or {})]
        if not scored:
            return None
        table = self._norm_table()

        def key(c: Candidate):
            e = c.eval() or {}
            # Turn games reward action efficiency; real-time games reward
            # survival. Fewer elapsed ticks must not reward an early death.
            realtime = (e.get("metadata", {}).get("game_kind") == "realtime"
                        or e.get("stale_fraction") is not None)
            steps = e.get("mean_steps") or 0.0
            mean_n, worst_n = self.selection_score(c, table)
            return (round(mean_n, 9), round(worst_n, 9), e.get("mean_score", 0.0),
                    -(e.get("stale_fraction") or 0.0), -e.get("system2_calls", 0),
                    steps if realtime else -steps)
        return max(scored, key=key)

    # -- docs -----------------------------------------------------------
    def write_task(self, text: str) -> None:
        (self.root / "task.md").write_text(text)

    def write_leaderboard(self) -> str:
        games = self.games()
        multi = len(games) > 1
        table = self._norm_table()
        head = ["#", "candidate"] + ([f"{g}" for g in games] if multi else ["mean score"]) + \
               (["selection (mean / worst norm.)"] if multi else []) + \
               ["episodes", "stale ticks", "S1 calls", "S2 calls", "failed S1/S2", "Jev→act", "Astra→act",
                "Jev p50/max s", "Astra p50/max s", "status"]
        rows = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]

        def fmt(x, pct=False):
            if x is None:
                return "–"
            return f"{100 * x:.0f}%" if pct else f"{x:.3f}"

        for c in self.candidates():
            e = c.eval()
            if e is None:
                rows.append("| " + " | ".join([f"{c.idx:04d}", c.dir.name] + ["–"] * (len(head) - 3) + ["not evaluated"]) + " |")
                continue
            why = ineligibility(e)
            status = "ok" if why is None else "✗ " + why[:120].replace("|", "/")
            ge = c.game_evals()
            cells = [f"{c.idx:04d}", c.dir.name]
            if multi:
                cells += [fmt(ge[g].get("mean_score")) if g in ge else "–" for g in games]
                mean_n, worst_n = self.selection_score(c, table)
                cells.append("–" if not _eligible(e) else f"{mean_n:.3f} / {worst_n:.3f}")
            else:
                cells.append(fmt(e.get("mean_score")))
            lat = latency_summary(e)

            def lat_cell(kind):
                return "–" if lat[f"{kind}_p50"] is None else f"{lat[f'{kind}_p50']:.1f} / {lat[f'{kind}_max']:.0f}"
            timeouts = (f" ({e.get('timeouts_s1', 0)} / {e.get('timeouts_s2', 0)} timed out)"
                        if e.get("timeouts_s1") or e.get("timeouts_s2") else "")
            cells += [str(e.get("n_episodes", 0)), fmt(e.get("stale_fraction"), pct=True),
                      str(e.get("system1_calls", 0)), str(e.get("system2_calls", 0)),
                      f"{e.get('failed_s1', 0)} / {e.get('failed_s2', 0)}{timeouts}",
                      fmt(e.get("s1_informed_fraction"), pct=True), fmt(e.get("s2_informed_fraction"), pct=True),
                      lat_cell("s1"), lat_cell("s2"), status]
            rows.append("| " + " | ".join(cells) + " |")
        best = self.best()
        text = "# Leaderboard\n\n" + "\n".join(rows) + "\n\n"
        if multi:
            text += ("Selection: each game's mean score is min-max normalised across eligible candidates; "
                     "candidates are ranked by the mean normalised score, then by their worst game. "
                     "A harness must work on every game.\n")
        text += ("`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 "
                 "answer since the previous action (reported, not used for selection). Latency columns are the "
                 "median per-episode p50 and the worst observed call: design timeouts and pipelining from these "
                 "measurements, not from assumptions.\n\n")
        text += (f"Best so far: `{best.dir.name}` (mean score {best.score:.3f})\n" if best else "No scored candidate yet.\n")
        text += ("Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation "
                 "cannot win, whatever its score. A used-up call budget or the deadline is not an error.\n")
        # per-seed outcome: which seeds failed, how each episode ended and how much of the deadline it used
        ep_rows = []
        for c in self.candidates():
            e = c.eval() or {}
            parts = []
            for ep in _episodes(e):
                seed = ep.get("seed")
                score = ep.get("score")
                part = f"seed {seed}: {'–' if score is None else f'{score:.2f}'}"
                if ep.get("ended_by"):
                    part += f" {ep['ended_by']}"
                if ep.get("wall_s") is not None:
                    part += f" {ep['wall_s']:.0f}s"
                if ep.get("error"):
                    part += f" ✗ {_last_line(ep['error'], 100)}"
                parts.append(part)
            if parts:
                ep_rows.append(f"- `{c.dir.name}`: " + " · ".join(parts))
        if ep_rows:
            text += "\nEpisodes (score, how it ended, wall time; ✗ = error):\n" + "\n".join(ep_rows) + "\n"
        calls = [(float(lat), int(tok)) for c in self.candidates() for ep in _episodes(c.eval() or {})
                 for lat, tok in (ep.get("calls_s2") or []) if tok]
        fit = latency_fit(calls)
        if fit:
            a, b, r, n = fit
            text += (f"\nSystem-2 latency ≈ {a:.1f} s + {1000 * b:.1f} ms × output tokens (fit over {n} archived calls, "
                     f"r = {r:.2f}; slowest {max(lat for lat, _ in calls):.0f} s): long answers are slow — size "
                     "timeouts from this, not from short baseline answers.\n")
        versions: dict[str, set[str]] = {}
        for c in self.candidates():
            for kind, names in ((c.eval() or {}).get("metadata", {}).get("resolved_models") or {}).items():
                versions.setdefault(kind, set()).update(names)
        drift = {k: sorted(v) for k, v in versions.items() if len(v) > 1}
        if drift:
            text += ("\nWarning: different model versions answered in this archive "
                     + "; ".join(f"{k}: {', '.join(v)}" for k, v in drift.items())
                     + " — scores across them are not strictly comparable.\n")
        # score composition, when the adaptor exposes it (mean over episodes)
        comp_rows = []
        for c in self.candidates():
            e = c.eval() or {}
            eps = [ep for g in (e["per_game"].values() if isinstance(e.get("per_game"), dict) else [e])
                   for ep in g.get("episodes", [])]
            bds = [(ep.get("info") or {}).get("score_breakdown") for ep in eps]
            bds = [b for b in bds if isinstance(b, dict)]
            if bds:
                keys = sorted({k for b in bds for k in b})
                comp_rows.append(f"- `{c.dir.name}`: " + ", ".join(
                    f"{k} {sum(b.get(k, 0.0) for b in bds) / len(bds):.2f}" for k in keys))
        if comp_rows:
            text += "\nScore composition (mean per episode, from the adaptor's score_breakdown):\n" + "\n".join(comp_rows) + "\n"
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

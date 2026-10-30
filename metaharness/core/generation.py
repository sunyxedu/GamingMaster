"""Planning a generation of proposals.

Which parent each proposer builds on and how (its *directive*), whether a proposal is new code
at all, and the reflection recorded after the generation ran.

Parent sampling follows ShinkaEvolve and the Darwin Gödel Machine: an eligible candidate is drawn
with weight ``sigmoid(4 (fitness - median)) / (1 + children)``, so good candidates are revisited,
not only the best, and a lineage that already has many children yields to others. Modes vary the
kind of change — a targeted refinement of the best, a structural rewrite, a crossover of two
parents, a free exploration — and a strong but ineligible candidate gets a repair slot. The
proposer still reads the whole archive; the directive fixes where it starts.
"""
from __future__ import annotations

import difflib
import math
import random
from pathlib import Path
from typing import Any, Optional

from .archive import Archive, Candidate, ineligibility

MODES = {
    "refine": "Keep the parent's architecture. Make one targeted change that the parent's worst episodes justify "
              "(quote the trace evidence), so the result isolates one hypothesis.",
    "rewrite": "Redesign the structure: keep what the parent's traces prove works, but change the division of "
               "labour between code, System-1 and System-2 (what is computed, what is asked, what is planned, "
               "and when). Not a parameter tweak.",
    "crossover": "Combine the two parents: identify the mechanism that earns each one its score (or its "
                 "robustness) and build one harness that has both.",
    "explore": "Test a hypothesis no candidate has tested yet (see lessons.md and the proposals). The parent is "
               "a starting point and a source of working plumbing, not a constraint.",
    "repair": "The parent scores well but is ineligible: fix the cause of its episode errors (its eval.json and "
              "traces show it) without changing its design, so its score can count.",
}
_ORDER = ("refine", "rewrite", "crossover", "explore")


def children_counts(archive: Archive) -> dict[int, int]:
    counts: dict[int, int] = {}
    for c in archive.candidates():
        for p in (c.directive() or {}).get("parents", []):
            counts[p] = counts.get(p, 0) + 1
    return counts


def _repairable(archive: Archive, best: Optional[Candidate]) -> Optional[Candidate]:
    """The highest-scoring candidate that beats the best eligible one but erred in some episode and
    has not had a repair attempt yet (one slot per candidate, not a slot every generation)."""
    bar = best.score if best is not None and best.score is not None else float("-inf")
    repaired = {p for c in archive.candidates() if (c.directive() or {}).get("mode") == "repair"
                for p in (c.directive() or {}).get("parents", [])}
    out, out_score = None, float("-inf")
    for c in archive.candidates():
        e = c.eval() or {}
        score = e.get("mean_score")
        if score is None or not e.get("errors") or e.get("fatal") or ineligibility(e) is None or score <= bar:
            continue
        if c.idx in repaired or (c.directive() or {}).get("mode") == "repair" or score <= out_score:
            continue
        out, out_score = c, score
    return out


def plan(archive: Archive, k: int, generation: int, rng: Optional[random.Random] = None) -> list[dict[str, Any]]:
    """``k`` directives for one generation: ``{"mode", "parents": [idx...], "parent_dirs", "generation"}``."""
    rng = rng or random.Random(generation)
    elig = [c for c in archive.candidates() if ineligibility(c.eval() or {}) is None]
    best = archive.best()
    if not elig or best is None:
        return [{"mode": "explore", "parents": [], "parent_dirs": [], "generation": generation} for _ in range(k)]
    children = children_counts(archive)
    table = archive._norm_table()
    fit = [archive.selection_score(c, table)[0] for c in elig]
    median = sorted(fit)[len(fit) // 2]
    weights = [(1.0 / (1.0 + math.exp(-4.0 * (f - median)))) / (1 + children.get(c.idx, 0)) for f, c in zip(fit, elig)]

    def sample(exclude: frozenset[int] = frozenset()) -> Candidate:
        pool = [(c, w) for c, w in zip(elig, weights) if c.idx not in exclude] or list(zip(elig, weights))
        return rng.choices([c for c, _ in pool], weights=[w for _, w in pool])[0]

    modes = [_ORDER[(generation + i) % len(_ORDER)] for i in range(k)]
    repair = _repairable(archive, best)
    if repair is not None:
        modes[-1] = "repair"
    by_idx = {c.idx: c for c in archive.candidates()}
    out = []
    for mode in modes:
        if mode == "refine":
            parents = [best]
        elif mode == "crossover" and len(elig) > 1:
            parents = [best, sample(frozenset({best.idx}))]
        elif mode == "repair":
            parents = [repair]
        else:
            mode = "rewrite" if mode == "crossover" else mode
            parents = [sample()]
        out.append({"mode": mode, "parents": [p.idx for p in parents],
                    "parent_dirs": [by_idx[p.idx].dir.name for p in parents], "generation": generation})
    return out


def describe(directive: dict[str, Any]) -> str:
    """The directive as the proposer reads it."""
    mode = directive.get("mode", "explore")
    dirs = directive.get("parent_dirs") or []
    parents = ", ".join(f"`candidates/{d}/`" for d in dirs) or "(none: the archive has no eligible candidate yet)"
    return f"**Mode: {mode}.** Parent(s): {parents}. {MODES.get(mode, '')}"


# -- novelty -------------------------------------------------------------------------------------
def code_lines(src: str) -> list[str]:
    return [ln.strip() for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def similarity(a: str, b: str) -> float:
    """Fraction of matching code lines (comments and blank lines ignored); 1.0 = identical."""
    la, lb = code_lines(a), code_lines(b)
    if not la or not lb:
        return 0.0
    return difflib.SequenceMatcher(None, la, lb, autojunk=False).ratio()


def duplicate_of(archive: Archive, cand: Candidate, threshold: float = 0.98) -> Optional[tuple[Candidate, float]]:
    """The earlier candidate this harness nearly repeats (similarity >= threshold), if any."""
    src = cand.harness_path.read_text()
    best: Optional[tuple[Candidate, float]] = None
    for other in archive.candidates():
        if other.idx >= cand.idx or not other.harness_path.exists():
            continue
        s = similarity(src, other.harness_path.read_text())
        if s >= threshold and (best is None or s > best[1]):
            best = (other, s)
    return best


# -- reflection ------------------------------------------------------------------------------------
def digest(c: Candidate, limit: int = 1500) -> str:
    e = c.eval() or {}
    d = c.directive() or {}
    prop = (c.dir / "proposal.md").read_text(errors="replace")[:limit] if (c.dir / "proposal.md").exists() else "(none)"
    eps = [ep for g in (e["per_game"].values() if isinstance(e.get("per_game"), dict) else [e]) for ep in g.get("episodes", [])]
    seeds = " · ".join(f"seed {ep.get('seed')}: {ep.get('score')} {ep.get('ended_by', '')}"
                       + (f" ✗ {str(ep.get('error'))[-120:]}" if ep.get("error") else "") for ep in eps)
    lines = [f"### candidates/{c.dir.name} — {d.get('mode', 'seed')} of {', '.join(d.get('parent_dirs') or []) or '–'}",
             f"status: {ineligibility(e) or 'ok'}; mean score {e.get('mean_score')}; episodes: {seeds or '–'}",
             f"calls S1/S2 {e.get('system1_calls', 0)}/{e.get('system2_calls', 0)}, failed {e.get('failed_s1', 0)}/"
             f"{e.get('failed_s2', 0)}, stale {e.get('stale_fraction')}, Jev→act {e.get('s1_informed_fraction')}, "
             f"Astra→act {e.get('s2_informed_fraction')}",
             "proposal (head):", prop]
    return "\n".join(lines)


def reflection_prompt(archive: Archive, generation: int, cands: list[Candidate]) -> str:
    lb = archive.write_leaderboard()
    return f"""# Reflection after generation {generation}

Working directory: `{archive.root}`. Read anything; write ONLY `lessons.md` (append; create it if missing).

You are the note-taker of the Meta-Harness search. Generation {generation} proposed the candidates below and
they have been evaluated. Compare each proposal's hypothesis with what actually happened (its eval.json, the
traces of its worst and best episodes: trace_seed*.jsonl), then append ONE section to `lessons.md`:

## Generation {generation}
- one bullet per candidate: what it tried (mode, parent, the mechanism), the outcome (numbers), and the trace
  evidence that explains the outcome;
- then at most 5 bullets of general lessons for the next proposers: mechanisms confirmed, hypotheses refuted,
  what remains untested, which contract or plumbing details cost episodes. Numbers, not adjectives.
- end with one bullet on where the remaining headroom is: compare the best candidate's score components (the
  leaderboard's score composition, if the game reports one) and how its episodes end with what the game
  description says is achievable, and name the component or failure that is furthest from it.

Do not edit candidates, the leaderboard or task.md; do not propose code. Keep the section under 60 lines.
If you cannot use tools, answer with the section text only.

{lb}

{chr(10).join(digest(c) for c in cands)}
"""


def read_lessons(root: Path, limit: int = 20000) -> str:
    p = root / "lessons.md"
    if not p.exists():
        return ""
    t = p.read_text(errors="replace")
    return t if len(t) <= limit else "…(earlier generations omitted)\n" + t[-limit:]


__all__ = ["MODES", "plan", "describe", "similarity", "duplicate_of", "reflection_prompt", "read_lessons",
           "children_counts"]

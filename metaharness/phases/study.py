"""Phase 1 — study the game.

Input: the game's name (plus whatever the user told us: platform, account
notes, a tier override). Output: ``study/`` with

    brief.md        rules, objective, scoring, controls, pace / latency profile
    interfaces.md   every way to drive the game programmatically that was found,
                    the chosen one and why, the software + credentials it needs
    strategies.md   prior knowledge from the internet: openings, heuristics,
                    known strong strategies and their counters, what should be
                    code vs. Astra reasoning vs. Jev judgment
    study.json      machine-readable summary (schema below) the later phases use

The model does the research with live web search inside ``study/``; the
orchestrator only validates ``study.json`` and asks for repairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from ..models.base import System2Model, TraceSink

STUDY_JSON_SCHEMA = """{
  "game": "<canonical game name>",
  "tier": 1 | 2,                       // 1: the world waits for the agent; 2: it ticks while we think
  "tier_rationale": "<one paragraph: how fast decisions must be and why>",
  "decision_latency_s": <float>,       // how long a decision may take before the game punishes it
  "interface": {
    "chosen": "<api | mod | bot library | emulator | screen+input | official env ...>",
    "why": "<why this beats the alternatives>",
    "software": ["<package or program to install>", ...],
    "env": [{"var": "ENV_VAR_NAME", "description": "<what to put there>", "required": true}],
    "setup_steps": ["<step the adaptor phase must perform>", ...],
    "observation": "<what the adaptor can observe and in what form (json state / text / screenshot)>",
    "actions": "<what the adaptor can do and how>"
  },
  "scoring": "<what the adaptor should report as score and why it reflects playing well>",
  "scenarios": "<how reset(seed) should vary starts / levels / opponents so held-out seeds are meaningful>",
  "strategy_summary": "<3-8 sentences: how strong players play and what to implement>",
  "risks": ["<anything that can block the adaptor: anti-cheat, ToS, cost, latency, GUI-only>", ...]
}"""

STUDY_SYSTEM = """You are the Meta-Harness *study* agent. Your job is to learn everything needed to let a
slow reasoning model (Astra) — and, for fast games, a fast typed-judgment model (Jev) — play a game
you are given only by name. Research on the web, read documentation and community knowledge, and write
your findings into files. Be concrete and cite URLs. Do not write any game-playing code yet."""


def study_prompt(game: str, notes: str, tier_hint: Optional[int], out_dir: Path) -> str:
    tier_txt = (f"The user says the game is tier {tier_hint}; verify and keep it unless clearly wrong."
                if tier_hint else "Decide the tier yourself and justify it.")
    return f"""# Study the game: **{game}**

Working directory: `{out_dir}` (write everything here). The evaluator's Python is `{sys.executable}`
(Python {sys.version.split()[0]}); judge software requirements against it, not the system python.
User notes (platform, account, constraints — may be empty): {notes or '(none)'}

We are building an agent for this game out of two foundation models plus a generated harness:
* Astra — a slow, strong reasoning model (seconds to a minute per call). Plans, explains rules, writes code.
* Jev — a fast (~0.5 s), cheap model that answers *typed* questions (choice / score / yes-no) about a
  JSON state, many independent questions per call. Recognises situations; does not plan or do arithmetic.
Games split into tiers: **tier 1** — the world waits (turn-based, puzzles, most card games): Astra plays
through a harness. **tier 2** — the world keeps moving (action, real-time strategy with seconds of slack):
Astra thinks while Jev acts. {tier_txt}

## Deliverables (all in this directory)
1. `brief.md` — rules, objective, how score/progress is measured, controls/actions, pace: how often the
   game state changes and how long a decision may take without penalty. Cite sources.
2. `interfaces.md` — every practical way to observe and control this game programmatically (official
   API or environment, mods/plugins, bot libraries, emulator hooks, screen capture + input injection...).
   For each: maturity, latency, what it exposes, legal/ToS risk, cost. Pick one and say why. List the
   software to install and the credentials / environment variables (server address, account, API key)
   the user must provide. Never write credential values into files.
3. `strategies.md` — what strong players do: openings/build orders/rotations, heuristics, common
   mistakes, known counters. For each, say whether it belongs in *code* (exact rules, search, bookkeeping),
   in *Astra prompts* (planning, adaptation), or in *Jev questions* (fast recognition/choice). Cite sources.
4. `study.json` — exactly this schema (JSON, no comments):
```
{STUDY_JSON_SCHEMA}
```
Spend most effort on 2 and 3: the next phase must be able to build a working adaptor from `interfaces.md`
alone, and the harness search will read `strategies.md` as its prior knowledge."""


REQUIRED_KEYS = ("game", "tier", "tier_rationale", "decision_latency_s", "interface", "scoring",
                 "scenarios", "strategy_summary")


def load_study(study_dir: Path) -> dict[str, Any]:
    data = json.loads((study_dir / "study.json").read_text())
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"study.json missing keys {missing}")
    if int(data["tier"]) not in (1, 2):
        raise ValueError("study.json: tier must be 1 or 2")
    for f in ("brief.md", "interfaces.md", "strategies.md"):
        if not (study_dir / f).exists() or len((study_dir / f).read_text()) < 200:
            raise ValueError(f"study/{f} missing or too short")
    return data


def study_text(study_dir: Path, max_chars: int = 24000) -> str:
    """Concatenated study documents for later prompts (truncated per file)."""
    parts = []
    for f in ("brief.md", "interfaces.md", "strategies.md"):
        p = study_dir / f
        if p.exists():
            t = p.read_text()
            parts.append(f"### study/{f}\n" + (t if len(t) <= max_chars // 3 else t[: max_chars // 3] + "\n…(truncated)"))
    return "\n\n".join(parts)


def run_study(model: System2Model, run_dir: Path, game: str, *, notes: str = "",
              tier_hint: Optional[int] = None, timeout_s: float = 2400.0, max_repairs: int = 2,
              log=print) -> dict[str, Any]:
    out = run_dir / "study"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "study.json").exists():
        try:
            data = load_study(out)
            log(f"study: reusing {out / 'study.json'} (tier {data['tier']})")
            return data
        except Exception as e:  # noqa: BLE001
            log(f"study: existing study.json invalid ({e}); redoing")
    trace = TraceSink(out / "study_trace.jsonl")
    prompt = study_prompt(game, notes, tier_hint, out)
    (out / "prompt.md").write_text(prompt)
    session = None
    try:
        for attempt in range(max_repairs + 1):
            res = model.run(prompt, cwd=out, system=STUDY_SYSTEM if attempt == 0 else None, agentic=True,
                            timeout_s=timeout_s, trace=trace, session_id=session, web_search=True)
            session = res.session_id
            (out / f"agent_output_{attempt}.md").write_text(res.text or "")
            try:
                data = load_study(out)
                if tier_hint:
                    data["tier"] = int(tier_hint)
                    (out / "study.json").write_text(json.dumps(data, indent=2))
                log(f"study: done — tier {data['tier']}, interface {data['interface'].get('chosen')}")
                return data
            except Exception as e:  # noqa: BLE001
                log(f"study: deliverables incomplete ({e}); asking for repair {attempt + 1}/{max_repairs}")
                prompt = (f"The study deliverables are incomplete: {e}. Fix the files in `{out}` so that "
                          f"study.json matches the schema exactly and all three markdown files exist.")
        raise RuntimeError("study phase did not produce valid deliverables")
    finally:
        trace.close()

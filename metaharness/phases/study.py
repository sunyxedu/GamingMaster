"""Phase 1 — study the game.

Input: the game's name (plus whatever the user told us: platform, account
notes, a pace override). Output: ``study/`` with

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
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from ..models.base import ModelTimeout, ModelUnavailable, System2Model, TraceSink

STUDY_JSON_SCHEMA = """{
  "game": "<canonical game name>",
  "realtime": true | false,            // true: the world keeps moving while we think; false: it waits
  "pace_rationale": "<one paragraph: how fast decisions must be and why>",
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
slow reasoning model (Astra) and a fast typed-judgment model (Jev) play a game
you are given only by name. Research on the web, read documentation and community knowledge, and write
your findings into files. Be concrete and cite URLs. Do not write any game-playing code yet."""


FINISH_PROMPT = ("Your time is up. Do not start new research: from what you have already found, write (or complete) "
                 "all four deliverables now — brief.md, interfaces.md, strategies.md and study.json exactly as "
                 "specified in the first message — then stop.")


def study_prompt(game: str, notes: str, realtime_hint: Optional[bool], out_dir: Path,
                 timeout_s: Optional[float] = None) -> str:
    pace_txt = ("Decide the pace yourself and justify it." if realtime_hint is None else
                f"The user says the game is {'real-time' if realtime_hint else 'turn-based'}; verify and keep it "
                "unless clearly wrong.")
    return f"""# Study the game: **{game}**

Working directory: `{out_dir}` (write everything here). The evaluator's Python is `{sys.executable}`
(Python {sys.version.split()[0]}); judge software requirements against it, not the system python.
User notes (platform, account, constraints — may be empty): {notes or '(none)'}

We are building an agent for this game out of two foundation models plus a generated harness:
* Astra — a slow, strong reasoning model (seconds to a minute per call). Plans, explains rules, writes code.
* Jev — a fast (~0.5 s), cheap model that answers *typed* questions (choice / score / yes-no) about a
  JSON state, many independent questions per call, with calibrated probabilities. Good at recognising
  a situation and picking among concrete options under time pressure; does not plan or do arithmetic.
The pace decides the adaptor's interface: **turn-based** — the world waits for the agent (puzzles, most
card and board games); **real-time** — it keeps moving while we think (action games, real-time strategy
with seconds of slack), so the adaptor ticks on its own clock. {pace_txt}

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
alone, and the harness search will read `strategies.md` as its prior knowledge.
{_budget(timeout_s)}"""


def _budget(timeout_s: Optional[float]) -> str:
    if not timeout_s:
        return ""
    return (f"Time: about {timeout_s / 60:.0f} minutes. Write a first complete version of all four files within the "
            "first third of it, then deepen them: the files on disk are what the next phase gets.")


def _pace(realtime: bool) -> str:
    return "real-time" if realtime else "turn-based"


def _session_file(out: Path) -> Path:
    return out / "session.json"


def _saved_session(out: Path) -> Optional[str]:
    try:
        return json.loads(_session_file(out).read_text()).get("session_id")
    except (OSError, ValueError, AttributeError):
        return None


REQUIRED_KEYS = ("game", "realtime", "pace_rationale", "decision_latency_s", "interface", "scoring",
                 "scenarios", "strategy_summary")


def study_realtime(study: dict[str, Any]) -> bool:
    """Whether the study decided on a real-time adaptor (older studies recorded a tier: 2 = real-time)."""
    return bool(study["realtime"]) if "realtime" in study else int(study.get("tier") or 1) == 2


def load_study(study_dir: Path) -> dict[str, Any]:
    data = json.loads((study_dir / "study.json").read_text())
    if "tier" in data and "realtime" not in data:          # the schema before pace replaced tiers
        data["realtime"] = int(data["tier"]) == 2
        data.setdefault("pace_rationale", data.get("tier_rationale", ""))
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"study.json missing keys {missing}")
    if not isinstance(data["realtime"], bool):
        raise ValueError("study.json: realtime must be true or false")
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


PRIOR_FILES = ("strategies.md", "brief.md", "interfaces.md", "study.json", "ADAPTOR.md")
PRIOR_WEIGHTS = {"strategies.md": 3, "brief.md": 2, "interfaces.md": 1}
SUMMARY_FIELDS = (("strategy_summary",), ("scoring",), ("interface", "observation"), ("interface", "actions"))


def _clip(text: str, limit: int, name: str) -> str:
    """``text`` in about ``limit`` chars: its beginning and its end (where the conclusions and the
    advice to the search usually are), the cut marked."""
    if len(text) <= limit:
        return text
    mark = f"…(truncated; full text in {name if '/' in name else 'prior/' + name})"
    keep = max(0, limit - len(mark) - 4)
    n_head = keep * 2 // 3
    head, tail = text[:n_head], text[len(text) - (keep - n_head):] if keep > n_head else ""
    nl = head.rfind("\n")                          # whole lines where that costs little
    if nl > n_head * 3 // 4:
        head = head[:nl]
    nl = tail.find("\n")
    if 0 <= nl < len(tail) // 4:
        tail = tail[nl + 1:]
    return "\n\n".join(p for p in (head.rstrip(), mark, tail.strip()) if p)


def _share(need: dict[str, int], weight: dict[str, int], budget: float) -> dict[str, int]:
    """Weighted max-min split: a document needing less than its share keeps all of it and what it
    leaves is divided among the others by weight."""
    alloc = dict.fromkeys(need, 0)
    pending = [f for f in need if need[f] > 0]
    while pending:
        total_w = sum(weight[f] for f in pending)
        small = [f for f in pending if need[f] <= budget * weight[f] / total_w]
        if not small:
            alloc.update({f: int(budget * weight[f] / total_w) for f in pending})
            break
        for f in small:
            alloc[f] = need[f]
            budget -= need[f]
            pending.remove(f)
    return alloc


def prior_documents(study_dir: Optional[Path], adaptor_dir: Optional[Path], archive_root: Path, *,
                    study_budget: int = 24000, adaptor_budget: int = 12000, label: str = "") -> str:
    """Prior knowledge for the search proposer: the inline digest for task.md, and the full texts.

    Copies ``study/{brief,interfaces,strategies}.md``, ``study/study.json`` and ``adaptor/ADAPTOR.md``
    (whichever exist) into ``<archive_root>/prior/`` — the proposer reads them there in full — and
    returns the digest: the study.json summary fields (strategy_summary, scoring,
    interface.observation, interface.actions) first, then strategies.md, brief.md, ADAPTOR.md,
    interfaces.md. The study part gets ``study_budget`` characters, split 3:2:1 between strategies,
    brief and interfaces; ADAPTOR.md gets up to ``adaptor_budget``. Whatever a shorter document leaves
    goes to the others, across both parts. A document over its share keeps its beginning and its end,
    and every cut is marked ``…(truncated; full text in prior/<file>)``. Deterministic for the same
    files: the search manifest compares task.md when an archive is resumed. ``label`` (a game name)
    puts the documents under ``prior/<label>/`` when a search trains on several games."""
    prior = Path(archive_root) / "prior" / label if label else Path(archive_root) / "prior"
    where = f"prior/{label}" if label else "prior"
    texts: dict[str, str] = {}
    for f in PRIOR_FILES:
        base = adaptor_dir if f == "ADAPTOR.md" else study_dir
        src = Path(base) / f if base else None
        if src is not None and src.is_file():
            prior.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, prior / f)
            texts[f] = src.read_text(encoding="utf-8", errors="replace").strip()
    if not texts:
        return ""
    summary = ""
    try:
        data = json.loads(texts.get("study.json", "{}"))
    except ValueError:
        data = {}
    lines = []
    for keys in SUMMARY_FIELDS:
        v: Any = data
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        if v not in (None, "", [], {}):
            v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            lines.append(f"- {'.'.join(keys)}: {_clip(v, 2000, where + '/study.json')}")
    if lines:
        summary = "### study/study.json (summary)\n" + "\n".join(lines)
    docs = [f for f in PRIOR_WEIGHTS if f in texts]
    need = {f: len(texts[f]) for f in docs}
    study_left = max(0, study_budget - len(summary))
    a_need = len(texts.get("ADAPTOR.md", ""))
    a_alloc = min(a_need, adaptor_budget + max(0, study_left - sum(need.values())))
    alloc = _share(need, PRIOR_WEIGHTS, study_left + max(0, adaptor_budget - a_need))
    alloc["ADAPTOR.md"] = a_alloc
    listed = ", ".join(f"`{where}/{f}`" for f in PRIOR_FILES if f in texts)
    parts = [f"The full documents are readable in this archive under `{where}/` ({listed}). Below: excerpts "
             f"within a size budget; every cut is marked `…(truncated; full text in {where}/<file>)`."]
    if summary:
        parts.append(summary)
    for f in ("strategies.md", "brief.md", "ADAPTOR.md", "interfaces.md"):
        if f in texts:
            parts.append(f"### {'adaptor' if f == 'ADAPTOR.md' else 'study'}/{f}\n"
                         + _clip(texts[f], alloc[f], f"{where}/{f}"))
    return "\n\n".join(parts)


def run_study(model: System2Model, run_dir: Path, game: str, *, notes: str = "",
              realtime_hint: Optional[bool] = None, timeout_s: float = 2400.0, max_repairs: int = 2,
              log=print) -> dict[str, Any]:
    """``realtime_hint``: the user's pace override (True real-time, False turn-based, None: the study decides)."""
    out = run_dir / "study"
    out.mkdir(parents=True, exist_ok=True)
    notes_md = out / "notes.md"                    # the notes this study was made with
    if (out / "study.json").exists():
        try:
            data = load_study(out)
        except Exception as e:  # noqa: BLE001
            log(f"study: existing study.json invalid ({e}); redoing")
        else:
            if notes_md.exists() and notes_md.read_text().strip() != notes.strip():
                log("study: WARNING — --notes differ from the notes this study was made with (study/notes.md); "
                    "reusing it anyway — delete study/study.json to redo the study with the new notes")
            if realtime_hint is not None and data["realtime"] != realtime_hint:   # the override a fresh study gets
                log(f"study: --pace {_pace(realtime_hint)} overrides the study's {_pace(data['realtime'])}")
                data["realtime"] = realtime_hint
                (out / "study.json").write_text(json.dumps(data, indent=2))
            log(f"study: reusing {out / 'study.json'} ({_pace(data['realtime'])})")
            return data
    notes_md.write_text(notes)
    trace = TraceSink(out / "study_trace.jsonl")
    prompt = study_prompt(game, notes, realtime_hint, out, timeout_s)
    (out / "prompt.md").write_text(prompt)
    # A study cut off by its timeout (in this run or an earlier one) continues that conversation: its
    # research is in the session, and starting over would repeat it.
    session = _saved_session(out)
    if session:
        log(f"study: resuming the session {session} that ran out of time")
        prompt = FINISH_PROMPT
    try:
        for attempt in range(max_repairs + 1):
            try:
                res = model.run(prompt, cwd=out, system=STUDY_SYSTEM if session is None else None, agentic=True,
                                timeout_s=timeout_s if prompt is not FINISH_PROMPT else min(timeout_s, 1800.0),
                                trace=trace, session_id=session, web_search=True)
                session, timed_out = res.session_id, False
                (out / f"agent_output_{attempt}.md").write_text(res.text or "")
            except ModelTimeout as e:
                if not e.session_id:
                    raise
                session, timed_out = e.session_id, True
                log(f"study: the agent ran out of time ({timeout_s / 60:.0f} min); resuming its session to write "
                    "the deliverables")
            except ModelUnavailable:
                raise
            except Exception as e:  # noqa: BLE001 — a failed call: check what is on disk, then try again
                timed_out = session is not None
                log(f"study: the agent call failed ({type(e).__name__}: {str(e)[:160]})")
            if session:
                _session_file(out).write_text(json.dumps({"session_id": session}))
            try:
                data = load_study(out)
                if realtime_hint is not None:
                    data["realtime"] = realtime_hint
                    (out / "study.json").write_text(json.dumps(data, indent=2))
                _session_file(out).unlink(missing_ok=True)
                log(f"study: done — {_pace(data['realtime'])}, interface {data['interface'].get('chosen')}")
                return data
            except Exception as e:  # noqa: BLE001
                if timed_out:
                    prompt = FINISH_PROMPT
                    continue
                log(f"study: deliverables incomplete ({e}); asking for repair {attempt + 1}/{max_repairs}")
                prompt = (f"The study deliverables are incomplete: {e}. Fix the files in `{out}` so that "
                          f"study.json matches the schema exactly and all three markdown files exist.")
        raise RuntimeError("study phase did not produce valid deliverables")
    finally:
        trace.close()

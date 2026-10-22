"""Phase 2 — build the game adaptor.

Astra (agentic, full access so it can install software / start servers) turns
``study/interfaces.md`` into ``adaptor/`` satisfying ``games/adaptor.py``. The
orchestrator runs the generic smoke test, feeds failures back, and stops when
the adaptor passes or the user has to supply something (credentials, a running
game client) — that is written to ``adaptor/NEEDS.md``. A rerun validates an
existing adaptor first and only calls the agent when it fails.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from ..config import load_dotenv
from ..games import adaptor as adaptor_mod
from ..games import base as base_mod
from ..models.base import System2Model, TraceSink

ADAPT_SYSTEM = """You are the Meta-Harness *adapt* agent: you write the code that connects a real game to a
fixed Python interface so that model-driven harnesses can observe and act. You may install packages,
download tools, and start local servers/emulators. You never hard-code credentials: read them from
environment variables and document them in env.example. Verify your adaptor actually works against the
real game (or a faithful local instance) before declaring it done."""


def adapt_prompt(game: str, study: dict[str, Any], study_dir: Path, out_dir: Path, feedback: str = "",
                 notes: str = "", kwargs: Optional[dict[str, Any]] = None) -> str:
    contract = inspect.getdoc(adaptor_mod) or ""
    base_src = Path(base_mod.__file__).read_text()
    tier = int(study["tier"])
    check = f"validate_adaptor('{out_dir}', expected_tier={tier}" + (f", kwargs={kwargs!r})" if kwargs else ")")
    return f"""# Build the adaptor for **{game}**

Working directory: `{out_dir}`. Study documents from the previous phase are in `{study_dir}`
(read `interfaces.md` first; `study.json` summarises the chosen interface, credentials and scenarios).
User notes (platform, account, constraints — may be empty): {notes or '(none)'}

## Contract
{contract}

`metaharness/games/base.py` (import `ActionSpec` from `metaharness.games.base`):
```python
{base_src}
```

## Requirements
* Tier {tier}: {'real-time — implement RealtimeGame: tick(action) applies the *current* action, tick_s = the real cadence, ticks counts applied ticks; the runner calls tick on its own clock' if tier == 2 else 'turn-based — implement TurnGame: step(action) blocks until the game has applied the action and returns the next observation'}.
* `reset(seed)` must start a fresh episode and vary it by seed as described in study.json "scenarios":
  {study.get('scenarios', '')}
  If only some seeds are valid (a fixed set of levels / maps), declare them in `SEEDS`.
* `score` must reflect playing well as described in study.json "scoring": {study.get('scoring', '')}
* `describe()` tells a model everything it may know a priori: rules, objective, action semantics,
  observation fields. `action_spec` lists every legal action name with a one-line meaning and a `noop`.
  Actions with parameters use a compact string grammar documented in `describe()` (e.g. `"move 3 4"`).
* Observations are strict JSON (no NaN / inf, sets or objects) and compact (< 200 kB): structured state
  where the interface gives it; otherwise text; screenshots as file paths under this directory.
* If the game needs a running client/server/account, script the setup (`setup.sh` or Python) and run it.
  Every process, server or connection an instance starts is released by its `close()`.
  If something needs the user (account login, purchase, a GUI step), write `NEEDS.md` listing exactly
  what and stop (the orchestrator removes it once the adaptor validates).
* Write `ADAPTOR.md`, `requirements.txt` (if any), `env.example`. The evaluator's interpreter is
  `{sys.executable}` (Python {sys.version.split()[0]}; `metaharness` is importable there). Install
  requirements into *that* environment: `{sys.executable} -m pip install ...`.
* Test it yourself: `{sys.executable} -c "from metaharness.games.adaptor import validate_adaptor; print({check}.to_markdown())"`
  must print PASS (read its [!] warnings too). Then play a few random actions and check the observation and
  score make sense.
{('## Validation feedback from the orchestrator' + chr(10) + feedback) if feedback else ''}"""


def env_status(out_dir: Path) -> tuple[list[str], list[str]]:
    """(present, missing) environment variables declared in env.example."""
    load_dotenv()
    ex = out_dir / "env.example"
    if not ex.exists():
        return [], []
    names = []
    for line in ex.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name = line.split("=", 1)[0].strip()
            names.append(name[len("export "):].strip() if name.startswith("export ") else name)
    present = [n for n in names if os.environ.get(n)]
    return present, [n for n in names if n not in present]


def _fingerprint(p: Path) -> Optional[tuple[int, bytes]]:
    try:
        return p.stat().st_mtime_ns, p.read_bytes()
    except OSError:
        return None


def run_adapt(model: System2Model, run_dir: Path, game: str, study: dict[str, Any], *, notes: str = "",
              timeout_s: float = 3600.0, max_attempts: int = 4, kwargs: Optional[dict[str, Any]] = None,
              log=print) -> adaptor_mod.ValidationReport:
    """Build / repair ``run_dir/adaptor`` until it validates (``kwargs``: the user's game kwargs).

    An existing ``game.py`` is validated first; the agent is only called with the failing report.
    The run stops for the user only when validation fails and NEEDS.md was written or changed during
    this run (by the agent, or by the orchestrator for unset env.example variables); a NEEDS.md
    left over from an earlier run is fed back like any failure, and removed once the adaptor passes."""
    out = run_dir / "adaptor"
    study_dir = run_dir / "study"
    out.mkdir(parents=True, exist_ok=True)
    tier = int(study["tier"])
    needs = out / "NEEDS.md"
    needs_at_start = _fingerprint(needs)

    def validate() -> adaptor_mod.ValidationReport:
        rep = adaptor_mod.validate_adaptor(out, kwargs=kwargs, expected_tier=tier)
        (out / "validation.md").write_text(rep.to_markdown())
        (out / "validation.json").write_text(json.dumps(rep.to_json(), indent=2))
        if rep.ok:
            if needs.exists():
                needs.unlink()
                log("adapt: removed adaptor/NEEDS.md — the adaptor validates")
            missing = env_status(out)[1]
            if missing:
                log(f"adapt: env.example marks {missing} as required but they are unset; the adaptor validates without them")
        return rep

    def wait_for_user(rep: adaptor_mod.ValidationReport) -> bool:
        """After a failed validation: does the user have to act (NEEDS.md written this run)?"""
        missing = env_status(out)[1]
        if missing:
            text = "Set these variables in .env and rerun:\n" + "".join(f"- {m}\n" for m in missing)
            if needs.exists() and _fingerprint(needs) != needs_at_start:     # keep what the agent asked for
                text = needs.read_text().rstrip() + "\n\n" + text
            needs.write_text(text)
            why = f"missing environment variables {missing}"
        elif _fingerprint(needs) != needs_at_start and needs.exists():
            why = "the model needs user input"
        else:
            return False
        log(f"adapt: {why} — see adaptor/NEEDS.md")
        rep.error = f"user input required ({why}); adaptor/NEEDS.md:\n{needs.read_text()[:2000]}" + (
            f"\n\nvalidation error:\n{rep.error}" if rep.error else "")
        return True

    feedback = ""
    if (out / "game.py").exists():                 # resume: the adaptor may pass as it is (env vars now set...)
        rep = validate()
        if rep.ok:
            log("adapt: reusing validated adaptor")
            return rep
        if wait_for_user(rep):
            return rep
        feedback = rep.to_markdown()
        log("adapt: the existing adaptor fails validation; repairing")
    trace = TraceSink(out / "adapt_trace.jsonl")
    session = None
    rep = adaptor_mod.ValidationReport(ok=False, error="not run")
    try:
        for attempt in range(max_attempts):
            prompt = adapt_prompt(game, study, study_dir, out, feedback, notes=notes, kwargs=kwargs)
            (out / f"prompt_{attempt}.md").write_text(prompt)
            res = model.run(prompt if session is None else f"{feedback}\n\nFix the adaptor in `{out}` accordingly.",
                            cwd=out, system=ADAPT_SYSTEM if session is None else None, agentic=True,
                            timeout_s=timeout_s, trace=trace, session_id=session, web_search=True, full_access=True)
            session = res.session_id
            (out / f"agent_output_{attempt}.md").write_text(res.text or "")
            rep = validate()
            if rep.ok:
                log(f"adapt: adaptor validated after {attempt + 1} attempt(s)")
                break
            if wait_for_user(rep):
                break
            feedback = rep.to_markdown()
            log(f"adapt: validation failed (attempt {attempt + 1}/{max_attempts}); feeding back")
    finally:
        trace.close()
    return rep

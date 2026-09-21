"""Phase 2 — build the game adaptor.

Astra (agentic, full access so it can install software / start servers) turns
``study/interfaces.md`` into ``adaptor/`` satisfying ``games/adaptor.py``. The
orchestrator runs the generic smoke test, feeds failures back, and stops when
the adaptor passes or the user has to supply something (credentials, a running
game client) — that is written to ``adaptor/NEEDS.md``.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Optional

from ..config import load_dotenv
from ..games import adaptor as adaptor_mod
from ..games import base as base_mod
from ..models.base import System2Model, TraceSink
from .study import study_text

ADAPT_SYSTEM = """You are the Meta-Harness *adapt* agent: you write the code that connects a real game to a
fixed Python interface so that model-driven harnesses can observe and act. You may install packages,
download tools, and start local servers/emulators. You never hard-code credentials: read them from
environment variables and document them in env.example. Verify your adaptor actually works against the
real game (or a faithful local instance) before declaring it done."""


def adapt_prompt(game: str, study: dict[str, Any], study_dir: Path, out_dir: Path, feedback: str = "") -> str:
    contract = inspect.getdoc(adaptor_mod) or ""
    base_src = Path(base_mod.__file__).read_text()
    return f"""# Build the adaptor for **{game}**

Working directory: `{out_dir}`. Study documents from the previous phase are in `{study_dir}`
(read `interfaces.md` first; `study.json` summarises the chosen interface, credentials and scenarios).

## Contract
{contract}

`metaharness/games/base.py` (import `ActionSpec` from `metaharness.games.base`):
```python
{base_src}
```

## Requirements
* Tier {study['tier']}: {'real-time — implement RealtimeGame: tick(action) applies the *current* action, tick_s = the real cadence, ticks counts applied ticks; the runner calls tick on its own clock' if int(study['tier']) == 2 else 'turn-based — implement TurnGame: step(action) blocks until the game has applied the action and returns the next observation'}.
* `reset(seed)` must start a fresh episode and vary it by seed as described in study.json "scenarios":
  {study.get('scenarios', '')}
* `score` must reflect playing well as described in study.json "scoring": {study.get('scoring', '')}
* `describe()` tells a model everything it may know a priori: rules, objective, action semantics,
  observation fields. `action_spec` lists every legal action name with a one-line meaning and a `noop`.
  Actions with parameters use a compact string grammar documented in `describe()` (e.g. `"move 3 4"`).
* Observations are JSON-serialisable and compact (< 200 kB): structured state where the interface
  gives it; otherwise text; screenshots as file paths under this directory.
* If the game needs a running client/server/account, script the setup (`setup.sh` or Python) and run it.
  If something needs the user (account login, purchase, a GUI step), write `NEEDS.md` listing exactly
  what and stop.
* Write `ADAPTOR.md`, `requirements.txt` (if any), `env.example`. Install requirements into the
  current Python environment (`uv pip install ...` or `pip install ...`).
* Test it yourself: `python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('{out_dir}').to_markdown())"`
  must print PASS. Then play a few random actions and check the observation and score make sense.
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
            names.append(line.split("=", 1)[0].strip())
    present = [n for n in names if os.environ.get(n)]
    return present, [n for n in names if n not in present]


def run_adapt(model: System2Model, run_dir: Path, game: str, study: dict[str, Any], *,
              timeout_s: float = 3600.0, max_attempts: int = 4, kwargs: Optional[dict[str, Any]] = None,
              log=print) -> adaptor_mod.ValidationReport:
    out = run_dir / "adaptor"
    study_dir = run_dir / "study"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "validation.json").exists():
        prev = json.loads((out / "validation.json").read_text())
        if prev.get("ok"):
            rep = adaptor_mod.validate_adaptor(out, kwargs=kwargs)
            (out / "validation.md").write_text(rep.to_markdown())
            if rep.ok:
                log("adapt: reusing validated adaptor")
                return rep
            log("adapt: previously validated adaptor no longer passes; repairing")
    trace = TraceSink(out / "adapt_trace.jsonl")
    session = None
    feedback = ""
    rep = adaptor_mod.ValidationReport(ok=False, error="not run")
    try:
        for attempt in range(max_attempts):
            prompt = adapt_prompt(game, study, study_dir, out, feedback)
            (out / f"prompt_{attempt}.md").write_text(prompt)
            res = model.run(prompt if session is None else f"{feedback}\n\nFix the adaptor in `{out}` accordingly.",
                            cwd=out, system=ADAPT_SYSTEM if session is None else None, agentic=True,
                            timeout_s=timeout_s, trace=trace, session_id=session, web_search=True, full_access=True)
            session = res.session_id
            (out / f"agent_output_{attempt}.md").write_text(res.text or "")
            if (out / "NEEDS.md").exists() and not (out / "game.py").exists():
                log("adapt: the model needs user input — see adaptor/NEEDS.md")
                rep = adaptor_mod.ValidationReport(ok=False, error="user input required: " + (out / "NEEDS.md").read_text()[:2000])
                break
            present, missing = env_status(out)
            if missing:
                (out / "NEEDS.md").write_text("Set these variables in .env and rerun:\n" + "".join(f"- {m}\n" for m in missing))
                log(f"adapt: missing environment variables {missing} — see adaptor/NEEDS.md")
                rep = adaptor_mod.ValidationReport(ok=False, error=f"missing env vars: {missing}")
                break
            rep = adaptor_mod.validate_adaptor(out, kwargs=kwargs)
            (out / "validation.md").write_text(rep.to_markdown())
            (out / "validation.json").write_text(json.dumps({"ok": rep.ok, "checks": rep.checks, "error": rep.error}, indent=2))
            if rep.ok:
                log(f"adapt: adaptor validated after {attempt + 1} attempt(s)")
                break
            feedback = rep.to_markdown()
            log(f"adapt: validation failed (attempt {attempt + 1}/{max_attempts}); feeding back")
    finally:
        trace.close()
    return rep

"""The proposer: a coding model (Astra via codex) that reads the archive and
writes the next harness candidate.

Two modes, same prompt:
* agentic — the model works inside ``runs/<exp>/`` with shell + file tools,
  reads whatever it wants (all prior harnesses, traces, scores) and writes
  ``candidates/NNNN/harness.py`` + ``proposal.md`` itself.
* text — (scripted/test models, or when tools are unavailable) the model
  answers with a fenced ``python`` block that we write to the same place.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Optional

from ..config import REPO_ROOT
from ..models.base import ModelUnavailable, System2Model, TraceSink
from .archive import Archive, Candidate
from .generation import describe as describe_directive
from .generation import duplicate_of, read_lessons

PROPOSER_SYSTEM = """You are the Meta-Harness proposer. You improve the *harness* — the code that connects
foundation models to a game — not the models. Work like a researcher: read the archive, form a
hypothesis about why the best candidate fails, write ONE new harness that tests it, and explain the
hypothesis in proposal.md so later iterations can build on it. Prefer bold structural changes over
parameter tweaks when the leaderboard is flat; prefer careful fixes when a candidate errors out."""


def _fenced_python(text: str) -> Optional[str]:
    m = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return max(m, key=len) if m else None


def build_prompt(archive: Archive, candidate: Candidate, contract_doc: str, task_md: str, extra: str = "",
                 iteration: Optional[tuple[int, int]] = None, timeout_s: Optional[float] = None,
                 directive: Optional[dict] = None, siblings: tuple[str, ...] = ()) -> str:
    lb = archive.write_leaderboard()
    skip = set(siblings) | {candidate.dir.name}
    prior = "\n".join(f"- candidates/{c.dir.name}/ (harness.py, proposal.md, eval.json, digest.md, trace_seed*.jsonl)"
                      for c in archive.candidates() if c.dir.name not in skip)
    if (archive.root / "prior").is_dir():
        prior += "\n- prior/ (the full study documents and the adaptor's interface documentation)"
    lessons = read_lessons(archive.root)
    if lessons:
        prior += "\n- lessons.md (the evaluator's record per generation: what each candidate tried, what happened, why)"
    when = ""
    if iteration:
        i, n = iteration
        when = (f"\nThis is generation {i} of {n}"
                + (" — the last one: prefer a careful, testable improvement over a risky rewrite" if i == n else "")
                + ".")
    if timeout_s:
        when += (f" You have {timeout_s / 60:.0f} minutes. Write `harness.py` and `proposal.md` first, then refine: "
                 "a harness written before the deadline is still evaluated, an unwritten one is lost.")
    when += recent_latency_text(archive)
    if siblings:
        when += (f"\nSibling proposers are writing {', '.join(f'candidates/{s}/' for s in siblings)} right now: those "
                 "directories are unfinished and unevaluated — ignore them and never write into them.")
    directive_md = f"\n## Your directive\n{describe_directive(directive)}\n" if directive else ""
    lessons_md = f"\n## Lessons so far (lessons.md)\n{lessons}\n" if lessons else ""
    return f"""# Meta-Harness generation -> candidate `{candidate.dir.name}`{when}

Working directory: `{archive.root}` (you have full read/write access here).

## Task
{task_md}

## Harness contract (must be satisfied exactly)
```
{contract_doc}
```
{extra}{directive_md}
## Archive
{lb}
Prior candidates (read their code, proposals, eval.json and the JSONL traces — traces contain every
model call with request/response/latency, every game tick/step, observations at the start and end of each
episode, on every turn-game step and whenever a real-time score changes, and how each episode ended):
{prior or '- (none yet)'}
{lessons_md}
## What to do
1. Read your parent candidate(s), the best candidate and the traces of their worst episodes (each evaluation's
   `digest.md` indexes its traces: every model call, step and score change). Diagnose the failure mode
   concretely (quote evidence from traces). Check lessons.md so you do not repeat a refuted hypothesis. Look
   at the `seeds median / min` column and any validation results: a leader with identical scores on every
   seed is a scenario-independent script — ask what happens on a scenario it was not written for, and make
   the plan detect its own failure (`ctx.progress()`) and re-plan through System-2 instead of tuning the script.
2. Write `candidates/{candidate.dir.name}/harness.py` — a complete module with `build(ctx)`.
   It must import only the standard library and `metaharness.*`. A harness whose code repeats an existing
   candidate's (comments aside) is not evaluated.
3. Write `candidates/{candidate.dir.name}/proposal.md`: parent(s), hypothesis, what changed, the lesson or
   evidence it acts on, what evidence would confirm/refute it, and (tier 2) the System-1 state schema,
   questions, thresholds and fallback policy.
4. If you cannot use tools, reply with the full harness in ONE ```python fenced block and the
   proposal text after it.
Before the real evaluation the candidate is *pre-flighted*: one short episode with stand-in models
that return arbitrary text / first options on a virtual clock. Any exception there (a contract
misuse, an unhandled parse failure of model output) is sent back to you once for repair; a harness
must survive garbage model output without crashing.
Do not modify any other candidate. To watch your harness play before the evaluation, run (from `{REPO_ROOT}`)
`{sys.executable} -m metaharness try {archive.root}/candidates/{candidate.dir.name}/harness.py --seed S`: one
episode (`--minutes`, default 5) on training seed S with this archive's game and settings, Jev live, Astra off
unless `--astra`; it prints the score and writes the trace under `candidates/{candidate.dir.name}/tries/`. Only
training seeds are accepted. Write the harness first: tries play in real time and count against your deadline.
Read only this archive and the interface documentation above. Do not inspect game implementation
source, other experiment archives, credentials, or held-out games/results. Infer semantics from the
public game description and observations supplied through ctx. Do not call model backends directly:
all inference must go through ctx.think/ctx.decide so the evaluator can account for it.
Model calls still in flight when the game ends or run_episode returns are cancelled by the evaluator;
any other background work must stop when ctx.done. The default evaluation resets both memory and module
globals per episode; only an explicit shared-memory protocol allows cross-episode caching. The manifest is
authoritative."""


def recent_latency_text(archive: Archive, last: int = 3) -> str:
    """System-2 latency observed in the archive's most recent evaluations: a shared endpoint's latency
    drifts with load, so the calibration in the task text goes stale and timeouts sized from it lose
    calls (a Jericho candidate lost 35 of 74 calls to an 85 s timeout while answers took up to 82 s)."""
    evals = [c.eval() for c in archive.candidates()]
    evals = [e for e in evals if e and e.get("episodes")][-last:]
    p50s, maxs, failed, total = [], [], 0, 0
    for e in evals:
        failed += int(e.get("failed_s2") or 0)
        total += int(e.get("system2_calls") or 0)
        for ep in e["episodes"]:
            st = ep.get("latency_s2") or {}
            if st.get("n"):
                p50s.append(float(st["p50"]))
                maxs.append(float(st["max"]))
    if not p50s:
        return ""
    p50s.sort()
    return (f"\nSystem-2 latency in the last {len(evals)} evaluation(s) of this archive: median {p50s[len(p50s) // 2]:.0f} s, "
            f"worst {max(maxs):.0f} s per call; {failed} of {total + failed} calls failed (timeouts included). Size call "
            "timeouts above the worst observed latency, not from the calibration in the task text.")


def tools_text(python: str, repo_root: str, dev_seeds: list[int], dev_evals: int) -> str:
    """What the proposer may run itself (the ``extra`` of the prompt): the evaluator's pre-flight and,
    on dedicated dev seeds, a real-model evaluation of its draft. The dev seeds are neither training
    nor held-out seeds, so what it learns there is procedure, not answers."""
    cmd = f"PYTHONPATH={repo_root} {python} -m metaharness"
    text = f"""
## Tools you may run yourself (from the archive root)
- Pre-flight (seconds, no model calls, the same check the evaluator runs first):
  `{cmd} preflight candidates/NNNN/harness.py`
  Run it before you finish; a failure costs you a repair round, a second failure the whole candidate."""
    if dev_seeds and dev_evals > 0:
        text += f"""
- Dev evaluation (real models, the evaluation's budgets, one episode per dev seed, output under
  `candidates/NNNN/dev/<timestamp>/` with eval.json, digest.md and the traces):
  `{cmd} dev-eval candidates/NNNN/harness.py` (dev seeds {dev_seeds}; add `--seeds` to pick some of them)
  Use at most {dev_evals} dev evaluation(s) per candidate: they take as long as a real episode, so start
  one only when the harness is complete, and read its digest and traces before deciding what to fix.
  Dev results are not the score: only the evaluator's training seeds count."""
    return text


def propose(
    model: System2Model,
    archive: Archive,
    *,
    contract_doc: str,
    task_md: str,
    extra: str = "",
    agentic: bool = True,
    trace: Optional[TraceSink] = None,
    timeout_s: float = 1800.0,
    candidate: Optional[Candidate] = None,
    iteration: Optional[tuple[int, int]] = None,
    directive: Optional[dict] = None,
    siblings: tuple[str, ...] = (),
    full_access: bool = False,
) -> Candidate:
    """``full_access``: the agent may run the evaluator's own tools (pre-flight, a dev evaluation through
    the model backends), which need the network and directories outside the archive."""
    cand = candidate or archive.new_candidate()
    if directive:
        (cand.dir / "directive.json").write_text(json.dumps(directive, indent=2) + "\n")
    prompt = build_prompt(archive, cand, contract_doc, task_md, extra, iteration=iteration, timeout_s=timeout_s,
                          directive=directive, siblings=siblings)
    (cand.dir / "prompt.md").write_text(prompt)
    try:
        res = model.run(prompt, cwd=archive.root, system=PROPOSER_SYSTEM, agentic=agentic,
                        timeout_s=timeout_s, trace=trace, full_access=full_access)
    except TimeoutError as exc:
        # A CLI can finish writing the deliverable and time out during its final checks. Keep
        # whatever harness it wrote as an explicitly incomplete proposal: pre-flight (load errors
        # get a repair round in the same session) and the evaluator establish whether it works.
        # One that wrote nothing gets its session back once to write it from what it worked out.
        session = getattr(exc, "session_id", None)
        (cand.dir / "proposal_error.json").write_text(json.dumps({
            "error": str(exc), "type": type(exc).__name__, "status": "timed_out", "session_id": session,
        }, indent=2) + "\n")
        if session:
            (cand.dir / "session.json").write_text(json.dumps({"session_id": session}))
        copy = duplicate_of(archive, cand) if cand.harness_path.exists() else None
        if session and (not cand.harness_path.exists() or copy):
            # a proposer typically starts from a copy of its parent: out of time before editing, the copy is all
            why = "Your time is up." + (f" Your harness.py is still an unmodified copy of candidates/{copy[0].dir.name}/."
                                        if copy else "")
            _finish(model, archive, cand, session, why, agentic=agentic, trace=trace,
                    timeout_s=min(900.0, timeout_s / 2), full_access=full_access)
        if cand.harness_path.exists():
            if not (cand.dir / "proposal.md").exists():
                (cand.dir / "proposal.md").write_text("(the proposer timed out before writing proposal.md)\n")
            return cand
        raise
    (cand.dir / "proposer_output.md").write_text(res.text or "")
    if res.session_id:
        (cand.dir / "session.json").write_text(json.dumps({"session_id": res.session_id}))
    if res.session_id and not cand.harness_path.exists() and not _fenced_python(res.text or ""):
        # an agent can end its turn after planning without writing the file: once more, same session
        _finish(model, archive, cand, res.session_id, "You ended without writing the harness.", agentic=agentic,
                trace=trace, timeout_s=min(900.0, timeout_s / 2), full_access=full_access)
    elif res.session_id and cand.harness_path.exists():
        copy = duplicate_of(archive, cand)
        if copy:
            _finish(model, archive, cand, res.session_id, f"Your harness.py is an unmodified copy of "
                    f"candidates/{copy[0].dir.name}/.", agentic=agentic, trace=trace, timeout_s=min(900.0, timeout_s / 2),
                    full_access=full_access)
    if isinstance(res.raw, list):
        (cand.dir / "proposer_events.jsonl").write_text(
            "".join(json.dumps(event, default=str) + "\n" for event in res.raw))
    if not cand.harness_path.exists():
        code = _fenced_python(res.text or "")
        if code:
            cand.harness_path.write_text(code)
            if not (cand.dir / "proposal.md").exists():
                after = (res.text or "").split("```")[-1].strip()
                (cand.dir / "proposal.md").write_text(after or "(no proposal text)\n")
    if not (cand.dir / "proposal.md").exists():
        (cand.dir / "proposal.md").write_text("(proposer wrote no proposal.md)\n")
    return cand


def _finish(model: System2Model, archive: Archive, cand: Candidate, session: str, why: str, *, agentic: bool,
            trace: Optional[TraceSink], timeout_s: float, full_access: bool = False) -> None:
    """Ask a proposer that produced no harness to write it now, from what it worked out (same session)."""
    from ..models._cli import STOPPING
    if STOPPING.is_set():
        return
    prompt = (f"{why} From what you have already worked out, write `candidates/{cand.dir.name}/harness.py` (a "
              f"complete module with build(ctx)) and `candidates/{cand.dir.name}/proposal.md` now, then stop. A copy "
              "of an existing candidate is not evaluated: write your own change, even a partial one.")
    try:
        res = model.run(prompt, cwd=archive.root, session_id=session, agentic=agentic, timeout_s=timeout_s,
                        trace=trace, full_access=full_access)
    except ModelUnavailable:
        raise
    except Exception:  # noqa: BLE001 — the original outcome is what gets reported
        return
    (cand.dir / "proposer_finish_output.md").write_text(res.text or "")
    if not cand.harness_path.exists():
        code = _fenced_python(res.text or "")
        if code:
            cand.harness_path.write_text(code)


def repair(
    model: System2Model,
    cand: Candidate,
    report: str,
    *,
    agentic: bool = True,
    trace: Optional[TraceSink] = None,
    timeout_s: float = 900.0,
    full_access: bool = False,
) -> bool:
    """One repair round after a failed pre-flight: same session if we have one."""
    session = None
    sp = cand.dir / "session.json"
    if sp.exists():
        session = json.loads(sp.read_text()).get("session_id")
    prompt = (f"Pre-flight of `candidates/{cand.dir.name}/harness.py` failed. Fix the harness IN PLACE "
              f"(same file) so it survives this and similar model-output/contract problems; keep the design. "
              f"Append one line about the cause to proposal.md.\n\nPre-flight report:\n```\n{report[-6000:]}\n```"
              + ("" if agentic else "\nReply with the complete fixed harness in ONE ```python fenced block."))
    (cand.dir / "repair_prompt.md").write_text(prompt)
    try:
        res = model.run(prompt, cwd=cand.dir.parent.parent, session_id=session, agentic=agentic,
                        timeout_s=timeout_s, trace=trace, full_access=full_access)
    except ModelUnavailable:
        raise                     # the search must stop, not evaluate an unrepaired candidate
    except Exception as exc:  # noqa: BLE001 — a failed repair is data
        (cand.dir / "repair_error.json").write_text(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        return False
    (cand.dir / "repair_output.md").write_text(res.text or "")
    if not agentic:
        code = _fenced_python(res.text or "")
        if code:
            cand.harness_path.write_text(code)
    return True

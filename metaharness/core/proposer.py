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
from typing import Optional

from ..models.base import ModelUnavailable, System2Model, TraceSink
from .archive import Archive, Candidate

PROPOSER_SYSTEM = """You are the Meta-Harness proposer. You improve the *harness* — the code that connects
foundation models to a game — not the models. Work like a researcher: read the archive, form a
hypothesis about why the best candidate fails, write ONE new harness that tests it, and explain the
hypothesis in proposal.md so later iterations can build on it. Prefer bold structural changes over
parameter tweaks when the leaderboard is flat; prefer careful fixes when a candidate errors out."""


def _fenced_python(text: str) -> Optional[str]:
    m = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return max(m, key=len) if m else None


def build_prompt(archive: Archive, candidate: Candidate, contract_doc: str, task_md: str, extra: str = "",
                 iteration: Optional[tuple[int, int]] = None, timeout_s: Optional[float] = None) -> str:
    lb = archive.write_leaderboard()
    prior = "\n".join(f"- candidates/{c.dir.name}/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)"
                      for c in archive.candidates() if c.idx != candidate.idx)
    if (archive.root / "prior").is_dir():
        prior += "\n- prior/ (the full study documents and the adaptor's interface documentation)"
    when = ""
    if iteration:
        i, n = iteration
        when = (f"\nThis is iteration {i} of {n}"
                + (" — the last one: prefer a careful, testable improvement over a risky rewrite" if i == n else "")
                + ".")
    if timeout_s:
        when += (f" You have {timeout_s / 60:.0f} minutes. Write `harness.py` and `proposal.md` first, then refine: "
                 "a harness written before the deadline is still evaluated, an unwritten one is lost.")
    return f"""# Meta-Harness iteration -> candidate `{candidate.dir.name}`{when}

Working directory: `{archive.root}` (you have full read/write access here).

## Task
{task_md}

## Harness contract (must be satisfied exactly)
```
{contract_doc}
```
{extra}

## Archive
{lb}
Prior candidates (read their code, proposals, eval.json and the JSONL traces — traces contain every
model call with request/response/latency, every game tick/step, observations at the start and end of each
episode, on every turn-game step and whenever a real-time score changes, and how each episode ended):
{prior or '- (none yet)'}

## What to do
1. Read the best candidate(s) and the traces of their worst episodes. Diagnose the failure mode
   concretely (quote evidence from traces).
2. Write `candidates/{candidate.dir.name}/harness.py` — a complete module with `build(ctx)`.
   It must import only the standard library and `metaharness.*`.
3. Write `candidates/{candidate.dir.name}/proposal.md`: hypothesis, what changed vs. the parent
   candidate, what evidence would confirm/refute it.
4. Append 3-6 bullet lessons to `notes.md` at the archive root (a running lab notebook: what
   generalised, what was a dead end, which contract details bit you). Read it first.
5. If you cannot use tools, reply with the full harness in ONE ```python fenced block and the
   proposal text after it.
Before the real evaluation the candidate is *pre-flighted*: one short episode with stand-in models
that return arbitrary text / first options on a virtual clock. Any exception there (a contract
misuse, an unhandled parse failure of model output) is sent back to you once for repair; a harness
must survive garbage model output without crashing.
Do not modify any other candidate. Do not run long experiments yourself; the evaluator will.
Read only this archive and the interface documentation above. Do not inspect game implementation
source, other experiment archives, credentials, or held-out games/results. Infer semantics from the
public game description and observations supplied through ctx. Do not call model backends directly:
all inference must go through ctx.think/ctx.decide so the evaluator can account for it.
Model calls still in flight when the game ends or run_episode returns are cancelled by the evaluator;
any other background work must stop when ctx.done. The default evaluation resets both memory and module
globals per episode; only an explicit shared-memory protocol allows cross-episode caching. The manifest is
authoritative."""


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
) -> Candidate:
    cand = candidate or archive.new_candidate()
    prompt = build_prompt(archive, cand, contract_doc, task_md, extra, iteration=iteration, timeout_s=timeout_s)
    (cand.dir / "prompt.md").write_text(prompt)
    try:
        res = model.run(prompt, cwd=archive.root, system=PROPOSER_SYSTEM, agentic=agentic,
                        timeout_s=timeout_s, trace=trace)
    except TimeoutError as exc:
        # A CLI can finish writing the deliverable and time out during its final checks. Keep
        # whatever harness it wrote as an explicitly incomplete proposal: pre-flight (load errors
        # get a repair round in the same session) and the evaluator establish whether it works.
        session = getattr(exc, "session_id", None)
        (cand.dir / "proposal_error.json").write_text(json.dumps({
            "error": str(exc), "type": type(exc).__name__, "status": "timed_out", "session_id": session,
        }, indent=2) + "\n")
        if session:
            (cand.dir / "session.json").write_text(json.dumps({"session_id": session}))
        if cand.harness_path.exists():
            if not (cand.dir / "proposal.md").exists():
                (cand.dir / "proposal.md").write_text("(the proposer timed out before writing proposal.md)\n")
            return cand
        raise
    (cand.dir / "proposer_output.md").write_text(res.text or "")
    if res.session_id:
        (cand.dir / "session.json").write_text(json.dumps({"session_id": res.session_id}))
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


def repair(
    model: System2Model,
    cand: Candidate,
    report: str,
    *,
    agentic: bool = True,
    trace: Optional[TraceSink] = None,
    timeout_s: float = 900.0,
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
                        timeout_s=timeout_s, trace=trace)
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

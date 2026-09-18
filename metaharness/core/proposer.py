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

import re
from pathlib import Path
from typing import Optional

from ..models.base import System2Model, TraceSink
from .archive import Archive, Candidate

PROPOSER_SYSTEM = """You are the Meta-Harness proposer. You improve the *harness* — the code that connects
foundation models to a game — not the models. Work like a researcher: read the archive, form a
hypothesis about why the best candidate fails, write ONE new harness that tests it, and explain the
hypothesis in proposal.md so later iterations can build on it. Prefer bold structural changes over
parameter tweaks when the leaderboard is flat; prefer careful fixes when a candidate errors out."""


def _fenced_python(text: str) -> Optional[str]:
    m = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return max(m, key=len) if m else None


def build_prompt(archive: Archive, candidate: Candidate, contract_doc: str, task_md: str, extra: str = "") -> str:
    lb = archive.write_leaderboard()
    prior = "\n".join(f"- candidates/{c.dir.name}/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)"
                      for c in archive.candidates() if c.idx != candidate.idx)
    return f"""# Meta-Harness iteration -> candidate `{candidate.dir.name}`

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
model call with request/response/latency and every game tick/step):
{prior or '- (none yet)'}

## What to do
1. Read the best candidate(s) and the traces of their worst episodes. Diagnose the failure mode
   concretely (quote evidence from traces).
2. Write `candidates/{candidate.dir.name}/harness.py` — a complete module with `build(ctx)`.
   It must import only the standard library and `metaharness.*`.
3. Write `candidates/{candidate.dir.name}/proposal.md`: hypothesis, what changed vs. the parent
   candidate, what evidence would confirm/refute it.
4. If you cannot use tools, reply with the full harness in ONE ```python fenced block and the
   proposal text after it.
Do not modify any other candidate. Do not run long experiments yourself; the evaluator will."""


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
) -> Candidate:
    cand = archive.new_candidate()
    prompt = build_prompt(archive, cand, contract_doc, task_md, extra)
    (cand.dir / "prompt.md").write_text(prompt)
    res = model.run(prompt, cwd=archive.root, system=PROPOSER_SYSTEM, agentic=agentic,
                    timeout_s=timeout_s, trace=trace)
    (cand.dir / "proposer_output.md").write_text(res.text or "")
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

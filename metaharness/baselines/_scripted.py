"""Offline stand-ins for dry runs and tests (no network).

* ``heuristic_snake_policy`` — a greedy-but-safe snake policy used as the fake
  Jev: moves toward the food unless that cell is dangerous.
* ``scripted_system2`` — fake Astra that answers snake prompts with a greedy
  move and puzzle prompts with ACTION1.
* ``scripted_proposer`` — fake proposer that "proposes" the bundled reference
  harness for the tier, so the whole loop can be exercised offline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..models.mock import HeuristicSystem1, ScriptedSystem2

_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _greedy(obs: Mapping[str, Any]) -> str:
    dx, dy = obs.get("food_delta", [0, 0])
    danger = obs.get("danger", {})
    heading = obs.get("heading", "right")
    prefs = []
    if abs(dx) >= abs(dy):
        prefs += ["right" if dx > 0 else "left", "down" if dy > 0 else "up"]
    else:
        prefs += ["down" if dy > 0 else "up", "right" if dx > 0 else "left"]
    prefs += [d for d in ("up", "down", "left", "right") if d not in prefs]
    for d in prefs:
        if d != _OPP.get(heading) and not danger.get(d, False):
            return d
    return heading


def heuristic_snake_policy(state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
    obs = state if isinstance(state, Mapping) else {}
    if "obs" in obs and isinstance(obs["obs"], Mapping):
        obs = obs["obs"]
    move = _greedy(obs) if "danger" in obs else None
    out: dict[str, Any] = {}
    for qid, q in questions.items():
        qj = q.to_json() if hasattr(q, "to_json") else dict(q)
        if qj.get("type") == "choice":
            opts = list(qj["criteria"].keys())
            out[qid] = move if move in opts else opts[0]
        elif qj.get("type") == "noul":
            out[qid] = 0.5
        else:
            out[qid] = 0
    return out


def scripted_system2() -> ScriptedSystem2:
    def fn(prompt: str) -> str:
        if "head=" in prompt and "danger=" in prompt:
            m = re.search(r"head=\[(-?\d+), (-?\d+)\].*?food=\[(-?\d+), (-?\d+)\].*?danger=(\{.*?\})", prompt, re.S)
            if m:
                hx, hy, fx, fy = map(int, m.groups()[:4])
                danger = json.loads(m.group(5).replace("'", '"').replace("True", "true").replace("False", "false"))
                return _greedy({"food_delta": [fx - hx, fy - hy], "danger": danger, "heading": "right"})
        if "ACTION1..ACTION6" in prompt or "ACTION" in prompt:
            return "ACTION1"
        return "PLAN: move toward the food along the longer axis first; avoid cells marked dangerous; " \
               "never reverse. Re-plan every 20 ticks."
    return ScriptedSystem2(fn=fn, latency_s=2.0)


def scripted_proposer(tier_mod) -> ScriptedSystem2:
    ref = Path(__file__).resolve().parent / ("tier2_reference.py" if "tier2" in tier_mod.__name__ else "tier1_reference.py")
    code = ref.read_text()
    reply = f"Hypothesis: see module docstring.\n\n```python\n{code}\n```\nProposal: the reference two-timescale harness."
    return ScriptedSystem2(replies=[reply], latency_s=5.0)

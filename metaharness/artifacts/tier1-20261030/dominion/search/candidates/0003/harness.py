"""Batch planning harness for the Dominion turn adaptor.

The model is asked for a short, decision-id tagged action plan.  Commands are
executed one at a time and are discarded as soon as the observed prompt no
longer matches the plan.  This keeps the latency advantage of planning several
steps ahead without trusting a stale plan through an attack or nested effect.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def _json(value: Any, limit: int = 11000) -> str:
    try:
        text = json.dumps(value, separators=(",", ":"), default=str)
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...(cut)"


def _prompt(obs: Any) -> dict[str, Any]:
    return obs.get("prompt", {}) if isinstance(obs, dict) else {}


def _did(obs: Any) -> Any:
    """Decision IDs are top-level in the adaptor observation."""
    if isinstance(obs, dict):
        if obs.get("decision_id") is not None:
            return obs.get("decision_id")
        return _prompt(obs).get("decision_id")
    return None


def _fallback(obs: Any) -> str | None:
    """Return a legal, conservative action when Astra is unavailable/garbled."""
    p = _prompt(obs)
    if not isinstance(p, dict):
        return "wait"
    did = _did(obs)
    if did is None:
        return "wait"
    opts = p.get("options") or []
    ids = [str(x.get("id")) for x in opts if isinstance(x, dict) and x.get("id") is not None]
    try:
        lo = int(p.get("min", 0))
        hi = int(p.get("max", 0))
    except Exception:
        lo, hi = 0, 0
    if lo <= 0:
        # Passing is useful for optional action, trash, and reaction prompts.
        return f"choose {did}"
    if len(ids) >= lo:
        return "choose " + str(did) + " " + " ".join(ids[: min(hi, len(ids))])
    return "wait"


def _commands(text: str) -> list[tuple[int, list[str]]]:
    """Extract only command-looking lines; explanations and code fences are ignored."""
    out: list[tuple[int, list[str]]] = []
    for raw in (text or "").splitlines():
        line = raw.strip().strip("`*-").strip()
        m = re.match(r"(?i)^choose\s+(\d+)(?:\s+(.+?))?\s*$", line)
        if not m:
            continue
        did = int(m.group(1))
        tail = m.group(2) or ""
        # IDs are opaque strings in the adaptor, but current Dominion IDs are
        # simple tokens.  Remove JSON punctuation commonly emitted by models.
        ids = [x.strip("[]{}\"',") for x in tail.split()]
        ids = [x for x in ids if x and x not in ("#",)]
        out.append((did, ids))
    return out


def _valid(obs: Any, did: int, ids: list[str]) -> bool:
    p = _prompt(obs)
    if not isinstance(p, dict) or str(_did(obs)) != str(did):
        return False
    options = p.get("options") or []
    legal = {str(x.get("id")) for x in options if isinstance(x, dict) and x.get("id") is not None}
    try:
        lo, hi = int(p.get("min", 0)), int(p.get("max", 0))
    except Exception:
        return False
    return lo <= len(ids) <= hi and len(ids) == len(set(ids)) and set(ids) <= legal


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if ctx.system2 is None:
            return
        obs = game.observe()
        session = None
        plan: list[tuple[int, list[str]]] = []
        previous = None
        failures = 0
        while not ctx.done and not getattr(game, "done", False):
            p = _prompt(obs)
            did = _did(obs)
            # A plan is valid only at the exact prompt for which it was made.
            if plan and (did is None or str(plan[0][0]) != str(did) or not _valid(obs, *plan[0])):
                plan = []
            if not plan:
                progress = {}
                try:
                    progress = ctx.progress()
                except Exception:
                    pass
                delta = ""
                if previous is not None:
                    delta = "\nPrevious observation (compare for failed assumptions):\n" + _json(previous, 7000)
                prompt = f"""You are the planning player in a turn-based Dominion game.
Return a short executable plan, one command per line, with no explanation.
Every line must be `choose DECISION_ID OPTION_ID ...`; use zero option IDs for an
optional prompt (equivalent to pass).  Use only IDs present in the current prompt.
Give 3-8 lines when future prompts are predictable, but stop before an uncertain
attack, discard, draw, or nested effect.  The harness checks every line and will
replan if a decision ID changes.  Prefer winning VP margin against BMU-1: build
coherent draw/thinning/economy, then green cards when Provinces or piles are under
pressure.  Never invent hidden cards or seed information.

Current game description and action contract:
{game.describe()}

Current observation:
{_json(obs)}
Progress/failure counters:
{_json(progress)}
{delta}
"""
                try:
                    remaining = ctx.budget.remaining().get("wall_s", 300.0)
                    timeout = max(20.0, min(145.0, float(remaining) - 2.0))
                    res = ctx.think(prompt, session_id=session, timeout_s=timeout)
                    session = res.session_id
                    plan = _commands(res.text)
                    # Do not execute an answer that does not start at this prompt.
                    if not plan or did is None or str(plan[0][0]) != str(did) or not _valid(obs, *plan[0]):
                        plan = []
                    failures = 0
                except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                    plan = []
                    failures += 1
                    if ctx.done:
                        break
            if not plan:
                action = _fallback(obs)
                if action is None:
                    break
            else:
                d, ids = plan.pop(0)
                action = "choose " + str(d) + ((" " + " ".join(ids)) if ids else "")
            previous = obs
            try:
                obs = ctx.act(action)
            except Exception:
                # Illegal actions never mutate the game.  Reobserve and send the
                # evidence to Astra on the next iteration rather than escaping.
                plan = []
                failures += 1
                try:
                    obs = game.observe()
                except Exception:
                    break
            try:
                pr = ctx.progress()
                if pr.get("illegal_actions", 0) or pr.get("since_obs_change", 0) > 2:
                    plan = []
            except Exception:
                pass


def build(ctx):
    return Harness(ctx)

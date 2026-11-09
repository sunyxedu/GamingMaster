"""Batch planning with a small, observation-only walk solver.

The walk solver is deliberately conservative: it is used only when a single
controlled object and an already-active goal are visible and no rule tile or
blocking object needs to be crossed.  All rule manipulation remains Astra's
job.  Every queued action is checked against the returned observation before
the next one is trusted.
"""
from __future__ import annotations

import json
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


_DIRS = (("up", 0, -1), ("right", 1, 0), ("down", 0, 1), ("left", -1, 0))
_RULE_TYPES = {"rule_object", "rule_is", "rule_property"}


def _json(obs):
    try:
        return json.dumps(obs, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        return str(obs)


def _objects(obs):
    """Return (agent positions, goal positions, blocked positions) when known."""
    if not isinstance(obs, dict) or not isinstance(obs.get("board"), list):
        return [], [], set()
    rules = obs.get("rules") if isinstance(obs.get("rules"), dict) else {}
    agents = set()
    goals = set()
    for r in rules.get("_rule_", []) if isinstance(rules.get("_rule_"), list) else []:
        if isinstance(r, dict):
            if r.get("property") == "is_agent" and r.get("object"):
                agents.add(r["object"])
            if r.get("property") == "is_goal" and r.get("object"):
                goals.add(r["object"])
    # The indexed maps are convenient, but _rule_ is the authority when both
    # are present.  Include them as a fallback for slightly different builds.
    for name, dest in (("is_agent", agents), ("is_goal", goals)):
        d = rules.get(name)
        if isinstance(d, dict):
            dest.update(k for k, v in d.items() if v)
    apos, gpos, blocked = [], [], set()
    for y, row in enumerate(obs["board"]):
        if not isinstance(row, list):
            continue
        for x, cell in enumerate(row):
            if not isinstance(cell, list):
                continue
            ids = []
            for obj in cell:
                if not isinstance(obj, dict):
                    continue
                typ = obj.get("type")
                if typ == "wall":
                    blocked.add((x, y))
                if typ in _RULE_TYPES:
                    # Walking through text changes rules, so the conservative
                    # direct walker never chooses such a square.
                    blocked.add((x, y))
                # Rule words carry an ``object`` field too, but are not the
                # physical controlled/goal object at that coordinate.
                oid = obj.get("object") or obj.get("id")
                if typ not in _RULE_TYPES:
                    if oid:
                        ids.append(oid)
                    if obj.get("type") in agents or oid in agents:
                        apos.append((x, y))
                    if obj.get("type") in goals or oid in goals:
                        gpos.append((x, y))
            # Physical non-goal objects are treated as obstacles.  This avoids
            # relying on unverified push/collision semantics in the cheap path.
            for obj in cell:
                if isinstance(obj, dict) and obj.get("type") not in _RULE_TYPES:
                    if obj.get("type") not in ("wall",) and (x, y) not in gpos:
                        blocked.add((x, y))
    if not apos:
        p = obs.get("agent_pos")
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, int) for v in p):
            apos.append((p[0], p[1]))
    return apos, gpos, blocked


def _cheap_path(obs):
    """Find a safe walk to an existing goal, or return None."""
    apos, gpos, blocked = _objects(obs)
    if len(apos) != 1 or not gpos:
        return None
    start = apos[0]
    targets = set(gpos)
    width, height = obs.get("width"), obs.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        height = len(obs.get("board", []))
        width = max((len(r) for r in obs.get("board", []) if isinstance(r, list)), default=0)
    q = deque([start])
    prev = {start: None}
    via = {}
    while q:
        p = q.popleft()
        if p in targets:
            out = []
            while prev[p] is not None:
                out.append(via[p])
                p = prev[p]
            out.reverse()
            return out
        for action, dx, dy in _DIRS:
            n = (p[0] + dx, p[1] + dy)
            if not (0 <= n[0] < width and 0 <= n[1] < height):
                continue
            if n in prev or (n in blocked and n not in targets):
                continue
            prev[n] = p
            via[n] = action
            q.append(n)
    return None


def _plan_actions(text, names, limit=8):
    """Parse only an explicit PLAN line, tolerating arbitrary model prose."""
    if not isinstance(text, str):
        return []
    # Requiring the marker prevents words such as "up" in an explanation from
    # becoming accidental game actions during preflight or a malformed reply.
    m = re.search(r"(?im)^\s*plan\s*:\s*(.*)$", text)
    body = m.group(1) if m else (text.strip() if text.strip().lower() in names else "")
    if not body:
        return []
    found = []
    for n in sorted(names, key=len, reverse=True):
        # Names are currently simple words, but this also handles spaces.
        for hit in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])", body, re.I):
            found.append((hit.start(), n))
    found.sort(key=lambda z: z[0])
    return [n for _, n in found[:limit]]


def _progress_signature(obs):
    """Ignore turn counters and facing when deciding whether a move did work.

    The engine can update an object's facing (and ``has_moved``) even when a
    wall rejects its displacement.  Those presentation fields must not make a
    stale batch look productive.
    """
    ignored = {"step_count", "agent_dir", "last_error", "dir", "has_moved"}

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k not in ignored}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return _json(scrub(obs))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system2 is None:
            return
        names = list(game.action_spec.names)
        if not names:
            return
        session = None
        obs = game.observe()
        queue = _cheap_path(obs) or []
        recent = []
        failed_plans = 0

        while not ctx.done and not getattr(game, "done", False):
            if not queue:
                # Recompute the conservative path after every model batch: a
                # rule push can create/remove an active goal.
                queue = _cheap_path(obs) or []
            if not queue:
                rem = ctx.budget.remaining()
                if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 5:
                    break
                prompt = (
                    "You are the planning controller for Baba Is AI. The board below is the "
                    "authoritative current observation. Choose a feasible win, including any "
                    "word pushes needed to create or change IS WIN. Track every YOU object, "
                    "the side needed to push each word, and the state after each action. "
                    "Return ONLY one line of the form PLAN: action action ... with at most "
                    "8 primitive actions, stopping before an uncertain or irreversible step. "
                    "Use only these action names: " + ", ".join(names) + ". A short plan is "
                    "better than speculation; after it executes you will see a fresh board.\n\n"
                    "Game:\n" + str(game.describe()) + "\n\nObservation:\n" + _json(obs) +
                    "\n\nRecent actions/results:\n" + " | ".join(recent[-8:]) +
                    "\nProgress:\n" + str(ctx.progress())
                )
                try:
                    res = ctx.think(prompt, session_id=session,
                                    system=("Output only PLAN: followed by zero to eight action names. "
                                            "Do not include explanations or other text."),
                                    timeout_s=250.0)
                    session = res.session_id
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break
                except (TimeoutError, RuntimeError):
                    failed_plans += 1
                    if failed_plans >= 3:
                        # Give a turn back to the engine, then let a later call
                        # reconsider from the resulting observation.
                        queue = [game.action_spec.noop] if game.action_spec.noop in names else [names[0]]
                    continue
                parsed = _plan_actions(getattr(res, "text", ""), names)
                if parsed:
                    queue = parsed
                    failed_plans = 0
                else:
                    failed_plans += 1
                    if failed_plans >= 3:
                        queue = [game.action_spec.noop] if game.action_spec.noop in names else [names[0]]
                    continue

            action = queue.pop(0)
            before = _progress_signature(obs)
            try:
                new_obs = ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                break
            except Exception as exc:
                # Illegal actions are recoverable contract events.  Clear the
                # stale batch and ask Astra from the unchanged observation.
                recent.append(f"{action}->error:{type(exc).__name__}")
                queue = []
                try:
                    obs = game.observe()
                except Exception:
                    break
                continue
            if new_obs is None:
                try:
                    new_obs = game.observe()
                except Exception:
                    break
            after = _progress_signature(new_obs)
            prog = ctx.progress()
            recent.append(f"{action}:changed={before != after},illegal={prog.get('illegal_actions', 0)}")
            obs = new_obs
            # A directional action that changes neither the logical board nor
            # the score is evidence that the queued plan hit a blocker.  Do
            # not continue executing stale actions after that mismatch.
            if action != getattr(game.action_spec, "noop", "noop") and before == after:
                queue = []
            if prog.get("illegal_actions", 0):
                queue = []


def build(ctx):
    return Harness(ctx)

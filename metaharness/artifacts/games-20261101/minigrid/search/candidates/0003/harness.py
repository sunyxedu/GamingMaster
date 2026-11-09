"""Batching controller for synchronous MiniGrid/BabyAI episodes.

The model chooses a short action prefix.  The executor applies one action at a
time, checks the returned public observation and progress counters, and asks
again when the prefix is stale or contradicted.  A tiny local policy keeps a
slow, malformed, or unavailable model call from blocking the episode.
"""
from __future__ import annotations

import json
import re
from collections import deque


ACTION_RE = re.compile(r"\b(left|right|forward|pickup|drop|toggle|done)\b", re.I)


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def json_object(text):
    """Find the first usable object in arbitrary model output."""
    text = str(text or "")
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return {}


def grid_cells(obs):
    grid = obs.get("grid", []) if isinstance(obs, dict) else []
    out = {}
    if not isinstance(grid, list):
        return out
    for row in grid:
        items = row if isinstance(row, list) else [row]
        for cell in items:
            if isinstance(cell, dict) and "x" in cell and "y" in cell:
                try:
                    out[(int(cell["x"]), int(cell["y"]))] = cell
                except (TypeError, ValueError):
                    pass
    return out


def compact_obs(obs):
    """Keep all public state, while excluding bulky non-state metadata."""
    if not isinstance(obs, dict):
        return {"raw": str(obs)[:3000]}
    keep = {}
    for key in ("environment_id", "mission", "direction", "direction_name",
                "view_convention", "grid", "inventory", "steps_used",
                "max_steps", "remaining_steps", "terminated", "truncated",
                "last_action", "last_reward"):
        if key in obs:
            keep[key] = obs[key]
    return keep or obs


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = deque()
        self.session = None
        self.obs = {}
        self.history = deque(maxlen=8)
        self.last_obs_key = None
        self.no_change = 0
        self.last_illegal = 0
        self.calls = 0

    def obs_key(self, obs):
        if not isinstance(obs, dict):
            return repr(obs)
        # Counters and rewards are deliberately omitted: movement can preserve
        # the rendered view in symmetric rooms.
        return repr((obs.get("mission"), obs.get("direction"),
                     obs.get("inventory"), obs.get("grid")))

    def front(self):
        return grid_cells(self.obs).get((0, 1), {})

    def local_action(self, names):
        """A conservative action when inference is unavailable or malformed."""
        available = set(names)
        front = self.front()
        obj = str(front.get("object", "")).lower()
        state = str(front.get("state", "")).lower()
        inv = self.obs.get("inventory")
        if obj in ("key", "ball", "box") and not inv and "pickup" in available:
            return "pickup"
        if obj == "door" and state == "open" and "forward" in available:
            return "forward"
        if obj == "door" and state == "closed" and "toggle" in available:
            return "toggle"
        if obj == "door" and state == "locked" and "toggle" in available:
            key = inv if isinstance(inv, dict) else {}
            if key.get("object") == "key" and key.get("color") == front.get("color"):
                return "toggle"
        if obj == "box" and "toggle" in available:
            return "toggle"
        if obj in ("empty", "floor", "goal", "agent") and "forward" in available:
            return "forward"
        # Turning exposes a new egocentric view and is safer than repeatedly
        # submitting a blocked interaction.  Prefer right for deterministic
        # tie-breaking, then any legal advertised action.
        for action in ("right", "left", "forward", "done"):
            if action in available:
                return action
        return names[0] if names else "done"

    def parse_actions(self, text, names):
        allowed = set(names)
        data = json_object(text)
        raw = data.get("actions")
        if isinstance(raw, str):
            raw = ACTION_RE.findall(raw.lower())
        elif isinstance(raw, list):
            raw = [str(x).lower().strip() for x in raw]
        else:
            raw = []
        actions = [a for a in raw if a in allowed]
        if not actions:
            # Stand-in models and real models occasionally ignore JSON.  A
            # plain action sequence is still recoverable without raising.
            actions = [a.lower() for a in ACTION_RE.findall(str(text)) if a.lower() in allowed]
        return actions[:10]

    def prompt(self, game, names, reason):
        progress = {}
        try:
            progress = self.ctx.progress()
        except Exception:
            pass
        payload = {
            "observation": compact_obs(self.obs),
            "recent": list(self.history),
            "previous_plan_remaining": list(self.plan),
            "progress": progress,
            "replan_reason": reason,
        }
        return (
            game.describe() + "\nAvailable actions: " + ", ".join(names) +
            "\nYou control a synchronous turn-based MiniGrid/BabyAI agent. "
            "The public grid uses front=(0,1), x right, and unknown is not empty. "
            "Interpret the exact mission and preserve ordered subgoals. Return ONLY "
            "JSON of the form {\"actions\":[\"left\",\"forward\"],\"subgoal\":\"...\"}. "
            "Give 4 to 10 actions when that prefix is safe (fewer only when the task "
            "requires a decisive interaction). Plan only from supplied observations; use turns "
            "before movement and face an adjacent object before pickup/toggle. Avoid "
            "forward into unknown, walls, or lava. The executor verifies every action "
            "and will replan if your prefix is contradicted.\nSTATE=" +
            json.dumps(payload, separators=(",", ":"), default=str)
        )

    def ask(self, game, names, reason):
        if self.ctx.done:
            return False
        try:
            budget = self.ctx.budget.remaining()
        except Exception:
            budget = {"system2_calls": 1, "wall_s": 180.0}
        if budget.get("system2_calls", 0) <= 0 or budget.get("wall_s", 0) < 8:
            return False
        # The archive's worst observed call is 102s.  A bounded timeout prevents
        # one stalled answer from consuming the entire turn-game deadline.
        timeout = min(155.0, max(8.0, float(budget.get("wall_s", 155.0)) - 3.0))
        try:
            result = self.ctx.think(self.prompt(game, names, reason),
                                    session_id=self.session, timeout_s=timeout)
            self.calls += 1
            self.session = getattr(result, "session_id", None)
            self.plan = deque(self.parse_actions(getattr(result, "text", ""), names))
            self.ctx.log("batch_plan", actions=list(self.plan), reason=reason,
                         calls=self.calls)
            return bool(self.plan)
        except Exception as exc:
            # TimeoutError, BudgetExceeded, ModelUnavailable and malformed backend
            # responses are all recoverable at this layer.
            self.session = None
            self.ctx.log("batch_plan_failed", error=type(exc).__name__, reason=reason)
            self.plan.clear()
            return False

    def apply(self, action, names):
        before = self.obs
        before_key = self.obs_key(before)
        before_front = self.front()
        before_inv = before.get("inventory") if isinstance(before, dict) else None
        pre_evidence = None
        front_obj = str(before_front.get("object", "")).lower()
        front_state = str(before_front.get("state", "")).lower()
        if action == "forward" and front_obj in ("wall", "lava"):
            pre_evidence = "planned forward is blocked or hazardous"
        elif action == "forward" and front_obj == "door" and front_state != "open":
            pre_evidence = "planned forward faces a closed or locked door"
        elif action == "pickup" and (before_inv or front_obj not in ("key", "ball", "box")):
            pre_evidence = "planned pickup has no legal visible object"
        elif action == "toggle" and front_obj not in ("door", "box"):
            pre_evidence = "planned toggle has no visible interactable object"
        try:
            result = self.ctx.act(action)
        except Exception as exc:
            if self.ctx.done:
                return False, "episode_over"
            self.ctx.log("action_failed", action=action, error=type(exc).__name__)
            return True, "act_exception"
        self.obs = as_dict(result if result is not None else self.ctx.observe())
        after_key = self.obs_key(self.obs)
        self.no_change = self.no_change + 1 if after_key == before_key else 0
        try:
            progress = self.ctx.progress()
        except Exception:
            progress = {}
        illegal = int(progress.get("illegal_actions", 0) or 0)
        evidence = pre_evidence
        if illegal > self.last_illegal:
            evidence = "adaptor rejected an action"
        self.last_illegal = illegal
        # Manipulation should produce observable evidence.  A blocked forward
        # can be detected from the unchanged pose/view only after a few actions.
        if action == "pickup" and self.obs.get("inventory") == before_inv:
            evidence = "pickup produced no inventory change"
        elif action == "drop" and self.obs.get("inventory") == before_inv:
            evidence = "drop produced no inventory change"
        elif action == "toggle":
            new_front = self.front()
            if (before_front.get("object"), before_front.get("color"), before_front.get("state")) == (
                    new_front.get("object"), new_front.get("color"), new_front.get("state")):
                evidence = "toggle produced no visible state change"
        if self.no_change >= 5:
            evidence = evidence or "five actions without a public observation change"
        self.history.append({"action": action, "before": compact_obs(before),
                             "after": compact_obs(self.obs), "evidence": evidence})
        self.ctx.log("verified_batch_step", action=action, evidence=evidence,
                     progress=progress)
        return not (self.ctx.done or self.obs.get("terminated") or self.obs.get("truncated")), evidence

    def run_episode(self, game):
        names = list(getattr(game.action_spec, "names", ()) or
                     ("left", "right", "forward", "pickup", "drop", "toggle", "done"))
        self.obs = as_dict(game.observe())
        if not names:
            return
        reason = "initial plan"
        while not self.ctx.done and not self.obs.get("terminated") and not self.obs.get("truncated"):
            if not self.plan:
                self.ask(game, names, reason)
            if not self.plan:
                action = self.local_action(names)
                reason = "model unavailable or empty response"
            else:
                action = self.plan.popleft()
                reason = "verified batch"
            if action not in names:
                action = self.local_action(names)
            alive, evidence = self.apply(action, names)
            if not alive:
                break
            if evidence:
                self.plan.clear()
                reason = evidence
            elif not self.plan:
                reason = "batch exhausted"


def build(ctx):
    return Harness(ctx)

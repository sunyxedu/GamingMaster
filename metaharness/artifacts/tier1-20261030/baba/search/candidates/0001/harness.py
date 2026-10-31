"""Batch-and-verify controller for Baba Is AI.

The model chooses a short primitive sequence.  The adapter executes it one
turn at a time and checks the resulting symbolic state after every turn; a
stalled sequence is discarded and sent back to Astra with evidence.
"""
from __future__ import annotations

import json
import re

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


_ALIASES = {
    "north": "up", "n": "up", "east": "right", "e": "right",
    "south": "down", "s": "down", "west": "left", "w": "left",
    "wait": "noop", "stay": "noop", "none": "noop",
}


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(value)


def _cell_text(cell):
    """A lossless-enough textual cell: every stack member and its rule name."""
    if not cell:
        return "."
    out = []
    for item in cell:
        if not isinstance(item, dict):
            out.append(str(item))
            continue
        name = item.get("name") or item.get("type") or "?"
        # Keep rule words distinguishable from physical objects.
        typ = str(item.get("type", ""))
        if typ.startswith("rule_"):
            name = str(name).upper()
        out.append(str(name))
    return "/".join(out)


def _board_summary(obs):
    """Compact deterministic rendering retaining coordinates and all layers."""
    if not isinstance(obs, dict):
        return _json(obs)
    lines = []
    board = obs.get("board")
    if isinstance(board, list):
        for y, row in enumerate(board):
            cells = []
            for x, cell in enumerate(row):
                label = _cell_text(cell)
                if label != ".":
                    cells.append(f"({x},{y})={label}")
            if cells:
                lines.append(" ".join(cells))
    head = {k: obs.get(k) for k in (
        "game", "level_id", "seed", "width", "height", "agent_pos", "agent_dir",
        "rules", "step_count", "max_steps", "success", "terminated",
        "termination_reason", "last_error") if k in obs}
    return _json(head) + "\nCELLS " + ("\n".join(lines) if lines else "(empty)")


def _rule_signature(obs):
    return _json(obs.get("rules")) if isinstance(obs, dict) else ""


def _actions(text, names):
    """Extract a bounded sequence from both strict and conversational replies."""
    allowed = {str(n).lower(): str(n) for n in names}
    canonical = dict(allowed)
    for alias, target in _ALIASES.items():
        if target in allowed:
            canonical[alias] = allowed[target]

    raw = str(getattr(text, "text", text) or "")
    # Prefer the explicitly requested line, which prevents words in an
    # explanation (for example "move right") from becoming extra moves.
    segment = None
    for line in raw.splitlines():
        m = re.search(r"(?:actions?|moves?|sequence)\s*[:=-]\s*(.*)$", line, re.I)
        if m:
            segment = m.group(1)
            break
    if segment is None:
        for line in raw.splitlines():
            words = re.findall(r"[A-Za-z_]+", line.lower())
            if words and sum(w in canonical for w in words) >= 1 and all(
                    w in canonical or w in {"and", "then", "next"} for w in words):
                segment = line
                break
    if segment is None:
        segment = raw
    found = []
    for word in re.findall(r"[A-Za-z_]+", segment.lower()):
        if word in canonical:
            found.append(canonical[word])
    # A bad or empty answer is still a valid turn; noop is safer than an
    # arbitrary direction and keeps the pre-flight garbage-model contract safe.
    if not found:
        fallback = allowed.get("noop") or (next(iter(allowed.values())) if allowed else "noop")
        found = [fallback]
    return found[:8]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _prompt(self, game, obs, prior, evidence, first=False):
        spec = game.action_spec
        actions = ", ".join(str(x) for x in spec.names)
        exact_description = ""
        if first:
            try:
                exact_description = "\nExact environment description:\n" + str(game.describe())
            except Exception:
                exact_description = ""
        intro = (
            "You are the planning controller for Baba Is AI. The board is fully "
            "observable and each submitted action advances exactly one turn. "
            "Rule words are movable tiles; active rules and all stacked cells are "
            "listed below. Find a real win (success=true), considering changing "
            "WIN or YOU when walking to a current goal is blocked. Check pushing "
            "sides and all controlled objects."
        )
        instruction = (
            "Reply with exactly one line beginning `ACTIONS:` followed by 1-8 "
            "primitive names separated by spaces. Choose a short executable "
            "prefix toward a win. Do not include prose on that line."
        )
        if not first:
            instruction += " Replan from this observed state; the previous prefix produced the evidence below."
        return (
            intro + "\nAllowed actions: " + actions + "\n" + instruction +
            "\nPrevious proposed prefix: " + (prior or "none") +
            "\nVerification evidence: " + (evidence or "none") +
            exact_description
            + "\nSTATE (coordinates are x,y, origin top-left):\n" + _board_summary(obs)
        )

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        names = list(game.action_spec.names)
        obs = game.observe()
        session = None
        plan = []
        prior = ""
        evidence = ""
        previous_hash = _json(obs)
        previous_rules = _rule_signature(obs)
        stale = 0
        call_failures = 0

        while not ctx.done and not getattr(game, "done", False):
            if not plan:
                try:
                    remaining = ctx.budget.remaining()
                    if remaining.get("system2_calls", 0) <= 0:
                        break
                    result = ctx.think(
                        self._prompt(game, obs, prior, evidence, first=(prior == "")),
                        session_id=session,
                        timeout_s=min(260.0, max(30.0, remaining.get("wall_s", 260.0) - 2.0)),
                    )
                    session = result.session_id
                    plan = _actions(result.text, names)
                    prior = " ".join(plan)
                    evidence = ""
                    call_failures = 0
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break
                except (TimeoutError, RuntimeError) as exc:
                    # Timeout/error is a normal model event; do not let it make
                    # the candidate ineligible. Retry while the episode remains open.
                    call_failures += 1
                    evidence = "model call failed: " + type(exc).__name__
                    if call_failures >= 3:
                        break
                    continue
                except Exception as exc:
                    # Stand-in models may return unusual objects during preflight.
                    call_failures += 1
                    evidence = "model response problem: " + type(exc).__name__
                    if call_failures >= 3:
                        break
                    continue

            if not plan or ctx.done:
                continue
            action = plan.pop(0)
            try:
                new_obs = ctx.act(action)
            except Exception as exc:
                evidence = "action %s raised %s; choose another legal action" % (action, type(exc).__name__)
                plan = []
                continue
            old_hash = previous_hash
            new_hash = _json(new_obs)
            changed = new_hash != old_hash
            new_rules = _rule_signature(new_obs)
            if changed:
                stale = 0
            else:
                stale += 1
            try:
                progress = ctx.progress()
            except Exception:
                progress = {}
            if progress.get("illegal_actions", 0):
                plan = []
                evidence = "illegal action count=%s after %s" % (progress.get("illegal_actions"), action)
            elif action != "noop" and stale >= 2:
                plan = []
                evidence = "state unchanged for %d turns after %s; likely blocked" % (stale, action)
            elif new_rules != previous_rules:
                # A rule push changes the physics; do not replay a plan made for
                # the old grammar.
                plan = []
                evidence = "active rules changed after %s; reassess control and goal" % action
            previous_hash = new_hash
            previous_rules = new_rules
            obs = new_obs
            if isinstance(new_obs, dict) and (new_obs.get("success") or new_obs.get("terminated")):
                break


def build(ctx):
    return Harness(ctx)

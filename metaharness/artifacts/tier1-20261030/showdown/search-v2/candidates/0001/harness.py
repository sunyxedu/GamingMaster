"""Batch-planning Astra harness for the turn-based Showdown game.

The baseline spent nearly the whole episode waiting for one model call per
choice.  Astra is instead asked for a short line of legal choices.  Every
choice is checked against the *new* request before it is posted; a forced
switch, rejection, or unexpected state change discards the remainder and
causes a fresh plan.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def _state_view(obs: Any) -> str:
    """Render the decision-critical part of a Showdown observation."""
    if not isinstance(obs, dict):
        return json.dumps(obs, default=str, separators=(",", ":"))
    # Keep the complete private request and public caches, but omit the huge
    # dex entries and old log.  The tail of the log is the useful causal trace.
    out = {
        "turn": obs.get("turn"),
        "done": obs.get("done"),
        "request": obs.get("request"),
        "legal_actions": obs.get("legal_actions", []),
        "last_error": obs.get("last_error"),
        "active": obs.get("active"),
        "opponent_revealed": obs.get("opponent_revealed", []),
        "public_summary": obs.get("public_summary", {}),
        "recent_events": obs.get("recent_events", [])[-80:],
    }
    return json.dumps(out, default=str, separators=(",", ":"))


def _extract_actions(text: str, legal: list[str], limit: int = 3) -> list[str]:
    """Extract an ordered, duplicate-free line from arbitrary model text."""
    if not text:
        return []
    # Longest first prevents ``move 1`` from stealing ``move 1 terastallize``.
    choices = sorted((str(x) for x in legal), key=len, reverse=True)
    found: list[tuple[int, int, str]] = []
    for name in choices:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", re.I)
        for m in pat.finditer(text):
            found.append((m.start(), m.end(), name))
    # Prefer a longer command at the same location, and suppress the shorter
    # substring match (e.g. ``move 1`` inside ``move 1 terastallize``).
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    result: list[str] = []
    spans: list[tuple[int, int]] = []
    for start, end, action in found:
        if any(start < b and end > a for a, b in spans):
            continue
        spans.append((start, end))
        if action not in result:
            result.append(action)
        if len(result) >= limit:
            break
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _plan(self, game, obs, session):
        ctx = self.ctx
        legal = [str(x) for x in getattr(game.action_spec, "names", ())]
        # The actual request list is authoritative and can be narrower than
        # ActionSpec while a replacement or a trap is being resolved.
        if isinstance(obs, dict) and isinstance(obs.get("legal_actions"), list):
            legal = [str(x) for x in obs["legal_actions"]]
        prompt = f"""{game.describe()}

You are the strategic controller for one Pokémon Showdown random battle.
Use the authoritative legal_actions list in the state.  Choose immediate
progress while preserving a needed counter, respecting revealed speed,
priority, hazards, status, PP, and the one-use Tera.  Do not assume hidden
opponent moves.  A switch request may be forced, and the team slot order is
the current request order.

Return up to THREE proposed next choices, one exact command per line under
the heading ACTIONS.  Copy commands character-for-character from
legal_actions.  The first choice must be best now; later choices should be
included only when they are a robust continuation after the preceding
resolution.  You may add at most two short lines of reasoning after the
commands.  Never invent a command.

CURRENT STATE:
{_state_view(obs)}
"""
        try:
            res = ctx.think(prompt, session_id=session, timeout_s=260.0)
            actions = _extract_actions(res.text, legal)
            return actions, res.session_id
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
            # A model failure must not turn into a harness error.  The first
            # currently legal action is a safe transport fallback.
            return (legal[:1] if legal else []), session

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if ctx.system2 is None:
            return
        obs = game.observe()
        session = None
        plan: list[str] = []
        previous_view = _state_view(obs)
        while not ctx.done and not getattr(game, "done", False):
            legal = []
            if isinstance(obs, dict):
                legal = [str(x) for x in obs.get("legal_actions", [])]
            if not legal:
                # Terminal/empty requests need no action.  Avoid spinning on
                # malformed stand-in observations during pre-flight.
                break

            # A plan is valid only for the request from which it was made.
            # Replan immediately after a rejection, forced replacement, or an
            # observation that did not advance as expected.
            if not plan or plan[0] not in legal:
                plan, session = self._plan(game, obs, session)
                if not plan:
                    plan = legal[:1]
            action = plan.pop(0)
            if action not in legal:
                plan = []
                continue
            try:
                new_obs = ctx.act(action)
            except (BudgetExceeded, ModelUnavailable, RuntimeError, TimeoutError):
                # Illegal actions and transient transport failures are normal
                # recovery events; the next request supplies the truth.
                plan = []
                if ctx.done:
                    break
                try:
                    new_obs = game.observe()
                except Exception:
                    break
            except Exception:
                # Do not let a malformed arbitrary pre-flight model response
                # escape the harness.  Re-read and let the planner recover.
                plan = []
                if ctx.done:
                    break
                new_obs = game.observe()

            obs = new_obs if new_obs is not None else game.observe()
            current_view = _state_view(obs)
            progress = {}
            try:
                progress = ctx.progress()
            except Exception:
                pass
            rejected = bool(progress.get("illegal_actions", 0)) if isinstance(progress, dict) else False
            stale = False
            if isinstance(progress, dict):
                # A line that produces several accepted actions without any
                # score/observation movement is probably a bad continuation
                # (for example a stalled Protect/recovery loop).  Give Astra
                # the evidence and let it choose a different line.
                stale = (progress.get("since_score_change", 0) >= 4 and
                         progress.get("since_obs_change", 0) >= 4)
            changed = current_view != previous_view
            if rejected or stale or not changed or (isinstance(obs, dict) and obs.get("last_error")):
                plan = []
            previous_view = current_view


def build(ctx):
    return Harness(ctx)

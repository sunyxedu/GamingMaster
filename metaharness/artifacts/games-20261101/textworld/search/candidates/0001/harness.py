"""Verified replanning harness for generic TextWorld quests.

The model proposes a short sequence, while this module owns command matching,
progress checks, and recovery from stale or rejected steps.  Plans are never
executed blindly: every command is resolved against the current admissible
list and the observation after the command is checked before continuing.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


def _field(obs: Any, key: str, default: Any = "") -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _short(value: Any, limit: int = 900) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            out = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            out = str(value)
    else:
        out = str(value)
    return out if len(out) <= limit else out[:limit] + "..."


def _signature(obs: Any) -> str:
    """A stable, public-observation signature for detecting no-effect actions."""
    parts = []
    for key in ("description", "inventory", "objective", "feedback", "score",
                "won", "lost", "terminated", "done"):
        parts.append(_short(_field(obs, key, ""), 500))
    return "\x1f".join(parts)


def _normalise_line(line: str) -> str:
    line = line.strip().strip("` ")
    line = re.sub(r"^(?:command|action|next)\s*:\s*", "", line,
                  flags=re.IGNORECASE)
    line = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", line)
    return line.strip().strip("` ")


def _extract_plan(text: Any, names: list[str], max_actions: int = 6) -> list[str]:
    """Extract exact commands while tolerating terse prose and markdown.

    Exact line matches are preferred.  For a line such as ``1. go north`` we
    also accept a contained command, choosing the longest matching name so
    ``take X from chest`` cannot be shortened to ``take X``.
    """
    if not isinstance(text, str):
        return []
    allowed = list(dict.fromkeys([str(x) for x in names] + ["look", "inventory"]))
    by_lower = {x.lower(): x for x in allowed}
    found: list[str] = []
    # First pass preserves the model's requested order, one command per line.
    for raw in text.splitlines():
        line = _normalise_line(raw)
        if not line:
            continue
        exact = by_lower.get(line.lower())
        if exact is not None:
            found.append(exact)
            continue
        matches = [x for x in allowed if x.lower() in line.lower()]
        if matches:
            # Longest first avoids selecting a prefix of a qualified command.
            matches.sort(key=len, reverse=True)
            found.append(matches[0])
    # If the answer was one paragraph, recover exact commands from it.
    if not found:
        lower = text.lower()
        hits = []
        for command in allowed:
            start = 0
            while True:
                pos = lower.find(command.lower(), start)
                if pos < 0:
                    break
                hits.append((pos, -len(command), command))
                start = pos + 1
        for _, _, command in sorted(hits):
            if not found or found[-1] != command:
                found.append(command)
    return found[:max_actions]


def _is_destructive(command: str) -> bool:
    return command.lower().split(" ", 1)[0] in {"eat", "cut", "cook", "slice", "chop", "dice"}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.memory = ctx.memory

    def _prompt(self, game: Any, obs: Any, plan: list[str], failures: list[str]) -> str:
        names = list(getattr(game.action_spec, "names", ()) or ())
        history = self.memory.get("history", [])[-8:]
        objective = _field(obs, "objective", "")
        return (
            "You are solving a generated TextWorld quest. Read the literal objective and "
            "preserve quest objects; admissible means executable, not safe. Plan backward "
            "from the final relation and its prerequisites. Return ONLY 1-6 exact commands, "
            "one per line, copied character-for-character from the current list (look or "
            "inventory are also allowed). Do not explain. Never eat/cook/cut unless the "
            "objective explicitly requires that transformation. A previous plan may be stale; "
            "use the latest feedback and do not repeat a rejected/no-effect command.\n\n"
            f"OBJECTIVE: {_short(objective)}\n"
            f"CURRENT OBSERVATION: {_short(obs, 2600)}\n"
            f"CURRENT COMMANDS: {_short(names, 2600)}\n"
            f"RECENT VERIFIED ACTIONS: {_short(history, 1800)}\n"
            f"FAILED OR STALE PLAN EVIDENCE: {_short(failures[-4:], 1000)}\n"
            f"UNFINISHED CACHED PLAN (reconsider it): {_short(plan, 700)}\n"
            "Select the next useful exact commands now."
        )

    def _record(self, action: str, before: Any, after: Any, progress: Any, error: str = ""):
        item = {
            "action": action,
            "feedback": _short(_field(after, "feedback", ""), 420),
            "score": _field(after, "score", None),
            "changed": _signature(before) != _signature(after),
            "progress": progress,
        }
        if error:
            item["error"] = error[:300]
        self.memory.setdefault("history", []).append(item)
        self.memory["history"] = self.memory["history"][-16:]

    def _fallback(self, names: list[str], recent: list[str], objective: str) -> str:
        # A stand-in model can return garbage during preflight.  Make one
        # conservative, non-repeating move rather than crashing or eating a goal.
        safe = [x for x in names if not _is_destructive(x)]
        unseen = [x for x in safe if x not in recent]
        if unseen:
            # Interactions are generally more useful than another noop.
            rank = {"unlock": 0, "open": 1, "take": 2, "insert": 3,
                    "put": 4, "close": 5, "go ": 6, "examine": 7, "drop": 8}
            unseen.sort(key=lambda x: min((v for k, v in rank.items() if x.startswith(k)), default=20))
            return unseen[0]
        return "inventory" if "inventory" not in recent else "look"

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            obs = game.observe()
        except Exception:
            return
        plan: list[str] = []
        failures: list[str] = []
        session = None
        recent_actions: list[str] = []
        calls_without_plan = 0

        while not ctx.done:
            try:
                names = list(getattr(game.action_spec, "names", ()) or ())
            except Exception:
                names = []
            allowed = list(dict.fromkeys(names + ["look", "inventory"]))

            # Ask for a fresh short plan whenever the previous one is empty or
            # contradicted by the current command list.
            if not plan:
                rem = {}
                try:
                    rem = ctx.budget.remaining()
                except Exception:
                    pass
                if rem.get("system2_calls", 1) <= 0 or rem.get("wall_s", 1) <= 2:
                    action = self._fallback(names, recent_actions, str(_field(obs, "objective", "")))
                else:
                    try:
                        result = ctx.think(self._prompt(game, obs, plan, failures),
                                           session_id=session, timeout_s=150.0)
                        session = getattr(result, "session_id", None)
                        plan = _extract_plan(getattr(result, "text", ""), allowed)
                    except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                        break
                    except (TimeoutError, RuntimeError, Exception) as exc:
                        failures.append("model error: " + str(exc))
                        session = None
                        plan = []
                    if plan:
                        calls_without_plan = 0
                        action = plan.pop(0)
                    else:
                        calls_without_plan += 1
                        action = self._fallback(names, recent_actions,
                                                str(_field(obs, "objective", "")))
                        if calls_without_plan > 2 and not names:
                            break

            # Plans are state-conditioned.  Resolve against this turn's exact
            # names and discard stale commands rather than submitting guesses.
            if action not in allowed:
                failures.append("stale command: " + str(action))
                plan = []
                session = None
                continue
            objective = str(_field(obs, "objective", "")).lower()
            if _is_destructive(action) and "eat" not in objective and not any(
                    word in objective for word in ("cook", "slice", "chop", "dice")):
                failures.append("destructive command deferred: " + action)
                plan = []
                session = None
                continue

            before = obs
            before_sig = _signature(before)
            try:
                obs = ctx.act(action)
            except Exception as exc:
                failures.append(f"{action}: {exc}")
                self._record(action, before, before, {}, str(exc))
                recent_actions.append(action)
                recent_actions = recent_actions[-12:]
                plan = []
                session = None
                if ctx.done:
                    break
                continue
            recent_actions.append(action)
            recent_actions = recent_actions[-12:]
            try:
                prog = ctx.progress()
            except Exception:
                prog = {}
            self._record(action, before, obs, prog)
            illegal = bool(isinstance(prog, dict) and prog.get("illegal_actions", 0))
            changed = before_sig != _signature(obs)
            if illegal or not changed or _field(obs, "done", False) or _field(obs, "terminated", False):
                # No-effect and rejected actions are direct evidence that any
                # remaining model plan was conditioned on the wrong state.
                if illegal or not changed:
                    failures.append(f"{action}: illegal={illegal}, changed={changed}")
                    plan = []
                    session = None


def build(ctx):
    return Harness(ctx)

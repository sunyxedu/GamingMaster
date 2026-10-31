"""Command-plan harness for the Jericho turn game.

The action channel is slightly unusual here: ``action_spec.names`` contains
the channel name (``command``), while the value passed to ``ctx.act`` must be
the parser sentence itself.  This controller therefore keeps parser-command
extraction separate from action-channel parsing.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


_DIRECTIONS = {"north", "south", "east", "west", "up", "down", "in", "out"}


def _text(obs: Any) -> str:
    if isinstance(obs, dict):
        return str(obs.get("text", ""))
    return str(obs)


def _clean_command(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").replace("\n", " ").strip()
    # Remove common wrappers emitted despite an exact-output instruction.
    s = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", s, flags=re.I).strip()
    s = re.sub(r"^(?:command|action)\s*(?::|=|\s)\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^<command>\s*|\s*</command>$", "", s, flags=re.I).strip()
    if (len(s) >= 2 and s[0] == s[-1] and s[0] in "'\""):
        s = s[1:-1].strip()
    # A function-like wrapper is common in model answers.
    m = re.fullmatch(r"(?:command|action)\s*\(\s*[\"'](.*?)[\"']\s*\)", s, re.I)
    if m:
        s = m.group(1).strip()
    if not s or len(s) > 200 or "\n" in s or "\x00" in s:
        return None
    # Never submit the action-channel name as a parser command.
    if s.casefold() in {"command", "action", "noop"}:
        return None
    return s


def _commands_from(value: Any) -> list[str]:
    """Extract parser lines from JSON-ish or plain Astra output."""
    found: list[str] = []
    if isinstance(value, str):
        c = _clean_command(value)
        if c:
            found.append(c)
        return found
    if isinstance(value, list):
        for item in value:
            found.extend(_commands_from(item))
        return found
    if isinstance(value, dict):
        # Prefer explicit command/action fields and recurse through plans.
        for key in ("command", "commands", "action", "actions", "arguments", "args", "input", "plan", "next"):
            if key in value:
                found.extend(_commands_from(value[key]))
        return found
    return found


def parse_commands(reply: str) -> list[str]:
    if not isinstance(reply, str) or not reply.strip():
        return []
    # Try the whole response, then fenced/embedded JSON objects or arrays.
    candidates = [reply.strip()]
    candidates.extend(re.findall(r"```(?:json|text)?\s*(.*?)```", reply, flags=re.I | re.S))
    for match in re.finditer(r"(?:\{|\[).*?(?:\}|\])", reply, flags=re.S):
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        got = _commands_from(obj)
        if got:
            return got
    # Plain replies: choose the last short command-like line, ignoring prose.
    lines = [x.strip(" \t*-•") for x in reply.replace("\r", "").splitlines()]
    plain: list[str] = []
    for line in lines:
        line = re.sub(r"^(?:command|action)\s*(?::|=|\s)\s*", "", line, flags=re.I).strip()
        c = _clean_command(line)
        if c and len(c.split()) <= 12 and not re.search(r"\b(?:because|should|means|plan)\b", c, re.I):
            plain.append(c)
    if plain:
        return plain
    # A one-line answer can be wrapped in prose; use a quoted command if one exists.
    quoted = re.findall(r"['\"]([^'\"]{1,200})['\"]", reply)
    return [c for q in quoted if (c := _clean_command(q))]


def infer_opening_plan(observation: Any) -> list[str]:
    """Compile the explicit objective in the opening transcript when present."""
    text = _text(observation)
    low = text.casefold()
    plan: list[str] = []
    dm = re.search(r"(?:go|move|head|travel)\s+to\s+(?:the\s+)?(north|south|east|west|up|down|in|out)\b", low)
    if dm:
        plan.append(dm.group(1))
    pm = re.search(
        r"(?:place|put)\s+(?:the\s+)?([a-z][a-z0-9_-]*)\s+(?:inside|in|into)\s+(?:the\s+)?([a-z][a-z0-9_-]*)",
        low,
    )
    if pm:
        item, container = pm.groups()
        # The opening room in the observed fixture does not contain the item;
        # taking it after the movement is the safe parser sequence.
        if not re.search(r"\b(?:carry|carrying|have|holding)\b[^.\n]*\b" + re.escape(item) + r"\b", low):
            plan.append("take " + item)
        plan.append("put " + item + " in " + container)
    cm = re.search(
        r"(?:make it so that|make sure|then)\s+(?:the\s+)?([a-z][a-z0-9_-]*)\b[^.\n]{0,80}?\bclosed\b",
        low,
    )
    if cm:
        plan.append("close " + cm.group(1))
    return plan


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _ask(self, game, obs, session):
        prompt = (
            "You control a turn-based interactive-fiction parser. Return JSON only: "
            '{"commands":["one parser command", ...]}. Commands are ordinary text such as '
            'east, take carrot, put carrot in chest, or close chest. Do not return the '
            'channel name `command`; each string must be submitted directly to the parser. '
            "Give at most four commands, in dependency order, using only facts in the transcript.\n\n"
            + str(game.describe()) + "\nCURRENT OBSERVATION:\n" + json.dumps(obs, ensure_ascii=False, default=str)
        )
        try:
            result = self.ctx.think(prompt, session_id=session, timeout_s=240.0)
            return parse_commands(result.text), result.session_id
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            return [], session

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if ctx.system2 is None:
            return
        obs = game.observe()
        queue = infer_opening_plan(obs)
        session = None
        failures = 0
        last_obs_text = _text(obs)
        while not ctx.done and not getattr(game, "done", False):
            if not queue:
                queue, session = self._ask(game, obs, session)
                if not queue:
                    # A garbage stand-in response must still leave a legal,
                    # bounded run.  `look` is an ordinary parser command.
                    queue = ["look"]
            command = _clean_command(queue.pop(0)) or "look"
            try:
                obs = ctx.act(command)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError, ValueError):
                break
            now_text = _text(obs)
            if now_text == last_obs_text and re.search(r"not a verb|don't understand|doesn't understand", now_text, re.I):
                failures += 1
            else:
                failures = 0
            last_obs_text = now_text
            # A failed compiled command invalidates the remainder; ask Astra
            # for a fresh plan rather than cycling the same grammar.
            if failures >= 1:
                queue = []
            if isinstance(obs, dict) and (obs.get("terminated") or obs.get("truncated") or obs.get("won") or obs.get("lost")):
                break


def build(ctx):
    return Harness(ctx)

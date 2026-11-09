"""Evidence-backed planner for the turn based Jericho command channel.

The opening text in TextWorld-style stories often states a short quest in plain
English.  We compile only relations that are explicitly present in that text;
all other decisions are delegated to Astra.  In particular, ``command`` is the
action *field* name, not a parser command, so actions passed to ``ctx.act`` are
raw strings.
"""
from __future__ import annotations

import json
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


_DIRECTIONS = {
    "north": "north", "south": "south", "east": "east", "west": "west",
    "up": "up", "down": "down", "in": "in", "out": "out",
}


def _obs_text(obs):
    if isinstance(obs, dict):
        return str(obs.get("text", ""))
    return str(obs)


def _initial_plan(text):
    """Compile unambiguous, observed imperative relations into parser commands."""
    low = text.lower()
    out = []

    # This deliberately requires a direction to occur in an imperative phrase.
    m = re.search(r"\b(?:go|walk|head|move|travel|proceed)\s+(?:to\s+)?(?:the\s+)?"
                  r"(north|south|east|west|up|down|in|out)\b", low)
    if m:
        out.append(_DIRECTIONS[m.group(1)])

    # Preserve the object names supplied by the story, rather than inventing
    # synonyms.  TextWorld uses "inside"/"into" as well as "in".
    put = re.search(r"\b(?:put|place|insert)\s+(?:the\s+)?([a-z][a-z0-9 -]*?)\s+"
                    r"(?:inside|into|in)\s+(?:the\s+)?([a-z][a-z0-9 -]*?)(?:[.!?,]|\s+after\b|\s+then\b|$)", low)
    if put:
        obj = re.sub(r"\s+", " ", put.group(1)).strip()
        dst = re.sub(r"\s+", " ", put.group(2)).strip()
        if obj and dst:
            out.extend(["take " + obj, "put " + obj + " in " + dst])

    closed = re.search(r"\b(?:make|leave|ensure|keep)\s+(?:it\s+so\s+that\s+)?"
                       r"(?:the\s+)?([a-z][a-z0-9 -]*?)\s+(?:is|be)\s+closed\b", low)
    if closed:
        obj = re.sub(r"\s+", " ", closed.group(1)).strip()
        # Location qualifiers describe where the object is, not part of its
        # parser noun ("the chest within the kitchen" -> "chest").
        obj = re.split(r"\s+(?:within|inside|in|at)\s+(?:the\s+)?", obj, maxsplit=1)[0].strip()
        if obj:
            out.append("close " + obj)

    # Deduplicate while preserving the dependency order above.
    seen = set()
    return [x for x in out if x and not (x in seen or seen.add(x))]


def _commands_from_reply(raw):
    """Accept JSON, fenced JSON, bullets, or a bare command without crashing."""
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    values = []
    candidates = [s]
    # JSON is preferred because it keeps multiword commands intact.
    for blob in re.findall(r"\{.*?\}|\[.*?\]", s, flags=re.S):
        try:
            candidates.append(json.loads(blob))
        except Exception:
            pass

    def collect(x):
        if isinstance(x, dict):
            for k in ("actions", "commands", "plan", "command", "action"):
                if k in x:
                    collect(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x:
                collect(y)
        elif isinstance(x, str):
            for line in x.splitlines():
                line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                line = re.sub(r"^```(?:json|text)?\s*|```$", "", line, flags=re.I).strip()
                if line:
                    # Remove common explanatory prefixes, but retain parser text.
                    line = re.sub(r"^(?:action|command)\s*[:=]\s*", "", line, flags=re.I)
                    line = line.strip(" `\"'")
                    if line and line.lower() not in {"command", "action", "none", "null"}:
                        values.append(line[:200].replace("\n", " "))

    for c in candidates:
        collect(c)
    # A JSON object may have been collected as its whole source above; discard
    # that and keep only plausible parser strings.
    result = []
    for v in values:
        if v.startswith("{") or v.startswith("["):
            continue
        if v not in result:
            result.append(v)
    return result


def _bad_response(obs):
    t = _obs_text(obs).lower()
    return any(p in t for p in (
        "not a verb", "don't understand", "do not understand", "can't see",
        "cannot see", "no such", "not understood", "i don't know how",
    ))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _ask(self, game, obs, transcript, session):
        remaining = self.ctx.budget.remaining()
        if self.ctx.done or remaining.get("system2_calls", 0) <= 0:
            return [], session
        prompt = (
            "You control a turn-based Jericho parser. Choose a short ordered plan "
            "from the exact observations below. Return ONLY JSON of the form "
            '{"actions":["one parser command", ...]}. Each action is one raw '
            "command such as east, take carrot, put carrot in chest, or close chest. "
            "Do not write the field name command as an action. Use at most six actions; "
            "if uncertain return {\"actions\":[\"look\"]}. Replan around parser failures.\n\n"
            f"Game: {game.describe()}\nCurrent observation: {json.dumps(obs, ensure_ascii=False, default=str)}\n"
            f"Recent transcript: {json.dumps(transcript[-4:], ensure_ascii=False, default=str)}"
        )
        try:
            timeout = min(180.0, max(5.0, float(remaining.get("wall_s", 180.0)) - 2.0))
            res = self.ctx.think(prompt, session_id=session, timeout_s=timeout)
            return _commands_from_reply(getattr(res, "text", "")), getattr(res, "session_id", session)
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
            return [], session

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system2 is None:
            return
        obs = game.observe()
        transcript = [{"obs": obs}]
        plan = deque(_initial_plan(_obs_text(obs)))
        session = None
        replans = 0
        while not ctx.done and not getattr(game, "done", False):
            if not plan and replans < 5:
                new, session = self._ask(game, obs, transcript, session)
                replans += 1
                plan.extend(new)
            if not plan:
                # Do not burn the whole evaluator cap repeating an observation
                # command after the planner has exhausted its evidence.
                return
            action = plan.popleft().strip()[:200].replace("\n", " ").replace("\x00", "")
            if not action:
                continue
            try:
                obs = ctx.act(action)
            except Exception as exc:
                ctx.log("act_error", action=action, error=type(exc).__name__)
                plan.clear()
                transcript.append({"action": action, "error": type(exc).__name__})
                continue
            transcript.append({"action": action, "obs": obs})
            if _bad_response(obs):
                plan.clear()
            if isinstance(obs, dict) and (obs.get("terminated") or obs.get("won") or obs.get("lost")):
                break


def build(ctx):
    return Harness(ctx)

"""Frontier-guided control with versioned transcript-derived world state.

The model is used for puzzle interpretation and choosing an objective.  The
controller keeps a small transcript-derived frontier of exits and refuses to
spend a route repeatedly on an unchanged textual state.  This makes progress
 through an unfamiliar map an explicit code invariant rather than a hope in a
 long Astra conversation.  The recurrence guard is reopened after substantive
 actions because room prose and score do not encode inventory or spell state.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, deque
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


LIFECYCLE = re.compile(
    r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I
)
COMPOUND = re.compile(r"[\x00\r\n;{}\[\]]|\b(?:then|and then)\b", re.I)
PARSER = re.compile(
    r"not a verb|don't know the word|do not know the word|don't understand|"
    r"do not understand|not a sentence|not recogni[sz]|can't see any such|"
    r"cannot see any such|no such (?:thing|object)|not here|aren't here|"
    r"can't go|cannot go|not holding|not carrying|don't have|do not have|"
    r"which .{0,120}(?:mean|refer)|be more specific|yes or no|\by/n\b|"
    r"nothing happens|doesn't seem to work|won't open|is locked|is closed", re.I | re.S
)
HAZARD = re.compile(
    r"pitch[- ]black|in the darkness|grue|lamp.{0,80}(?:dim|flicker|out|dying|fading)|"
    r"attacks? you|strikes? you|wounded|bleeding|flood|stolen|disarmed|"
    r"you are dead|you have died|game over|congratulations|you win", re.I | re.S
)
MENU = re.compile(r"q\s*=\s*resume game|press\s+space|character input|help menu", re.I)

# Only commands explicitly represented by the current prose become local
# exploration candidates.  This prevents the generic noun-probe failure seen
# in an earlier candidate (words such as "can't" became commands).
DIRECTIONS = (
    "northwest", "northeast", "southwest", "southeast", "north", "south",
    "east", "west", "up", "down", "in", "out", "inside", "outside",
)
DIR_RE = re.compile(r"\b(" + "|".join(DIRECTIONS) + r")\b", re.I)
DIRECTION_ALIASES = {
    "northeast": "ne", "north-east": "ne", "northwest": "nw", "north-west": "nw",
    "southeast": "se", "south-east": "se", "southwest": "sw", "south-west": "sw",
    "inside": "in", "outside": "out",
}


def obs_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"text": "" if value is None else str(value)}


def text_of(obs: Any) -> str:
    return str(obs.get("text", "")) if isinstance(obs, dict) else str(obs or "")


def terminal(obs: dict[str, Any]) -> bool:
    return any(bool(obs.get(k)) for k in ("terminated", "truncated", "won", "lost"))


def clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    s = re.sub(r"^```(?:json|text|command)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()
    for _ in range(2):
        old = s
        s = re.sub(r"^(?:[-*>]|\d+[.)])\s*", "", s).strip()
        s = re.sub(r"^(?:command|action|input|cmd)\s*[:=|]\s*", "", s, flags=re.I).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
            s = s[1:-1].strip()
        if s == old:
            break
    s = s.rstrip(".").strip()
    if not s or len(s) > 200 or LIFECYCLE.match(s) or COMPOUND.search(s):
        return None
    if s.casefold() in {"command", "action", "none", "null", "noop"}:
        return None
    return s


def decode(raw: Any) -> tuple[list[str], str, str]:
    """Total decoding for model output, including arbitrary pre-flight text."""
    if not isinstance(raw, str):
        return [], "", ""
    source = raw.strip()[:30000]
    obj: Any = None
    for candidate in (source, re.sub(r"^```[^\n]*\n?|\n?```$", "", source).strip()):
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
                break
            except Exception:
                pass
        if obj is not None:
            break
    values: list[Any] = []
    if isinstance(obj, dict):
        for key in ("commands", "actions", "plan", "steps", "command", "action", "input"):
            value = obj.get(key)
            if key in ("command", "action") and isinstance(value, str) and value.casefold() in {"command", "action"}:
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item = item.get("command", item.get("action", item.get("input", item.get("cmd"))))
                    values.append(item)
            elif isinstance(value, str):
                values.append(value)
    elif isinstance(obj, list):
        values = obj
    elif isinstance(obj, str):
        values = [obj]
    if not values:
        for line in source.splitlines():
            if re.match(r"^\s*(?:command|action|input|cmd)\s*[:=]", line, re.I):
                values.append(line)
        if not values and "\n" not in source:
            values.append(source)
    commands: list[str] = []
    for value in values:
        command = clean(value)
        if command:
            commands.append(command)
        if len(commands) >= 8:
            break
    memory = obj.get("beliefs", obj.get("memory", "")) if isinstance(obj, dict) else ""
    if isinstance(memory, (dict, list)):
        memory = json.dumps(memory, ensure_ascii=False)
    if not isinstance(memory, str):
        memory = ""
    goal = obj.get("goal", "") if isinstance(obj, dict) else ""
    return commands, memory[:2400], str(goal)[:700]


def base_state_key(obs: dict[str, Any]) -> str:
    lines = [re.sub(r"\s+", " ", x).strip() for x in text_of(obs).splitlines() if x.strip()]
    return (" | ".join(lines[:3])[:500] + "\u241f" + str(obs.get("score")))


def exits(text: str) -> list[str]:
    out: list[str] = []
    for item in DIR_RE.findall(text):
        item = item.casefold()
        if item not in out:
            out.append(item)
    return out


def parser_repair(command: str, response: str) -> str | None:
    if not isinstance(command, str) or not isinstance(response, str):
        return None
    if not re.search(r"don't know the word|do not know the word|not a verb|can't go|cannot go", response, re.I):
        return None
    token = command.strip().casefold()
    replacement = DIRECTION_ALIASES.get(token)
    if replacement is None:
        return None
    if re.search(r"can't go|cannot go", response, re.I) and token not in {
        "northeast", "northwest", "southeast", "southwest", "north-east", "north-west", "south-east", "south-west"
    }:
        return None
    return replacement


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rows: list[dict[str, Any]] = []
        self.beliefs = ""
        self.goal = ""
        self.frontier: deque[tuple[str, str]] = deque()
        self.seen_frontier: set[tuple[str, str]] = set()
        self.attempted: set[tuple[str, str]] = set()
        self.edges: Counter[tuple[str, str, str]] = Counter()
        self.state_visits: Counter[str] = Counter()
        self.failed: deque[dict[str, Any]] = deque(maxlen=16)
        self.reason = "Start from the opening text and find a reachable scoring prerequisite."
        self.model_errors = 0
        self.invalid = 0
        # A room description and score are insufficient state keys: learning a
        # spell, opening a container, or changing inventory can leave both
        # unchanged.  Epochs reopen edges after every substantive action while
        # transition counts still stop genuine west/east oscillations.
        self.epoch = 0
        self.inventory_hint = ""
        self.transitions: Counter[tuple[str, str, str]] = Counter()

    def key(self, obs: dict[str, Any]) -> str:
        inv = re.sub(r"\s+", " ", self.inventory_hint).strip()[:700]
        return base_state_key(obs) + "\u241e" + str(self.epoch) + "\u241e" + inv

    def record(self, command: str | None, obs: dict[str, Any]) -> None:
        row = {"step": obs.get("step_count", len(self.rows)), "command": command,
               "score": obs.get("score"), "delta": obs.get("score_delta"),
               "moves": obs.get("native_moves"), "text": text_of(obs)[:4200]}
        self.rows.append(row)
        self.ctx.memory["transcript"] = self.rows[-80:]

    def refresh_frontier(self, obs: dict[str, Any]) -> None:
        key = self.key(obs)
        for direction in exits(text_of(obs)):
            pair = (key, direction)
            if pair not in self.seen_frontier:
                self.seen_frontier.add(pair)
                self.frontier.append(pair)

    def packet(self, game, obs: dict[str, Any], rem: dict[str, Any]) -> str:
        data = {
            "status": {k: obs.get(k) for k in ("game_id", "score", "max_score", "score_delta", "native_moves", "step_count", "terminated", "truncated", "won", "lost")},
            "current_text": text_of(obs)[:4800],
            "opening_text": self.rows[0]["text"][:2200] if self.rows else "",
            "recent": self.rows[-8:],
            "frontier": list(self.frontier)[:18],
            "repeated_edges": [{"state": a, "command": b, "response": c, "n": n} for (a, b, c), n in self.edges.items() if n >= 1][-14:],
            "failed": list(self.failed)[-8:], "beliefs": self.beliefs[:2200], "goal": self.goal,
            "world_context": {"epoch": self.epoch, "inventory_hint": self.inventory_hint[:900]},
            "why_replan": self.reason,
            "budget": {"calls": rem.get("system2_calls"), "wall_s": round(float(rem.get("wall_s", 0) or 0), 1)},
        }
        return (
            "You are Astra controlling a turn-based offline Jericho interactive-fiction story. "
            "Maximize FINAL native score. Thinking is free, but each parser command consumes one "
            "turn. The action value is the parser text, never the label `command`. Return ONLY "
            'JSON: {"goal":"...","commands":["..."],"beliefs":"..."}. '
            "Choose one reachable objective and give 3-8 ordinary commands whose later steps are "
            "safe direct consequences. Use nouns and clues in the evidence. Do not use lifecycle "
            "or compound commands. Stop before a risky irreversible action unless the transcript "
            "supports it. The controller tracks a directed frontier and a versioned world context. "
            "An edge that failed before may become valid after learning a spell, opening a container, "
            "or changing inventory; use current evidence before rejecting a retry. Do not oscillate "
            "through the same transition three times. If a parser error, "
            "clarification, hazard, score loss, or changed objective appears, the controller will "
            "replan. Keep beliefs provisional and concise.\n\n"
            f"GAME:\n{str(game.describe())[:2200]}\n"
            f"EVIDENCE PACKET:\n{json.dumps(data, ensure_ascii=False, default=str)}\n"
            "Reason silently and emit compact JSON."
        )

    def local_fallback(self, obs: dict[str, Any]) -> str | None:
        key = self.key(obs)
        while self.frontier:
            old_key, command = self.frontier.popleft()
            if old_key != key or (old_key, command) in self.attempted:
                # A frontier entry from a prior room is stale; fresh text will
                # repopulate it.  This also prevents alternating old exits.
                continue
            return command
        # Observation commands are bounded and only used once per exact state.
        if self.state_visits[key] == 1:
            return "look"
        if self.state_visits[key] == 2:
            return "inventory"
        return None

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            obs = obs_dict(game.observe())
        except Exception:
            obs = {"text": ""}
        self.record(None, obs)
        self.state_visits[self.key(obs)] += 1
        self.refresh_frontier(obs)
        pending: list[str] = []
        last_score = obs.get("score")
        while not ctx.done and not terminal(obs):
            if not pending:
                rem = ctx.budget.remaining()
                wall = float(rem.get("wall_s", 0) or 0)
                if rem.get("system2_calls", 0) <= 0 or wall < 8:
                    return
                try:
                    result = ctx.think(self.packet(game, obs, rem), session_id=None,
                                       timeout_s=max(1.0, min(75.0, wall - 5.0)))
                    pending, memory, goal = decode(getattr(result, "text", ""))
                    if memory:
                        self.beliefs = memory
                        ctx.memory["beliefs"] = memory
                    if goal:
                        self.goal = goal
                    self.model_errors = 0
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    ctx.log("think_failed", error=type(exc).__name__)
                    self.model_errors += 1
                    pending = []
                    if ctx.done or self.model_errors >= 3:
                        return
                except Exception as exc:
                    ctx.log("think_parse_failed", error=type(exc).__name__)
                    self.model_errors += 1
                    pending = []
                    if self.model_errors >= 3:
                        return
                if not pending:
                    self.invalid += 1
                    fallback = self.local_fallback(obs)
                    if fallback:
                        pending = [fallback]
                    elif self.invalid >= 3:
                        return
                else:
                    self.invalid = 0
                    ctx.log("plan", goal=self.goal, commands=pending[:])

            if not pending:
                continue
            command = clean(pending.pop(0))
            if not command:
                pending.clear()
                continue
            before_key = self.key(obs)
            # A repeated exact edge is a cycle signal.  One retry is allowed
            # after the story has changed score or text substantially.
            if (before_key, command.casefold()) in self.attempted:
                self.reason = "The proposed edge was already tried in this unchanged state; choose a new frontier or puzzle test."
                pending.clear()
                continue
            try:
                out = ctx.act(command)
                obs = obs_dict(out if out is not None else game.observe())
            except Exception as exc:
                ctx.log("act_failed", command=command, error=type(exc).__name__)
                self.failed.append({"command": command, "error": type(exc).__name__})
                pending.clear()
                if ctx.done:
                    return
                continue
            response = text_of(obs)
            delta = obs.get("score_delta")
            if not isinstance(delta, (int, float)) and isinstance(obs.get("score"), (int, float)) and isinstance(last_score, (int, float)):
                delta = obs.get("score") - last_score
            # Inventory responses are high-value state evidence.  Preserve a
            # compact exact hint so a later retry is distinguished even when
            # the room prose and native score are unchanged.
            if re.search(r"you are carrying|carrying:\s*$|spell book", response, re.I | re.S) and (
                    command.casefold() in {"inventory", "i", "spells", "read spell book", "read book", "open book"}
                    or re.search(r"you are carrying", response, re.I)):
                self.inventory_hint = response[:1200]
            # Any command capable of changing the world advances a local
            # version.  Pure observations do not; score changes always do.
            observational = command.casefold() in {
                "look", "l", "inventory", "i", "score", "spells", "read spell book",
                "read book", "open book", "examine", "x"
            } or command.casefold().startswith(("examine ", "read ", "look ", "how do "))
            if (not observational) or (isinstance(delta, (int, float)) and delta != 0):
                self.epoch += 1
            after_key = self.key(obs)
            before_base = base_state_key(self.rows[-1]) if self.rows else before_key
            after_base = base_state_key(obs)
            self.record(command, obs)
            self.attempted.add((before_key, command.casefold()))
            self.edges[(before_key, command.casefold(), re.sub(r"\s+", " ", response).strip()[:500])] += 1
            self.transitions[(before_base, command.casefold(), after_base)] += 1
            self.state_visits[after_key] += 1
            self.refresh_frontier(obs)
            interrupted = bool(PARSER.search(response) or HAZARD.search(response) or MENU.search(response))
            if isinstance(delta, (int, float)) and delta < 0:
                interrupted = True
            if self.state_visits[after_key] >= 3 and after_key == before_key:
                interrupted = True
            if self.transitions[(before_base, command.casefold(), after_base)] >= 3:
                interrupted = True
            alias = parser_repair(command, response)
            if alias and (before_key, alias.casefold()) not in self.attempted:
                pending.clear()
                pending.append(alias)
                interrupted = False
                self.reason = f"Parser rejected {command!r}; retrying observed direction alias {alias!r}."
                ctx.log("repair", command=command, replacement=alias, response=response[:300])
            if interrupted:
                pending.clear()
                self.failed.append({"command": command, "response": response[:350], "delta": delta})
                self.reason = "The latest response invalidated the route; reassess from exact evidence."
            last_score = obs.get("score")
            ctx.log("step", command=command, score=obs.get("score"), delta=delta)
            if terminal(obs) or ctx.done:
                return


def build(ctx):
    return Harness(ctx)

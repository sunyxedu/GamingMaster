"""Bounded-context Astra controller for turn-based Jericho stories."""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, deque
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable

LIFECYCLE = re.compile(r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I)
COMPOUND = re.compile(r"[\r\n\x00;{}\[\]]|\b(?:then|and then)\b", re.I)
PARSER_FAILURE = re.compile(
    r"not a verb|don't know the word|do not know the word|don't understand|do not understand|"
    r"didn't understand|not a sentence|not recogni[sz]|can't see any such|cannot see any such|"
    r"no such (?:thing|object)|not here|can't go|cannot go|can't do|cannot do|not holding|"
    r"not carrying|don't have|do not have|won't open|will not open|is locked|is closed|"
    r"nothing happens|doesn't seem to work|which .{0,100}(?:do you mean|one|refer)|"
    r"what do you mean|be more specific|please .{0,40}(?:answer|specify)|yes or no|\(y/n\)|\[y/n\]",
    re.I | re.S,
)
HAZARD = re.compile(
    r"pitch[- ](?:black|dark)|in the darkness|grue|lamp.{0,70}(?:dim|flicker|out|fading|dying)|"
    r"attacks? you|strikes? you|you are wounded|bleeding|rising water|(?:stolen|steals|disarmed)|"
    r"you are dead|you have died|game over|congratulations|you win|victory", re.I | re.S)
MENU = re.compile(r"q\s*=\s*resume game|return\s*=\s*read subject", re.I)


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


def text_of(obs: dict[str, Any]) -> str:
    return str(obs.get("text", ""))


def terminal(obs: dict[str, Any]) -> bool:
    return any(bool(obs.get(k)) for k in ("terminated", "truncated", "won", "lost"))


def clean_command(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    s = re.sub(r"^```(?:json|text|command)?\s*|\s*```$", "", s, flags=re.I).strip()
    for _ in range(3):
        old = s
        s = re.sub(r"^(?:[-*>]\s*|\d+[.)]\s*)", "", s).strip()
        s = re.sub(r"^(?:command|action|input|cmd)\s*[:=|]\s*", "", s, flags=re.I).strip()
        m = re.fullmatch(r"<(?:command|action)>(.*?)</(?:command|action)>", s, re.I | re.S)
        if m:
            s = m.group(1).strip()
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


def _walk(value: Any, result: list[str], depth: int = 0) -> None:
    if depth > 8 or len(result) >= 12:
        return
    if isinstance(value, str):
        command = clean_command(value)
        if command:
            result.append(command)
    elif isinstance(value, list):
        for item in value[:12]:
            _walk(item, result, depth + 1)
            if len(result) >= 12:
                return
    elif isinstance(value, dict):
        for key in ("commands", "plan", "actions", "steps", "command", "input", "cmd", "value"):
            if key not in value:
                continue
            if key == "command" and isinstance(value[key], str) and value[key].casefold() == "command":
                continue
            before = len(result)
            _walk(value[key], result, depth + 1)
            if len(result) > before:
                return


def decode(raw: Any) -> tuple[list[str], str]:
    """Total decoder: arbitrary stand-in output must never escape the harness."""
    if not isinstance(raw, str):
        return [], ""
    source = raw.strip()[:50000]
    obj: Any = None
    for candidate in (source, re.sub(r"^```[^\n]*\n?|\n?```$", "", source).strip()):
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
                break
            except Exception:
                continue
        if obj is not None:
            break
    memory = ""
    if isinstance(obj, dict) and isinstance(obj.get("memory"), (str, dict, list)):
        memory = obj["memory"] if isinstance(obj["memory"], str) else json.dumps(obj["memory"], ensure_ascii=False)
        memory = memory[:3000]
    commands: list[str] = []
    if obj is not None:
        _walk(obj, commands)
    if not commands:
        for line in source.splitlines():
            if re.match(r"^\s*(?:command|action|input|cmd)\s*[:=|]", line, re.I):
                command = clean_command(line)
                if command:
                    commands.append(command)
                    break
        if not commands and "\n" not in source:
            command = clean_command(source)
            if command:
                commands.append(command)
    return commands[:12], memory


def _nouns(text: str) -> list[str]:
    stop = {"the", "and", "you", "are", "was", "were", "this", "that", "with", "from", "have", "has", "here", "there", "into", "your", "about", "what", "which", "then", "only", "for", "not", "can", "will", "would", "could", "room", "look", "score", "inventory", "game", "story", "text", "command", "latest", "current"}
    out: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text):
        w = word.casefold()
        if w not in stop and w not in out and len(w) <= 24:
            out.append(w)
        if len(out) >= 8:
            break
    return out


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.records: list[dict[str, Any]] = []
        self.failed: deque[str] = deque(maxlen=16)
        self.notebook = ""
        self.reason = "Start from the opening description; find a reachable clue, tool, or scoring prerequisite."
        self.pairs: Counter[tuple[str, str, str]] = Counter()
        self.commands_seen: Counter[str] = Counter()
        self.probes: deque[str] = deque()
        self.game_packet = ""
        self.model_errors = 0
        self.invalid = 0

    def record(self, command: str | None, obs: dict[str, Any]) -> None:
        row = {"step": obs.get("step_count", len(self.records)), "command": command, "text": text_of(obs)[:4500], "score": obs.get("score"), "delta": obs.get("score_delta"), "moves": obs.get("native_moves")}
        self.records.append(row)
        self.ctx.memory["transcript"] = self.records[-80:]

    def packet(self, obs: dict[str, Any], remaining: dict[str, Any]) -> dict[str, Any]:
        blocked = [{"command": c, "response": r[:260], "score": s, "count": n} for (c, r, s), n in list(self.pairs.items())[-18:] if n]
        return {"status": {k: obs.get(k) for k in ("game_id", "score", "max_score", "score_delta", "native_moves", "step_count", "terminated", "truncated", "won", "lost")}, "opening": self.records[0].get("text", "")[:3500] if self.records else "", "current_exact": {**obs, "text": text_of(obs)[:5000]}, "recent_transcript": self.records[-8:], "score_events": [r for r in self.records if r.get("delta") not in (None, 0)][-8:], "failed_commands": list(self.failed)[-12:], "blocked_exact_pairs": blocked[-12:], "notebook_unverified": self.notebook[:2600], "why_replanning": self.reason, "budget": {"system2_calls": remaining.get("system2_calls"), "wall_s": round(float(remaining.get("wall_s", 0)), 1)}}

    def prompt(self, obs: dict[str, Any], remaining: dict[str, Any]) -> str:
        return ("You control a turn-based offline Jericho interactive-fiction story. Maximize FINAL native score and survive. Thinking consumes wall time but no game turn; every parser command consumes one turn. The action value is ordinary parser text, never the literal action name `command`. Use only supplied evidence.\n\nReturn ONLY JSON: {\"goal\":\"short\",\"memory\":\"evidence-based notes\",\"commands\":[\"one parser line\", ...]}. Give 4-10 commands only when later commands are direct, clue-supported consequences. The harness executes them one at a time and interrupts on parser failure, clarification, hazard, score loss, or terminal state. Positive score can confirm a route. Use exact nouns and verbs from the transcript; retain light, access, unique tools, and treasure dependencies. Change the hypothesis after an unchanged response or blocked pair. Do not use save/restore/restart/quit/undo, menus, compound commands, or invented walkthroughs. Each command is one line at most 200 characters.\n\nGAME CONTRACT:\n" + self.game_packet + "\nEVIDENCE PACKET:\n" + json.dumps(self.packet(obs, remaining), ensure_ascii=False, default=str))

    def make_probe(self, obs: dict[str, Any]) -> None:
        if self.probes:
            return
        for noun in _nouns(text_of(obs)):
            for verb in ("examine", "read", "open", "take"):
                command = clean_command(f"{verb} {noun}")
                if command and self.commands_seen[command.casefold()] == 0:
                    self.probes.append(command)
        for direction in ("north", "south", "east", "west", "up", "down", "in", "out"):
            if self.commands_seen[direction] == 0:
                self.probes.append(direction)

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            self.game_packet = str(game.describe())[:4500] + "\n" + str(game.action_spec.describe())[:1800]
        except Exception:
            self.game_packet = "One action named command; its value is a single parser string."
        try:
            obs = obs_dict(ctx.observe())
        except Exception:
            obs = obs_dict(game.observe())
        self.record(None, obs)
        pending: deque[str] = deque()
        while not ctx.done and not terminal(obs):
            if not pending:
                remaining = ctx.budget.remaining()
                wall = float(remaining.get("wall_s", 0) or 0)
                if remaining.get("system2_calls", 0) <= 0 or wall < 10:
                    return
                try:
                    result = ctx.think(self.prompt(obs, remaining), session_id=None, timeout_s=max(8.0, min(105.0, wall - 6.0)))
                    commands, memory = decode(getattr(result, "text", ""))
                    pending.extend(commands)
                    if memory:
                        self.notebook = memory
                        ctx.memory["notebook"] = memory
                    self.model_errors = 0
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    ctx.log("think_failed", error=type(exc).__name__)
                    self.model_errors += 1
                    self.reason = "A bounded reasoning call failed; recover from unchanged exact evidence."
                    if ctx.done or self.model_errors >= 3:
                        return
                except Exception as exc:
                    ctx.log("think_failed", error=type(exc).__name__)
                    self.model_errors += 1
                    if ctx.done or self.model_errors >= 3:
                        return
                if not pending:
                    self.invalid += 1
                    self.reason = "Planner output was empty or malformed; use strict JSON with actual parser commands."
                    self.make_probe(obs)
                    if self.probes:
                        pending.append(self.probes.popleft())
                    elif self.invalid >= 3:
                        return
                else:
                    self.invalid = 0
                    self.reason = "Replan from the latest exact response and preserve verified prerequisites."
            if not pending:
                continue
            command = clean_command(pending.popleft())
            if not command:
                continue
            before = text_of(obs)
            pair_key = (command.casefold(), re.sub(r"\s+", " ", before).strip()[:1000], str(obs.get("score")))
            if self.pairs[pair_key] >= 2:
                self.failed.append(command)
                pending.clear()
                self.reason = "The exact command/response/score pair repeated; choose a new discriminating action."
                continue
            try:
                raw = ctx.act(command)
                obs = obs_dict(raw if raw is not None else ctx.observe())
            except Exception as exc:
                ctx.log("act_failed", command=command, error=type(exc).__name__)
                self.failed.append(command)
                pending.clear()
                self.reason = "The interface rejected the command; repair grammar or find its prerequisite."
                if ctx.done:
                    return
                continue
            self.record(command, obs)
            response = text_of(obs)
            self.commands_seen[command.casefold()] += 1
            self.pairs[(command.casefold(), re.sub(r"\s+", " ", response).strip()[:1000], str(obs.get("score")))] += 1
            if terminal(obs) or ctx.done:
                return
            if MENU.search(response):
                pending.clear()
                self.reason = "The response is an interactive menu; answer it or restore normal parser mode."
            elif PARSER_FAILURE.search(response) or HAZARD.search(response):
                pending.clear()
                self.failed.append(command)
                self.reason = "The latest response invalidates the route: parser/precondition, hazard, or terminal evidence."
            elif isinstance(obs.get("score_delta"), (int, float)) and obs.get("score_delta") < 0:
                pending.clear()
                self.reason = "Native score fell; abandon the suffix and reassess irreversible loss or danger."
            elif not response.strip():
                pending.clear()
                self.reason = "The parser response was empty; inspect the current input mode before proceeding."


def build(ctx):
    return Harness(ctx)

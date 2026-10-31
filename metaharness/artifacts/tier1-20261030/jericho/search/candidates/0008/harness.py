"""Fresh-context, short-commitment controller for Jericho parser stories.

Each planning call receives a bounded evidence packet and starts a fresh Astra
session.  This avoids hidden conversation growth while still amortizing the
slow model call over a few commands.  Code owns transcript, recurrence checks,
and parser/menu interruption handling; Astra owns puzzle interpretation.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


LIFECYCLE = re.compile(
    r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I
)
COMPOUND = re.compile(r"[\x00\r\n;{}\[\]]|\b(?:then|and then)\b", re.I)
PARSER_FAILURE = re.compile(
    r"not a verb|don't know the word|do not know the word|don't understand|"
    r"do not understand|not a sentence|not recogni[sz]|can't see any such|"
    r"cannot see any such|no such (?:thing|object)|not here|aren't here|"
    r"you (?:can't|cannot) (?:go|take|open|do)|not holding|not carrying|"
    r"you (?:don't|do not) have|which .{0,100}(?:mean|refer)|be more specific|"
    r"yes or no|\by/n\b|nothing happens|doesn't seem to work|won't open", re.I | re.S
)
HAZARD = re.compile(
    r"pitch[- ]black|in the darkness|grue|lamp.{0,70}(?:dim|flicker|out|dying|fading)|"
    r"(?:attacks|strikes|wounds|hits) you|you are wounded|flood|stolen|disarmed|"
    r"you are dead|you have died|game over|congratulations|you win", re.I | re.S
)
MENU = re.compile(r"q\s*=\s*resume game|press\s+space|character input|help menu", re.I)


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


def clean_command(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    s = re.sub(r"^```(?:json|text|command)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()
    s = re.sub(r"^(?:[-*>]|\d+[.)])\s*", "", s).strip()
    s = re.sub(r"^(?:command|action|input|cmd)\s*[:=|]\s*", "", s, flags=re.I).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    if not s or len(s) > 200 or LIFECYCLE.match(s) or COMPOUND.search(s):
        return None
    if s.casefold() in {"command", "action", "none", "null", "noop"}:
        return None
    return s.rstrip(".").strip() or None


def decode(raw: Any) -> tuple[list[str], str]:
    """Total decoding for strict JSON, fenced JSON, tagged lines, or garbage."""
    if not isinstance(raw, str):
        return [], ""
    s = raw.strip()[:30000]
    obj: Any = None
    candidates = [s, re.sub(r"^```[^\n]*\n?|\n?```$", "", s).strip()]
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
                break
            except Exception:
                pass
        if obj is not None:
            break

    values: list[Any] = []

    def collect(x: Any, depth: int = 0) -> None:
        if depth > 6 or len(values) >= 8:
            return
        if isinstance(x, str):
            values.append(x)
        elif isinstance(x, list):
            for y in x:
                collect(y, depth + 1)
        elif isinstance(x, dict):
            for key in ("commands", "actions", "plan", "steps", "command", "action", "input", "cmd"):
                if key not in x:
                    continue
                val = x[key]
                if key in ("action", "command") and isinstance(val, str) and val.casefold() in {"command", "action"}:
                    continue
                collect(val, depth + 1)

    if obj is not None:
        collect(obj)
    if not values:
        for line in s.splitlines():
            if re.match(r"^\s*(?:command|action|input|cmd)\s*[:=]", line, re.I):
                values.append(line)
        if not values and "\n" not in s:
            values.append(s)

    commands: list[str] = []
    for value in values:
        command = clean_command(value)
        if command:
            commands.append(command)
        if len(commands) >= 5:
            break

    memory = ""
    if isinstance(obj, dict):
        candidate_memory = obj.get("beliefs", obj.get("memory", ""))
        if isinstance(candidate_memory, (dict, list)):
            candidate_memory = json.dumps(candidate_memory, ensure_ascii=False)
        if isinstance(candidate_memory, str):
            memory = candidate_memory[:2400]
    return commands, memory


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.records: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.beliefs = ""
        self.recurrences: Counter[tuple[str, str, str]] = Counter()
        self.model_failures = 0
        self.invalid_outputs = 0

    def packet(self, game, obs: dict[str, Any], remaining: dict[str, Any]) -> str:
        recent = []
        for row in self.records[-8:]:
            recent.append({
                "step": row.get("step"), "command": row.get("command"),
                "score": row.get("score"), "delta": row.get("delta"),
                "text": str(row.get("text", ""))[:900],
            })
        status = {
            "game_id": obs.get("game_id"), "score": obs.get("score"),
            "max_score": obs.get("max_score"), "score_delta": obs.get("score_delta"),
            "native_moves": obs.get("native_moves"), "step_count": obs.get("step_count"),
            "calls_left": remaining.get("system2_calls"),
            "wall_left": round(float(remaining.get("wall_s", 0) or 0), 1),
        }
        return (
            "You are Astra controlling a turn-based offline Jericho interactive-fiction story. "
            "Maximize FINAL native score. Thinking takes no game turn; every posted parser line "
            "does. Use only supplied evidence and infer puzzle dependencies from the prose. "
            "The sole action value is the parser command text, never the label `command`.\n\n"
            "Return ONLY JSON of the form {\"commands\":[\"...\"],\"beliefs\":\"...\"}. "
            "Give 2-5 ordinary one-line commands, <=200 characters each. The first must fit the "
            "current observation; later commands must be safe direct consequences. Do not use "
            "lifecycle commands, semicolon/compound commands, or guessed nouns absent from the "
            "evidence. Stop a route after a clarification, parser failure, hazard, or changed "
            "objective. Keep beliefs concise and label uncertainty; never claim an unverified fact.\n\n"
            f"GAME SUMMARY:\n{str(game.describe())[:1800]}\n"
            f"STATUS:\n{json.dumps(status, ensure_ascii=False)}\n"
            f"CURRENT OBSERVATION (verbatim text):\n{text_of(obs)[:4200]}\n"
            f"OPENING OBSERVATION:\n{str(self.records[0].get('text', ''))[:2600] if self.records else text_of(obs)[:2600]}\n"
            f"RECENT TRANSCRIPT:\n{json.dumps(recent, ensure_ascii=False)}\n"
            f"FAILED OR INTERRUPTED ACTIONS:\n{json.dumps(self.failed[-8:], ensure_ascii=False)}\n"
            f"UNVERIFIED BELIEFS:\n{self.beliefs[:2400]}\n"
            "Reason silently, then emit compact JSON."
        )

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            obs = obs_dict(game.observe())
        except Exception:
            obs = {"text": ""}
        self.records.append({"step": obs.get("step_count", 0), "command": None,
                             "score": obs.get("score"), "delta": obs.get("score_delta", 0),
                             "text": text_of(obs)})
        pending: list[str] = []
        last_score = obs.get("score")
        fallback_used = False

        while not ctx.done and not terminal(obs):
            if not pending:
                rem = ctx.budget.remaining()
                wall = float(rem.get("wall_s", 0) or 0)
                if rem.get("system2_calls", 0) <= 0 or wall < 7:
                    return
                try:
                    # Fresh session is intentional: the bounded packet is the
                    # complete context and prevents cumulative hidden history.
                    result = ctx.think(
                        self.packet(game, obs, rem), session_id=None,
                        timeout_s=max(1.0, min(100.0, wall - 4.0)),
                    )
                    pending, memory = decode(getattr(result, "text", ""))
                    if memory:
                        self.beliefs = memory
                        ctx.memory["beliefs"] = memory
                    self.model_failures = 0
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    ctx.log("think_failed", error=type(exc).__name__)
                    self.model_failures += 1
                    pending = []
                    if ctx.done or self.model_failures >= 3:
                        return
                except Exception as exc:
                    ctx.log("think_parse_failed", error=type(exc).__name__)
                    self.model_failures += 1
                    pending = []
                    if self.model_failures >= 3:
                        return
                if not pending:
                    self.invalid_outputs += 1
                    # Only a bounded, generic observation fallback is allowed;
                    # noun probes derived from prose can turn words like
                    # "can't" into commands and waste the episode.
                    if not fallback_used:
                        pending = ["look"]
                        fallback_used = True
                    elif self.invalid_outputs >= 3:
                        return
                else:
                    self.invalid_outputs = 0
                    fallback_used = False
                    ctx.log("plan", commands=pending[:])

                # A repeated malformed/timeout response may intentionally
                # leave no fallback after the bounded recovery attempt.
                if not pending:
                    continue

            command = clean_command(pending.pop(0))
            if not command:
                continue
            before_text = text_of(obs)
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
            row = {"step": obs.get("step_count", len(self.records)), "command": command,
                   "score": obs.get("score"), "delta": delta, "text": response}
            self.records.append(row)
            ctx.memory["transcript"] = self.records[-80:]
            key = (command.casefold(), re.sub(r"\s+", " ", response).strip(), str(obs.get("score")))
            self.recurrences[key] += 1
            interrupted = bool(PARSER_FAILURE.search(response) or HAZARD.search(response) or MENU.search(response))
            # Preserve a suffix across positive rewards, but cut it after a
            # loss or any explicit parser/context interruption.
            if isinstance(delta, (int, float)) and delta < 0:
                interrupted = True
            if self.recurrences[key] >= 2:
                interrupted = True
            if interrupted:
                pending.clear()
                self.failed.append({"command": command, "response": response[:350], "delta": delta})
            elif before_text == response and command.casefold() in {"look", "inventory", "i", "l"}:
                # Repeated observation with no new information should force a
                # fresh hypothesis rather than consume an entire macro.
                pending.clear()
            last_score = obs.get("score")
            ctx.log("step", command=command, score=obs.get("score"), delta=delta)
            if terminal(obs) or ctx.done:
                return


def build(ctx):
    return Harness(ctx)

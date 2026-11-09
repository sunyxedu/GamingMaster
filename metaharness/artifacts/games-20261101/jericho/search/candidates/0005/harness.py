"""Long-horizon, evidence-checked control for turn based parser stories.

The previous controller threw away a batch whenever the native score changed.
That is a poor signal in these stories: taking a treasure or crossing a scored
room is often exactly the evidence that the next command in a route is ready.
This controller keeps a short route through positive score changes and only
cuts it on parser/context evidence.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


RESERVED = re.compile(
    r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)\b",
    re.I,
)
BAD = re.compile(
    r"not a verb|don't know the word|do not know the word|don't understand|"
    r"do not understand|didn't understand|not a sentence|beg your pardon|"
    r"not recogni[sz]|can't see any such|cannot see any such|no such (?:thing|object)|"
    r"not something you need to refer|can't go|cannot go|can't do|cannot do|"
    r"not holding|not carrying|don't have|do not have|won't open|will not open|"
    r"is locked|is closed|nothing happens|doesn't seem to work|"
    r"which .{0,90}(?:do you mean|one)|what do you mean|be more specific|"
    r"please .{0,30}(?:answer|specify)|yes or no|yes/no|\by/n\b|"
    r"you are (?:dead|dying)|you have died|game over|congratulations|you win|"
    r"\*\*\*", re.I,
)
HAZARD = re.compile(
    r"pitch black|in the darkness|lamp .*?(?:dim|flicker|out)|light .*?(?:fading|dying)|"
    r"attacks? you|strikes? you|you are wounded|bleeding|flood|stolen|disarmed|"
    r"you will be eaten", re.I,
)


def _obs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            x = json.loads(value)
            if isinstance(x, dict):
                return x
        except Exception:
            pass
        return {"text": value}
    return {"text": "" if value is None else str(value)}


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    s = re.sub(r"^```(?:json|text|command)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()
    s = re.sub(r"^(?:[-*>]|\d+[.)])\s*", "", s).strip()
    s = re.sub(r"^(?:command|action|input)\s*[:=]\s*", "", s, flags=re.I).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    s = s.rstrip(".").strip()
    if not s or len(s) > 200 or "\n" in s or "\r" in s or RESERVED.match(s):
        return None
    if s.casefold() in {"command", "action", "none", "null", "noop", "q"}:
        return None
    # A compound command would bypass the one-turn contract.
    if any(c in s for c in "{}[];") or re.search(r"\bthen\b|\band then\b", s, re.I):
        return None
    return s


def _decode(text: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Total parser for strict JSON plus harmless stand-in model garbage."""
    if not isinstance(text, str):
        return [], None
    raw: Any = None
    s = text.strip()
    for candidate in (s, re.sub(r"^```[^\n]*\n?|\n?```$", "", s).strip()):
        for loader in (json.loads, ast.literal_eval):
            try:
                raw = loader(candidate)
                break
            except Exception:
                continue
        if raw is not None:
            break
    if raw is None:
        # Permit explicit command lines and a single bare parser command.
        lines = []
        for line in s.splitlines():
            line = line.strip()
            if re.match(r"^(?:command|action|input)\s*[:=]", line, re.I):
                lines.append(re.sub(r"^(?:command|action|input)\s*[:=]\s*", "", line, flags=re.I))
        if not lines and "\n" not in s:
            lines = [s]
        raw = lines
    memory = None
    if isinstance(raw, dict):
        memory = raw.get("memory")
        raw = raw.get(
            "commands", raw.get("plan", raw.get("actions", [raw] if any(
                k in raw for k in ("command", "action", "input")
            ) else []))
        )
    elif isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    plan: list[dict[str, Any]] = []
    for item in raw[:12]:
        if isinstance(item, dict):
            command = _clean(item.get("command", item.get("action", item.get("input"))))
            expect = item.get("expect", [])
            stop = bool(item.get("stop", False))
        else:
            command, expect, stop = _clean(item), [], False
        if not command:
            # Do not execute a dependent suffix after malformed model output.
            break
        if isinstance(expect, str):
            expect = [expect]
        if not isinstance(expect, list):
            expect = []
        expect = [str(x).casefold()[:120] for x in expect[:4] if isinstance(x, str) and x.strip()]
        plan.append({"command": command, "expect": expect, "stop": stop})
    if isinstance(memory, (dict, list)):
        memory = json.dumps(memory, ensure_ascii=False)
    if not isinstance(memory, str):
        memory = None
    return plan, memory[:3500] if memory else None


def _text(obs: dict[str, Any]) -> str:
    return str(obs.get("text", ""))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.history: list[dict[str, Any]] = []
        self.notebook = ""
        self.session = None
        self.feedback = ""

    def _prompt(self, game, obs: dict[str, Any], remaining: dict[str, Any]) -> str:
        recent = self.history[-12:]
        status = {k: obs.get(k) for k in (
            "game_id", "score", "max_score", "score_delta", "native_moves",
            "step_count", "terminated", "truncated")}
        return (
            "You are Astra controlling an offline Jericho interactive-fiction story. "
            "Maximize final native score and stay alive. The world advances one turn per "
            "ordinary parser command; thinking is free. The action value is the parser text, "
            "never the literal action name `command`.\n\n"
            f"GAME:\n{game.describe()}\nACTION SPEC:\n{game.action_spec.describe()}\n"
            f"STATUS:\n{json.dumps(status, ensure_ascii=False)}\n"
            f"BUDGET:\n{json.dumps(remaining)}\n"
            "Return ONLY JSON: {\"commands\":[{\"command\":\"...\",\"expect\":[\"short literal\"],\"stop\":false}],\"memory\":\"...\"}. "
            "Give 4-12 concise, sequential commands when the route is supported by the transcript. "
            "Each command is one parser line, <=200 chars, and later commands must remain sensible "
            "after successful score changes. Use expect only for a short literal response that must "
            "be present; stop after a risky decision. Do not use save/restore/restart/quit/undo or "
            "compound commands. Preserve useful inventory, routes, clues and failed hypotheses in "
            "memory, citing evidence steps.\n"
            f"NOTEBOOK (hypotheses, verify): {self.notebook}\n"
            f"FEEDBACK FROM LAST ROUTE: {self.feedback or 'New episode: inspect the premise and pursue the best reachable prerequisite.'}\n"
            f"EXACT RECENT TRANSCRIPT: {json.dumps(recent, ensure_ascii=False)}\n"
            "Reason silently and emit compact JSON."
        )

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        obs = _obs(game.observe())
        previous_score = obs.get("score")
        pending: list[dict[str, Any]] = []
        bad_calls = 0
        invalid = 0
        repeats: dict[tuple[str, str], int] = {}
        while not ctx.done and not obs.get("terminated") and not obs.get("truncated"):
            if not pending:
                rem = ctx.budget.remaining()
                if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 8:
                    return
                try:
                    result = ctx.think(
                        self._prompt(game, obs, rem), session_id=self.session,
                        timeout_s=max(1.0, min(180.0, float(rem.get("wall_s", 8)) - 3.0)),
                    )
                    self.session = getattr(result, "session_id", self.session)
                    pending, memory = _decode(getattr(result, "text", ""))
                    if memory:
                        self.notebook = memory
                        ctx.memory["notebook"] = memory
                    bad_calls = 0
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    ctx.log("think_failed", error=type(exc).__name__)
                    bad_calls += 1
                    if ctx.done or bad_calls >= 3:
                        return
                    self.feedback = "Reasoning call failed; recover from the unchanged exact state."
                    continue
                except Exception as exc:
                    ctx.log("think_parse_failed", error=type(exc).__name__)
                    bad_calls += 1
                    if bad_calls >= 3:
                        return
                    continue
                if not pending:
                    invalid += 1
                    self.feedback = "Output was unusable. Return JSON commands with actual parser text."
                    if invalid >= 3:
                        return
                    continue
                invalid = 0
                self.feedback = ""
                ctx.log("route", commands=[x["command"] for x in pending])

            item = pending.pop(0)
            command = item["command"]
            before = _text(obs)
            key = (command.casefold(), before)
            if repeats.get(key, 0) >= 2:
                self.feedback = "This command produced the same response twice; change the hypothesis."
                pending.clear()
                continue
            try:
                out = ctx.act(command)
            except Exception as exc:
                ctx.log("act_failed", command=command, error=type(exc).__name__)
                pending.clear()
                self.feedback = "The interface rejected the last command; diagnose it before retrying."
                if ctx.done:
                    return
                continue
            obs = _obs(out if out is not None else game.observe())
            response = _text(obs)
            self.history.append({"step": obs.get("step_count", len(self.history) + 1),
                                 "command": command, "text": response,
                                 "score": obs.get("score"), "delta": obs.get("score_delta"),
                                 "moves": obs.get("native_moves")})
            ctx.memory["transcript"] = self.history[-80:]
            pair = (command.casefold(), response)
            repeats[pair] = repeats.get(pair, 0) + 1
            new_score = obs.get("score")
            # Positive score is evidence that this route is working. A loss is
            # evidence of a changed/unsafe state and invalidates its suffix.
            if isinstance(previous_score, (int, float)) and isinstance(new_score, (int, float)) and new_score < previous_score:
                pending.clear()
                self.feedback = "Native score fell; abandon the old route and reassess hazards/resources."
            previous_score = new_score
            low = response.casefold()
            interrupted = bool(BAD.search(response) or HAZARD.search(response))
            if item.get("expect") and not any(x in low for x in item["expect"]):
                interrupted = True
            if interrupted or item.get("stop") or repeats[pair] >= 2:
                pending.clear()
                self.feedback = "Route stopped after the exact response to " + repr(command) + ": " + response[:300]
            if obs.get("terminated") or obs.get("truncated") or ctx.done:
                return


def build(ctx):
    return Harness(ctx)

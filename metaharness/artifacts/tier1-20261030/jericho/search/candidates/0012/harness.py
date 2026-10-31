"""Guarded transactional routes for turn based Jericho stories.

The model may suggest a short route, but code commits each transition only when
the response matches the model's declared evidence.  This keeps a successful
first command from silently carrying a stale suffix into a new parser state.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


LIFE = re.compile(
    r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I
)
COMPOUND = re.compile(r"[\x00\r\n;{}\[\]]|\b(?:then|and then)\b", re.I)
PARSER = re.compile(
    r"not a verb|don't know the word|do not know the word|don't understand|"
    r"do not understand|not a sentence|not recogni[sz]|can't see any such|"
    r"cannot see any such|no such (?:thing|object)|can't go|cannot go|not here|"
    r"not holding|not carrying|don't have|do not have|which .{0,120}(?:mean|refer)|"
    r"be more specific|yes or no|\by/n\b|nothing happens|doesn't seem to work|"
    r"won't open|is locked|is closed", re.I | re.S
)
HAZARD = re.compile(
    r"pitch[- ]black|in the darkness|grue|lamp.{0,80}(?:dim|flicker|out|dying|fading)|"
    r"attacks? you|strikes? you|wounded|bleeding|flood|stolen|disarmed|"
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


def clean(value: Any, allow_menu_answer: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    s = re.sub(r"^```(?:json|text|command)?\s*|\s*```$", "", s, flags=re.I).strip()
    s = re.sub(r"^(?:[-*>]|\d+[.)])\s*", "", s).strip()
    s = re.sub(r"^(?:command|action|input|cmd)\s*[:=|]\s*", "", s, flags=re.I).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    if not s or len(s) > 200 or LIFE.match(s) or COMPOUND.search(s):
        return None
    if not allow_menu_answer and s.casefold() in {"command", "action", "none", "null", "noop", "q"}:
        return None
    return s.rstrip(".").strip() or None


def _json_value(raw: str) -> Any:
    """Decode strict/fenced Python-like output without ever raising."""
    source = raw.strip()[:40000]
    for candidate in (source, re.sub(r"^```[^\n]*\n?|\n?```$", "", source).strip()):
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(candidate)
            except Exception:
                pass
    return None


def decode(raw: Any) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(raw, str):
        return [], ""
    obj = _json_value(raw)
    route: Any = None
    memory: Any = ""
    if isinstance(obj, dict):
        route = obj.get("route", obj.get("commands", obj.get("plan", obj.get("actions"))))
        memory = obj.get("beliefs", obj.get("memory", ""))
        if route is None and any(k in obj for k in ("command", "action", "input", "cmd")):
            route = [obj]
    elif isinstance(obj, list):
        route = obj
    elif isinstance(obj, str):
        route = [obj]
    if route is None:
        tagged = []
        for line in raw.splitlines():
            if re.match(r"^\s*(?:command|action|input|cmd)\s*[:=]", line, re.I):
                tagged.append(line)
        route = tagged or ([raw] if "\n" not in raw else [])
    if isinstance(route, dict):
        route = [route]
    if not isinstance(route, list):
        route = []
    out: list[dict[str, Any]] = []
    for item in route[:10]:
        if isinstance(item, str):
            command, expect, avoid, risk, stop = clean(item), [], [], "unknown", False
        elif isinstance(item, dict):
            command = clean(item.get("command", item.get("action", item.get("input", item.get("cmd")))))
            expect, avoid = item.get("expect", item.get("success", [])), item.get("avoid", item.get("abort", []))
            risk, stop = str(item.get("risk", "unknown"))[:20].lower(), bool(item.get("stop", False))
        else:
            continue
        if not command:
            break
        if isinstance(expect, str):
            expect = [expect]
        if isinstance(avoid, str):
            avoid = [avoid]
        if not isinstance(expect, list):
            expect = []
        if not isinstance(avoid, list):
            avoid = []
        out.append({
            "command": command,
            "expect": [str(x).casefold()[:180] for x in expect[:6] if isinstance(x, str) and x.strip()],
            "avoid": [str(x).casefold()[:180] for x in avoid[:6] if isinstance(x, str) and x.strip()],
            "risk": risk,
            "stop": stop,
        })
    # A suffix without an evidence contract is not transactional.  Retain the
    # first command so arbitrary stand-ins remain harmless and total.
    if len(out) > 1:
        out = out[:1] + [x for x in out[1:] if x["expect"]]
    if isinstance(memory, (dict, list)):
        memory = json.dumps(memory, ensure_ascii=False)
    if not isinstance(memory, str):
        memory = ""
    return out, memory[:2800]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rows: list[dict[str, Any]] = []
        self.beliefs = ""
        self.failed: list[dict[str, Any]] = []
        self.recurrence: Counter[tuple[str, str]] = Counter()
        self.model_failures = 0
        self.invalid = 0
        self.menu_used = False

    def _packet(self, game, obs: dict[str, Any], rem: dict[str, Any]) -> str:
        recent = [
            {"step": r.get("step"), "command": r.get("command"), "score": r.get("score"),
             "delta": r.get("delta"), "text": r.get("text", "")[:1100]}
            for r in self.rows[-9:]
        ]
        status = {k: obs.get(k) for k in ("game_id", "score", "max_score", "score_delta", "native_moves", "step_count")}
        status.update({"calls_left": rem.get("system2_calls"), "wall_left": round(float(rem.get("wall_s", 0) or 0), 1)})
        return (
            "You are Astra controlling a turn-based offline Jericho parser story. Maximize final "
            "native score and survive. Thinking consumes no game turn; each posted parser line does. "
            "The action value is the raw command text, never the label command.\n\n"
            "Return ONLY JSON: {\"route\":[{\"command\":\"...\",\"expect\":[\"literal text expected "
            "in the next response\"],\"avoid\":[\"danger text\"],\"risk\":\"low\",\"stop\":false}],"
            "\"beliefs\":\"compact provisional notes\"}. Give 2-6 commands, each one parser line "
            "<=200 chars. Every command after the first MUST include a short literal expect anchor "
            "that code can verify in the immediate response. Use exact nouns and exits from evidence. "
            "A route is cut on a missing anchor, parser error, ambiguity, menu, hazard, score loss, "
            "or changed objective. Mark irreversible or uncertain commands risk=high and stop=true. "
            "Do not use lifecycle or compound commands; label beliefs as hypotheses and cite prior steps.\n\n"
            f"GAME:\n{str(game.describe())[:1800]}\nSTATUS:\n{json.dumps(status, ensure_ascii=False)}\n"
            f"CURRENT RESPONSE (verbatim):\n{text_of(obs)[:4500]}\n"
            f"OPENING RESPONSE:\n{self.rows[0].get('text', '')[:2600] if self.rows else text_of(obs)[:2600]}\n"
            f"RECENT EXACT TRANSCRIPT:\n{json.dumps(recent, ensure_ascii=False)}\n"
            f"FAILED OR GUARD-MISMATCHED ACTIONS:\n{json.dumps(self.failed[-8:], ensure_ascii=False)}\n"
            f"PROVISIONAL NOTEBOOK:\n{self.beliefs[:2800]}\nReason silently, then emit compact JSON."
        )

    def _record(self, command: str | None, obs: dict[str, Any]) -> None:
        row = {"step": obs.get("step_count", len(self.rows)), "command": command,
               "score": obs.get("score"), "delta": obs.get("score_delta"),
               "moves": obs.get("native_moves"), "text": text_of(obs)[:5000]}
        self.rows.append(row)
        self.ctx.memory["transcript"] = self.rows[-80:]

    def _menu_exit(self, game, obs: dict[str, Any]) -> dict[str, Any] | None:
        if not MENU.search(text_of(obs)) or self.menu_used or self.ctx.done:
            return None
        # q is reserved as a lifecycle token in normal model output, but is a
        # contextual answer when the live parser explicitly asks to resume.
        self.menu_used = True
        try:
            out = self.ctx.act("q")
            nxt = obs_dict(out if out is not None else game.observe())
            self._record("q", nxt)
            return nxt
        except Exception as exc:
            self.ctx.log("menu_exit_failed", error=type(exc).__name__)
        return obs

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            obs = obs_dict(game.observe())
        except Exception:
            obs = {"text": ""}
        self._record(None, obs)
        pending: list[dict[str, Any]] = []
        previous_score = obs.get("score")
        while not ctx.done and not terminal(obs):
            menu_obs = self._menu_exit(game, obs)
            if menu_obs is not None:
                # The contextual answer changed the observation; ask Astra to
                # interpret it rather than executing the prior route.
                obs = menu_obs
                pending.clear()
                continue
            if not pending:
                rem = ctx.budget.remaining()
                wall = float(rem.get("wall_s", 0) or 0)
                if rem.get("system2_calls", 0) <= 0 or wall < 8:
                    return
                try:
                    result = ctx.think(self._packet(game, obs, rem), session_id=None,
                                       timeout_s=max(1.0, min(85.0, wall - 4.0)))
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
                    if ctx.done or self.model_failures >= 3:
                        return
                except Exception as exc:
                    ctx.log("think_parse_failed", error=type(exc).__name__)
                    self.model_failures += 1
                    if self.model_failures >= 3:
                        return
                if not pending:
                    self.invalid += 1
                    if self.invalid >= 3:
                        return
                    # A harmless observation is preferable to a malformed
                    # model payload becoming an episode error.
                    pending = [{"command": "look", "expect": [], "avoid": [], "risk": "low", "stop": True}]
                else:
                    self.invalid = 0
                    ctx.log("guarded_route", commands=[x["command"] for x in pending])
            item = pending.pop(0)
            command = item.get("command")
            if not command:
                pending.clear()
                continue
            before_text = text_of(obs)
            state = (re.sub(r"\s+", " ", before_text).strip()[:900], str(obs.get("score")))
            self.recurrence[state] += 1
            if self.recurrence[state] >= 3:
                pending.clear()
                self.failed.append({"command": command, "reason": "repeated state"})
            try:
                out = ctx.act(command)
                obs = obs_dict(out if out is not None else game.observe())
            except Exception as exc:
                ctx.log("act_failed", command=command, error=type(exc).__name__)
                self.failed.append({"command": command, "reason": type(exc).__name__})
                pending.clear()
                if ctx.done:
                    return
                continue
            response = text_of(obs)
            self._record(command, obs)
            low = response.casefold()
            delta = obs.get("score_delta")
            if not isinstance(delta, (int, float)) and isinstance(previous_score, (int, float)) and isinstance(obs.get("score"), (int, float)):
                delta = obs.get("score") - previous_score
            mismatch = bool(item.get("expect")) and not any(x in low for x in item["expect"])
            avoid_hit = any(x in low for x in item.get("avoid", []))
            interrupted = bool(PARSER.search(response) or HAZARD.search(response) or MENU.search(response) or mismatch or avoid_hit)
            if isinstance(delta, (int, float)) and delta < 0:
                interrupted = True
            if mismatch or avoid_hit:
                self.failed.append({"command": command, "reason": "guard mismatch", "response": response[:360]})
            if interrupted or item.get("stop"):
                pending.clear()
            previous_score = obs.get("score")
            ctx.log("step_guard", command=command, score=obs.get("score"), delta=delta,
                    guard_ok=not mismatch, interrupted=interrupted)
            if ctx.done or terminal(obs):
                return


def build(ctx):
    return Harness(ctx)

"""Fresh-context, response-checked Astra controller for Jericho stories."""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


FAIL = re.compile(r"not a verb|don't understand|do not understand|didn't understand|can't see|cannot see|not here|isn't here|aren't here|can't go|cannot go|not carrying|don't have|do not have|can't do|cannot do|nothing happens|doesn't seem to work|not something|is locked|is closed|which .*do you mean|what do you mean|be more specific|please .*answer|yes or no|\(y/n\)|\[y/n\]", re.I | re.S)
DANGER = re.compile(r"pitch[- ](?:black|dark)|too dark|grue|you are dead|you have died|you are dying|lamp.*(?:dim|flicker|out|running)|attacks you|strikes you|wounds you|stolen|steals|disarmed|rising water|game over|victory|congratulations|you win", re.I | re.S)
MENU = re.compile(r"q\s*=\s*resume game|return\s*=\s*read subject", re.I)
RESERVED = re.compile(r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I)


def _text(obs: Any) -> str:
    return str(obs.get("text", "")) if isinstance(obs, dict) else str(obs or "")


def _obs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            x = json.loads(value)
            if isinstance(x, dict):
                return x
        except Exception:
            pass
    return {"text": _text(value)}


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    s = re.sub(r"^```(?:json|text|command)?\s*|\s*```$", "", s, flags=re.I).strip()
    for _ in range(3):
        old = s
        s = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", s).strip()
        s = re.sub(r"^(?:command|action|input)\s*[:=]\s*", "", s, flags=re.I).strip()
        m = re.fullmatch(r"<(?:command|action)>(.*?)</(?:command|action)>", s, re.I | re.S)
        if m:
            s = m.group(1).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
            s = s[1:-1].strip()
        if s == old:
            break
    if not s or len(s) > 200 or any(c in s for c in "\r\n\x00;{}[]"):
        return None
    if RESERVED.match(s) or s.casefold() in {"command", "action", "none", "null"}:
        return None
    if re.search(r"\bthen\b|\.(?=\s*\w)", s, re.I):
        return None
    return s


def _walk(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 8 or len(out) >= 8:
        return
    if isinstance(value, str):
        c = _clean(value)
        if c:
            out.append(c)
    elif isinstance(value, list):
        for v in value[:8]:
            _walk(v, out, depth + 1)
            if len(out) >= 8:
                break
    elif isinstance(value, dict):
        for key in ("commands", "command", "actions", "action", "plan", "steps", "input", "value"):
            if key in value:
                if key == "action" and value[key] == "command":
                    continue
                _walk(value[key], out, depth + 1)
                if out:
                    return


def extract(raw: Any) -> list[str]:
    """Total parser for arbitrary stand-in/model replies."""
    if not isinstance(raw, str):
        return []
    out: list[str] = []
    s = raw.strip()
    for candidate in (s, re.sub(r"^```(?:json|text)?\s*|\s*```$", "", s, flags=re.I).strip()):
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
            except Exception:
                continue
            _walk(obj, out)
            if out:
                return out[:8]
    for line in raw.splitlines():
        if re.match(r"^\s*(?:command|action|input)\s*[:=]", line, re.I):
            c = _clean(line)
            if c:
                return [c]
    for line in raw.splitlines():
        line = line.strip().strip("`\"'")
        if line and len(line) <= 200 and not re.search(r"\b(?:reason|because|plan|observation)\s*:", line, re.I):
            c = _clean(line)
            if c:
                return [c]
    return []


def _terminal(obs: dict[str, Any]) -> bool:
    return any(bool(obs.get(k)) for k in ("terminated", "truncated", "won", "lost"))


def _brief(obs: dict[str, Any]) -> dict[str, Any]:
    return {k: obs.get(k) for k in ("game_id", "score", "max_score", "score_delta", "native_moves", "step_count", "terminated", "truncated", "won", "lost", "last_action", "text")}


POLICY = """You control a turn based offline Jericho parser story. Maximize FINAL native score.
Return JSON only: {\"commands\":[\"one parser command\", ...],\"goal\":\"short\"}.
Each command is one ordinary parser input, at most 200 characters. Give 2-6 commands only
when later commands are direct, clue-supported consequences; otherwise give one command.
The harness executes them one at a time and stops on parser failure, clarification, danger,
negative score, an interactive menu, or terminal text. Positive score may mean the suffix is
still useful. Use exact nouns and grammar from evidence. Preserve unique tools, light and
access prerequisites; inspect objects or read clues before guessing. Do not use save/restore/
restart/undo/quit or compound commands. Repeated movement with no new room or clue is a failed
hypothesis: change objective or inspect a salient object. Treat the notebook as hypotheses and
verify it against the exact transcript. Thinking costs no game turn.
"""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rows: list[dict[str, Any]] = []
        self.failed: list[str] = []
        self.reason = "Start from the opening description and choose a reachable scoring prerequisite."
        self.notebook = ""
        self.ctx_game = ""

    def _record(self, command: str | None, obs: dict[str, Any]) -> None:
        row = {"id": len(self.rows), "command": command, "text": _text(obs)[:5000], "score": obs.get("score"), "delta": obs.get("score_delta"), "moves": obs.get("native_moves"), "step": obs.get("step_count")}
        self.rows.append(row)
        self.ctx.memory["transcript"] = self.rows[-80:]

    def _prompt(self, obs: dict[str, Any], remaining: dict[str, Any]) -> str:
        packet = {
            "status": _brief(obs),
            "opening": self.rows[0].get("text", "")[:7000] if self.rows else "",
            "recent_exact_transcript": self.rows[-12:],
            "older_score_events": [r for r in self.rows if r.get("delta") not in (None, 0)][-8:],
            "failed_commands": self.failed[-10:],
            "why_replanning": self.reason,
            "working_notebook_unverified": self.notebook[:2500],
            "budget": {"system2_calls": remaining.get("system2_calls"), "wall_s": round(float(remaining.get("wall_s", 0)), 1)},
        }
        return POLICY + "\nGAME CONTRACT:\n" + self.ctx_game + "\nEVIDENCE PACKET:\n" + json.dumps(packet, ensure_ascii=False, default=str)

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            self.ctx_game = str(game.describe())[:7000] + "\n" + str(game.action_spec.describe())[:4000]
        except Exception:
            self.ctx_game = "One action is a parser command string."
        try:
            obs = _obs(ctx.observe())
        except Exception:
            obs = _obs(game.observe())
        self._record(None, obs)
        pending: list[str] = []
        last_pair: tuple[str, str, Any] | None = None
        while not ctx.done and not _terminal(obs):
            if not pending:
                rem = ctx.budget.remaining()
                if rem.get("system2_calls", 0) <= 0 or float(rem.get("wall_s", 0)) <= 3:
                    return
                try:
                    # Fresh session is intentional: the packet is compact and authoritative.
                    result = ctx.think(self._prompt(obs, rem), session_id=None, timeout_s=max(1.0, min(180.0, float(rem.get("wall_s", 180)) - 2.0)))
                    raw_text = getattr(result, "text", "")
                    pending = extract(raw_text)
                    self.notebook = ""
                    try:
                        parsed = json.loads(raw_text)
                        if isinstance(parsed, dict) and isinstance(parsed.get("goal"), str):
                            self.notebook = parsed["goal"][:1200]
                    except Exception:
                        pass
                except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError) as exc:
                    if ctx.done:
                        return
                    self.reason = "Model call failed; reconsider from unchanged evidence (" + type(exc).__name__ + ")."
                    pending = ["look"]
                except Exception as exc:
                    ctx.log("think_parse_error", error=type(exc).__name__)
                    pending = ["look"]
                if not pending:
                    pending = ["look"]
                ctx.log("plan", commands=pending[:8], fresh_session=True)

            command = _clean(pending.pop(0)) or "look"
            before = _text(obs)
            try:
                obs = _obs(ctx.act(command))
            except Exception as exc:
                ctx.log("act_error", command=command, error=type(exc).__name__)
                self.failed.append(command)
                pending.clear()
                self.reason = "Action was rejected; choose a different grammar or prerequisite."
                if ctx.done:
                    return
                continue
            self._record(command, obs)
            now_text = _text(obs)
            delta = obs.get("score_delta")
            pair = (command.casefold(), now_text, obs.get("score"))
            if pair == last_pair:
                self.failed.append(command)
                pending.clear()
                self.reason = "Identical command and response repeated; abandon that route."
            last_pair = pair
            if FAIL.search(now_text) or MENU.search(now_text) or DANGER.search(now_text) or (isinstance(delta, (int, float)) and delta < 0):
                self.failed.append(command)
                pending.clear()
                self.reason = "Latest response requires replan: parser/precondition, prompt, hazard, menu, or score loss."
            elif not now_text.strip():
                pending.clear()
                self.reason = "Empty parser response; inspect the current input mode."
            elif before == now_text and command.casefold() in {"look", "l", "inventory", "i"}:
                self.reason = "Observation did not change; select a new object or objective."
            if _terminal(obs) or ctx.done:
                return


def build(ctx):
    return Harness(ctx)

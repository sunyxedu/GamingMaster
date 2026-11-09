"""Astra planner for Jericho's single string parser channel.

The adaptor exposes one action name (``command``), but the action value itself
is the parser text.  This harness asks Astra for a short command plan, extracts
string commands from several safe response formats, and submits the strings
one at a time.  A parser failure or an unexpected score/state change discards
the cached suffix and causes a fresh plan.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


_ERROR_MARKERS = (
    "not a verb", "don't understand", "do not understand", "can't see",
    "cannot see", "which " , "what do you want", "i don't know", "unknown",
    "nothing happens", "you can't", "you cannot", "not here",
)


def _obs_text(obs: Any) -> str:
    try:
        return json.dumps(obs, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return str(obs)


def _valid_command(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.replace("\\n", " ").replace("\\r", " ").strip()
    value = re.sub(r"^```(?:text|json)?\s*|\s*```$", "", value, flags=re.I).strip()
    value = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", value).strip()
    # Common wrappers produced despite the exact-output instruction.
    m = re.match(r"(?is)^(?:command|action)\s*\(\s*[\"']?(.*?)\s*[\"']?\s*\)$", value)
    if m:
        value = m.group(1).strip()
    m = re.match(r"(?is)^(?:command|action)\s*[:=]\s*[\"']?(.*?)[\"']?$", value)
    if m:
        value = m.group(1).strip()
    m = re.match(r"(?is)^(?:command|action)\s+(.+)$", value)
    if m and not value.lower().startswith(("command command", "action action")):
        value = m.group(1).strip()
    value = value.strip(" \t\"'`[]{}")
    if not value or len(value) > 200 or "\n" in value or "\x00" in value:
        return None
    # Never submit an action schema name as the parser command.
    if value.lower() in {"command", "action", "none", "null", "unknown"}:
        return None
    return value


def _from_json(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            k = str(key).lower()
            if k in {"command", "action", "next_command", "parser_command"}:
                c = _valid_command(val)
                if c:
                    out.append(c)
            elif k in {"commands", "plan", "actions", "steps", "route"}:
                out.extend(_from_json(val))
            else:
                out.extend(_from_json(val))
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            c = _valid_command(val)
            if c:
                out.append(c)
            else:
                out.extend(_from_json(val))
    return out


def _extract_commands(text: str) -> list[str]:
    """Extract parser strings without ever raising on arbitrary model text."""
    if not isinstance(text, str):
        return []
    found: list[str] = []
    # Whole JSON and fenced JSON are the most useful plan formats.
    candidates = [text.strip()]
    for block in re.findall(r"```(?:json|text)?\s*(.*?)```", text, flags=re.I | re.S):
        candidates.append(block.strip())
    for raw in candidates:
        try:
            found.extend(_from_json(json.loads(raw)))
        except Exception:
            pass
    # Also recover a command field from prose containing a JSON fragment.
    for m in re.finditer(r"[\"'](?:command|action|next_command|parser_command)[\"']\s*:\s*[\"']([^\"']+)", text, flags=re.I):
        c = _valid_command(m.group(1))
        if c:
            found.append(c)

    # Plain one-command-per-line output, including numbered plans.
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        # Structured JSON has already been handled above; treating its raw
        # punctuation as parser text would create commands like
        # ``commands":["east"``.
        if any(ch in line for ch in "{}[]"):
            continue
        # Strip a short explanation prefix while preserving ordinary parser text.
        line = re.sub(r"(?i)^(?:next\s+)?(?:command|action)\s*[:=-]\s*", "", line).strip()
        line = re.sub(r"(?i)^(?:use|enter|type)\s+", "", line).strip()
        c = _valid_command(line)
        if c and not c.startswith(("The ", "I ", "Because ", "Here ", "Reply ")):
            found.append(c)
    # A bare answer may have no line break; prefer the complete text only when
    # it resembles a parser command rather than an explanation.
    if not found:
        c = _valid_command(text.strip())
        if c:
            found.append(c)
    # Preserve order while deduplicating accidental JSON + line duplicates.
    unique: list[str] = []
    seen: set[str] = set()
    for c in found:
        key = c.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:12]


def _looks_like_failure(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    text = str(obs.get("text", "")).lower()
    return any(marker in text for marker in _ERROR_MARKERS)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session: str | None = None
        self.transcript: list[dict[str, Any]] = []
        self.plan: list[str] = []
        self.last_score: Any = None
        self.stop = False

    def _prompt(self, game: Any, obs: Any) -> str:
        recent = self.transcript[-6:]
        history = "\n".join(
            f"{i + 1}. command={r['command']!r} response={r['response']}"
            for i, r in enumerate(recent)
        ) or "(no commands yet)"
        return (
            "You are the action planner for an offline Jericho interactive-fiction game. "
            "The action channel is unusual: submit the parser sentence itself, such as "
            "east, take carrot, put carrot in chest, or close chest. Do not submit the "
            "literal word command. Read the mission and latest response, then propose a "
            "short safe plan. Return JSON only in the form "
            '{"commands":["first parser command","second parser command"]}. '
            "Use at most 6 commands; each is one ordinary parser line with no newline. "
            "If the previous command failed, repair its wording or change the hypothesis. "
            "Do not include explanations.\n\n"
            f"Game description:\n{game.describe()}\n\n"
            f"Latest observation:\n{_obs_text(obs)}\n\n"
            f"Recent transcript:\n{history}\n"
        )

    def _ask(self, game: Any, obs: Any) -> list[str]:
        try:
            result = self.ctx.think(
                self._prompt(game, obs), session_id=self.session, timeout_s=240.0
            )
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            # Budget exhaustion and episode closure are terminal for this run;
            # remember them even if a pre-flight context leaves ctx.done false.
            self.stop = True
            return []
        except TimeoutError:
            # A transient/slow model failure can be retried on the next loop.
            return []
        except RuntimeError as exc:
            # Some contract shims expose BudgetExceeded as a RuntimeError with
            # only a ``kind`` field, so recognize that shape too.
            kind = str(getattr(exc, "kind", "")).lower()
            if kind in {"system2", "wall"} or "budget" in str(exc).lower():
                self.stop = True
            return []
        try:
            self.session = result.session_id
            return _extract_commands(result.text)
        except Exception:
            return []

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", None) is None:
            return
        try:
            obs = game.observe()
        except Exception:
            try:
                obs = ctx.observe()
            except Exception:
                return
        while not ctx.done and not getattr(game, "done", False):
            if not self.plan:
                self.plan = self._ask(game, obs)
                if self.stop:
                    break
                if not self.plan:
                    # A harmless observation command gives the next planner a
                    # fresh response and keeps garbage stand-in models safe.
                    self.plan = ["look"]
            command = _valid_command(self.plan.pop(0)) or "look"
            before = obs
            try:
                obs = ctx.act(command)
            except (EpisodeOver, BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                break
            self.transcript.append({"command": command, "response": _obs_text(obs)})
            score = obs.get("score") if isinstance(obs, dict) else None
            changed_score = self.last_score is not None and score != self.last_score
            self.last_score = score
            # A failed command, score change, terminal flag, or large observation
            # discontinuity can invalidate the remaining planned suffix.
            if _looks_like_failure(obs) or changed_score:
                self.plan.clear()
            if isinstance(obs, dict) and (
                obs.get("terminated") or obs.get("truncated") or obs.get("won") or obs.get("lost")
            ):
                break
            # Avoid blindly repeating a stale plan after a materially different
            # response even when it was not recognized as a parser error.
            if isinstance(before, dict) and isinstance(obs, dict):
                old_text = str(before.get("text", ""))
                new_text = str(obs.get("text", ""))
                if old_text and new_text and old_text == new_text and command.lower() in {"look", "l"}:
                    self.plan.clear()


def build(ctx):
    return Harness(ctx)

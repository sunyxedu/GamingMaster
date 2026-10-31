"""Astra controller for Jericho parser games.

The adaptor exposes one action named ``command``, whose *value* is the
parser line.  The generic baseline treated the action name as the value, so
this controller deliberately extracts the value from Astra's response.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


def _json_values(value: Any):
    """Yield strings that look like parser commands from nested JSON."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        # Prefer the fields used in the prompt, while accepting common model
        # spellings and nested tool-call shaped responses.
        for key in ("commands", "command", "actions", "action", "plan", "steps", "input", "arguments", "args", "value"):
            if key in value:
                # In tool-call-shaped output this is the channel name, not a
                # parser command.  The sibling ``command`` field carries the
                # actual value.
                if key == "action" and value[key] == "command":
                    continue
                yield from _json_values(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _json_values(item)


def _clean_command(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    # Remove one layer of presentation markup, numbering, and labels.
    s = re.sub(r"^```(?:text|json|command)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()
    s = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", s).strip()
    s = re.sub(r"^(?:command|action|input)\s*[:=]\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^<command>\s*(.*?)\s*</command>$", r"\1", s, flags=re.I).strip()
    s = re.sub(r"^(?:command|action)\s*\(\s*[\"']?(.*?)[\"']?\s*\)$", r"\1", s, flags=re.I).strip()
    s = re.sub(r"^(?:command|action)\s+", "", s, flags=re.I).strip()
    s = s.replace("%20", " ")
    s = re.sub(r"^(?:command|action)\s+", "", s, flags=re.I).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if len(s) > 200:
        s = s[:200].rstrip()
    if not s or "\n" in s or "\r" in s:
        return None
    return s


def extract_commands(text: Any) -> list[str]:
    """Parse strict JSON, tagged lines, or plain Astra output defensively."""
    if not isinstance(text, str):
        return []
    out: list[str] = []

    def add(v: Any) -> None:
        c = _clean_command(v)
        # Preserve intentional repetitions (for example repeated ``wait`` or
        # combat commands); the controller's state checks decide whether a
        # suffix is stale.
        if c:
            out.append(c)

    stripped = text.strip()
    for candidate in (stripped, re.sub(r"^```(?:json|text)?\s*|\s*```$", "", stripped, flags=re.I)):
        if not candidate:
            continue
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
            except Exception:
                continue
            for val in _json_values(obj):
                # A list item is one command; a scalar may contain a tagged
                # line, which is handled below if it is not a command itself.
                add(val)
            if out:
                return out[:6]

    # Explicit command lines are the safest recovery from prose.
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(?:command|action|input)\s*[:=]", line, flags=re.I):
            add(line)
    if out:
        return out[:6]

    # Finally use the first short, non-explanatory line.  This keeps garbage
    # stand-in model output harmless while allowing a bare "take lamp" reply.
    for line in text.splitlines():
        line = line.strip().strip('`"\'')
        if not line or line.startswith(("#", "//")):
            continue
        if len(line) <= 200 and not re.search(r"\b(?:reason|because|observation|plan)\b\s*:", line, re.I):
            add(line)
            if out:
                break
    return out[:6]


def _render(obs: Any, limit: int = 7000) -> str:
    try:
        s = json.dumps(obs, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        s = repr(obs)
    return s if len(s) <= limit else s[:limit] + " …"


def _text_of(obs: Any) -> str:
    if isinstance(obs, dict):
        return str(obs.get("text", ""))
    return str(obs)


def _needs_replan(obs: Any) -> bool:
    """Parser failures and prompts make any remaining batch stale."""
    t = _text_of(obs).lower()
    markers = (
        "not a verb", "don't understand", "do not understand", "which do you",
        "which one", "what do you mean", "can't see", "cannot see", "not here",
        "you can't", "you cannot", "not something", "doesn't", "does not",
        "none at all", "nothing happens", "no way", "bars your way",
        "game over", "you are dead",
        "victory", "congratulations", "you win",
    )
    return any(m in t for m in markers)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        session = None
        obs = game.observe()
        pending: list[str] = []
        history: list[dict[str, Any]] = []
        failed: list[str] = []
        # Seed the score baseline from the initial observation; otherwise a
        # zero-valued first response would incorrectly discard the whole plan.
        last_score = obs.get("score") if isinstance(obs, dict) else None

        while not ctx.done and not getattr(game, "done", False):
            # A plan is deliberately short: it amortizes Astra latency but is
            # discarded as soon as the parser reports a changed context.
            if not pending:
                recent = history[-8:]
                prompt = (
                    "You are the controller for an offline Jericho interactive-fiction story. "
                    "The world advances one turn per parser command. Solve for native score and "
                    "survival. The action value is the command text itself, not the action name.\n\n"
                    f"GAME DESCRIPTION:\n{game.describe()}\n\n"
                    f"ACTION SPECIFICATION:\n{game.action_spec.describe()}\n\n"
                    "Return strict JSON with a key `commands` containing 1-4 ordinary parser "
                    "commands. Each list item is exactly one command line, at most 200 characters; "
                    "do not include explanations. Choose the first command for the current state. "
                    "Later commands should be short, well-supported consequences of the first and "
                    "must be abandoned if the response is a parser error, clarification, hazard, "
                    "score change, or terminal state. Use wording from the story.\n\n"
                    f"CURRENT OBSERVATION:\n{_render(obs)}\n"
                    f"RECENT TRANSCRIPT:\n{_render(recent, 5000)}\n"
                    f"RECENT FAILED COMMANDS:\n{failed[-8:]}\n"
                    "Reason silently, then output only JSON."
                )
                try:
                    res = ctx.think(prompt, session_id=session, timeout_s=180.0)
                    session = getattr(res, "session_id", session)
                    pending = extract_commands(getattr(res, "text", ""))
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break
                except (TimeoutError, RuntimeError):
                    # A transient/slow call is a normal episode event.  A
                    # bounded fallback keeps the harness from spinning.
                    if ctx.done:
                        break
                    pending = ["look"]
                except Exception:
                    # Pre-flight stand-ins may return unusual result objects;
                    # never let response parsing become an episode error.
                    pending = ["look"]
                if not pending:
                    pending = ["look"]

            command = _clean_command(pending.pop(0)) or "look"
            try:
                obs = ctx.act(command)
            except Exception as exc:
                ctx.log("act_error", command=command, error=type(exc).__name__)
                failed.append(command)
                pending.clear()
                break

            score = obs.get("score") if isinstance(obs, dict) else None
            response = _text_of(obs)
            history.append({"command": command, "response": response, "score": score})
            if score is not None and score != last_score:
                pending.clear()
            last_score = score
            if _needs_replan(obs):
                failed.append(command)
                pending.clear()
            # Identical action/response pairs are a strong signal that the
            # current plan is stuck; force Astra to formulate a new test.
            if len(history) >= 2 and history[-1]["command"] == history[-2]["command"] and response == history[-2]["response"]:
                failed.append(command)
                pending.clear()
            if isinstance(obs, dict) and (obs.get("terminated") or obs.get("truncated")):
                break


def build(ctx):
    return Harness(ctx)

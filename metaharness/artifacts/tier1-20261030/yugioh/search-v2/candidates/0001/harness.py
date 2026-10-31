"""Prompt-aware Astra harness for the EDOPro Tier-1 benchmark.

The baseline parsed against action_spec.names (``choose``, ``select``, ...),
which discarded the prompt token and made every response stale.  This harness
derives the complete legal action from the current prompt, asks Astra for a
short tactical choice, and validates the answer against that prompt before
posting it.  A small deterministic controller handles singleton prompts,
garbled replies, and exhausted model budgets.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def _compact(obs: Any) -> str:
    """Keep the information Astra needs while avoiding huge repeated catalogues."""
    if not isinstance(obs, dict):
        return json.dumps(obs, default=str, separators=(",", ":"))[:10000]
    view = {}
    for k in ("round", "your_deck", "agent_seat", "turn_player", "turn", "phase",
              "players", "chain", "prompt", "last_action_error", "duel_result",
              "completed_duels"):
        if k in obs:
            view[k] = obs[k]
    # The last events explain what just changed; the card catalogue is static
    # and already described in the system prompt.
    if "recent_events" in obs:
        view["recent_events"] = obs["recent_events"][-14:]
    return json.dumps(view, default=str, separators=(",", ":"))[:14000]


def _prompt(obs: Any) -> dict[str, Any] | None:
    return obs.get("prompt") if isinstance(obs, dict) and isinstance(obs.get("prompt"), dict) else None


def _actions(obs: Any) -> list[str]:
    """Return exact protocol actions legal for this observation."""
    p = _prompt(obs)
    if not p:
        return []
    out = []
    for c in p.get("choices", []) or []:
        if isinstance(c, dict) and isinstance(c.get("action"), str):
            out.append(c["action"])
        elif isinstance(c, str):
            out.append(c)
    if out:
        return out
    token = str(p.get("token", ""))
    kind = str(p.get("kind", ""))
    cand = p.get("candidates", []) or []
    inds = []
    for i, c in enumerate(cand):
        if isinstance(c, dict) and isinstance(c.get("index"), int):
            inds.append(c["index"])
        else:
            inds.append(i)
    if kind.startswith("select"):
        # Satisfy the minimum count/weight with the cheapest legal prefix.
        minimum = int(p.get("minimum", p.get("min_count", 0)) or 0)
        maximum = int(p.get("maximum", p.get("max_count", len(inds))) or len(inds))
        weights = p.get("weights", []) or []
        min_weight = p.get("min_weight")
        if min_weight is not None or weights:
            total = 0
            chosen = []
            for pos, (i, c) in enumerate(zip(inds, cand)):
                chosen.append(i)
                if pos < len(weights):
                    total += int(weights[pos])
                else:
                    total += int(c.get("weight", 1)) if isinstance(c, dict) else 1
                if min_weight is not None and total >= int(min_weight):
                    break
                if len(chosen) >= maximum:
                    break
            inds = chosen
        else:
            inds = inds[:max(minimum, 0)]
        if len(inds) < minimum:
            inds = inds[:maximum]
        if not inds and minimum == 0:
            return [f"select {token} -"]
        return [f"select {token} {','.join(map(str, inds))}"] if inds else []
    if kind.startswith("order"):
        return [f"order {token} {','.join(map(str, inds))}"] if inds else [f"order {token} -"]
    # A prompt with no decoded choices cannot be safely answered.  Returning
    # noop here would spin forever on a turn game because noop leaves the
    # decision pending; let the episode end cleanly instead.
    return []


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().strip('`').strip()).lower()


def _parse_reply(text: str, legal: list[str], obs: Any) -> str | None:
    """Map arbitrary Astra text to one exact current legal action."""
    if not legal:
        return None
    raw = str(text or "")
    low = _norm(raw)
    # Exact action strings are preferred; this preserves round/prompt tokens.
    for a in sorted(legal, key=len, reverse=True):
        if _norm(a) in low:
            return a
    p = _prompt(obs) or {}
    token = re.escape(str(p.get("token", "")))
    # Recover command + arguments when Astra omitted punctuation or quotes.
    m = re.search(r"\b(choose|select|order)\s+" + token + r"(?:\s+([-\d, ]+))?", raw, re.I)
    if m:
        candidate = _norm(m.group(0))
        for a in legal:
            if _norm(a) == candidate:
                return a
        # For choose prompts, an index in the answer can identify the legal
        # choice even if spacing/extra prose differed.
        nums = re.findall(r"-?\d+", m.group(2) or "")
        if nums:
            idx = nums[0]
            for a in legal:
                if re.search(r"\s" + re.escape(idx) + r"\s*$", a):
                    return a
    # Common prose forms: "choice 3", "index 3", or a quoted label.
    nums = re.findall(r"(?:choice|index|option)\s*#?\s*(\d+)", raw, re.I)
    if nums:
        idx = nums[-1]
        for a in legal:
            if re.search(r"\s" + re.escape(idx) + r"\s*$", a):
                return a
    return None


def _fallback(obs: Any, legal: list[str]) -> str | None:
    """Cheap legal policy used for singleton/failed calls and as a safety net."""
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]
    p = _prompt(obs) or {}
    choices = p.get("choices", []) or []
    scored = []
    for pos, a in enumerate(legal):
        c = choices[pos] if pos < len(choices) and isinstance(choices[pos], dict) else {}
        label = str(c.get("label", "")).lower()
        card = c.get("card", {}) if isinstance(c, dict) else {}
        name = str(card.get("name", "")).lower() if isinstance(card, dict) else ""
        atk = int(card.get("attack", 0) or 0) if isinstance(card, dict) else 0
        score = 0
        if "attack" in label or "enter battle" in label:
            score += 80
        if "normal summon" in label:
            score += 50 + min(atk, 3000) // 100
        if "activate" in label:
            # Removal and defensive cards are useful when no better scripted
            # answer is available; leave optional chains to Astra when possible.
            if any(x in name for x in ("dark hole", "fissure", "compulsory", "mirror force", "trap hole")):
                score += 45
            else:
                score += 15
        if "set monster" in label:
            score += 25
        if "set spell" in label or "set trap" in label:
            score += 20
        if "end" in label or "cancel" in label or "decline" in label or "no" == label.strip() or a.strip().lower() == "noop":
            score -= 10
        scored.append((score, -pos, a))
    return max(scored)[2]


def _needs_model(obs: Any) -> bool:
    """Reserve Astra calls for decisions where card/turn context matters."""
    p = _prompt(obs) or {}
    kind = str(p.get("kind", "")).lower()
    # Zone and position placement are interchangeable bookkeeping choices in
    # this pool.  Main/battle/effect/target/chain decisions are strategic.
    if kind.startswith(("select_zone", "position", "zone")):
        return False
    if kind.startswith(("main", "battle", "chain", "activate", "effect", "select", "order", "tribute", "option", "yes", "no")):
        return True
    return len(p.get("choices", []) or []) > 2


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", None) is None:
            return
        session = None
        # Send the adaptor's authoritative rules/action description once.  A
        # session keeps it available to Astra while later requests stay small.
        try:
            rules = game.describe()
        except Exception:
            rules = ""
        introduced = False
        obs = game.observe()
        while not ctx.done:
            legal = _actions(obs)
            if not legal:
                break
            # Singleton prompts are deterministic and cost no model call.
            action = legal[0] if len(legal) == 1 else None
            if action is None and _needs_model(obs):
                prompt = (
                    "You are playing a turn-based Yu-Gi-Oh! duel. Choose exactly one "
                    "legal action from the CURRENT prompt. Think tactically: win or "
                    "survive first, then preserve useful cards and follow-up. Do not "
                    "invent actions or tokens. Reply with the exact action string only.\n"
                    + (("AUTHORITATIVE GAME DESCRIPTION:\n" + str(rules) + "\n") if not introduced else "") +
                    "CURRENT LEGAL ACTIONS:\n" + "\n".join(legal) +
                    "\nSTATE:\n" + _compact(obs)
                )
                try:
                    rem = ctx.budget.remaining()
                    # Keep enough time for retries and later duels.
                    timeout = max(5.0, min(90.0, float(rem.get("wall_s", 90.0)) - 1.0))
                    res = ctx.think(prompt, session_id=session, timeout_s=timeout)
                    session = getattr(res, "session_id", None) or session
                    introduced = True
                    action = _parse_reply(getattr(res, "text", ""), legal, obs)
                except (BudgetExceeded, ModelUnavailable):
                    action = None
                except (TimeoutError, RuntimeError):
                    action = None
                if action is None:
                    action = _fallback(obs, legal)
            elif action is None:
                action = _fallback(obs, legal)
            if action is None:
                break
            try:
                previous_token = (_prompt(obs) or {}).get("token")
                obs = ctx.act(action)
                # A malformed/stale action leaves the same prompt pending. Try
                # another legal option without burning another model call.
                for _ in range(2):
                    if ctx.done:
                        break
                    now_token = (_prompt(obs) or {}).get("token")
                    err = obs.get("last_action_error") if isinstance(obs, dict) else None
                    if now_token != previous_token or not err:
                        break
                    fresh = _actions(obs)
                    alt = _fallback(obs, [x for x in fresh if x != action])
                    if not alt:
                        break
                    action = alt
                    obs = ctx.act(action)
            except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                break


def build(ctx):
    return Harness(ctx)

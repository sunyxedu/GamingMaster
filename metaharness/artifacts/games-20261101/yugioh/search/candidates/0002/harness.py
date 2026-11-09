"""Tactical split controller for the fixed-pool Yu-Gi-Oh benchmark.

The native engine remains the rules authority.  We only automate transparent
main/battle command choices; effect timing, targets, and compound selections
are delegated to Astra, with every response checked against the live prompt.
"""
from __future__ import annotations

import json
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def _obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else {}
        except Exception:
            return {}
    return {}


def _fallback(obs):
    p = _obj(obs).get("prompt", {})
    choices = p.get("choices", [])
    if choices and isinstance(choices[0], dict):
        return choices[0].get("action", "noop")
    token = p.get("token", "")
    kind = p.get("kind", "")
    candidates = p.get("candidates", [])
    if "select" in str(kind).lower():
        limits = p.get("constraints", p)
        minimum = 0
        if isinstance(limits, dict):
            for key in ("min", "min_count", "minimum", "min_select", "min_tributes"):
                if key in limits:
                    try:
                        minimum = int(limits[key]); break
                    except (TypeError, ValueError):
                        pass
        ids = list(range(min(minimum, len(candidates))))
        return f"select {token} " + (",".join(map(str, ids)) if ids else "-")
    if "order" in str(kind).lower():
        return "order %s %s" % (token, ",".join(str(i) for i in range(len(candidates))))
    return "noop"


def _choice_action(c):
    return c.get("action") if isinstance(c, dict) else None


def _card_info(choice, obs):
    card = dict(choice.get("card", {}))
    if "attack" in card and "defense" in card:
        return card
    for player in _obj(obs).get("players", []):
        if player.get("role") != "you":
            continue
        for zone in ("hand", "monsters", "spells_traps"):
            entries = player.get(zone, [])
            if isinstance(entries, list):
                for known in entries:
                    if (isinstance(known, dict) and
                        ((card.get("slot") is not None and known.get("slot") == card.get("slot")) or
                         (card.get("code") is not None and known.get("code") == card.get("code")))):
                        return known
    return card


def _auto_action(obs):
    """Return a legal action string for obvious tactical command menus."""
    o = _obj(obs)
    p = o.get("prompt", {})
    choices = p.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return None
    labels = [(str(c.get("label", "")).lower(), c) for c in choices]
    kind = str(p.get("kind", "")).lower()
    # Main-phase command prompt: use the strongest ordinary body, set useful
    # interaction/Flip monsters, and stop rather than spending removal blind.
    if "main_command" in kind or (any("normal summon" in l for l, _ in labels)
                                   and any("end turn" in l for l, _ in labels)):
        # If a live spell/effect or a Tribute Summon is available, sequencing
        # and resource costs matter; leave the whole menu to Astra.
        if any("activate" in l or "tribute summon" in l for l, _ in labels):
            return None
        summons = [(int(_card_info(c, o).get("attack", 0)), c)
                   for l, c in labels if "normal summon" in l]
        if summons:
            # Keep Flip-control monsters face-down; otherwise summon the best
            # printed attacker available. Choices are current/legal by contract.
            flip_names = {"Man-Eater Bug", "Penguin Soldier", "Magician of Faith"}
            faceup = [x for x in summons if _card_info(x[1], o).get("name") not in flip_names]
            if faceup:
                return max(faceup, key=lambda x: x[0])[1].get("action")
        # Prefer setting a Flip monster in defense and set one backrow per
        # invocation; the engine offers a fresh prompt after each action.
        sets = [c for l, c in labels if "set monster" in l]
        flips = [c for c in sets if _card_info(c, o).get("name") in
                 {"Man-Eater Bug", "Penguin Soldier", "Magician of Faith"}]
        if flips:
            return flips[0].get("action")
        backrow = [c for l, c in labels if "set spell/trap" in l]
        if backrow:
            useful = {"Mirror Force": 9, "Trap Hole": 8,
                      "Compulsory Evacuation Device": 8, "Call of the Haunted": 7,
                      "Book of Moon": 7, "Rush Recklessly": 6,
                      "Mystical Space Typhoon": 5}
            eligible = [c for c in backrow if _card_info(c, o).get("name") in useful]
            if eligible:
                return max(eligible, key=lambda c: useful[_card_info(c, o).get("name")]).get("action")
        end = next((c for l, c in labels if "end turn" in l), None)
        if end:
            return end.get("action")
    # Battle menu: exact displayed attack candidates only. Prefer attacks that
    # destroy a weaker attack-position target; avoid attacks into stronger ATK.
    if "battle" in kind or any("attack" in l for l, _ in labels):
        attacks = [(l, c) for l, c in labels if "attack" in l]
        if attacks:
            monsters = {str(m.get("slot")): m for pl in o.get("players", [])
                        for m in pl.get("monsters", []) if isinstance(m, dict)}
            def battle_value(pair):
                label, c = pair
                card = c.get("card", {})
                atk = int(card.get("attack", 0))
                target = c.get("target", {})
                if isinstance(target, dict):
                    defn = int(target.get("defense", target.get("attack", 0)))
                    pos = str(target.get("position", "")).lower()
                    if "attack" in pos:
                        return (3 if atk > defn else (1 if atk == defn else -4), atk-defn)
                    return (2 if atk > defn else 0, 0)
                # Direct attacks are valuable. Fall back to the engine-provided
                # order when target identity is not represented in the choice.
                return (4 if "direct" in label else 1, atk)
            return max(attacks, key=battle_value)[1].get("action")
    return None


def _reply_action(text, obs):
    """Accept only a current legal choice or a well-formed current-token action."""
    if not isinstance(text, str):
        return None
    o = _obj(obs)
    p = o.get("prompt", {})
    token = str(p.get("token", ""))
    choices = p.get("choices", [])
    clean = text.strip().strip("` \"'")
    for c in choices:
        a = _choice_action(c)
        if a and (clean == a or a in clean.splitlines()):
            return a
    # A model may return only an action (or wrap it in a short explanation).
    m = re.search(rf"\b(choose|select|order)\s+{re.escape(token)}\s+([0-9,\s-]+)", clean)
    if not m:
        return None
    verb, tail = m.group(1), m.group(2).strip()
    if verb == "choose":
        candidate = f"choose {token} {tail}"
        return candidate if any(_choice_action(c) == candidate for c in choices) else None
    candidates = p.get("candidates", [])
    ids = [] if tail == "-" else [int(x) for x in tail.split(",") if x.strip().isdigit()]
    if any(i < 0 or i >= len(candidates) for i in ids) or len(set(ids)) != len(ids):
        return None
    if verb == "order" and (len(ids) != len(candidates) or set(ids) != set(range(len(candidates)))):
        return None
    return f"{verb} {token} {tail}"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None

    def _ask(self, game, obs, reason):
        ctx = self.ctx
        if ctx.system2 is None:
            return _fallback(obs)
        prompt = (f"{game.describe()}\nCurrent observation (JSON):\n"
                  f"{json.dumps(obs, ensure_ascii=False, separators=(',', ':'))}\n\n"
                  f"Reason for review: {reason}. Use only current legal prompt choices. "
                  "Return exactly one complete action string, copied from choices where available. "
                  "For select/order obey the displayed candidate constraints. Prioritize winning, "
                  "survival, efficient trades, and preserving follow-up.")
        try:
            res = ctx.think(prompt, session_id=self.session, timeout_s=180.0)
            self.session = res.session_id
            parsed = _reply_action(res.text, obs)
            if parsed:
                return parsed
            ctx.log("unparsed_model_action", reply=str(res.text)[:300], prompt_kind=_obj(obs).get("prompt", {}).get("kind"))
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError) as e:
            if ctx.done:
                return None
            ctx.log("model_call_recovery", error=type(e).__name__)
        return _fallback(obs)

    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        last_token = None
        stalled = 0
        while not ctx.done and not game.done:
            progress = ctx.progress()
            p = _obj(obs).get("prompt", {})
            token = p.get("token")
            auto = _auto_action(obs)
            # Replan if a prior action was rejected or the same request did not
            # advance. This guards against stale/incorrect cheap-policy effects.
            if progress.get("illegal_actions", 0) or token == last_token or stalled >= 1:
                action = self._ask(game, obs, f"progress check failed; progress={progress}")
            elif auto is not None:
                action = auto
            else:
                action = self._ask(game, obs, f"complex prompt kind={p.get('kind')} hint={p.get('hint', '')}")
            if action is None:
                break
            before_token = token
            try:
                new_obs = ctx.act(action)
            except (BudgetExceeded, ModelUnavailable):
                break
            except Exception as e:
                # Illegal/stale action leaves the request pending; send its
                # evidence to Astra on the next loop instead of propagating.
                ctx.log("action_rejected", action=action, error=type(e).__name__)
                new_obs = game.observe()
            if new_obs is None:
                break
            new_token = (_obj(new_obs).get("prompt") or {}).get("token")
            stalled = stalled + 1 if new_token == before_token else 0
            last_token = before_token if stalled else None
            obs = new_obs


def build(ctx):
    return Harness(ctx)

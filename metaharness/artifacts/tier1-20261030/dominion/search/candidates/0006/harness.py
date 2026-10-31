"""Verified local regime controller for Dominion.

The controller deliberately keeps ordinary turns out of the slow model loop.  It
classifies the public kingdom, executes a receding-horizon policy, and rechecks
the observation after every action.  Astra is reserved for an unfamiliar prompt
or repeated evidence that the local rules no longer fit.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _obj(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else {}
        except Exception:
            return {}
    return {}


def _num(x: Any, default: int = 0) -> int:
    try:
        return int(x or 0)
    except Exception:
        return default


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.regime = "money"
        self.failures = 0
        self.last_obs = None
        self.last_decision = None
        self.last_provinces = 8
        self.model_used = False
        self.plan_note = ""

    def _prompt(self, obs):
        p = obs.get("prompt")
        return p if isinstance(p, dict) else {}

    def _opts(self, obs):
        out = []
        for x in self._prompt(obs).get("options", []) or []:
            if isinstance(x, dict) and x.get("id") is not None:
                out.append((str(x["id"]), str(x.get("label", ""))))
        return out

    def _did(self, obs):
        p = self._prompt(obs)
        return str(obs.get("decision_id", p.get("decision_id", "")))

    def _bounds(self, obs):
        p = self._prompt(obs)
        lo = _num(p.get("min"), 0)
        hi = _num(p.get("max"), lo)
        return max(0, lo), max(lo, hi)

    def _choose(self, obs, ids):
        return "choose " + self._did(obs) + ((" " + " ".join(ids)) if ids else "")

    def _owned(self, obs):
        return _obj(obs.get("self")).get("owned", {}) or {}

    def _hand(self, obs):
        return _obj(obs.get("self")).get("hand", {}) or {}

    def _supply(self, obs):
        return obs.get("supply", {}) if isinstance(obs.get("supply"), dict) else {}

    def _count(self, obs, card):
        return _num(self._owned(obs).get(card), 0)

    def _classify(self, obs):
        kingdom = set(obs.get("kingdom", []) or [])
        if "Gardens" in kingdom and "Workshop" in kingdom:
            self.regime = "gardens"
        elif "Witch" in kingdom:
            self.regime = "witch"
        elif "Chapel" in kingdom:
            self.regime = "chapel"
        else:
            self.regime = "money"

    def _compact(self, obs, reason):
        owned = self._owned(obs)
        supply = self._supply(obs)
        return {
            "reason": reason, "regime": self.regime,
            "kingdom": obs.get("kingdom", []),
            "supply": {k: _obj(v).get("count", 0) for k, v in supply.items()},
            "owned": owned, "hand": self._hand(obs),
            "turns": obs.get("turns_taken", {}), "resources": obs.get("resources", {}),
            "derived": obs.get("derived", {}), "opponent": obs.get("opponent", {}),
        }

    def _ask_recovery(self, obs, reason):
        """A bounded advisory escape hatch; malformed output never affects legality."""
        if self.model_used or getattr(self.ctx, "done", False):
            return
        try:
            rem = self.ctx.budget.remaining()
            if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 125:
                return
            prompt = (
                "You are a recovery adviser for a Dominion controller. Use only this public JSON. "
                "Return one compact JSON object {mode:money|witch|chapel|gardens,note:string}; "
                "do not give actions or invent hidden cards. Current regime and state: "
                + json.dumps(self._compact(obs, reason), separators=(",", ":"))
            )
            self.model_used = True
            res = self.ctx.think(prompt, timeout_s=min(120.0, max(20.0, rem.get("wall_s", 120) - 3)))
            text = getattr(res, "text", "")
            parsed = None
            for m in re.finditer(r"\{", text if isinstance(text, str) else ""):
                try:
                    candidate, _ = json.JSONDecoder().raw_decode(text[m.start():])
                    if isinstance(candidate, dict):
                        parsed = candidate
                        break
                except Exception:
                    continue
            mode = str((parsed or {}).get("mode", "")).lower()
            if mode in {"money", "witch", "chapel", "gardens"}:
                self.regime = mode
                self.plan_note = str((parsed or {}).get("note", ""))[:160]
                try:
                    self.ctx.log("recovery_plan", mode=mode, reason=reason)
                except Exception:
                    pass
        except Exception:
            try:
                self.ctx.log("recovery_plan_failed", reason=reason)
            except Exception:
                pass

    def _cost(self, obs, card):
        return _num(_obj(self._supply(obs).get(card)).get("cost"), 99)

    def _available(self, obs, card):
        return _num(_obj(self._supply(obs).get(card)).get("count"), 0)

    def _total_cards(self, obs):
        return sum(_num(v) for v in self._owned(obs).values())

    def _vp_pressure(self, obs):
        d = _obj(obs.get("derived"))
        return _num(d.get("provinces_left"), 8)

    def _buy_score(self, card, obs, gain=False):
        if self._available(obs, card) <= 0:
            return -10000
        provinces = self._vp_pressure(obs)
        total = self._total_cards(obs)
        own = self._owned(obs)
        turn = _num(_obj(obs.get("turns_taken")).get("agent"), 0)
        if gain and self.regime == "gardens":
            # Workshop is a gain engine, not its own payload.  Early gains
            # should repair treasure/action density; only gain Gardens after
            # the deck has enough mass to make its variable VP real.
            if card == "Workshop" and own.get("Workshop", 0) >= 2:
                return -500
            if card == "Silver":
                return 520
            if card in {"Village", "Cellar"}:
                return 410 if own.get(card, 0) < 2 else 90
            if card == "Gardens":
                return 360 if total >= 25 and provinces <= 4 else -300
        # Ending cards have a hard priority once the race is visible.
        if card == "Province":
            return 1000
        if card == "Duchy":
            return 800 if provinces <= 3 else -40
        if card == "Estate":
            return 700 if provinces <= 1 else -50
        if card == "Curse":
            return -1000
        if self.regime == "gardens":
            # Build card mass, then deliberately contest Gardens.  Workshop
            # gains are free of Buy pressure and can empty the pile quickly.
            if card == "Gardens":
                # Gardens are only worthwhile once the deck is large enough;
                # before that, preserve the economy needed to race Provinces.
                return 620 if total >= 25 and provinces <= 4 else -20
            if card == "Workshop":
                return 500 if own.get("Workshop", 0) < 1 else -20
            if card in {"Village", "Festival", "Market", "Laboratory"}:
                return 250 if own.get(card, 0) < 2 else 80
            if card in {"Smithy", "Cellar"}:
                return 180 if own.get(card, 0) < 2 else 40
            if card == "Copper":
                return 20
        if self.regime == "witch":
            if card == "Witch":
                return 640 if own.get(card, 0) < 2 else 160
            if card in {"Chapel", "Sentry"}:
                return 560 if own.get(card, 0) < 1 else 100
            if card in {"Village", "Festival", "Laboratory", "Market"}:
                return 250 if own.get(card, 0) < 2 else 80
        if self.regime == "chapel":
            if card == "Chapel":
                return 600 if own.get(card, 0) < 1 else 50
            if card in {"Smithy", "Village", "Laboratory", "Market", "Festival"}:
                return 300 if own.get(card, 0) < 2 else 90
        # Money is deliberately conservative against BMU: Silver before
        # speculative terminals, Gold before green, and Province at $8.
        if card == "Gold":
            return 540
        if card == "Silver":
            return 340 if turn < 12 or sum(_num(own.get(x)) for x in ("Gold", "Silver")) < 6 else 230
        if card in {"Smithy", "Militia", "Witch"}:
            return 250 if own.get(card, 0) < 2 else 60
        if card in {"Market", "Laboratory", "Festival", "Village", "Merchant"}:
            return 190 if own.get(card, 0) < 1 else 50
        return 20

    def _trash(self, obs, options, hi):
        # Remove curses/Estates first. Copper is removed only when the current
        # hand still contains enough treasure to make a useful next buy.
        hand = self._hand(obs)
        coins = sum(_num(hand.get(c)) * v for c, v in (("Copper", 1), ("Silver", 2), ("Gold", 3)))
        provinces = self._vp_pressure(obs)
        rank = {"Curse": 0, "Estate": 1, "Copper": 2, "Gardens": 8, "Duchy": 9, "Province": 10}
        selected = []
        for oid, card in sorted(options, key=lambda z: rank.get(z[1], 5)):
            if card == "Curse" or (card == "Estate" and provinces > 2):
                selected.append(oid)
            elif card == "Copper" and provinces > 2 and coins - len(selected) >= 4:
                selected.append(oid)
            if len(selected) >= hi:
                break
        lo, _ = self._bounds(obs)
        for oid, _ in options:
            if len(selected) >= lo:
                break
            if oid not in selected:
                selected.append(oid)
        return selected[:hi]

    def _discard(self, obs, options, lo):
        value = {"Curse": -10, "Estate": -4, "Copper": 1, "Silver": 4, "Gold": 6,
                 "Province": 10, "Duchy": 8, "Gardens": 7}
        return [oid for oid, _ in sorted(options, key=lambda z: value.get(z[1], 3))[:lo]]

    def _action(self, obs, options):
        if not options:
            return []
        hand = self._hand(obs)
        actions = _num(_obj(obs.get("resources")).get("actions"), 1)
        provinces = self._vp_pressure(obs)
        def score(card):
            s = 0
            if card in {"Village", "Festival", "Market", "Laboratory", "Merchant", "Sentry", "Cellar"}:
                s += 90
            if card == "Workshop":
                s += 125 if self.regime == "gardens" and self._count(obs, "Workshop") < 1 and self._total_cards(obs) < 24 else 18
            if card == "Cellar" and self.regime == "gardens":
                s -= 65
            if card == "Chapel":
                junk = sum(_num(hand.get(c)) for c in ("Curse", "Estate"))
                s += 150 if junk else -20
            if card == "Witch":
                s += 140 if self.regime == "witch" and provinces > 1 else 55
            if card == "Militia":
                s += 70
            if card == "Smithy":
                s += 80
            if card in {"Village", "Festival"} and actions <= 1:
                s += 35
            # Do not spend the last action on a terminal when an action maker
            # is in hand and there is another terminal to feed.
            if actions <= 1 and card in {"Smithy", "Witch", "Militia", "Chapel", "Workshop", "Mine", "Remodel"}:
                if any(c in hand for c in ("Village", "Festival", "Market", "Laboratory", "Merchant")):
                    s -= 35
            return s
        best = max(options, key=lambda z: score(z[1]))
        return [best[0]] if score(best[1]) > 0 else []

    def _pick(self, obs):
        p = self._prompt(obs)
        options = self._opts(obs)
        if not p:
            return "wait"
        lo, hi = self._bounds(obs)
        kind = str(p.get("kind", "")).lower()
        if not options:
            return self._choose(obs, []) if lo == 0 else "wait"
        if kind in {"yes_no", "reaction"} or all(x[1].lower() in {"true", "false", "yes", "no"} for x in options):
            text = (str(p.get("instruction", "")) + " " + str(p.get("source", ""))).lower()
            want = "moat" in text or "reveal" in text or "reaction" in kind
            target = {"true", "yes"} if want else {"false", "no"}
            return self._choose(obs, [next((oid for oid, lab in options if lab.lower() in target), options[0][0])])
        if "play_action" in kind or kind in {"action", "play"}:
            return self._choose(obs, self._action(obs, options))
        if "trash" in kind:
            return self._choose(obs, self._trash(obs, options, min(hi, len(options))))
        if "discard" in kind:
            ids = self._discard(obs, options, min(lo, len(options))) if lo else []
            return self._choose(obs, ids)
        if "order" in kind or "topdeck" in kind or "sequence" in kind:
            # Keep the most valuable card on top; IDs are local to this prompt.
            value = {"Gold": 8, "Silver": 6, "Copper": 2, "Estate": 0, "Curse": -5}
            ids = [oid for oid, _ in sorted(options, key=lambda z: value.get(z[1], 3), reverse=True)]
            return self._choose(obs, ids[:hi]) if lo else self._choose(obs, ids[:hi])
        if kind in {"buy", "gain"}:
            source = str(p.get("source", ""))
            ranked = sorted(options, key=lambda z: self._buy_score(z[1], obs, source == "Workshop"), reverse=True)
            if lo == 0 and (not ranked or self._buy_score(ranked[0][1], obs) < 0):
                return self._choose(obs, [])
            ids = [ranked[0][0]] if ranked else []
            # Normally only one card is legal; satisfy a forced lower bound safely.
            while len(ids) < lo:
                ids.append(next((oid for oid, _ in options if oid not in ids), options[0][0]))
            return self._choose(obs, ids[:hi])
        if lo == 0:
            return self._choose(obs, [])
        return self._choose(obs, [oid for oid, _ in options[:min(lo, hi)]])

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "done", False):
            return
        try:
            obs = _obj(game.observe())
        except Exception:
            return
        self._classify(obs)
        while not getattr(ctx, "done", False) and not getattr(game, "done", False):
            obs = _obj(obs)
            provinces = self._vp_pressure(obs)
            if provinces != self.last_provinces:
                self.last_provinces = provinces
            kind = str(self._prompt(obs).get("kind", "")).lower()
            known = (not kind or any(x in kind for x in ("buy", "gain", "play", "trash", "discard", "reaction", "yes", "order", "topdeck", "sequence")))
            if not known:
                self._ask_recovery(obs, "unfamiliar prompt " + kind)
            action = self._pick(obs)
            before = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            did = self._did(obs)
            try:
                nxt = ctx.act(action)
                obs = _obj(nxt if nxt is not None else ctx.observe())
            except Exception as exc:
                if getattr(ctx, "done", False):
                    return
                self.failures += 1
                try:
                    ctx.log("local_reject", decision=did, action=action, error=type(exc).__name__)
                except Exception:
                    pass
                try:
                    obs = _obj(ctx.observe())
                except Exception:
                    return
            after = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            try:
                progress = ctx.progress()
            except Exception:
                progress = {}
            illegal = _num(progress.get("illegal_actions"), 0) if isinstance(progress, dict) else 0
            if illegal or (str(obs.get("decision_id", "")) == did and after == before):
                self.failures += 1
                if self.failures >= 2:
                    self._ask_recovery(obs, "rejected or unchanged action evidence")
                    self.failures = 0
            else:
                self.failures = 0


def build(ctx):
    return Harness(ctx)

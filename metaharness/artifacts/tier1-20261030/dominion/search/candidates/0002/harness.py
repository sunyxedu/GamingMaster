"""Regime-planning Dominion harness.

One short Astra call selects a broad regime.  The rest of the episode is a
validated local controller: it recomputes card occurrence IDs at every prompt,
plays a conservative action order, and buys from a receding-horizon priority
list.  Astra is called again only when the public Province race changes or the
local controller presents evidence that its assumptions stopped working.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def _dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else {}
        except Exception:
            return {}
    return {}


def _count(m: Any, name: str) -> int:
    return int((_dict(m).get(name, 0) or 0))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mode = "money"
        self.priority: list[str] = []
        self.targets: dict[str, int] = {}
        self.session = None
        self.reviews = 0
        self.next_review = 0
        self.last_race = None
        self.last_turn = 0
        self.bad: set[tuple[str, str]] = set()
        self.failures = 0

    def _prompt(self, obs: dict) -> dict:
        p = obs.get("prompt")
        return p if isinstance(p, dict) else {}

    def _options(self, obs: dict) -> list[tuple[str, str]]:
        out = []
        for x in self._prompt(obs).get("options", []) or []:
            if isinstance(x, dict) and x.get("id") is not None:
                out.append((str(x["id"]), str(x.get("label", ""))))
        return out

    def _did(self, obs: dict) -> str:
        p = self._prompt(obs)
        return str(obs.get("decision_id", p.get("decision_id", "")))

    def _bounds(self, obs: dict) -> tuple[int, int]:
        p = self._prompt(obs)
        try:
            lo = int(p.get("min", 0) or 0)
            hi = int(p.get("max", lo) if p.get("max") is not None else lo)
        except Exception:
            lo, hi = 0, 0
        return max(0, lo), max(0, hi)

    def _action(self, obs: dict, ids: list[str]) -> str:
        did = self._did(obs)
        return "choose " + did + (" " + " ".join(ids) if ids else "")

    def _safe_progress(self) -> dict:
        try:
            p = self.ctx.progress()
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}

    def _regime_prompt(self, obs: dict, reason: str) -> str:
        kingdom = obs.get("kingdom", [])
        supply = obs.get("supply", {}) or {}
        compact_supply = {k: (v.get("count", 0) if isinstance(v, dict) else v)
                          for k, v in supply.items() if k in kingdom or k in {"Province", "Duchy", "Gold", "Silver", "Curse"}}
        state = {
            "kingdom": kingdom,
            "supply": compact_supply,
            "turn": (obs.get("turns_taken") or {}).get("agent", 0),
            "resources": obs.get("resources", {}),
            "owned": (obs.get("self") or {}).get("owned", {}),
            "provinces_left": (obs.get("derived") or {}).get("provinces_left"),
            "opponent": obs.get("opponent", {}),
            "reason": reason,
        }
        return (
            "You are a Dominion strategic adviser. Against BMU-1, maximize final VP margin. "
            "Choose one regime and a short acquisition priority. Use only the supplied state. "
            "Return ONLY JSON, under 100 tokens: {\"mode\":\"money|engine|witch|gardens\","
            "\"priority\":[card names],\"targets\":{card:number},\"note\":\"brief\"}. "
            "Money is the default; engine requires both action capacity and draw; Witch is "
            "valuable when the pile and cycling support it; Gardens requires fast Workshop/gains. "
            "Province race always overrides building.\nSTATE=" +
            json.dumps(state, separators=(",", ":"))
        )

    def _review(self, obs: dict, reason: str) -> None:
        if self.reviews >= 5 or self.ctx.done:
            return
        try:
            rem = self.ctx.budget.remaining()
            if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 140:
                return
        except Exception:
            return
        self.reviews += 1
        try:
            timeout = min(145.0, max(25.0, float(rem.get("wall_s", 145)) - 3.0))
            result = self.ctx.think(self._regime_prompt(obs, reason),
                                    session_id=self.session, timeout_s=timeout)
            self.session = getattr(result, "session_id", self.session)
            text = getattr(result, "text", "")
            parsed = None
            if isinstance(text, str):
                for m in re.finditer(r"\{", text):
                    try:
                        value, _ = json.JSONDecoder().raw_decode(text[m.start():])
                        if isinstance(value, dict):
                            parsed = value
                            break
                    except Exception:
                        continue
            if not parsed:
                return
            kingdom = set(obs.get("kingdom", []) or [])
            mode = str(parsed.get("mode", "money")).lower()
            if mode not in {"money", "engine", "witch", "gardens"}:
                mode = "money"
            # Do not accept a regime that names no card present on this board.
            rawp = parsed.get("priority", [])
            if isinstance(rawp, str):
                rawp = re.split(r"[,>\n]", rawp)
            priority = []
            if isinstance(rawp, list):
                for c in rawp:
                    hit = next((k for k in kingdom if str(k).lower() == str(c).strip().lower()), None)
                    if hit and hit not in priority:
                        priority.append(hit)
            targets = {}
            if isinstance(parsed.get("targets"), dict):
                for k, v in parsed["targets"].items():
                    if k in kingdom and isinstance(v, (int, float)) and not isinstance(v, bool):
                        targets[k] = max(0, min(6, int(v)))
            if priority or mode == "money":
                self.mode, self.priority, self.targets = mode, priority[:8], targets
                self.ctx.log("regime", mode=mode, priority=self.priority, reason=reason)
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError) as exc:
            try:
                self.ctx.log("regime_failed", error=type(exc).__name__, reason=reason)
            except Exception:
                pass

    def _cards(self, obs: dict) -> dict:
        return (_dict(obs.get("self"))).get("owned", {}) or {}

    def _default_regime(self, obs: dict) -> None:
        kingdom = set(obs.get("kingdom", []) or [])
        # This fallback is deliberately broad and scenario-independent.
        if "Witch" in kingdom and ("Chapel" in kingdom or "Sentry" in kingdom):
            self.mode = "witch"
            self.priority = [x for x in ("Witch", "Chapel", "Sentry", "Village", "Smithy") if x in kingdom]
        elif "Gardens" in kingdom and "Workshop" in kingdom:
            self.mode = "gardens"
            self.priority = [x for x in ("Workshop", "Gardens", "Cellar", "Village", "Smithy") if x in kingdom]
        elif "Chapel" in kingdom and ("Village" in kingdom or "Laboratory" in kingdom):
            self.mode = "engine"
            self.priority = [x for x in ("Chapel", "Village", "Smithy", "Laboratory", "Market", "Festival") if x in kingdom]
        else:
            self.mode = "money"
            self.priority = [x for x in ("Smithy", "Militia", "Merchant", "Silver") if x in kingdom]

    def _rank_buy(self, card: str, obs: dict) -> float:
        owned = self._cards(obs)
        kingdom = set(obs.get("kingdom", []) or [])
        supply = obs.get("supply", {}) or {}
        p = int((_dict(obs.get("derived")).get("provinces_left", 8) or 8))
        total = sum(int(v or 0) for v in owned.values())
        turn = int((_dict(obs.get("turns_taken")).get("agent", 0) or 0))
        available = int((_dict(supply.get(card)).get("count", 0) or 0))
        if available <= 0:
            return -1000
        if card == "Province":
            return 1000
        if card == "Duchy":
            return 420 if p <= 3 else -20
        if card == "Estate":
            return 300 if p <= 1 else -20
        if card == "Curse":
            return -1000
        if card == "Gold":
            return 190
        if card == "Silver":
            return 120 if turn < 10 else 100
        if card == "Gardens":
            return 260 if self.mode == "gardens" and (p <= 3 or total >= 15) else -15
        target = self.targets.get(card, 0)
        if _count(owned, card) >= target and target > 0:
            return 8
        base = {"Witch": 185, "Chapel": 180, "Sentry": 155, "Smithy": 145,
                "Village": 135, "Festival": 130, "Laboratory": 135,
                "Market": 120, "Merchant": 105, "Militia": 115,
                "Workshop": 120, "Cellar": 55, "Mine": 65, "Remodel": 60,
                "Moat": 35}.get(card, 20)
        if self.mode == "money" and card in {"Village", "Festival", "Laboratory", "Market"}:
            base -= 55
        if self.mode == "engine" and card in {"Village", "Festival", "Laboratory", "Market", "Smithy", "Chapel"}:
            base += 35
        if self.mode == "witch" and card in {"Witch", "Chapel", "Sentry"}:
            base += 55
        if self.mode == "gardens" and card in {"Workshop", "Gardens", "Cellar"}:
            base += 65
        if card in self.priority:
            base += max(5, 28 - 3 * self.priority.index(card))
        # Keep a small number of terminal actions; buying them forever loses the race.
        terminals = sum(_count(owned, x) for x in ("Smithy", "Witch", "Militia", "Chapel", "Remodel", "Mine"))
        if card in {"Smithy", "Witch", "Militia", "Chapel", "Remodel", "Mine"} and terminals >= max(2, total // 9):
            base -= 45
        return base

    def _choose_action(self, obs: dict, options: list[tuple[str, str]]) -> str:
        owned = self._cards(obs)
        actions = int((_dict(obs.get("resources")).get("actions", 1) or 0))
        hand = (_dict(obs.get("self"))).get("hand", {}) or {}
        junk = _count(hand, "Curse") + _count(hand, "Estate") + max(0, _count(hand, "Copper") - 2)
        def value(pair):
            card = pair[1]
            v = 10.0
            if card in {"Village", "Festival", "Market", "Laboratory", "Merchant", "Sentry", "Cellar"}:
                v += 80 if actions <= 1 else 110
            if card == "Chapel":
                v += 100 if junk else -30
            if card == "Witch":
                v += 84 if self.mode in {"witch", "engine"} else 50
            if card == "Militia":
                v += 60
            if card == "Smithy":
                v += 62
            if card in {"Mine", "Remodel", "Workshop"}:
                v += 20
            if card in {"Village", "Festival"} and actions > 1:
                v += 30
            if card in self.priority:
                v += 10
            return v
        if not options:
            # An empty Action option list still has a legal pass response.
            return self._action(obs, [])
        best = max(options, key=value)
        lo, _ = self._bounds(obs)
        if lo == 0 and value(best) < 20:
            return self._action(obs, [])
        return self._action(obs, [best[0]])

    def _card_value(self, card: str, obs: dict) -> float:
        values = {"Curse": -100, "Estate": -35, "Gardens": -30,
                  "Copper": 5, "Silver": 25, "Gold": 45,
                  "Chapel": 30, "Village": 35, "Festival": 35,
                  "Smithy": 32, "Witch": 34, "Militia": 30,
                  "Laboratory": 36, "Market": 34, "Merchant": 28,
                  "Sentry": 34, "Workshop": 20, "Cellar": 15,
                  "Mine": 18, "Remodel": 18, "Moat": 18}
        return values.get(card, 10)

    def _choose_cards(self, obs: dict, options: list[tuple[str, str]], kind: str) -> str:
        lo, hi = self._bounds(obs)
        if kind == "trash":
            # Curses and Estates are always dead; remove Copper only while the
            # hand still has at least four dollars of future treasure.
            ranked = sorted(options, key=lambda x: self._card_value(x[1], obs))
            selected = []
            for oid, card in ranked:
                if card in {"Curse", "Estate"} or (card == "Copper" and len(selected) < hi - 1):
                    selected.append(oid)
                if len(selected) >= hi:
                    break
        elif kind == "discard":
            ranked = sorted(options, key=lambda x: self._card_value(x[1], obs))
            selected = [oid for oid, _ in ranked[:lo]]
        else:  # topdeck/order or an unknown ordered card prompt
            ranked = sorted(options, key=lambda x: self._card_value(x[1], obs), reverse=True)
            selected = [oid for oid, _ in ranked[:max(lo, 1)]]
        # Forced selections must always satisfy the exact lower bound.
        for oid, _ in options:
            if len(selected) >= lo:
                break
            if oid not in selected:
                selected.append(oid)
        return self._action(obs, selected[:hi])

    def _choose_buy(self, obs: dict, options: list[tuple[str, str]]) -> str:
        lo, hi = self._bounds(obs)
        if not options:
            return self._action(obs, []) if lo == 0 else "wait"
        ranked = sorted(options, key=lambda x: self._rank_buy(x[1], obs), reverse=True)
        if lo == 0 and self._rank_buy(ranked[0][1], obs) < 0:
            return self._action(obs, [])
        chosen = [ranked[0][0]] if ranked and self._rank_buy(ranked[0][1], obs) >= 0 else []
        while len(chosen) < lo:
            chosen.append(next((oid for oid, _ in options if oid not in chosen), options[0][0]))
        return self._action(obs, chosen[:hi])

    def _choose(self, obs: dict) -> str:
        p = self._prompt(obs)
        options = self._options(obs)
        if not options and not p:
            return "wait"
        kind = str(p.get("kind", "")).lower()
        lo, hi = self._bounds(obs)
        if kind == "buy" or kind == "gain":
            return self._choose_buy(obs, options)
        if "play_action" in kind or kind == "action":
            return self._choose_action(obs, options)
        if "trash" in kind:
            return self._choose_cards(obs, options, "trash")
        if "discard" in kind:
            return self._choose_cards(obs, options, "discard")
        if any(x in kind for x in ("topdeck", "order", "sequence")):
            return self._choose_cards(obs, options, "order")
        labels = {label.lower(): oid for oid, label in options}
        if labels and all(x in {"true", "false", "yes", "no"} for x in labels):
            instruction = str(p.get("instruction", "")).lower()
            source = str(p.get("source", "")).lower()
            want = "moat" in source or "reaction" in kind or "reveal" in instruction
            target = ("true", "yes") if want else ("false", "no")
            oid = next((labels[x] for x in target if x in labels), options[0][0])
            return self._action(obs, [oid])
        if lo == 0:
            return self._action(obs, [])
        return self._action(obs, [x[0] for x in options[:min(lo, hi)]])

    def run_episode(self, game) -> None:
        ctx = self.ctx
        try:
            obs = _dict(game.observe())
        except Exception:
            return
        self._default_regime(obs)
        # A single regime call supplies strategic context without consuming the
        # dozens of calls that the naive controller spent on routine prompts.
        self._review(obs, "opening board")
        while not ctx.done and not getattr(game, "done", False):
            obs = _dict(obs)
            turn = int((_dict(obs.get("turns_taken")).get("agent", 0) or 0))
            prov = int((_dict(obs.get("derived")).get("provinces_left", 8) or 8))
            race = 0 if prov > 5 else (1 if prov > 3 else (2 if prov > 1 else 3))
            if self.last_race is not None and race != self.last_race:
                self._review(obs, "Province race changed")
            elif turn >= self.next_review and turn >= self.last_turn + 6:
                self._review(obs, "periodic public-state review")
            self.last_race, self.last_turn, self.next_review = race, turn, turn + 6
            action = self._choose(obs)
            key = (self._did(obs), action)
            if key in self.bad:
                # Rebuild a legal alternative from the current prompt.
                p = self._prompt(obs)
                opts = self._options(obs)
                lo, hi = self._bounds(obs)
                if lo == 0:
                    action = self._action(obs, [])
                elif opts:
                    action = self._action(obs, [x[0] for x in opts[:min(lo, hi)]])
            before = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            did = self._did(obs)
            progress_before = self._safe_progress()
            try:
                nxt = ctx.act(action)
                obs = _dict(nxt if nxt is not None else ctx.observe())
            except (BudgetExceeded, ModelUnavailable, TimeoutError):
                return
            except Exception as exc:
                self.bad.add((did, action))
                self.failures += 1
                if ctx.done:
                    return
                try:
                    obs = _dict(ctx.observe())
                except Exception:
                    return
                try:
                    ctx.log("local_action_rejected", action=action, error=type(exc).__name__)
                except Exception:
                    pass
            progress = self._safe_progress()
            after = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            try:
                old_illegal = int(progress_before.get("illegal_actions", 0) or 0)
                new_illegal = int(progress.get("illegal_actions", old_illegal) or 0)
            except Exception:
                old_illegal = new_illegal = 0
            if new_illegal > old_illegal or (str(obs.get("decision_id")) == did and after == before):
                self.bad.add((did, action))
                self.failures += 1
                if self.failures >= 2:
                    self._review(obs, "rejected or unchanged action evidence")
                    self.failures = 0
            else:
                self.failures = 0


def build(ctx):
    return Harness(ctx)

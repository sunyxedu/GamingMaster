"""Opponent-aware Dominion controller.

The strategic unit is a regime (money, engine, witch, or gardens), rather than
an action suffix.  All card occurrences and bounds are rebuilt from the current
prompt so attacks and nested effects cannot invalidate a cached response.
"""
from __future__ import annotations

import json
import re


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


def _count(x, key):
    try:
        return int((_obj(x).get(key, 0) or 0))
    except Exception:
        return 0


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mode = "money"
        self.priority = []
        self.targets = {}
        self.review_count = 0
        self.last_review_turn = -1
        self.last_provinces = None
        self.last_illegal = 0
        self.stall = 0
        self.bad = set()
        self.seen_attack = False

    def _prompt(self, obs):
        return _obj(obs).get("prompt") or {}

    def _opts(self, obs):
        out = []
        for x in self._prompt(obs).get("options", []) or []:
            if isinstance(x, dict) and x.get("id") is not None:
                out.append((str(x["id"]), str(x.get("label", ""))))
        return out

    def _did(self, obs):
        return str(_obj(obs).get("decision_id", self._prompt(obs).get("decision_id", "")))

    def _bounds(self, obs):
        p = self._prompt(obs)
        try:
            lo = max(0, int(p.get("min", 0) or 0))
        except Exception:
            lo = 0
        try:
            hi = int(p.get("max", lo) if p.get("max") is not None else lo)
        except Exception:
            hi = lo
        return lo, max(lo, min(hi, len(self._opts(obs))))

    def _cmd(self, obs, ids=()):
        return "choose " + self._did(obs) + ((" " + " ".join(ids)) if ids else "")

    def _progress(self):
        try:
            p = self.ctx.progress()
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}

    def _default_regime(self, obs):
        k = set(_obj(obs).get("kingdom", []) or [])
        if "Gardens" in k and "Workshop" in k:
            self.mode = "gardens"
            self.priority = [x for x in ("Workshop", "Gardens", "Cellar", "Village", "Smithy") if x in k]
        elif "Witch" in k:
            self.mode = "witch"
            self.priority = [x for x in ("Witch", "Chapel", "Sentry", "Village", "Smithy") if x in k]
        elif "Chapel" in k and any(x in k for x in ("Village", "Festival", "Laboratory", "Market")):
            self.mode = "engine"
            self.priority = [x for x in ("Chapel", "Village", "Smithy", "Laboratory", "Market", "Festival") if x in k]
        else:
            self.mode = "money"
            self.priority = [x for x in ("Smithy", "Militia", "Merchant", "Silver") if x in k]

    def _review(self, obs, reason):
        """Ask Astra for a board regime; arbitrary or late output is harmless."""
        if self.review_count >= 3 or getattr(self.ctx, "done", False):
            return
        try:
            rem = self.ctx.budget.remaining()
            if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 150:
                return
        except Exception:
            return
        view = {
            "kingdom": _obj(obs).get("kingdom", []),
            "supply": _obj(obs).get("supply", {}),
            "self": _obj(obs).get("self", {}),
            "opponent": _obj(obs).get("opponent", {}),
            "resources": _obj(obs).get("resources", {}),
            "derived": _obj(obs).get("derived", {}),
            "turns": _obj(obs).get("turns_taken", {}),
            "events": (_obj(obs).get("public_log_delta", []) or [])[-10:],
            "reason": reason,
            "progress": self._progress(),
        }
        prompt = (
            "You are the strategic planner for two-player Dominion versus BMU-1. "
            "Return ONLY compact JSON with mode (money, engine, witch, gardens), "
            "priority (exact card names, at most 6), and targets (total owned counts). "
            "Witch pressure should favor Witch plus Chapel/Sentry and a fast Province race; "
            "Gardens is viable only with Workshop and rapid pile pressure. Preserve payload "
            "and switch to green cards when Provinces are scarce. Never return action IDs.\n" +
            json.dumps(view, separators=(",", ":"))
        )
        self.review_count += 1
        try:
            result = self.ctx.think(prompt, session_id=None, timeout_s=130.0)
            text = getattr(result, "text", "") or ""
            data = None
            for m in re.finditer(r"\{", text):
                try:
                    q, _ = json.JSONDecoder().raw_decode(text[m.start():])
                    if isinstance(q, dict):
                        data = q
                        break
                except Exception:
                    continue
            if not data:
                return
            kingdom = set(_obj(obs).get("kingdom", []) or [])
            mode = str(data.get("mode", self.mode)).lower()
            if mode not in {"money", "engine", "witch", "gardens"}:
                mode = self.mode
            raw = data.get("priority", [])
            if isinstance(raw, str):
                raw = re.split(r"[,>\n]", raw)
            pri = []
            for item in raw if isinstance(raw, list) else []:
                hit = next((c for c in kingdom if c.lower() == str(item).strip().lower()), None)
                if hit and hit not in pri:
                    pri.append(hit)
            targets = {}
            if isinstance(data.get("targets"), dict):
                for card, value in data["targets"].items():
                    if card in kingdom and isinstance(value, (int, float)) and not isinstance(value, bool):
                        targets[card] = max(0, min(8, int(value)))
            self.mode, self.priority, self.targets = mode, pri[:6], targets
            try:
                self.ctx.log("regime_review", mode=mode, reason=reason)
            except Exception:
                pass
        except Exception as exc:
            try:
                self.ctx.log("regime_review_failed", error=type(exc).__name__, reason=reason)
            except Exception:
                pass

    def _buy_score(self, card, obs):
        state = _obj(obs)
        own = (state.get("self", {}) or {}).get("owned", {}) or {}
        supply = state.get("supply", {}) or {}
        total = sum(int(v or 0) for v in own.values())
        prov = int((state.get("derived", {}) or {}).get("provinces_left", 8) or 8)
        turn = int((state.get("turns_taken", {}) or {}).get("agent", 0) or 0)
        if _count(supply.get(card), "count") <= 0:
            return -10000
        if card == "Province":
            return 2000
        if card == "Duchy":
            return 900 if prov <= 3 else -80
        if card == "Estate":
            return 700 if prov <= 1 else -120
        if card == "Curse":
            return -10000
        if card == "Gold":
            return 600 if prov > 3 else 480
        if card == "Silver":
            return 320 if turn < 12 else 250
        if card == "Gardens":
            return 850 if self.mode == "gardens" and total >= 12 else -100
        if self.mode == "gardens":
            base = {"Workshop": 520, "Cellar": 125, "Village": 120, "Smithy": 105, "Market": 100}.get(card, 20)
        elif self.mode == "witch":
            base = {"Witch": 520, "Chapel": 430, "Sentry": 350, "Village": 190, "Smithy": 150,
                    "Laboratory": 130, "Market": 120, "Militia": 115, "Moat": 80}.get(card, 20)
        elif self.mode == "engine":
            base = {"Chapel": 380, "Village": 300, "Smithy": 270, "Laboratory": 240,
                    "Market": 220, "Festival": 210, "Sentry": 180}.get(card, 30)
        else:
            base = {"Smithy": 180, "Militia": 165, "Merchant": 145, "Market": 125,
                    "Witch": 120, "Chapel": 100, "Village": 95}.get(card, 25)
        if card in self.priority:
            base += 35 - 5 * self.priority.index(card)
        if card in self.targets and _count(own, card) >= self.targets[card]:
            base -= 160
        terminals = sum(_count(own, x) for x in ("Smithy", "Witch", "Militia", "Chapel", "Remodel", "Mine"))
        if card in ("Smithy", "Witch", "Militia", "Chapel", "Remodel", "Mine") and terminals >= max(3, total // 9):
            base -= 100
        if prov <= 3 and card not in ("Province", "Duchy", "Estate", "Gold", "Silver", "Gardens"):
            base -= 120
        return base

    def _trash_ids(self, obs, opts, hi):
        state = _obj(obs)
        prov = int((state.get("derived", {}) or {}).get("provinces_left", 8) or 8)
        # Keep two Copper-equivalents while building; remove Curses and Estates first.
        copper = sum(1 for _, c in opts if c == "Copper")
        chosen = []
        rank = {"Curse": 0, "Estate": 1 if prov > 2 else 5, "Copper": 2}
        for oid, card in sorted(opts, key=lambda z: rank.get(z[1], 9)):
            if len(chosen) >= hi:
                break
            if card in ("Curse", "Estate") or (card == "Copper" and prov > 2 and copper > 2):
                chosen.append(oid)
                if card == "Copper":
                    copper -= 1
        return chosen

    def _keep_value(self, card):
        return {"Curse": -20, "Estate": -8, "Copper": 0, "Silver": 5, "Gold": 9,
                "Province": 12, "Duchy": 10, "Gardens": 8}.get(card, 3)

    def _pick(self, obs):
        p, opts = self._prompt(obs), self._opts(obs)
        if not opts:
            return "pass"
        lo, hi = self._bounds(obs)
        kind = str(p.get("kind", "")).lower()
        source = str(p.get("source", ""))
        state = _obj(obs)
        hand = (state.get("self", {}) or {}).get("hand", {}) or {}
        own = (state.get("self", {}) or {}).get("owned", {}) or {}
        labels = {x.lower() for _, x in opts}
        if labels <= {"true", "false", "yes", "no"}:
            attack = "reaction" in kind or "moat" in source.lower() or "reveal" in str(p.get("instruction", "")).lower()
            want = attack and _count(hand, "Moat") > 0
            yes = {"true", "yes"} if want else {"false", "no"}
            return self._cmd(obs, [next((i for i, l in opts if l.lower() in yes), opts[0][0])])
        if "trash" in kind:
            if source == "Mine":
                ranked = sorted(opts, key=lambda z: {"Copper": 0, "Silver": 1, "Gold": 9}.get(z[1], 3))
                ids = [ranked[0][0]] if ranked and ranked[0][1] != "Gold" else []
            elif source == "Remodel":
                ids = [max(opts, key=lambda z: {"Curse": 90, "Estate": 80, "Copper": 65, "Silver": 20}.get(z[1], 0))[0]]
            else:
                ids = self._trash_ids(obs, opts, hi)
            while len(ids) < lo:
                ids += [i for i, _ in opts if i not in ids][:lo - len(ids)]
            return self._cmd(obs, ids[:hi])
        if "discard" in kind:
            ids = [i for i, c in sorted(opts, key=lambda z: self._keep_value(z[1]))[:max(lo, hi)]]
            while len(ids) < lo:
                ids += [i for i, _ in opts if i not in ids][:lo - len(ids)]
            return self._cmd(obs, ids[:hi])
        if "order" in kind or "topdeck" in kind:
            ids = [i for i, _ in sorted(opts, key=lambda z: self._keep_value(z[1]))[:max(lo, 1)]]
            return self._cmd(obs, ids[:hi])
        if kind in ("buy", "gain") or "buy" in kind or "gain" in kind:
            if source == "Mine":
                return self._cmd(obs, [max(opts, key=lambda z: {"Gold": 3, "Silver": 2, "Copper": 1}.get(z[1], 0))[0]])
            best = max(opts, key=lambda z: self._buy_score(z[1], obs))
            if lo == 0 and self._buy_score(best[1], obs) <= 0:
                return self._cmd(obs)
            return self._cmd(obs, [best[0]])
        if "action" in kind:
            actions = int((state.get("resources", {}) or {}).get("actions", 1) or 0)
            def score(z):
                card = z[1]
                v = {"Village": 240, "Festival": 230, "Market": 220, "Laboratory": 215,
                     "Merchant": 190, "Cellar": 175, "Sentry": 180}.get(card, 100)
                if card == "Chapel": v += 130 if _count(hand, "Curse") + _count(hand, "Estate") else -70
                if card == "Witch": v += 120 if self.mode == "witch" else 40
                if card == "Militia": v += 75
                if card == "Smithy": v += 65
                if card in ("Village", "Festival") and actions <= 1: v += 45
                if card in self.priority: v += 25
                return v
            return self._cmd(obs, [max(opts, key=score)[0]])
        if lo == 0:
            return self._cmd(obs)
        return self._cmd(obs, [i for i, _ in opts[:lo]])

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "done", False):
            return
        try:
            obs = _obj(game.observe())
        except Exception:
            return
        self._default_regime(obs)
        self._review(obs, "opening board")
        while not getattr(ctx, "done", False) and not getattr(game, "done", False):
            state = _obj(obs)
            prov = int((state.get("derived", {}) or {}).get("provinces_left", 8) or 8)
            turn = int((state.get("turns_taken", {}) or {}).get("agent", 0) or 0)
            events = str(state.get("public_log_delta", []))
            if "Witch" in events or "Curse" in events:
                self.seen_attack = True
                if self.mode == "money" and "Witch" in set(state.get("kingdom", []) or []):
                    self.mode = "witch"
                    self.priority = [x for x in ("Witch", "Chapel", "Sentry", "Village", "Smithy") if x in set(state.get("kingdom", []) or [])]
                    self._review(obs, "public Witch/Curse pressure")
            if self.last_provinces is not None and prov != self.last_provinces and prov <= 3:
                self._review(obs, "Province race changed")
            self.last_provinces = prov
            if turn >= self.last_review_turn + 8 and self.review_count < 2:
                self._review(obs, "periodic deck review")
                self.last_review_turn = turn
            action = self._pick(obs)
            key = (self._did(obs), action)
            if key in self.bad:
                opts = self._opts(obs); lo, _ = self._bounds(obs)
                action = self._cmd(obs, [i for i, _ in opts[:lo]]) if lo else self._cmd(obs)
            before = json.dumps(state, sort_keys=True, separators=(",", ":"))
            before_id = self._did(obs)
            try:
                nxt = ctx.act(action)
            except Exception as exc:
                if getattr(ctx, "done", False):
                    return
                self.bad.add(key)
                self.stall += 1
                try:
                    ctx.log("action_rejected", action=action, error=type(exc).__name__)
                    nxt = ctx.observe()
                except Exception:
                    return
            obs = _obj(nxt if nxt is not None else ctx.observe())
            prog = self._progress()
            after = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            illegal = int(prog.get("illegal_actions", 0) or 0)
            if illegal > self.last_illegal or (self._did(obs) == before_id and after == before):
                self.bad.add(key)
                self.stall += 1
            else:
                self.stall = 0
            self.last_illegal = illegal
            if self.stall >= 2:
                self._review(obs, "local policy stalled; evidence=" + json.dumps(prog, separators=(",", ":")))
                self.stall = 0


def build(ctx):
    return Harness(ctx)

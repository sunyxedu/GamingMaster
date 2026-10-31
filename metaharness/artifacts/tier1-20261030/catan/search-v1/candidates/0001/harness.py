"""Catan harness: legal-ID control with a small production-aware policy.

The engine's action IDs are revision-local.  Astra is used only to resolve
strategic choices among the IDs currently offered; forced actions and routine
build/trade choices are handled in code so a malformed or stale model answer
cannot strand the game.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
TERRAIN_WEIGHT = {"WHEAT": 1.35, "ORE": 1.30, "WOOD": 1.12, "BRICK": 1.10, "SHEEP": 0.98}
RESOURCE_COSTS = {
    "build_road": {"WOOD": 1, "BRICK": 1},
    "build_settlement": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "build_city": {"ORE": 3, "WHEAT": 2},
    "buy_development_card": {"ORE": 1, "SHEEP": 1, "WHEAT": 1},
}


def _actions(obs):
    xs = obs.get("legal_actions", []) if isinstance(obs, dict) else []
    return [x for x in xs if isinstance(x, dict) and isinstance(x.get("id"), str)]


def _board_features(obs):
    """Return hexes, vertices and a production feature for each vertex."""
    board = (obs or {}).get("board") or {}
    hexes = board.get("hexes") or []
    by_id = {h.get("id"): h for h in hexes if isinstance(h, dict)}
    vertices = board.get("vertices") or []
    # Build from the explicit adjacent_hexes field when present; this also
    # handles coastal vertices that touch only one or two land hexes.
    vf = {}
    for v in vertices:
        if not isinstance(v, dict):
            continue
        vid = v.get("id")
        hs = v.get("adjacent_hexes")
        if hs is None:
            hs = [h.get("id") for h in hexes if vid in (h.get("vertices") or [])]
        vals = []
        resources = set()
        pips = 0
        for hid in hs or []:
            h = by_id.get(hid, {})
            terrain = h.get("terrain")
            number = h.get("number")
            if terrain in TERRAIN_WEIGHT and number in PIPS:
                p = PIPS[number]
                if not h.get("robber"):
                    pips += p
                resources.add(terrain)
                vals.append((p, terrain, bool(h.get("robber"))))
        vf[vid] = {
            "pips": pips,
            "diversity": len(resources),
            "resources": resources,
            "port": v.get("port"),
            "building": v.get("building"),
            "owner": v.get("owner"),
            "hexes": hs or [],
            "vals": vals,
        }
    return by_id, vf, board


def _vertex_score(feat, opening=False):
    s = float(feat.get("pips", 0)) + 0.65 * float(feat.get("diversity", 0))
    for p, terrain, blocked in feat.get("vals", []):
        s += 0.09 * p * TERRAIN_WEIGHT.get(terrain, 1.0)
        if blocked:
            s -= 0.4
    # Grain/ore support the city/development path; wood/brick preserve an
    # expansion route.  This is a tie-breaker, never a legality assumption.
    if opening:
        s += 0.35 * sum(TERRAIN_WEIGHT.get(r, 0) for r in feat.get("resources", ()))
    if feat.get("port"):
        s += 0.25
    return s


def _param_vertex(a):
    p = a.get("parameters")
    return p if isinstance(p, int) else None


def _resource_counts(obs):
    r = ((obs or {}).get("self") or {}).get("resources") or {}
    return {str(k): int(v or 0) for k, v in r.items()}


def _can_pay(resources, typ):
    cost = RESOURCE_COSTS.get(typ)
    return bool(cost) and all(resources.get(k, 0) >= n for k, n in cost.items())


def _effective_resources(resources, params):
    """Apply a maritime trade parameter tuple, if it is well formed."""
    r = dict(resources)
    if not isinstance(params, list) or len(params) < 2:
        return None
    give = params[:-1]
    get = params[-1]
    if not isinstance(get, str) or not all(isinstance(x, str) for x in give):
        return None
    for x in give:
        r[x] = r.get(x, 0) - 1
    if any(v < 0 for v in r.values()):
        return None
    r[get] = r.get(get, 0) + 1
    return r


def _opening_choice(obs, typ):
    _, vf, _ = _board_features(obs)
    candidates = []
    for a in _actions(obs):
        if a.get("type") != typ:
            continue
        if typ == "build_initial_settlement" or typ == "build_settlement":
            v = _param_vertex(a)
            candidates.append((_vertex_score(vf.get(v, {}), opening=True), a))
        else:
            p = a.get("parameters")
            if isinstance(p, list) and len(p) >= 2:
                ends = [vf.get(x, {}) for x in p[:2]]
                score = max((_vertex_score(x, opening=True) for x in ends), default=0)
                candidates.append((score, a))
            else:
                candidates.append((0, a))
    if not candidates:
        return None
    return max(candidates, key=lambda z: (z[0], z[1].get("id", "")))[1].get("id")


def _robber_choice(obs):
    by_id, vf, board = _board_features(obs)
    opponents = {x.get("id"): x for x in ((obs or {}).get("opponents") or []) if isinstance(x, dict)}
    best = None
    for a in _actions(obs):
        if a.get("type") not in ("move_robber", "play_knight_card"):
            continue
        p = a.get("parameters")
        coord = p[0] if isinstance(p, list) and p else None
        victim = p[1] if isinstance(p, list) and len(p) > 1 else None
        h = next((x for x in (board.get("hexes") or []) if x.get("coordinate") == coord), None)
        if h is None:
            # Some builds expose a hex id instead of a coordinate.
            h = by_id.get(coord, {})
        pip = PIPS.get((h or {}).get("number"), 0)
        hid = (h or {}).get("id")
        exposed = 0
        for v in (board.get("vertices") or []):
            if hid in (v.get("adjacent_hexes") or []) and v.get("owner") in opponents:
                exposed += 1
        threat = float((opponents.get(victim) or {}).get("public_vp", 0))
        hand = float((opponents.get(victim) or {}).get("resource_count", 0))
        score = 2.0 * pip * exposed + 0.8 * threat + 0.05 * hand
        # Never prefer moving to the current robber hex when alternatives exist.
        if (h or {}).get("robber"):
            score -= 100
        key = (score, a.get("id", ""))
        if best is None or key > best[0]:
            best = (key, a.get("id"))
    return best[1] if best else None


def _local_choice(obs):
    """Choose an action guaranteed to be one of this observation's IDs."""
    acts = _actions(obs)
    if not acts:
        return None
    phase = str((obs or {}).get("phase", ""))
    # Setup choices have no resource information and deserve production-aware
    # ranking.  The actual engine type is build_initial_* in this adaptor.
    if phase == "build_initial_settlement":
        x = _opening_choice(obs, "build_settlement")
        if x:
            return x
    if phase == "build_initial_road":
        x = _opening_choice(obs, "build_road")
        if x:
            return x
    if phase in ("discard", "discard_resources"):
        for a in acts:
            if a.get("type") == "discard":
                return a["id"]
    if phase == "move_robber":
        x = _robber_choice(obs)
        if x:
            return x
    # Any exact singleton is a forced engine transition (roll, discard, or
    # a sequential Road Building choice).
    if len(acts) == 1:
        return acts[0]["id"]

    resources = _resource_counts(obs)
    _, vf, _ = _board_features(obs)
    # Immediate build ordering: VP city/settlement first, then a road that can
    # reach a productive vertex.  The legal list already enforces piece counts
    # and costs, but checking resources makes this robust to adaptor variants.
    for typ in ("build_city", "build_settlement"):
        cand = [a for a in acts if a.get("type") == typ and _can_pay(resources, typ)]
        if cand:
            if typ == "build_city":
                return max(cand, key=lambda a: _vertex_score(vf.get(_param_vertex(a), {}), False))["id"]
            return max(cand, key=lambda a: _vertex_score(vf.get(_param_vertex(a), {}), False))["id"]
    roads = [a for a in acts if a.get("type") == "build_road" and _can_pay(resources, "build_road")]
    if roads:
        def rs(a):
            p = a.get("parameters") or []
            return max((_vertex_score(vf.get(x, {}), False) for x in p[:2]), default=0)
        return max(roads, key=lambda a: (rs(a), a["id"]))["id"]
    # A trade is useful only if it creates a buildable hand.  This avoids
    # burning four cards merely because maritime_trade appears in legal list.
    for a in acts:
        if a.get("type") != "maritime_trade":
            continue
        r2 = _effective_resources(resources, a.get("parameters"))
        if r2 and any(_can_pay(r2, t) for t in ("build_city", "build_settlement", "build_road", "buy_development_card")):
            return a["id"]
    # Buy a development card after expansion options; it is a useful fallback
    # in ore/wheat/sheep positions and legal only when affordable.
    for a in acts:
        if a.get("type") == "buy_development_card" and _can_pay(resources, "buy_development_card"):
            return a["id"]
    # A playable old development card is generally preferable to ending a turn.
    for typ in ("play_year_of_plenty", "play_monopoly", "play_road_building", "play_knight_card"):
        for a in acts:
            if a.get("type") == typ:
                return a["id"]
    # Keep the turn moving.  Prefer end_turn over a malformed/unknown action.
    for a in acts:
        if a.get("type") == "end_turn":
            return a["id"]
    return acts[0]["id"]


def _compact_prompt(obs, description, acts, plan):
    by_id, vf, board = _board_features(obs)
    hsummary = []
    for h in (board.get("hexes") or []):
        if h.get("terrain") in TERRAIN_WEIGHT:
            hsummary.append([h.get("id"), h.get("terrain"), h.get("number"), bool(h.get("robber"))])
    vsummary = []
    for vid, f in vf.items():
        if f.get("building") or f.get("pips", 0) >= 9:
            vsummary.append([vid, round(_vertex_score(f, False), 2), f.get("building"), f.get("owner"), sorted(f.get("resources", ()))])
    view = {
        "phase": obs.get("phase"), "turn": obs.get("turn_number"),
        "self": obs.get("self"), "opponents": obs.get("opponents"),
        "awards": obs.get("awards"), "hexes": hsummary, "key_vertices": vsummary,
        "legal_actions": [{"id": a.get("id"), "type": a.get("type"), "parameters": a.get("parameters")} for a in acts],
        "plan": plan,
    }
    return ("You are the strategic Catan controller. Use only this supplied state. "
            "Choose exactly one ID from legal_actions, copying its id literally (for example a7). "
            "IDs are revision-local; never invent coordinates or action names. Prefer an immediate legal VP build, "
            "then a resource-producing expansion, then development/ trade. Return one token `aN` and at most one "
            "short sentence.\n\nRules/context:\n" + description + "\n\nSTATE:\n" +
            json.dumps(view, separators=(",", ":"), default=str) + "\nPersistent plan: " + str(plan))


def _model_id(text, valid):
    if not isinstance(text, str):
        return None
    valid = set(valid)
    # Prefer an explicit ACTION= marker if the model ignored the one-token rule.
    m = re.search(r"(?:ACTION|CHOICE)\s*[:=]\s*(a\d+)\b", text, re.I)
    if m and m.group(1) in valid:
        return m.group(1)
    for m in re.finditer(r"\b(a\d+)\b", text, re.I):
        if m.group(1) in valid:
            return m.group(1)
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = "Prioritize productive settlement/city; preserve a route and block the leader."
        self.calls = 0
        self.last_model_turn = None
        self.last_fingerprint = None

    def _ask(self, obs, game, acts):
        if self.ctx.system2 is None or self.calls >= 16 or self.ctx.done:
            return None
        rem = self.ctx.budget.remaining()
        if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 110:
            return None
        prompt = _compact_prompt(obs, game.describe(), acts, self.plan)
        self.calls += 1
        try:
            res = self.ctx.think(prompt, session_id=None, timeout_s=min(105.0, max(5.0, rem.get("wall_s", 105.0) - 5.0)))
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            return None
        chosen = _model_id(getattr(res, "text", ""), [a["id"] for a in acts])
        # Keep only a tiny, bounded memory note; never allow the model's prose
        # to become an unbounded conversation that increases latency.
        txt = getattr(res, "text", "") or ""
        if len(txt) > 240:
            txt = txt[:240]
        if txt:
            self.plan = txt
        return chosen

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if ctx.system2 is None:
            return
        obs = game.observe()
        while not ctx.done and not (isinstance(obs, dict) and obs.get("terminal")):
            acts = _actions(obs)
            if not acts:
                break
            local = _local_choice(obs)
            phase = str(obs.get("phase", ""))
            turn = obs.get("turn_number")
            meaningful = {a.get("type") for a in acts} - {"roll", "end_turn", "discard"}
            # One Astra call at the first meaningful decision of each turn and
            # during setup; code handles every forced/remaining action.
            ask = False
            if phase in ("build_initial_settlement", "build_initial_road"):
                ask = len(acts) > 1
            elif phase == "play_turn" and meaningful and turn != self.last_model_turn:
                ask = len(acts) > 1
            elif phase in ("move_robber", "play_development_card") and len(acts) > 1:
                ask = True
            chosen = self._ask(obs, game, acts) if ask else None
            if chosen:
                # Do not let a strategic answer throw away a guaranteed legal
                # build/trade when it says end_turn.
                types = {a.get("type") for a in acts}
                if chosen not in {a["id"] for a in acts}:
                    chosen = None
                elif chosen and next((a.get("type") for a in acts if a["id"] == chosen), None) == "end_turn" and (types & {"build_city", "build_settlement", "build_road", "buy_development_card", "maritime_trade"}):
                    chosen = local
            action = chosen or local
            if not action:
                break
            if ask:
                self.last_model_turn = turn
            before_count = obs.get("action_count") if isinstance(obs, dict) else None
            try:
                new_obs = ctx.act(action)
            except Exception:
                # An adaptor may report a rejected action rather than raise;
                # if it does raise, immediately retry a different current ID.
                alternatives = [a["id"] for a in acts if a["id"] != action]
                if not alternatives:
                    break
                try:
                    new_obs = ctx.act(alternatives[0])
                except Exception:
                    break
            # Catanatron's adaptor can return a new observation after a
            # rejected/stale action without raising.  An unchanged action
            # counter and legal-ID set is a concrete signal that the plan did
            # not advance; immediately try a different current ID once.
            if isinstance(obs, dict) and isinstance(new_obs, dict):
                same_count = before_count is not None and new_obs.get("action_count") == before_count
                old_ids = tuple(a.get("id") for a in acts)
                new_ids = tuple(a.get("id") for a in _actions(new_obs))
                if same_count and old_ids == new_ids and len(acts) > 1:
                    alt = next((a.get("id") for a in acts if a.get("id") != action), None)
                    if alt:
                        try:
                            new_obs = ctx.act(alt)
                        except Exception:
                            pass
            obs = new_obs


def build(ctx):
    return Harness(ctx)

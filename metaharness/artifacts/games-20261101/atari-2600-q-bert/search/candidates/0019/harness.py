"""0019: landing-ahead advice plus a bounded rolling-threat shadow.

A one-hop motor stays deterministic and neutralized.  The main structural
change is that Jev is asked about the *next junction while a hop is in flight*,
so a normal ~0.2 s answer can be applied on arrival instead of being stale at
the current node.  A localized memory of rendered rolling balls augments the
parent corridor score without treating purple_ball_state as an enemy position.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice


DIRS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE = ("DOWN", "RIGHT", "UP", "LEFT")
ORDER = (1, 4, 7, 11, 6, 10, 16, 12, 17, 18, 13, 9, 14, 19, 20,
         8, 2, 0, 3, 5, 15)
GRAPH = {n: {} for n in range(21)}
ROW = {}
for r in range(6):
    for c in range(r + 1):
        n = r * (r + 1) // 2 + c
        ROW[n] = r
        if r:
            if c < r:
                GRAPH[n]["UP"] = n - r
            if c:
                GRAPH[n]["LEFT"] = n - r - 1
        if r < 5:
            first = (r + 1) * (r + 2) // 2
            GRAPH[n]["DOWN"] = first + c
            GRAPH[n]["RIGHT"] = first + c + 1


def make_distances():
    all_dist = []
    for root in range(21):
        d = [99] * 21
        d[root] = 0
        todo = deque([root])
        while todo:
            here = todo.popleft()
            for there in GRAPH[here].values():
                if d[there] == 99:
                    d[there] = d[here] + 1
                    todo.append(there)
        all_dist.append(d)
    return all_dist


DIST = make_distances()


def num(x, default=0.0):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(x, default=0):
    if isinstance(x, bool):
        return default
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError):
        return int(num(x, default))


def node(x):
    if isinstance(x, bool) or not isinstance(x, int):
        return None
    return x if 0 <= x < 21 else None


def tick_of(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)))


def motion_of(obs):
    m = obs.get("qbert_motion")
    return m if isinstance(m, dict) else {"phase": str(m or "unknown")}


def is_standing(obs):
    p = node(obs.get("qbert_node"))
    m = motion_of(obs)
    phase = str(m.get("phase", "unknown")).lower()
    return (p is not None and not obs.get("qbert_in_hop")
            and phase not in ("hopping", "falling", "not_visible")
            and not m.get("falling"))


def colors_of(obs):
    v = obs.get("cube_colors")
    if isinstance(v, list) and len(v) == 21 and all(
            isinstance(c, int) and not isinstance(c, bool) for c in v):
        return tuple(v)
    return None


def safe_json(x):
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (list, tuple)):
        return [safe_json(v) for v in x]
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    return str(x)


def parse_json_object(text):
    if not isinstance(text, str):
        return {}
    dec = json.JSONDecoder()
    for i, ch in enumerate(text[:12000]):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except (TypeError, ValueError):
            continue
    return {}


def answer_value(answer):
    v = getattr(answer, "value", None) if answer is not None else None
    return v if isinstance(v, str) else None


def option_probability(answer, key):
    p = getattr(answer, "probabilities", None) if answer is not None else None
    return num(p.get(key), 0.0) if isinstance(p, dict) else 0.0


class PaletteLedger:
    """Infer board epochs from local scoring changes and global flash tallies."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.prev = None
        self.score = None
        self.goal = None
        self.need = [1] * 21
        self.flash = False
        self.flash_last = -999
        self.flash_score = 0.0
        self.flash_origin = None
        self.epoch = 0
        self.clears = 0
        self.all_done_since = None
        self.last_progress = 0
        self.evidence = deque(maxlen=8)

    def update(self, obs):
        now = tick_of(obs)
        current = colors_of(obs)
        score = num(obs.get("score"), self.score if self.score is not None else 0.0)
        delta = score - self.score if self.score is not None else 0.0
        old_score = self.score if self.score is not None else score
        changed = ([i for i in range(21)
                    if self.prev is not None and current is not None
                    and current[i] != self.prev[i]])

        if self.prev is not None and len(changed) >= 16:
            if not self.flash:
                self.flash = True
                self.flash_score = old_score
                self.flash_origin = node(obs.get("qbert_node"))
                self.ctx.log("board_flash_started", tick=now, score=score,
                             reported_round=obs.get("round"))
            self.flash_last = now

        if self.flash:
            quiet = now - self.flash_last
            returned_top = node(obs.get("qbert_node")) == 0 and self.flash_origin != 0
            if quiet >= 4 and (returned_top or quiet >= 7):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.flash = False
                self.goal = None
                self.need = [1] * 21
                self.all_done_since = None
                self.last_progress = now
                self.evidence.clear()
                self.ctx.log("board_epoch", tick=now, epoch=self.epoch,
                             inferred_clears=self.clears, tally=tally)
                self.prev, self.score = current, score
                return True
            self.prev, self.score = current, score
            self.need = [1] * 21
            return False

        if current is None:
            raw = obs.get("needed_landings")
            if isinstance(raw, list) and len(raw) == 21:
                self.need = [max(0, integer(v, 1)) for v in raw]
            self.prev, self.score = current, score
            return False

        # A 325-point observation can be a 300-point Sam catch plus a cube.
        # The modulo test avoids interpreting unrelated large events as a cube.
        if delta > 0 and int(round(delta)) % 100 == 25 and 0 < len(changed) <= 3:
            q = node(obs.get("qbert_node"))
            landed = q if q in changed else (changed[0] if len(changed) == 1 else None)
            if landed is not None:
                self.goal = current[landed]
                self.last_progress = now
                self.evidence.append({"node": landed, "color": current[landed],
                                      "reward": delta,
                                      "phase": (obs.get("cube_phase") or [None] * 21)[landed]})
                self.ctx.log("destination_palette_learned", tick=now, epoch=self.epoch,
                             node=landed, color=current[landed], reward=delta)

        if self.goal is None:
            raw = obs.get("needed_landings")
            if isinstance(raw, list) and len(raw) == 21 and any(integer(v, 0) > 0 for v in raw):
                self.need = [max(0, integer(v, 1)) for v in raw]
            else:
                self.need = [1] * 21
        else:
            self.need = [0 if c == self.goal else 1 for c in current]

        if not any(self.need):
            if self.all_done_since is None:
                self.all_done_since = now
            elif now - self.all_done_since >= 24:
                # Later cycles can require a return pass; this is a probe, not
                # a claim that an intermediate colour is the final destination.
                self.goal = None
                self.need = [1] * 21
                self.all_done_since = None
                self.last_progress = now
                self.ctx.log("colour_cycle_probe", tick=now, epoch=self.epoch)
        else:
            self.all_done_since = None

        self.prev, self.score = current, score
        return False


class ThreatMemory:
    """Visible threats plus a short spatial cone from rendered rolling balls."""

    def __init__(self):
        self.visible = {}
        self.rolling = deque(maxlen=5)
        self.latest_rolling = {}
        self._late_mode = False

    @staticmethod
    def down_frontier(nodes):
        out = set()
        for n in nodes:
            for a in ("DOWN", "RIGHT"):
                if a in GRAPH[n]:
                    out.add(GRAPH[n][a])
        return out

    def clear(self):
        self.visible.clear()
        self.rolling.clear()
        self.latest_rolling.clear()

    def update(self, obs, now, late=False):
        self._late_mode = bool(late)
        live = {}
        hazards = obs.get("hazards")
        if not isinstance(hazards, list):
            hazards = []
        for h in hazards:
            if not isinstance(h, dict) or not h.get("lethal"):
                continue
            kind = str(h.get("kind", "hazard"))
            if kind not in ("purple_ball", "red_ball", "coily"):
                continue
            n = node(h.get("estimated_node", h.get("node")))
            if n is None:
                continue
            item = {"kind": kind, "node": n, "tick": now,
                    "dx": num(h.get("dx")), "dy": num(h.get("dy")),
                    "visible": True}
            live[kind] = item
            if kind in ("purple_ball", "red_ball"):
                previous = self.latest_rolling.get(kind)
                if previous is None or previous["node"] != n:
                    self.rolling.append({"kind": kind, "node": n, "tick": now})
                self.latest_rolling[kind] = {"kind": kind, "node": n,
                                             "tick": now, "visible": True}

        # Keep enemy memory finite. A stale slot is never promoted to a visible
        # enemy; the RGB-verified adaptor remains the visibility authority.
        for kind, old in list(self.visible.items()):
            if kind not in live and now - old["tick"] <= (7 if kind == "coily" else 3):
                remembered = dict(old)
                remembered["visible"] = False
                live[kind] = remembered
        self.visible = live

        for kind, item in list(self.latest_rolling.items()):
            if kind in live and live[kind]["visible"]:
                item["tick"] = now
                item["node"] = live[kind]["node"]
                item["visible"] = True
            else:
                item["visible"] = False
            if now - item["tick"] > 60:
                self.latest_rolling.pop(kind, None)
        self.rolling = deque((r for r in self.rolling if now - r["tick"] <= 60), maxlen=5)

    def shadow(self, destination, now, horizon=8):
        """Return localized possible descendants, not an exact hidden position."""
        found = []
        for item in self.rolling:
            age = max(0, now - item["tick"])
            if age > 60:
                continue
            source = item["node"]
            depth = min(4, max(1, (age + horizon + 11) // 12))
            frontier = {source}
            descendants = {source}
            for _ in range(depth):
                frontier = self.down_frontier(frontier)
                descendants.update(frontier)
            if destination in descendants:
                found.append({"kind": item["kind"], "source": source,
                              "age": age, "depth": depth,
                              "last_seen_visible": bool(self.latest_rolling.get(item["kind"], {}).get("visible"))})
        return found

    def possible_at(self, kind, now, horizon):
        """Bounded possibility cone from a rendered or recent sighting."""
        item = self.visible.get(kind)
        if item is None:
            return set()
        source = item.get("node")
        if source not in GRAPH:
            return set()
        age = max(0, now - integer(item.get("tick"), now))
        steps = min(2, max(0, (age + horizon) // 9))
        possible = {source}
        frontier = {source}
        if kind in ("purple_ball", "red_ball"):
            for _ in range(steps):
                frontier = self.down_frontier(frontier)
                possible.update(frontier)
        else:
            radius = min(2, max(1, (age + horizon) // 10))
            for _ in range(radius):
                frontier = {there for here in frontier for there in GRAPH[here].values()}
                possible.update(frontier)
        return possible

    def risk(self, origin, destination, now, horizon=8):
        value = 0.0
        causes = []
        for kind, h in self.visible.items():
            hn = h["node"]
            age = max(0, now - integer(h.get("tick"), now))
            source_distance = DIST[hn][origin] if origin is not None else 99
            if kind == "coily" and source_distance <= 1:
                value += 3.0
                causes.append("coily_takeoff")
            if kind in ("purple_ball", "red_ball") and source_distance == 0:
                value += 3.0
                causes.append("rolling_takeoff")
            arrival = self.possible_at(kind, now, horizon)
            if destination in arrival:
                value += 5.5 if age <= 3 else 3.0
                causes.append(kind + "_arrival")
            if kind in ("purple_ball", "red_ball"):
                next_branch = self.down_frontier(arrival)
                if destination in next_branch and destination not in arrival:
                    value += 2.0
                    causes.append(kind + "_branch")
            elif hn in GRAPH and DIST[hn][destination] <= 1:
                value += 0.7
                causes.append("coily_near_landing")

        # The shadow is deployed only in the later-board occlusion regime.
        shadowed = self.shadow(destination, now + horizon, horizon=5) if self._late_mode else []
        if shadowed:
            youngest = min(x["age"] for x in shadowed)
            value += 1.45 if youngest <= 36 else 0.85
            causes.append("recent_rolling_shadow")
        return value, causes, shadowed

    def semantic(self, origin, now):
        result = []
        for kind, item in self.visible.items():
            distance = DIST[origin][item["node"]] if origin is not None else 99
            result.append({"kind": kind,
                           "visibility": "rendered" if item["visible"] else "recent_memory",
                           "relation": "same_cube" if distance == 0 else
                                       "adjacent" if distance == 1 else
                                       "near" if distance <= 2 else "distant"})
        return result

    def signature(self, now):
        live = tuple(sorted((k, v["node"], bool(v["visible"])) for k, v in self.visible.items()))
        branch = tuple(sorted((x["kind"], x["node"], min(4, (now - x["tick"]) // 12))
                              for x in self.rolling if now - x["tick"] <= 60))
        return live, branch


class RouteEngine:
    def __init__(self, ctx, book, threats):
        self.ctx = ctx
        self.book = book
        self.threats = threats
        self.cursor = 0
        self.target = None
        self.posture = "center"
        self.sensitivity = "balanced"
        self.recent_nodes = deque(maxlen=18)
        self.edge_failures = Counter()
        self.policy_version = 0

    def reset_board(self):
        self.cursor = 0
        self.target = None
        self.recent_nodes.clear()
        self.edge_failures.clear()

    def choose_target(self, here, need=None):
        needs = self.book.need if need is None else need
        unfinished = {i for i, v in enumerate(needs) if v > 0}
        if self.target not in unfinished:
            self.target = None
        if self.target is None:
            for j in range(len(ORDER)):
                idx = (self.cursor + j) % len(ORDER)
                dest = ORDER[idx]
                if dest in unfinished and (dest != here or len(unfinished) == 1):
                    self.cursor = idx
                    self.target = dest
                    break
        return self.target

    def candidates(self, origin, allowed, now, need=None, virtual=False):
        if origin is None:
            return []
        needs = self.book.need if need is None else need
        target = self.choose_target(origin, needs)
        remaining = [i for i, v in enumerate(needs) if v > 0]
        if not remaining:
            return []
        actions = [a for a in TIE if a in allowed and a in GRAPH[origin]]
        choices = []
        seen = Counter(self.recent_nodes)
        for action in actions:
            dest = GRAPH[origin][action]
            risk, causes, shadow = self.threats.risk(origin, dest, now, horizon=8)
            exits = 0
            for back, next_node in GRAPH[dest].items():
                if next_node == origin:
                    continue
                exit_risk, _, _ = self.threats.risk(dest, next_node, now + 8, horizon=5)
                if exit_risk < 3.0:
                    exits += 1
            route = min(DIST[dest][goal] for goal in remaining)
            useful = needs[dest] > 0
            # The coverage order is persistent, while nearest unfinished cube
            # is the fallback target when a scripted target is no longer useful.
            target_dist = DIST[dest][target] if target is not None else route
            cost = 1.9 * route + (0.0 if useful else 0.52)
            cost += 10.5 * min(8.0, risk)
            cost -= 0.18 * min(2, exits)
            cost += 0.10 * seen[dest]
            if dest in (15, 20) and exits == 0 and not useful:
                cost += 0.8
            if self.sensitivity == "branch_aware" and shadow:
                cost += 1.2
            if self.posture == "upper" and useful:
                cost += 0.16 * ROW[dest]
            elif self.posture == "frontier" and useful:
                cost -= 0.13 * ROW[dest]
            if target is not None:
                cost += 0.12 * target_dist
            cost += min(2.0, 0.7 * self.edge_failures[(origin, dest)])
            choices.append({"action": action, "destination": dest, "cost": cost,
                            "risk": risk, "causes": causes, "shadow": shadow,
                            "useful": useful, "route": route, "target": target,
                            "exits": exits})

        choices.sort(key=lambda c: (c["risk"] >= 6.0, c["cost"],
                                    ORDER.index(c["destination"]) if c["destination"] in ORDER else 99,
                                    TIE.index(c["action"])))
        return choices

    def landing(self, actual, expected, origin):
        if actual is None:
            return
        self.recent_nodes.append(actual)
        if expected is not None and actual != expected and origin is not None:
            self.edge_failures[(origin, expected)] += 1
            self.ctx.log("landing_resynchronized", origin=origin, expected=expected,
                         observed=actual, count=self.edge_failures[(origin, expected)])
        if self.target == actual and self.book.need[actual] <= 0:
            self.target = None


class QbertController:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.book = PaletteLedger(ctx)
        self.threats = ThreatMemory()
        self.route = RouteEngine(ctx, self.book, self.threats)
        self.lives = None
        self.life_counter = None
        self.life_epoch = 0
        self.pending = None
        self.neutral_until = 0
        self.recover_until = 0
        self.advice_future = None
        self.advice_meta = None
        self.advice = None
        self.last_jev_tick = -999
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.astra_meta = None
        self.last_status = -999
        try:
            names = {str(v) for v in game.action_spec.names}
        except Exception:
            names = {"NOOP", *DIRS}
        self.names = names
        self.noop = "NOOP" if "NOOP" in names else next(iter(names), "NOOP")

    def ingest(self, obs):
        now = tick_of(obs)
        old_epoch = self.book.epoch
        self.book.update(obs)
        self.threats.update(obs, now, late=self.book.clears >= 4)
        lives = integer(obs.get("lives"), self.lives if self.lives is not None else 4)
        lc = integer(obs.get("life_lost_count"), self.life_counter if self.life_counter is not None else 0)
        lost = ((self.lives is not None and lives < self.lives)
                or (self.life_counter is not None and lc > self.life_counter))
        if self.book.epoch != old_epoch:
            self.route.reset_board()
            self.pending = None
            self.advice = None
            self.neutral_until = max(self.neutral_until, now + 2)
            self.threats.clear()
            self.ctx.log("controller_board_reset", tick=now, epoch=self.book.epoch,
                         clears=self.book.clears)
            if self.astra_calls < 2:
                self.astra_meta = "board_progression_review"
        if lost:
            self.life_epoch += 1
            self.pending = None
            self.advice = None
            self.route.target = None
            self.recover_until = max(self.recover_until, now + 4)
            self.neutral_until = max(self.neutral_until, now + 8)
            self.threats.clear()
            self.ctx.log("life_recovery", tick=now, lives=lives,
                         life_epoch=self.life_epoch, epoch=self.book.epoch)
            if self.astra_calls < 2:
                self.astra_meta = "life_loss_review"
        self.lives, self.life_counter = lives, lc

    def safe_actions(self, obs, here):
        raw = obs.get("safe_moves")
        if not isinstance(raw, list) or not raw:
            raw = obs.get("legal_moves", [])
        allowed = set(v for v in raw if isinstance(v, str)) if isinstance(raw, list) else set()
        return [a for a in TIE if a in allowed and a in GRAPH.get(here, {})]

    def make_jev_state(self, obs, projected_node, candidates, origin_action):
        now = tick_of(obs)
        origin = node(obs.get("qbert_node"))
        described = []
        for i, c in enumerate(candidates[:4]):
            described.append({"key": c["action"],
                              "landing": "unfinished" if c["useful"] else "transit",
                              "lethal_exposure": "concrete" if c["risk"] >= 4.0 else
                                                 "possible_branch" if c["shadow"] else "not_supported",
                              "escape_after_landing": "several" if c["exits"] >= 2 else
                                                      "one" if c["exits"] == 1 else "none",
                              "route": "near" if c["route"] <= 1 else "farther",
                              "code_rank": "first" if i == 0 else "alternate"})
        return safe_json({
            "game": "Atari 2600 Q*bert, six-row 21-cube pyramid",
            "decision_timing": "The current hop is already underway. Choose only among edges from projected_landing for the next junction.",
            "current_hop": {"action": origin_action, "projected_landing": projected_node},
            "next_position": {"standing_after_hop": projected_node,
                              "candidates": described},
            "threats_now": self.threats.semantic(origin, now),
            "progress": {"inferred_board_clears": self.book.clears,
                         "incomplete_bucket": "many" if sum(v > 0 for v in self.book.need) > 8 else
                                              "several" if sum(v > 0 for v in self.book.need) > 2 else "few",
                         "priority": "survival, finish required cube phases, no optional bonus chase"},
            "guard": "Do not invent actions. Code will recheck node, legality, danger, and route cost on arrival."
        })

    def start_jev_ahead(self, obs, from_action, landing):
        if self.advice_future is not None or self.ctx.done or landing is None:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        now = tick_of(obs)
        if now - self.last_jev_tick < 6:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 2:
                return
        except Exception:
            pass

        # Generate plausible next-junction corridors before the player arrives.
        # The answer normally completes during the eight-ish tick animation.
        projected = self.route.candidates(landing, list(GRAPH[landing]), now + 8)
        if len(projected) < 2:
            return
        shown = projected[:4]
        keys = [x["action"] for x in shown]
        options = {c["action"]: {
            "what": "Take this supplied legal edge for one hop after the current landing.",
            "landing": "unfinished cube" if c["useful"] else "transit cube",
            "threat": "concrete" if c["risk"] >= 4.0 else
                      "possible rolling branch" if c["shadow"] else "no supported immediate threat",
            "escape": "several routes" if c["exits"] >= 2 else
                      "one route" if c["exits"] == 1 else "no non-return exit",
            "not_for": "inventing a direction or chasing a bonus"
        } for c in shown}
        options["keep_code_choice"] = "No clear model-supported improvement; use the freshly recomputed code ranking."
        instruction = (
            "At the next junction, choose the safest useful edge from `next_position.candidates`. "
            "Prefer no_supported immediate threat over concrete exposure; a possible rolling branch is uncertain, "
            "not confirmed occupancy. When safety is similar, prefer more escape routes, then unfinished coverage. "
            "The current hop is already underway; do not judge or change it. Choose keep_code_choice if the evidence is tied."
        )
        criteria_a = {k: options[k] for k in keys}
        criteria_a["keep_code_choice"] = options["keep_code_choice"]
        reverse = list(reversed(keys)) + ["keep_code_choice"]
        criteria_b = {k: options[k] for k in reverse}
        questions = {
            "arrival_choice_a": Choice(instruction, criteria_a),
            "arrival_choice_b": Choice(instruction, criteria_b),
        }
        state = self.make_jev_state(obs, landing, shown, from_action)
        meta = {"created": now, "expected_node": landing,
                "epoch": self.book.epoch, "life": self.life_epoch,
                "policy": self.route.policy_version,
                "threat_signature": self.threats.signature(now),
                "keys": keys, "ranked": [dict(c) for c in shown]}
        self.last_jev_tick = now

        def call():
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception as exc:
                return {"_error": type(exc).__name__}

        try:
            self.advice_future = self.ctx.spawn(call)
            self.advice_meta = meta
            self.ctx.log("jev_arrival_review_started", tick=now, landing=landing,
                         next_options=keys)
        except Exception as exc:
            self.advice_future = None
            self.advice_meta = None
            self.ctx.log("jev_background_start_failed", error=type(exc).__name__)

    def collect_jev(self):
        if self.advice_future is None or not self.advice_future.done():
            return
        future, meta = self.advice_future, self.advice_meta
        self.advice_future = self.advice_meta = None
        try:
            answers = future.result()
            if not isinstance(answers, dict) or "_error" in answers or not isinstance(meta, dict):
                self.ctx.log("jev_arrival_result_invalid")
                return
            a = answer_value(answers.get("arrival_choice_a"))
            b = answer_value(answers.get("arrival_choice_b"))
            if a != b or a not in meta.get("keys", []) + ["keep_code_choice"]:
                self.ctx.log("jev_arrival_disagreement", first=a, second=b)
                return
            p = min(option_probability(answers.get("arrival_choice_a"), a),
                    option_probability(answers.get("arrival_choice_b"), b))
            self.advice = {**meta, "choice": a, "probability": p}
            self.ctx.log("jev_arrival_result", expected_node=meta.get("expected_node"),
                         choice=a, probability=p)
        except Exception as exc:
            self.ctx.log("jev_arrival_result_error", error=type(exc).__name__)

    def apply_advice(self, obs, candidates):
        advice = self.advice
        self.advice = None
        if not advice or not candidates:
            return candidates[0] if candidates else None
        now = tick_of(obs)
        here = node(obs.get("qbert_node"))
        if (advice.get("expected_node") != here or advice.get("epoch") != self.book.epoch
                or advice.get("life") != self.life_epoch
                or advice.get("policy") != self.route.policy_version
                or now < advice.get("created", now) or now - advice.get("created", now) > 12):
            self.ctx.log("jev_arrival_expired", tick=now, expected=advice.get("expected_node"), node=here)
            return candidates[0]
        key = advice.get("choice")
        if key == "keep_code_choice" or num(advice.get("probability")) < 0.60:
            return candidates[0]
        chosen = next((c for c in candidates if c["action"] == key), None)
        if chosen is None:
            return candidates[0]
        best = candidates[0]
        # Jev is a safety specialist, never a tie-breaker for the coverage
        # route. It can override only when the current recomputed certificate
        # shows a material risk reduction; equal-risk choices stay with code.
        signature_changed = self.threats.signature(now) != advice.get("threat_signature")
        required_improvement = 0.8 if signature_changed else 0.45
        materially_safer = chosen["risk"] <= best["risk"] - required_improvement
        better_escape = (num(advice.get("probability")) >= 0.72
                         and chosen["risk"] <= best["risk"] + 0.05
                         and chosen["exits"] >= best["exits"] + 1
                         and chosen["cost"] <= best["cost"] + 0.5
                         and (chosen["useful"] or not best["useful"]))
        if ((materially_safer or better_escape)
                and chosen["cost"] <= best["cost"] + 2.0
                and chosen["exits"] >= max(0, best["exits"] - 1)):
            if chosen["action"] != best["action"]:
                self.ctx.log("jev_changed_next_hop", tick=now, old=best["action"],
                             new=chosen["action"], probability=advice["probability"],
                             risk=chosen["risk"], best_risk=best["risk"])
            return chosen
        self.ctx.log("jev_rejected_by_code", tick=now, option=key,
                     risk=chosen["risk"], best_risk=best["risk"])
        return best

    def start_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_calls >= 2 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            pass
        now = tick_of(obs)
        summary = safe_json({"reason": reason, "tick": now,
                             "score": num(obs.get("score")), "lives": self.lives,
                             "inferred_clears": self.book.clears,
                             "unfinished": sum(v > 0 for v in self.book.need),
                             "last_colour_evidence": list(self.book.evidence)[-4:],
                             "recent_hazards": self.threats.semantic(node(obs.get("qbert_node")), now),
                             "policy": {"posture": self.route.posture,
                                        "shadow_sensitivity": self.route.sensitivity}})
        prompt = (
            "You are the slow strategic adviser for a real-time Atari 2600 Q*bert harness. "
            "Code owns the 21-node graph, palette evidence, and neutralized one-hop motor. "
            "Jev reviews the *next* junction while Q*bert is hopping; code rejects stale or unsafe advice. "
            "Choose a durable coverage posture: posture is center, upper, or frontier; "
            "shadow_sensitivity is balanced or branch_aware. branch_aware gives a small extra cost to "
            "destinations in the bounded descendant cone of a recently rendered rolling ball; it is not a "
            "claim of current occupancy. Favor survival and complete boards over optional bonuses. "
            "Do not return actions, routes, node numbers, FIRE, or prose. Return ONLY compact JSON such as "
            '{"posture":"center","shadow_sensitivity":"balanced"}. Invalid output is ignored.\n'
            + json.dumps(summary, separators=(",", ":")))
        self.astra_calls += 1
        self.astra_meta = None
        session = self.astra_session

        def call():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=110.0)
                return {"text": getattr(result, "text", ""),
                        "session": getattr(result, "session_id", None)}
            except Exception as exc:
                return {"error": type(exc).__name__}

        try:
            self.astra_future = self.ctx.spawn(call)
            self.ctx.log("astra_review_started", tick=now, reason=reason)
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_background_start_failed", error=type(exc).__name__)

    def collect_astra(self, obs):
        if self.astra_future is None or not self.astra_future.done():
            return
        future, self.astra_future = self.astra_future, None
        try:
            result = future.result()
            if not isinstance(result, dict) or "error" in result:
                self.ctx.log("astra_review_invalid")
                return
            if result.get("session"):
                self.astra_session = result["session"]
            obj = parse_json_object(result.get("text", ""))
            old = (self.route.posture, self.route.sensitivity)
            if obj.get("posture") in ("center", "upper", "frontier"):
                self.route.posture = obj["posture"]
            if obj.get("shadow_sensitivity") in ("balanced", "branch_aware"):
                self.route.sensitivity = obj["shadow_sensitivity"]
            new = (self.route.posture, self.route.sensitivity)
            if new != old:
                self.route.policy_version += 1
                self.advice = None
                self.ctx.log("astra_posture_updated", tick=tick_of(obs), old=list(old), new=list(new))
            else:
                self.ctx.log("astra_posture_unchanged", tick=tick_of(obs), parsed=bool(obj))
        except Exception as exc:
            self.ctx.log("astra_review_error", error=type(exc).__name__)

    def action(self, obs):
        now = tick_of(obs)
        self.collect_jev()
        self.collect_astra(obs)
        if self.book.flash:
            self.pending = None
            return self.noop

        here = node(obs.get("qbert_node"))
        phase = str(motion_of(obs).get("phase", "unknown")).lower()
        if here is None or phase in ("falling", "not_visible") or motion_of(obs).get("falling"):
            self.pending = None
            return self.noop

        if now < self.recover_until:
            return self.noop
        if self.recover_until and now >= self.recover_until:
            self.recover_until = 0

        if self.pending is not None:
            hop = self.pending
            age = now - hop["tick"]
            if obs.get("qbert_in_hop") or here != hop["origin"]:
                hop["moving"] = True
            if hop["moving"] and is_standing(obs):
                self.route.landing(here, hop["destination"], hop["origin"])
                matched = here == hop["destination"]
                self.ctx.log("hop_verified" if matched else "hop_mismatch", tick=now,
                             origin=hop["origin"], expected=hop["destination"],
                             observed=here, duration=age)
                self.pending = None
                self.neutral_until = max(self.neutral_until, now + 1)
                return self.noop
            if not hop["moving"] and is_standing(obs) and age <= 3 and hop["posts"] < 2:
                hop["posts"] += 1
                return hop["action"]
            if age > 18 and is_standing(obs):
                self.pending = None
                self.neutral_until = max(self.neutral_until, now + 3)
                self.ctx.log("hop_not_started", tick=now, origin=hop["origin"],
                             expected=hop["destination"])
            else:
                return self.noop

        if now < self.neutral_until or not is_standing(obs):
            return self.noop
        allowed = self.safe_actions(obs, here)
        candidates = self.route.candidates(here, allowed, now)
        if not candidates:
            return self.noop
        chosen = self.apply_advice(obs, candidates) or candidates[0]
        action = chosen["action"]
        if action not in allowed or action not in DIRS:
            return self.noop
        destination = GRAPH[here].get(action)
        if destination is None:
            return self.noop

        self.pending = {"origin": here, "destination": destination,
                        "action": action, "tick": now, "posts": 1, "moving": False}
        self.start_jev_ahead(obs, action, destination)
        if now - self.last_status >= 25:
            self.ctx.log("coverage_decision", tick=now, score=num(obs.get("score")),
                         lives=self.lives, epoch=self.book.epoch,
                         inferred_clears=self.book.clears,
                         incomplete=sum(v > 0 for v in self.book.need),
                         node=here, target=self.route.target,
                         action=action, destination=destination,
                         risk=chosen["risk"], escapes=chosen["exits"],
                         shadow=chosen["shadow"])
            self.last_status = now
        return action

    def run(self):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.ingest(obs)
        self.start_astra(obs, "initial coverage and hazard posture")
        while not ctx.done and not getattr(self.game, "done", False):
            obs = ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
            action = self.action(obs)
            if self.astra_meta is not None:
                self.start_astra(obs, self.astra_meta)
            try:
                ctx.act(action if action in self.names else self.noop)
                ctx.wait_for_tick()
            except Exception as exc:
                # Context closure at deadline/game end is normal. A valid
                # action should not otherwise fail; record and leave promptly.
                self.ctx.log("episode_loop_closed", error=type(exc).__name__)
                return
        self.ctx.log("candidate_summary", score=getattr(self.game, "score", None),
                     inferred_clears=self.book.clears, palette_epochs=self.book.epoch,
                     lives=self.lives, posture=self.route.posture,
                     shadow_sensitivity=self.route.sensitivity)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        QbertController(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)

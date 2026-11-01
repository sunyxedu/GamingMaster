"""0007: palette-epoch coverage, guarded motor, asynchronous tactical advisers.

Only public observations are used.  In particular, the local progress tracker
never changes the observation or score: it cross-checks misleading decoded
phase labels against observed palette transitions and score deltas.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRS = ("UP", "RIGHT", "LEFT", "DOWN")
ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
GRAPH = {}
ROWS = {}
for r in range(6):
    for c in range(r + 1):
        n = r * (r + 1) // 2 + c
        ROWS[n] = r
        GRAPH[n] = {}
        if r and c < r:
            GRAPH[n]["UP"] = n - r
        if c:
            GRAPH[n]["LEFT"] = n - r - 1
        if r < 5:
            GRAPH[n]["DOWN"] = n + r + 1
            GRAPH[n]["RIGHT"] = n + r + 2

DIST = []
for start in range(21):
    distances = [99] * 21
    distances[start] = 0
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in GRAPH[node].values():
            if distances[nxt] == 99:
                distances[nxt] = distances[node] + 1
                queue.append(nxt)
    DIST.append(distances)


def number(value, default=0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    return int(number(value, default))


def valid_node(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 21 else None


def tick(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)))


def motion(obs):
    value = obs.get("qbert_motion")
    return value if isinstance(value, dict) else {"phase": str(value or "unknown")}


def standing(obs):
    return (valid_node(obs.get("qbert_node")) is not None
            and not obs.get("qbert_in_hop")
            and motion(obs).get("phase") not in ("hopping", "falling", "not_visible")
            and not motion(obs).get("falling"))


def colors(obs):
    raw = obs.get("cube_colors")
    if isinstance(raw, list) and len(raw) == 21 and all(isinstance(x, int) for x in raw):
        return tuple(raw)
    return None


def probability(answer, key):
    p = getattr(answer, "probabilities", None)
    return number(p.get(key), 0) if isinstance(p, dict) else 0


def choice(answer):
    value = getattr(answer, "value", None)
    return value if isinstance(value, str) else None


def parse_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:12000]):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError):
                pass
    return {}


class PaletteProgress:
    """Learn the destination per board, not from an initial palette constant."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.score = None
        self.goal = None
        self.start = None
        self.flashing = False
        self.flash_tick = 0
        self.flash_score = 0
        self.flash_origin = None
        self.clears = 0
        self.epoch = 0
        self.need = [1] * 21
        self.complete_since = None
        self.last_progress = 0
        self.evidence = []
        self.mismatch_logged = False

    def update(self, obs):
        t = tick(obs)
        current = colors(obs)
        score = number(obs.get("score"))
        delta = score - self.score if self.score is not None else 0
        changed = [i for i in range(21) if self.previous and current
                   and self.previous[i] != current[i]]
        self.score = score
        if current is None:
            raw = obs.get("needed_landings")
            if isinstance(raw, list) and len(raw) == 21:
                self.need = [max(0, integer(x, 1)) for x in raw]
            return False
        if self.start is None:
            self.start = Counter(current).most_common(1)[0][0]
        new_epoch = False

        # A simultaneous whole-board recolouring is not 21 independent Sam
        # reversions. The parent traces show a 31 x 100-point clear tally.
        if len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_score = score - delta
                self.flash_origin = valid_node(obs.get("qbert_node"))
                self.ctx.log("palette_flash_begin", tick=t, score=score,
                             reported_rounds=obs.get("completed_rounds"))
            self.flash_tick = t
        if self.flashing:
            quiet = t - self.flash_tick
            relocated = (valid_node(obs.get("qbert_node")) == 0
                         and self.flash_origin != 0)
            if quiet >= 4 and (relocated or quiet >= 7):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.flashing = False
                self.start = Counter(current).most_common(1)[0][0]
                self.goal = None
                self.need = [1] * 21
                self.complete_since = None
                self.last_progress = t
                self.mismatch_logged = False
                new_epoch = True
                self.ctx.log("palette_epoch", tick=t, epoch=self.epoch,
                             inferred_clears=self.clears, tally_points=tally,
                             start_color=self.start,
                             reported_rounds=obs.get("completed_rounds"))
            else:
                self.previous = current
                return False

        # Only a local, scoring change is a destination-colour calibration.
        # 325 may be a simultaneous 300-point capture and 25-point landing.
        scored_landing = delta > 0 and int(round(delta)) % 100 == 25
        n = valid_node(obs.get("qbert_node"))
        if scored_landing and 0 < len(changed) <= 3:
            landed = n if n in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                destination = current[landed]
                if self.goal is None:
                    self.goal = destination
                    self.ctx.log("palette_goal_learned", tick=t, epoch=self.epoch,
                                 node=landed, color=destination, reward=delta)
                self.last_progress = t
                self.evidence.append({"epoch": self.epoch, "node": landed,
                                      "reward": delta,
                                      "decoded_phase": (obs.get("cube_phase") or [None] * 21)[landed]})
                self.evidence = self.evidence[-8:]
                if (not self.mismatch_logged and self.goal == destination
                        and isinstance(obs.get("needed_landings"), list)
                        and len(obs["needed_landings"]) == 21
                        and number(obs["needed_landings"][landed]) > 0):
                    self.mismatch_logged = True
                    self.ctx.log("decoder_progress_disagreement", tick=t, node=landed,
                                 color=destination, reward=delta, epoch=self.epoch)

        if self.goal is not None:
            self.need = [0 if value == self.goal else 1 for value in current]
        else:
            self.need = [1] * 21
        # A two-phase level can reward an intermediate change. If a uniformly
        # visited board does not flash, do not deadlock: probe another pass.
        if not any(self.need):
            if self.complete_since is None:
                self.complete_since = t
            elif t - self.complete_since > 22:
                self.ctx.log("palette_additional_pass", tick=t, epoch=self.epoch,
                             old_goal=self.goal)
                self.start = self.goal
                self.goal = None
                self.need = [1] * 21
                self.complete_since = None
        else:
            self.complete_since = None
        self.previous = current
        return new_epoch


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.progress = PaletteProgress(ctx)
        self.life_epoch = 0
        self.lives = None
        self.pulse = None
        self.neutral_until = 0
        self.last_node = None
        self.recent = deque(maxlen=18)
        self.last_hop_tick = -999
        self.centers = {}
        self.threat_memory = {}
        # Keep the last rendered position of rolling balls after the adaptor
        # correctly suppresses their stale sprite slots.  This is distinct
        # from the parent's four-tick immediate-threat memory.
        self.rolling_history = {}
        self.now = 0
        self.style = "nearest"
        self.review = "collision_first"
        self.policy_version = 0
        self.s1 = None
        self.s1meta = None
        self.advice = None
        self.last_request = -999
        self.s2 = None
        self.s2calls = 0
        self.s2session = None
        self.s2reason = None
        self.last_strategy_tick = -999
        self.last_logged = -999
        self.waiting_since = None

    def update(self, obs):
        t = tick(obs)
        self.now = t
        changed = self.progress.update(obs)
        lives = integer(obs.get("lives"), 4)
        life_loss = self.lives is not None and lives < self.lives
        if life_loss or changed:
            self.life_epoch += 1
            self.pulse = None
            self.advice = None
            self.recent.clear()
            # A life reset or a new board invalidates a trajectory from the
            # previous playfield; do not carry an old ball cone across it.
            self.rolling_history.clear()
            self.neutral_until = t + 2
            self.waiting_since = None
            self.s2reason = "life_loss" if life_loss else "observed_palette_epoch_change"
            self.ctx.log("control_epoch", tick=t, lives=lives,
                         palette_epoch=self.progress.epoch, life_loss=life_loss)
        self.lives = lives
        if not self.centers:
            details = obs.get("cube_details", [])
            if isinstance(details, list):
                for item in details:
                    if isinstance(item, dict) and valid_node(item.get("node")) is not None:
                        # Public cube coordinates plus Q*bert's observed initial
                        # 8x19 sprite: its centre is about (cube_x+10,cube_y-7).
                        self.centers[item["node"]] = (number(item.get("x")) + 10,
                                                      number(item.get("y")) - 7)
        if self.progress.flashing:
            self.pulse = None
            self.advice = None
            return
        raw = obs.get("hazards", [])
        if isinstance(raw, list):
            for h in raw:
                if not isinstance(h, dict) or not h.get("lethal"):
                    continue
                kind = str(h.get("kind", "hazard"))
                x = number(h.get("x")) + number(h.get("w"), 7) / 2
                y = number(h.get("y")) + number(h.get("h"), 8) / 2
                estimate = valid_node(h.get("estimated_node", h.get("node")))
                if self.centers:
                    estimate = min(self.centers, key=lambda n:
                                   (self.centers[n][0] - x) ** 2 +
                                   (self.centers[n][1] - y) ** 2)
                old = self.threat_memory.get(kind)
                dx = number(h.get("dx"))
                dy = number(h.get("dy"))
                if old and abs(dx) < 0.01 and abs(dy) < 0.01:
                    dx, dy = old["dx"], old["dy"]
                self.threat_memory[kind] = {"kind": kind, "node": estimate,
                                            "x": x, "y": y, "dx": dx, "dy": dy,
                                            "tick": t}
        self.threat_memory = {k: v for k, v in self.threat_memory.items()
                              if t - v["tick"] <= 4}
        # Store only currently rendered rolling hazards.  The adaptor's
        # palette cross-check prevents stale RAM placeholders from entering
        # this trajectory memory.
        if isinstance(raw, list):
            for h in raw:
                if not isinstance(h, dict) or h.get("kind") not in ("purple_ball", "red_ball"):
                    continue
                hx = number(h.get("x")) + number(h.get("w"), 7) / 2
                hy = number(h.get("y")) + number(h.get("h"), 8) / 2
                hn = valid_node(h.get("estimated_node", h.get("node")))
                if self.centers:
                    hn = min(self.centers, key=lambda n:
                             (self.centers[n][0] - hx) ** 2 +
                             (self.centers[n][1] - hy) ** 2)
                if hn is not None:
                    self.rolling_history[h["kind"]] = {"node": hn, "tick": t}
        self.rolling_history = {
            k: v for k, v in self.rolling_history.items()
            if 0 <= t - v["tick"] <= 60
        }

    def stale_ball_shadow(self, destination):
        """Return a recent ball's possible downward descendants, if any.

        Q*bert's red and purple balls descend through DOWN/RIGHT branches.
        The trace shows a ball last rendered near node 8 and absent for about
        55 ticks before a loss at node 13; short (4-10 tick) sprite memory
        cannot represent that possible route.  This bounded projection is a
        risk signal, not a claim that the ball occupies a particular node.
        """
        if self.progress.clears < 4:
            return None
        for kind, item in self.rolling_history.items():
            age = self.now - integer(item.get("tick"), self.now)
            source = valid_node(item.get("node"))
            if source is None or age < 5 or age > 60:
                continue
            # One additional row of possible descent per ~15 public ticks,
            # capped at the bottom row. Include every earlier reachable row.
            depth = min(4, 1 + (age - 5) // 15)
            frontier = {source}
            descendants = set()
            for _ in range(depth):
                following = set()
                for node in frontier:
                    following.update(value for action, value in GRAPH[node].items()
                                     if action in ("DOWN", "RIGHT"))
                descendants.update(following)
                frontier = following
            if destination in descendants:
                return {"kind": kind, "source": source, "age": age, "depth": depth}
        return None

    def safe_edges(self, obs, node=None):
        n = valid_node(obs.get("qbert_node")) if node is None else node
        if n is None:
            return []
        safe = obs.get("safe_moves", obs.get("legal_moves", []))
        if not isinstance(safe, list):
            return []
        return [a for a in ORDER if a in safe and a in GRAPH[n]]

    def danger(self, destination, origin):
        danger = 0.0
        # Unlike immediate rendered danger, a projected descendant is a soft
        # avoidance cost: it can reroute coverage but never declares an edge
        # illegal or starts an unbounded wait.
        if self.stale_ball_shadow(destination):
            danger += 2.5
        for h in self.threat_memory.values():
            hn = h["node"]
            if hn is None:
                continue
            if destination in self.centers:
                x, y = self.centers[destination]
                d = math.hypot(x - h["x"], y - h["y"])
                danger += 5 if d < 17 else 1.2 if d < 30 else 0
            if hn == destination:
                danger += 3
            if h["kind"] in ("purple_ball", "red_ball"):
                children = [v for a, v in GRAPH[hn].items() if a in ("DOWN", "RIGHT")]
                if destination in children:
                    danger += 1.5
            elif h["kind"] == "coily":
                nxt = list(GRAPH[hn].values())
                toward = min((DIST[x][origin] for x in nxt), default=99)
                if destination in nxt and DIST[destination][origin] == toward:
                    danger += 2
                if DIST[hn][destination] == 1:
                    danger += 0.4
        return danger

    def candidates(self, obs):
        origin = valid_node(obs.get("qbert_node"))
        if origin is None:
            return []
        targets = [i for i, n in enumerate(self.progress.need) if n > 0]
        if not targets:
            return []
        candidates = []
        visits = Counter(self.recent)
        for action in self.safe_edges(obs):
            nxt = GRAPH[origin][action]
            route = min(DIST[nxt][target] for target in targets)
            useful = bool(self.progress.need[nxt])
            danger = self.danger(nxt, origin)
            escapes = sum(self.danger(other, nxt) < 2
                          for other in GRAPH[nxt].values())
            cost = route * 2 + (0 if useful else 0.5) + 12 * danger
            cost += visits[nxt] * 0.12 - escapes * 0.12
            approaching_coily = any(h["kind"] == "coily" and h["node"] is not None
                                    and DIST[nxt][h["node"]] <= 2
                                    for h in self.threat_memory.values())
            # An apparently clear corner is not safe if its sole return cube
            # will be occupied when Q*bert lands. Defer that target and lure
            # the snake elsewhere instead of knowingly entering the trap.
            if approaching_coily and escapes == 0:
                cost += 14
            if self.style == "top_first" and useful:
                cost += ROWS[nxt] * 0.15
            elif self.style == "frontier" and useful:
                cost -= ROWS[nxt] * 0.12
            if self.review == "escape_space":
                cost -= escapes * 0.25
            shadow = self.stale_ball_shadow(nxt)
            candidates.append({"action": action, "node": nxt, "cost": cost,
                               "danger": danger, "useful": useful,
                               "distance": route, "escapes": escapes,
                               "rolling_shadow": shadow})
        candidates.sort(key=lambda x: (x["cost"], ORDER.index(x["action"])))
        # A descending ball may threaten BOTH parents of a bottom cube. A
        # blind least-danger hop still intersects it. Stay briefly while its
        # branch becomes observable, but never wait for an approaching Coily.
        current_danger = self.danger(origin, origin)
        descending = any(h["kind"] in ("purple_ball", "red_ball")
                         for h in self.threat_memory.values())
        coily_near = any(h["kind"] == "coily" and h["node"] is not None
                         and DIST[origin][h["node"]] <= 2
                         for h in self.threat_memory.values())
        wait_age = tick(obs) - self.waiting_since if self.waiting_since is not None else 0
        if (candidates and descending and not coily_near and wait_age < 18
                and candidates[0]["danger"] >= 1.2
                and current_danger + 0.4 < candidates[0]["danger"]):
            candidates.insert(0, {"action": "NOOP", "node": origin,
                                  "cost": candidates[0]["cost"] - 1,
                                  "danger": current_danger, "useful": False,
                                  "distance": min(DIST[origin][n] for n in targets),
                                  "escapes": 0})
        return candidates

    def signature(self):
        immediate = tuple(sorted((kind, h["node"]) for kind, h in self.threat_memory.items()))
        # Invalidate advice when a hidden-ball branch first becomes active,
        # expands to a new depth band, or expires.
        shadows = []
        if self.progress.clears >= 4:
            for kind, item in self.rolling_history.items():
                age = self.now - integer(item.get("tick"), self.now)
                if 5 <= age <= 60:
                    depth = min(4, 1 + (age - 5) // 15)
                    shadows.append(("rolling:" + kind, item.get("node"), depth))
        return immediate + tuple(sorted(shadows))

    def ask_jev(self, obs, candidates):
        if self.s1 is not None or not candidates or self.ctx.done:
            return
        t = tick(obs)
        if t - self.last_request < 5 or getattr(self.ctx, "system1", True) is None:
            return
        if self.ctx.budget.remaining().get("system1_calls", 1) <= 1:
            return
        origin = valid_node(obs.get("qbert_node"))
        described = []
        for c in candidates:
            described.append({"action": c["action"],
                              "landing": "wait_for_descending_ball_to_commit" if c["action"] == "NOOP"
                                         else "unfinished" if c["useful"] else "transit",
                              "danger": "occupied" if c["danger"] >= 5 else
                                        "intercept_possible" if c["danger"] >= 1 else "clear",
                              "escape_space": "multiple" if c["escapes"] >= 2 else "narrow",
                              "rolling_branch": "recent_hidden_descendant" if c.get("rolling_shadow") else "none",
                              "route": "on_target" if c["distance"] == 0 else
                                       "one_hop_away" if c["distance"] == 1 else "farther",
                              "code_preferred": c is candidates[0]})
        state = {
            "game": "Atari 2600 Q*bert; 21 cubes. Code supplies all geometry and arithmetic.",
            "policy": {"style": self.style, "review": self.review,
                       "priority": "survival, useful cube coverage, round completion; no bonus chasing"},
            "progress": {"interpretation": "destination learned from scoring palette changes; decoder labels may be stale",
                         "board_epoch": self.progress.epoch},
            "position": {"node": origin,
                         "motion": motion(obs).get("phase", "unknown")},
            "candidates": described,
            "proposed_action": candidates[0]["action"],
            "threats": ([{"kind": h["kind"],
                          "proximity": "adjacent" if h["node"] is not None and DIST[origin][h["node"]] <= 1
                                       else "farther",
                          "visibility": "recently_hidden" if self.now - h["tick"] > 0 else "rendered"}
                         for h in self.threat_memory.values()]
                        + [{"kind": kind, "proximity": "downstream_possible",
                            "visibility": "recently_hidden", "age_bucket":
                            "one_to_two_hops" if self.now - item["tick"] < 35 else "three_to_four_hops"}
                           for kind, item in self.rolling_history.items()
                           if self.progress.clears >= 4
                           and 5 <= self.now - item["tick"] <= 60])}
        criteria = {c["action"]: ("Wait neutral briefly for a descending ball to commit; "
                                   "code verifies the current cube is safer and bounds the wait."
                                   if c["action"] == "NOOP" else
                                   "Take this supplied edge if its immediate danger and escape-space facts support it; "
                                   "recent_hidden_descendant means uncertain downward-branch exposure, not confirmed occupancy.")
                    for c in candidates}
        criteria["unknown"] = "No supported preference; let the deterministic route continue."
        instruction = ("Choose the safest useful next hop from candidates. Prefer clear over occupied "
                       "or intercept_possible. A recent_hidden_descendant marks a plausible downward route "
                       "from a recently hidden rolling ball, not certain contact; when otherwise similar, "
                       "prefer an edge without that marker. Among similarly safe choices prefer code_preferred. "
                       "Do not compute distances. Return unknown if the evidence is insufficient. ")
        if self.review == "escape_space":
            instruction += "When danger is similar, prefer multiple escape space over narrow."
        questions = {
            "a": Choice(instruction, criteria),
            "b": Choice(instruction, dict(reversed(list(criteria.items())))),
            "danger": Noul("Do the candidate facts establish immediate lethal danger on proposed_action? "
                           "A projected recent_hidden_descendant alone is uncertain and is not confirmed occupancy; "
                           "an occupied landing or a concrete near collision is danger.",
                           {"true": "Proposed landing is occupied or a concrete near collision is stated.",
                            "false": "Clear landing, or only distant or uncertain threats."})}
        self.s1meta = {"node": origin, "tick": t, "life": self.life_epoch,
                       "board": self.progress.epoch, "policy": self.policy_version,
                       "threats": self.signature(), "proposed": candidates[0]["action"]}
        self.last_request = t

        def work():
            return self.ctx.decide(state, questions, timeout_s=1.0)

        self.s1 = self.ctx.spawn(work)

    def collect_jev(self, obs):
        if self.s1 is None or not self.s1.done():
            return
        future, meta = self.s1, self.s1meta
        self.s1 = None
        self.s1meta = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            a, b = choice(answers.get("a")), choice(answers.get("b"))
            agreed = a == b and a in DIRS + ("NOOP",)
            p = min(probability(answers.get("a"), a), probability(answers.get("b"), b))
            veto = number(getattr(answers.get("danger"), "value", 0)) >= 0.82
            self.advice = dict(meta, action=a if agreed and p >= 0.60 else None,
                               veto=veto, probability=p)
        except Exception as exc:
            self.ctx.log("jev_failure", error=type(exc).__name__)

    def advised_action(self, obs, candidates):
        fallback = candidates[0]["action"]
        meta = self.advice
        if not meta:
            return fallback
        self.advice = None
        fresh = (meta["node"] == valid_node(obs.get("qbert_node"))
                 and meta["life"] == self.life_epoch
                 and meta["board"] == self.progress.epoch
                 and meta["policy"] == self.policy_version
                 and meta["threats"] == self.signature()
                 and 0 <= tick(obs) - meta["tick"] <= 8)
        if not fresh:
            return fallback
        action = meta.get("action")
        by_action = {c["action"]: c for c in candidates}
        chosen = by_action.get(action)
        if (chosen and chosen["cost"] <= candidates[0]["cost"] + 2.0
                and chosen["danger"] <= candidates[0]["danger"] + 0.2):
            if action != fallback:
                self.ctx.log("jev_changes_action", tick=tick(obs), old=fallback,
                             new=action, probability=meta["probability"])
            return action
        if meta.get("veto") and meta["proposed"] == fallback and len(candidates) > 1:
            alternative = candidates[1]
            if alternative["danger"] <= candidates[0]["danger"] + 0.1:
                self.ctx.log("jev_veto_applied", tick=tick(obs), old=fallback,
                             new=alternative["action"])
                return alternative["action"]
        return fallback

    def ask_astra(self, obs, reason):
        if self.s2 is not None or self.s2calls >= 3 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 1) <= 1:
            return
        if self.s2calls and tick(obs) - self.last_strategy_tick < 300:
            return
        state = {"reason": reason, "score": number(obs.get("score")),
                 "lives": self.lives, "inferred_clears": self.progress.clears,
                 "reported_rounds": obs.get("completed_rounds"),
                 "evidence": self.progress.evidence,
                 "policy": {"style": self.style, "review": self.review}}
        prompt = (
            "You advise a real-time Atari 2600 Q*bert controller, six-row 21-node board. "
            "Code owns geometry, raw-colour bookkeeping, danger forecasts and neutralized "
            "single-hop execution. Archived observations show a 31x100 score tally and "
            "global flashing after all cubes finish, but the decoder still says round 1; "
            "on the next board rewarded colour changes are wrongly labeled incomplete. "
            "The controller learns a new destination palette per observed clear. "
            "Select one durable coverage policy: nearest (shortest useful landing), "
            "top_first (top-region tie bias), or frontier (deep-frontier tie bias). "
            "Select one Jev review criterion: collision_first or escape_space. "
            "Jev receives compact per-edge facts: clear/intercept_possible/occupied, "
            "unfinished/transit, multiple/narrow exits, and code_preferred. Two reordered "
            "typed Choices must agree with probabilities >=0.60; a Noul danger veto "
            "requires >=0.82. Code always rechecks legality, epoch and freshness. "
            "Jev does not generate plans or perform arithmetic. Your review selection "
            "changes its next question wording and code tie-breaking. Never recommend "
            "edge falls, guessed disc exits or FIRE. Return ONLY JSON, under 60 tokens: "
            '{"style":"nearest","review":"collision_first"}. '
            "Do not produce a long explanation. Evidence: " + json.dumps(state))
        session = self.s2session
        self.s2calls += 1
        self.last_strategy_tick = tick(obs)
        self.s2reason = None

        def work():
            return self.ctx.think(prompt, session_id=session, timeout_s=100.0)

        self.s2 = self.ctx.spawn(work)
        self.ctx.log("astra_requested", tick=tick(obs), reason=reason)

    def collect_astra(self, obs):
        if self.s2 is None or not self.s2.done():
            return
        future, self.s2 = self.s2, None
        try:
            result = future.result()
            self.s2session = getattr(result, "session_id", None)
            data = parse_object(getattr(result, "text", ""))
            old = (self.style, self.review)
            if data.get("style") in ("nearest", "top_first", "frontier"):
                self.style = data["style"]
            if data.get("review") in ("collision_first", "escape_space"):
                self.review = data["review"]
            if old != (self.style, self.review):
                self.policy_version += 1
                self.advice = None
                self.ctx.log("astra_policy_changed", tick=tick(obs), old=list(old),
                             style=self.style, review=self.review)
            else:
                self.ctx.log("astra_policy_unchanged", tick=tick(obs), parsed=bool(data))
        except Exception as exc:
            self.ctx.log("astra_failure", error=type(exc).__name__)

    def action(self, obs):
        t = tick(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)
        if self.progress.flashing or not any(self.progress.need):
            self.pulse = None
            return "NOOP"
        n = valid_node(obs.get("qbert_node"))
        phase = motion(obs).get("phase")
        if n is None or phase in ("falling", "not_visible") or motion(obs).get("falling"):
            self.pulse = None
            return "NOOP"
        candidates = self.candidates(obs)
        # Speculate during the incoming hop, not after it. Code owns the
        # clock; there is never an inference-induced wait before a safe hop.
        self.ask_jev(obs, candidates)
        if self.pulse:
            p = self.pulse
            age = t - p["tick"]
            if obs.get("qbert_in_hop") or n != p["origin"]:
                p["moving"] = True
            if (p["moving"] and standing(obs) and age >= 5):
                if n != p["origin"]:
                    self.recent.append(n)
                self.ctx.log("hop_landed", tick=t, origin=p["origin"], node=n,
                             expected=p["destination"], duration=age)
                self.pulse = None
                self.neutral_until = t + 1
            elif not p["moving"] and age <= 2 and p["action"] in self.safe_edges(obs):
                return p["action"]
            elif age > 18 and standing(obs):
                self.pulse = None
                self.neutral_until = t + 1
            else:
                return "NOOP"
        if not standing(obs) or t < self.neutral_until:
            return "NOOP"
        if not candidates:
            return "NOOP"
        action = self.advised_action(obs, candidates)
        if action == "NOOP":
            if self.waiting_since is None:
                self.waiting_since = t
                self.ctx.log("tactical_wait", tick=t, node=n,
                             threats=list(self.signature()))
            return "NOOP"
        if action not in self.safe_edges(obs):
            return "NOOP"
        self.waiting_since = None
        destination = GRAPH[n][action]
        if any(c.get("rolling_shadow") for c in candidates):
            self.ctx.log("rolling_shadow_decision", tick=t, origin=n,
                         action=action, destination=destination,
                         chosen_shadow=next((c.get("rolling_shadow") for c in candidates
                                             if c["action"] == action), None),
                         options=[{"action": c["action"], "node": c["node"],
                                   "cost": c["cost"], "danger": c["danger"],
                                   "shadow": c.get("rolling_shadow")}
                                  for c in candidates])
        self.pulse = {"origin": n, "destination": destination, "action": action,
                      "tick": t, "moving": False}
        self.last_hop_tick = t
        if t - self.last_logged >= 20:
            self.ctx.log("coverage_status", tick=t, score=number(obs.get("score")),
                         lives=self.lives, inferred_clears=self.progress.clears,
                         unfinished=sum(x > 0 for x in self.progress.need),
                         node=n, action=action, threats=list(self.signature()))
            self.last_logged = t
        return action


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        controller = Controller(ctx)
        obs = ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        controller.update(obs)
        controller.ask_astra(obs, "initial durable policy")
        while not ctx.done and not game.done:
            obs = ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            controller.update(obs)
            action = controller.action(obs)
            ctx.act(action if action in DIRS else "NOOP")
            if controller.s2reason:
                controller.ask_astra(obs, controller.s2reason)
            elif (tick(obs) - controller.progress.last_progress > 150
                  and controller.s2calls < 3):
                controller.ask_astra(obs, "no scoring progress; reconsider Jev review criterion")
            ctx.wait_for_tick()
        ctx.log("candidate_summary", score=game.score,
                inferred_clears=controller.progress.clears,
                learned_goal=controller.progress.goal,
                style=controller.style, review=controller.review)


def build(ctx):
    return HarnessProgram(ctx)

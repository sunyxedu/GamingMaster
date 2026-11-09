"""Adaptive temporal-safety controller for the Atari 2600 Q*bert port.

The controller deliberately separates three things which the earlier controllers
mostly conflated: a board-progress perception model, a one-hop motor, and a
short-lived hazard/edge-failure model.  It never asks a model to invent a board
edge.  Model calls are advisory and off the real-time action clock.
"""

import json
import math
from collections import deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
DEFAULT_NAMES = {"NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"}


def make_graph():
    graph = {n: {} for n in range(21)}
    for row in range(6):
        first = row * (row + 1) // 2
        for col in range(row + 1):
            node = first + col
            if row:
                if col < row:
                    graph[node]["UP"] = (row - 1) * row // 2 + col
                if col:
                    graph[node]["LEFT"] = (row - 1) * row // 2 + col - 1
            if row < 5:
                child = (row + 1) * (row + 2) // 2
                graph[node]["DOWN"] = child + col
                graph[node]["RIGHT"] = child + col + 1
    return graph


GRAPH = make_graph()
ROW = {}
for _r in range(6):
    for _c in range(_r + 1):
        ROW[_r * (_r + 1) // 2 + _c] = _r


def make_distances():
    result = {}
    for start in GRAPH:
        result[start] = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in GRAPH[node].values():
                if nxt not in result[start]:
                    result[start][nxt] = result[start][node] + 1
                    queue.append(nxt)
    return result


DIST = make_distances()

# A spanning-tree-like sweep.  It is intentionally not the nearest-cube order
# used by the parent: when a life is lost the mirrored sweep changes the timing
# of enemy encounters without introducing seed-specific actions.
LEFT_SWEEP = (1, 3, 6, 10, 15, 16, 11, 7, 4, 2, 5, 8, 12, 17, 18, 13, 9, 14, 19, 20)
RIGHT_SWEEP = tuple(
    _r * (_r + 1) // 2 + (_r - _c)
    for _r, _c in ((ROW[n], n - ROW[n] * (ROW[n] + 1) // 2) for n in LEFT_SWEEP)
)
# The first-life order follows the study's top-region rule.  It avoids
# sending the opening nearest tie through the bottom corner before the upper
# branches have been cleared.  Losses still switch to the adaptive sweeps.
OPENING_ORDER = (1, 4, 7, 11, 6, 10, 16, 12, 17, 18, 13, 9, 14, 19, 20,
                 15, 3, 5, 8, 2, 0)
OPENING_RANK = {n: i for i, n in enumerate(OPENING_ORDER)}
LEFT_RANK = {n: i for i, n in enumerate(LEFT_SWEEP)}
RIGHT_RANK = {n: i for i, n in enumerate(RIGHT_SWEEP)}


def as_int(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def as_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def node_of(value):
    value = as_int(value)
    return value if value is not None and 0 <= value < 21 else None


def tick_of(observation):
    value = as_int(observation.get("tick", observation.get("ticks")), 0)
    return 0 if value is None else value


def clean_names(value, allowed=DIRECTIONS):
    if not isinstance(value, (list, tuple)):
        return []
    allowed = set(allowed)
    return [item for item in value if isinstance(item, str) and item in allowed]


def motion_of(observation):
    motion = observation.get("qbert_motion")
    if isinstance(motion, dict):
        return motion
    return {"phase": str(motion or "unknown")}


def standing(observation):
    n = node_of(observation.get("qbert_node"))
    motion = motion_of(observation)
    return (
        n is not None
        and not bool(observation.get("qbert_in_hop"))
        and str(motion.get("phase", "")).lower() not in ("hopping", "falling", "not_visible")
        and not bool(motion.get("falling"))
    )


def json_safe(value):
    """Keep every model request strict JSON, including during pre-flight."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def answer_value(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value


def answer_probability(answer, option):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return 0.0
    return max(0.0, min(1.0, as_float(probabilities.get(option), 0.0)))


def object_json(text):
    """Find a JSON object in a bounded Astra response; garbage is harmless."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text[:10000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError):
            continue
    return {}


def phase_values(observation):
    value = observation.get("cube_phase", observation.get("cube_state"))
    if not isinstance(value, list):
        return ["unknown"] * 21
    answer = []
    for item in value[:21]:
        text = str(item).upper()
        answer.append(text if text in ("S", "I", "D") else "unknown")
    return (answer + ["unknown"] * 21)[:21]


def colors_of(observation):
    value = observation.get("cube_colors")
    if not isinstance(value, list) or len(value) != 21:
        return None
    answer = []
    for item in value:
        number = as_int(item)
        if number is None:
            return None
        answer.append(number)
    return tuple(answer)


def needed_of(observation):
    value = observation.get("needed_landings")
    if not isinstance(value, list) or len(value) != 21:
        return None
    answer = []
    for item in value:
        number = as_int(item, -1)
        answer.append(number if number >= 0 else None)
    return answer


def hazard_facts(observation):
    """Return small semantic hazard records, never raw RAM or image data."""
    value = observation.get("hazards")
    if not isinstance(value, list):
        return []
    answer = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("lethal", False)):
            continue
        answer.append({
            "kind": str(item.get("kind", "hazard")),
            "node": node_of(item.get("estimated_node", item.get("node"))),
            "risk": str(item.get("risk", item.get("risk_label", "unknown"))),
            "distance": str(item.get("distance", item.get("pixel_distance", "unknown"))),
            "dx": as_float(item.get("dx"), 0.0),
            "dy": as_float(item.get("dy"), 0.0),
        })
    return answer[:8]


class BoardModel:
    """Learn board-local destination colours and quarantine the clear flash."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.last_score = None
        self.goal_color = None
        self.need = [1] * 21
        self.epoch = 0
        self.mode = "play"
        self.mode_started = 0
        self.flash_last = -1
        self.flash_score = 0.0
        self.complete_since = None
        self.last_progress = 0
        self.last_event_key = None
        self.evidence = deque(maxlen=8)

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def update(self, observation):
        t = tick_of(observation)
        colors = colors_of(observation)
        score = as_float(observation.get("score"), self.last_score or 0.0)
        delta = 0.0 if self.last_score is None else score - self.last_score
        changed = []
        if self.previous is not None and colors is not None:
            changed = [i for i in range(21) if colors[i] != self.previous[i]]

        # The port's all-cube award is a visual transition, not 21 ordinary
        # landings.  Keep this state independent of the adaptor's phase labels.
        if len(changed) >= 16:
            if self.mode != "flash":
                self.mode = "flash"
                self.mode_started = t
                self.flash_score = score - delta
                self.log("temporal_flash_begin", tick=t, score=score,
                         epoch=self.epoch, reported_rounds=observation.get("completed_rounds"))
            self.flash_last = t
            self.complete_since = None

        if self.mode == "flash":
            quiet = t - self.flash_last
            at_apex = node_of(observation.get("qbert_node")) == 0
            if quiet >= 4 and (at_apex or t - self.mode_started >= 55):
                tally = score - self.flash_score
                self.epoch += 1
                self.mode = "play"
                self.goal_color = None
                self.need = [1] * 21
                self.complete_since = None
                self.last_progress = t
                self.log("board_epoch_started", tick=t, epoch=self.epoch,
                         tally=round(tally, 2), inferred_clear=tally >= 1000,
                         start_color=(colors[0] if colors else None))
            self.previous = colors if colors is not None else self.previous
            self.last_score = score
            return

        if colors is not None:
            # A scored local colour transition identifies the current board's
            # destination without assuming fixed Atari palette numbers.
            if self.goal_color is None and delta > 0 and int(round(delta)) % 100 == 25:
                qnode = node_of(observation.get("qbert_node"))
                candidates = [i for i in changed if 0 <= i < 21]
                selected = qnode if qnode in candidates else (candidates[0] if len(candidates) == 1 else None)
                if selected is not None:
                    self.goal_color = colors[selected]
                    self.evidence.append({
                        "tick": t, "node": selected, "color": self.goal_color,
                        "reward": delta, "decoded_phase": phase_values(observation)[selected],
                    })
                    self.log("destination_calibrated", tick=t, node=selected,
                             color=self.goal_color, reward=delta, epoch=self.epoch)
            if self.goal_color is not None:
                self.need = [0 if value == self.goal_color else 1 for value in colors]
            else:
                observed_need = needed_of(observation)
                # Immediately after a whole-board flash, the adaptor can still
                # expose the previous board's all-zero vector.  Keep the
                # freshly armed epoch alive until a local scored landing
                # calibrates its destination; never turn that stale zero into
                # an empty target set.
                if observed_need is not None and any(
                        value is not None and value > 0 for value in observed_need):
                    self.need = [1 if value is not None and value > 0 else 0
                                 for value in observed_need]
                elif not any(self.need):
                    self.need = [1] * 21

        # A Sam reversion or a multi-stage colour cycle naturally reopens a
        # target because need is recomputed from the current colour.  A stable
        # all-destination view starts a handoff; it must not become a one-cube
        # route during the next board's flashing palette.
        if self.goal_color is not None and not any(self.need):
            if self.complete_since is None:
                self.complete_since = t
                self.mode = "handoff"
                self.mode_started = t
                self.log("handoff_armed", tick=t, epoch=self.epoch)
        elif self.mode == "handoff":
            self.mode = "play"
            self.mode_started = t

        if delta > 0 or changed:
            self.last_progress = t
        self.previous = colors if colors is not None else self.previous
        self.last_score = score

    def stalled_reopen(self, t):
        """Bounded recovery if a decoder suppresses the palette flash."""
        if self.mode == "handoff" and self.complete_since is not None and t - self.complete_since > 60:
            self.mode = "play"
            self.goal_color = None
            self.need = [1] * 21
            self.complete_since = None
            self.epoch += 1
            self.log("handoff_recovery", tick=t, epoch=self.epoch)
        elif self.mode == "play" and self.goal_color is not None and not any(self.need):
            if self.complete_since is not None and t - self.complete_since > 28:
                self.goal_color = None
                self.need = [1] * 21
                self.complete_since = None
                self.log("second_pass_recovery", tick=t, epoch=self.epoch)

    def unfinished(self):
        return [i for i, value in enumerate(self.need) if value > 0]


class RiskMemory:
    """Predict short enemy shadows and remember edges that actually killed."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.visible = {}
        self.edge_failures = {}
        self.bad_nodes = {}
        self.last_lives = None
        self.losses = 0
        self.recent_launches = deque(maxlen=12)
        self.recent_landings = deque(maxlen=8)

    def note_launch(self, launch):
        if isinstance(launch, dict):
            self.recent_launches.append(dict(launch))

    def note_landing(self, landing):
        if isinstance(landing, dict):
            self.recent_landings.append(dict(landing))

    def update(self, observation, t):
        for fact in hazard_facts(observation):
            kind = fact["kind"]
            if fact["node"] is None:
                continue
            old = self.visible.get(kind)
            dx = fact["dx"]
            dy = fact["dy"]
            if old is not None and abs(dx) < 0.01 and abs(dy) < 0.01:
                dx, dy = old.get("dx", 0.0), old.get("dy", 0.0)
            self.visible[kind] = {
                "kind": kind, "node": fact["node"], "risk": fact["risk"],
                "distance": fact["distance"], "dx": dx, "dy": dy, "tick": t,
            }
        self.visible = {
            key: value for key, value in self.visible.items()
            if t - value.get("tick", t) <= 16
        }
        self.bad_nodes = {
            key: until for key, until in self.bad_nodes.items() if until >= t
        }
        new_lives = as_int(observation.get("lives"), self.last_lives)
        loss = self.last_lives is not None and new_lives is not None and new_lives < self.last_lives
        if loss:
            self.losses += self.last_lives - new_lives
            self._attribute_loss(t)
        if new_lives is not None:
            self.last_lives = new_lives
        return bool(loss)

    def _attribute_loss(self, t):
        # A contact is often reported after the sprite has disappeared.  The
        # most recent launched edge is therefore evidence, not just the latest
        # observation's qbert_node.
        chosen = None
        for item in reversed(self.recent_launches):
            age = t - as_int(item.get("tick"), t)
            if 0 <= age <= 22:
                chosen = item
                break
        if chosen is None and self.recent_landings:
            chosen = self.recent_landings[-1]
        if chosen is None:
            return
        origin = node_of(chosen.get("origin"))
        dest = node_of(chosen.get("destination", chosen.get("dest")))
        action = chosen.get("action")
        if origin is None or dest is None:
            return
        edge = (origin, dest)
        self.edge_failures[edge] = self.edge_failures.get(edge, 0) + 1
        # Avoiding only the exact edge is too weak when the enemy is invisible;
        # a short node shadow changes the next life without permanently banning
        # a needed cube.
        self.bad_nodes[dest] = t + 110
        self.bad_nodes[origin] = max(self.bad_nodes.get(origin, 0), t + 35)
        try:
            self.ctx.log("edge_failure_learned", tick=t, origin=origin,
                         destination=dest, action=action,
                         failures=self.edge_failures[edge])
        except Exception:
            pass

    def recent_hazard(self, t):
        return any(t - value.get("tick", t) <= 7 for value in self.visible.values())

    def signature(self, t):
        return tuple(sorted((key, value.get("node"), max(0, t - value.get("tick", t)))
                            for key, value in self.visible.items()))

    def danger(self, origin, dest, action, t):
        value = 0.0
        if (origin, dest) in self.edge_failures:
            value += 2.5 + 3.0 * self.edge_failures[(origin, dest)]
        if self.bad_nodes.get(dest, 0) >= t:
            value += 3.0
        for hazard in self.visible.values():
            age = max(0, t - hazard.get("tick", t))
            decay = max(0.15, 1.0 - age / 18.0)
            hn = node_of(hazard.get("node"))
            if hn is None:
                continue
            distance = DIST.get(dest, {}).get(hn, 99)
            kind = hazard.get("kind", "hazard")
            if dest == hn:
                value += 12.0 * decay
            elif distance == 1:
                value += 2.4 * decay
            elif distance == 2:
                value += 0.7 * decay
            if kind in ("purple_ball", "red_ball"):
                children = [n for edge, n in GRAPH[hn].items() if edge in ("DOWN", "RIGHT")]
                if dest in children:
                    value += 4.2 * decay
                grandchildren = set()
                for child in children:
                    grandchildren.update(GRAPH[child].values())
                if dest in grandchildren:
                    value += 1.0 * decay
            elif kind == "coily":
                # Coily follows rather than simply rolling downhill.  Keep a
                # larger shadow around the branch containing Q*bert.
                if distance == 1:
                    value += 3.0 * decay
                if DIST.get(hn, {}).get(origin, 99) <= 2 and distance <= 2:
                    value += 1.8 * decay
            if str(hazard.get("risk", "")).lower() in ("near", "imminent", "occupied"):
                value += 1.0 * decay
        return value


class RoutePlanner:
    """Choose a useful target while changing route timing after a death."""

    def __init__(self, board, risks, ctx):
        self.board = board
        self.risks = risks
        self.ctx = ctx
        # Match the validated nearest-coverage baseline for the first life;
        # only a witnessed loss changes the route timing.
        self.style = "nearest"
        self.target = None
        self.waypoint = None
        self.last_epoch = -1
        self.recent_nodes = deque(maxlen=16)
        self.waiting_since = None

    def reset_epoch(self):
        self.target = None
        self.waypoint = None
        self.waiting_since = None
        self.last_epoch = self.board.epoch

    def on_life_loss(self, t):
        # Each alternative is a structural perturbation, not a seed route.
        if self.risks.losses % 3 == 1:
            self.style = "right_sweep"
        elif self.risks.losses % 3 == 2:
            self.style = "center_escape"
        else:
            self.style = "nearest"
        self.target = None
        self.waypoint = None
        self.waiting_since = t
        try:
            self.ctx.log("route_mode_after_loss", tick=t, losses=self.risks.losses,
                         style=self.style)
        except Exception:
            pass

    def set_style(self, style):
        if style in ("opening", "left_sweep", "right_sweep", "center_escape", "nearest"):
            self.style = style
            self.target = None
            self.waypoint = None

    def rank(self, node):
        if self.style == "opening":
            return OPENING_RANK.get(node, 99)
        if self.style == "right_sweep":
            return RIGHT_RANK.get(node, 99)
        if self.style == "center_escape":
            return abs(ROW.get(node, 0) - 3) * 4 + abs((node - ROW.get(node, 0) * (ROW.get(node, 0) + 1) // 2) - ROW.get(node, 0) / 2)
        if self.style == "nearest":
            return 0
        return LEFT_RANK.get(node, 99)

    def choose_target(self, origin):
        unfinished = set(self.board.unfinished())
        if not unfinished or origin is None:
            return None
        if self.target in unfinished and self.target != origin:
            return self.target
        if self.target not in unfinished:
            self.target = None
            self.waypoint = None

        candidates = [n for n in unfinished if n != origin]
        if not candidates:
            self.target = origin
            return origin
        # Distance dominates, but the post-loss sweep is enough to choose a
        # different equally-short branch and avoid repeating a lethal rhythm.
        candidates.sort(key=lambda n: (
            DIST[origin].get(n, 99) * (1.0 if self.style == "nearest" else 1.12),
            self.rank(n),
            ROW.get(n, 9),
            n,
        ))
        self.target = candidates[0]
        self.waypoint = None
        return self.target

    def detour(self, origin, safe, t):
        options = []
        for action in safe:
            dest = GRAPH.get(origin, {}).get(action)
            if dest is None:
                continue
            danger = self.risks.danger(origin, dest, action, t)
            degree = len(GRAPH.get(dest, {}))
            options.append((danger * 4 - degree * 0.3, self.rank(dest), action, dest))
        if not options:
            return None
        options.sort()
        return options[0][3]

    def candidates(self, observation, t):
        origin = node_of(observation.get("qbert_node"))
        if origin is None:
            return []
        safe = clean_names(observation.get("safe_moves"))
        if not safe:
            return []
        if self.last_epoch != self.board.epoch:
            self.reset_epoch()
        target = self.choose_target(origin)
        if target is None:
            return []
        if target == origin:
            if self.waypoint is None:
                self.waypoint = self.detour(origin, safe, t)
            effective = self.waypoint if self.waypoint is not None else target
        else:
            effective = self.waypoint if self.waypoint is not None and origin != self.waypoint else target
            if self.waypoint is not None and origin == self.waypoint:
                self.waypoint = None
                effective = target

        targets = set(self.board.unfinished())
        result = []
        for action in safe:
            dest = GRAPH.get(origin, {}).get(action)
            if dest is None:
                continue
            route = DIST[dest].get(effective, 99)
            useful = dest in targets
            danger = self.risks.danger(origin, dest, action, t)
            exits = [self.risks.danger(dest, nxt, edge, t) for edge, nxt in GRAPH[dest].items()]
            clear_exits = sum(value < 2.0 for value in exits)
            backtrack = 1.0 if self.recent_nodes and dest == self.recent_nodes[-1] else 0.0
            dead_end = 1.2 if len(GRAPH[dest]) <= 2 and not useful else 0.0
            score = (
                route * 2.4 + danger * 8.0 + (0.0 if useful else 0.8)
                + backtrack * 0.7 + dead_end - clear_exits * 0.18
                + self.rank(dest) * 0.035
            )
            result.append({
                "action": action, "destination": dest, "cost": score,
                "danger": danger, "useful": useful, "route": route,
                "escape_space": "multiple" if clear_exits >= 2 else "narrow",
                "landing": "unfinished" if useful else "transit",
            })
        result.sort(key=lambda item: (item["cost"], DIRECTIONS.index(item["action"])))

        # A visible or recently visible descending hazard can make every edge
        # look equally bad.  One bounded neutral observation is safer than
        # committing to a predicted intersection; this is not an unbounded idle
        # policy, and never waits when Coily is already close.
        if result:
            coily_close = any(
                h.get("kind") == "coily" and DIST.get(origin, {}).get(node_of(h.get("node")), 99) <= 2
                for h in self.risks.visible.values()
            )
            has_ball = any(h.get("kind") in ("purple_ball", "red_ball") for h in self.risks.visible.values())
            current_danger = min((self.risks.danger(origin, nxt, edge, t)
                                  for edge, nxt in GRAPH[origin].items()), default=0.0)
            if has_ball and not coily_close and result[0]["danger"] >= 1.7 and current_danger + 0.35 < result[0]["danger"]:
                age = t - self.waiting_since if self.waiting_since is not None else 0
                if age < 12:
                    result.insert(0, {
                        "action": "NOOP", "destination": origin, "cost": result[0]["cost"] - 0.5,
                        "danger": current_danger, "useful": False, "route": 0,
                        "escape_space": "wait", "landing": "bounded_wait",
                    })
        return result


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.board = BoardModel(ctx)
        self.risks = RiskMemory(ctx)
        self.planner = RoutePlanner(self.board, self.risks, ctx)
        self.names = set(DEFAULT_NAMES)
        try:
            self.names = {str(value) for value in game.action_spec.names}
        except Exception:
            pass
        self.noop = "NOOP"
        try:
            candidate = str(getattr(game.action_spec, "noop", "NOOP"))
            if candidate in self.names:
                self.noop = candidate
        except Exception:
            pass
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else (next(iter(self.names)) if self.names else "NOOP")

        self.pulse = None
        self.last_launch = None
        self.hold_until = 0
        self.life_epoch = 0
        self.pending_jev = None
        self.jev_meta = None
        self.advice = None
        self.last_jev = -100
        self.pending_astra = None
        self.astra_session = None
        self.astra_calls = 0
        self.last_astra = -1000
        self.astra_style = None
        self.dwell_long = False
        self.last_observation = {}

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def start_astra(self, observation, reason):
        if self.pending_astra is not None or self.astra_calls >= 2 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if remaining.get("system2_calls", 1) <= 0:
                return
        except Exception:
            pass
        t = tick_of(observation)
        if t - self.last_astra < 300 and reason != "life_loss":
            return
        state = json_safe({
            "reason": reason,
            "score": as_float(observation.get("score")),
            "lives": as_int(observation.get("lives")),
            "board_epoch": self.board.epoch,
            "unfinished": self.board.unfinished(),
            "recent_hazards": hazard_facts(observation),
            "edge_failures": [[a, b, count] for (a, b), count in self.risks.edge_failures.items()],
            "route_style": self.planner.style,
            "dwell": "long" if self.dwell_long else "short",
        })
        prompt = (
            "You are an off-clock strategy auditor for real-time Atari 2600 Q*bert. "
            "The board is a six-row 21-node graph. Code owns the graph, palette learning, "
            "enemy shadow, edge-failure memory, legality, and the one-hop motor. It will "
            "never execute a raw action or FIRE from your answer. The controller's "
            "specific hypothesis is that a late collision can occur after a hazard vanishes "
            "from the decoded observation; after a life loss it changes sweep direction and "
            "penalizes the failed edge. Choose a durable policy only: style is one of "
            "left_sweep, right_sweep, center_escape, nearest; dwell is short or long. "
            "Long dwell means one or two extra neutral ticks after a confirmed landing, "
            "not indefinite waiting. Survival and useful cube coverage outrank bonuses. "
            "Return only JSON under 40 tokens, for example "
            '{"style":"center_escape","dwell":"long"}. Evidence: ' + json.dumps(state)
        )
        self.astra_calls += 1
        self.last_astra = t

        def work():
            try:
                result = self.ctx.think(prompt, session_id=self.astra_session, timeout_s=105.0)
                return {"result": result, "error": None}
            except Exception as exc:
                return {"result": None, "error": type(exc).__name__}

        try:
            self.pending_astra = self.ctx.spawn(work)
            self.log("astra_policy_requested", tick=t, reason=reason)
        except Exception as exc:
            self.pending_astra = None
            self.log("astra_policy_spawn_failed", error=type(exc).__name__)

    def collect_astra(self, observation):
        future = self.pending_astra
        if future is None or not future.done():
            return
        self.pending_astra = None
        try:
            result = future.result()
            if not isinstance(result, dict):
                return
            model_result = result.get("result")
            if model_result is None:
                return
            self.astra_session = getattr(model_result, "session_id", None)
            data = object_json(getattr(model_result, "text", ""))
            old = (self.astra_style, self.dwell_long)
            if data.get("style") in ("left_sweep", "right_sweep", "center_escape", "nearest"):
                self.astra_style = data["style"]
                if self.risks.losses == 0:
                    self.planner.set_style(self.astra_style)
            if data.get("dwell") in ("short", "long"):
                self.dwell_long = data["dwell"] == "long"
            new = (self.astra_style, self.dwell_long)
            if new != old:
                self.log("astra_policy_applied", tick=tick_of(observation), old=old, new=new)
        except Exception as exc:
            self.log("astra_policy_failed", error=type(exc).__name__)

    def compact_state(self, observation, candidates):
        t = tick_of(observation)
        described = []
        for item in candidates:
            described.append({
                "action": item["action"],
                "landing": item["landing"],
                "danger": "high" if item["danger"] >= 5 else "possible" if item["danger"] >= 1.2 else "clear",
                "escape_space": item["escape_space"],
                "route": "adjacent" if item["route"] <= 1 else "near" if item["route"] <= 3 else "far",
                "useful": bool(item["useful"]),
                "code_preferred": item is candidates[0],
            })
        return json_safe({
            "game": "Atari 2600 Q*bert; 21 cubes; four joystick edges; never FIRE",
            "clock": {"tick": t, "observation_age": 0},
            "position": {
                "node": node_of(observation.get("qbert_node")),
                "motion": motion_of(observation).get("phase", "unknown"),
                "safe_moves": clean_names(observation.get("safe_moves")),
            },
            "progress": {
                "board_epoch": self.board.epoch,
                "unfinished": self.board.unfinished(),
                "target": self.planner.target,
                "route_style": self.planner.style,
            },
            "candidates": described,
            "hazards": [{"kind": h["kind"], "node": h["node"], "risk": h["risk"],
                         "age": max(0, t - self.risks.visible.get(h["kind"], {}).get("tick", t))}
                        for h in hazard_facts(observation)],
            "recent_edge_failures": [[a, b, count] for (a, b), count in self.risks.edge_failures.items()],
            "proposed": candidates[0]["action"] if candidates else self.noop,
        })

    def start_jev(self, observation, candidates):
        if self.pending_jev is not None or not candidates or self.ctx.done:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        t = tick_of(observation)
        if t - self.last_jev < 6 or not standing(observation):
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 1) <= 1:
                return
        except Exception:
            pass
        options = []
        for item in candidates:
            action = item.get("action")
            if action not in options and (action in DIRECTIONS or action == "NOOP"):
                options.append(action)
        if not options:
            return
        if "unknown" not in options:
            options.append("unknown")
        first = list(options)
        # Put the deterministic proposal in a different position in the two
        # questions; the bridge requires agreement and does not trust ordering.
        proposed = candidates[0].get("action")
        if proposed in first:
            first.remove(proposed)
            first.insert(0, proposed)
        second = list(reversed(first))
        criteria_a = {}
        criteria_b = {}
        for action in first:
            criteria_a[action] = (
                "Post this supplied certified edge only if its candidate facts are safe; "
                "NOOP means a bounded wait and unknown means defer to code."
            )
        for action in second:
            criteria_b[action] = (
                "Choose this supplied option from the current candidates, never invent an edge "
                "and never press FIRE."
            )
        instruction = (
            "Choose the safest useful immediate option from candidates. A high danger or narrow "
            "escape is worse than clear; when close, prefer code_preferred and useful coverage. "
            "Do not calculate distances or infer an unavailable edge. Return unknown if evidence "
            "does not support changing the deterministic proposal."
        )
        state = self.compact_state(observation, candidates)
        questions = {
            "choice_a": Choice(instruction, criteria_a),
            "choice_b": Choice(instruction, criteria_b),
            "danger": Noul(
                "Is the proposed option exposed to a concrete immediate lethal contact or edge fall "
                "within this hop? Distant or stale visibility alone is false.",
                {"true": "The proposed landing or its immediate branch is occupied or concretely threatened.",
                 "false": "The supplied facts show a clear or only distant/uncertain branch."},
            ),
        }
        self.last_jev = t
        self.jev_meta = {
            "tick": t, "node": node_of(observation.get("qbert_node")),
            "epoch": self.board.epoch, "life": self.life_epoch,
            "hazards": self.risks.signature(t), "proposed": proposed,
            "candidate_costs": {item["action"]: item["cost"] for item in candidates},
        }

        def work():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.8), "error": None}
            except Exception as exc:
                return {"answers": None, "error": type(exc).__name__}

        try:
            self.pending_jev = self.ctx.spawn(work)
        except Exception as exc:
            self.pending_jev = None
            self.jev_meta = None
            self.log("jev_spawn_failed", error=type(exc).__name__)

    def collect_jev(self, observation):
        future = self.pending_jev
        meta = self.jev_meta
        if future is None or not future.done():
            return
        self.pending_jev = None
        self.jev_meta = None
        try:
            result = future.result()
            answers = result.get("answers") if isinstance(result, dict) else None
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            a = answers.get("choice_a")
            b = answers.get("choice_b")
            first = answer_value(a)
            second = answer_value(b)
            danger = as_float(answer_value(answers.get("danger")), 0.0)
            if first != second or first not in DIRECTIONS + ("NOOP",):
                return
            probability = min(answer_probability(a, first), answer_probability(b, second))
            if probability < 0.62:
                return
            t = tick_of(observation)
            if t - as_int(meta.get("tick"), t) > 5:
                return
            self.advice = dict(meta, action=first, probability=probability, danger=danger)
            self.log("jev_advice_ready", tick=t, action=first, probability=round(probability, 3))
        except Exception as exc:
            self.log("jev_result_failed", error=type(exc).__name__)

    def use_advice(self, observation, candidates):
        fallback = candidates[0]
        advice = self.advice
        self.advice = None
        if not isinstance(advice, dict):
            return fallback["action"]
        t = tick_of(observation)
        if (advice.get("node") != node_of(observation.get("qbert_node"))
                or advice.get("epoch") != self.board.epoch
                or advice.get("life") != self.life_epoch
                or t - as_int(advice.get("tick"), t) < 0
                or t - as_int(advice.get("tick"), t) > 5
                or advice.get("hazards") != self.risks.signature(t)):
            return fallback["action"]
        by_action = {item["action"]: item for item in candidates}
        chosen = by_action.get(advice.get("action"))
        if chosen is None:
            return fallback["action"]
        if chosen["action"] == "NOOP":
            if chosen["danger"] <= fallback["danger"] + 0.2 and fallback["danger"] >= 1.2:
                self.log("jev_wait_accepted", tick=t)
                return "NOOP"
            return fallback["action"]
        if (chosen["cost"] <= fallback["cost"] + 1.8
                and chosen["danger"] <= fallback["danger"] + 0.7):
            if chosen["action"] != fallback["action"]:
                self.log("jev_route_change", tick=t, old=fallback["action"], new=chosen["action"])
            if as_float(advice.get("danger"), 0.0) >= 0.84 and len(candidates) > 1:
                return candidates[1]["action"]
            return chosen["action"]
        return fallback["action"]

    def update(self, observation):
        t = tick_of(observation)
        before_epoch = self.board.epoch
        self.board.update(observation)
        self.board.stalled_reopen(t)
        if self.board.epoch != before_epoch:
            self.planner.reset_epoch()
            self.pulse = None
            self.advice = None
            self.hold_until = max(self.hold_until, t + 2)
        loss = self.risks.update(observation, t)
        if loss:
            self.life_epoch += 1
            self.planner.on_life_loss(t)
            self.pulse = None
            self.advice = None
            self.hold_until = max(self.hold_until, t + 5)
            self.start_astra(observation, "life_loss")
        self.last_observation = observation

    def action(self, observation):
        t = tick_of(observation)
        self.collect_jev(observation)
        self.collect_astra(observation)
        if self.board.mode in ("flash", "handoff"):
            self.pulse = None
            self.advice = None
            return self.noop
        if not standing(observation):
            return self.noop

        current = node_of(observation.get("qbert_node"))
        safe = set(clean_names(observation.get("safe_moves")))
        if current is None or not safe:
            return self.noop

        if self.pulse is not None:
            pulse = self.pulse
            age = t - as_int(pulse.get("tick"), t)
            now_node = current
            if now_node != pulse.get("origin") or bool(observation.get("qbert_in_hop")):
                pulse["departed"] = True
            if (standing(observation) and now_node == pulse.get("destination") and age >= 4):
                self.risks.note_landing({"tick": t, **pulse})
                self.planner.recent_nodes.append(now_node)
                self.pulse = None
                # Temporal deconfliction is the central experiment: a short
                # confirmed landing dwell, longer on lower rows or after loss.
                dwell = 2 if ROW.get(now_node, 0) >= 3 else 1
                if self.dwell_long:
                    dwell += 1
                if self.risks.recent_hazard(t):
                    dwell += 1
                self.hold_until = max(self.hold_until, t + dwell)
                return self.noop
            if standing(observation) and not pulse.get("departed") and age >= 3:
                # Sticky input did not launch a hop.  Release and retry from a
                # fresh observation rather than holding a lethal edge.
                self.pulse = None
                self.hold_until = max(self.hold_until, t + 1)
                return self.noop
            if age > 22:
                self.pulse = None
                self.hold_until = max(self.hold_until, t + 1)
                return self.noop
            # A direction is held for at most two ticks; all remaining hop
            # animation receives neutral input.
            if not pulse.get("departed") and age <= 2 and pulse.get("action") in safe:
                return pulse["action"]
            return self.noop

        if t < self.hold_until:
            return self.noop

        candidates = self.planner.candidates(observation, t)
        if not candidates:
            return self.noop
        self.start_jev(observation, candidates)
        action = self.use_advice(observation, candidates)
        if action == "NOOP":
            self.planner.waiting_since = self.planner.waiting_since if self.planner.waiting_since is not None else t
            return self.noop
        if action not in safe or action not in GRAPH.get(current, {}):
            action = candidates[0]["action"]
        if action not in safe or action == "NOOP":
            return self.noop

        destination = GRAPH[current][action]
        pulse = {
            "origin": current, "destination": destination, "action": action,
            "tick": t, "departed": False,
        }
        self.pulse = pulse
        self.last_launch = dict(pulse)
        self.risks.note_launch(pulse)
        return action

    def run(self, game):
        self.start_astra({}, "initial policy")
        while not self.ctx.done and not getattr(game, "done", False):
            try:
                observation = self.ctx.observe()
            except Exception:
                return
            if not isinstance(observation, dict):
                observation = {}
            self.update(observation)
            action = self.action(observation)
            if action not in self.names:
                action = self.noop
            try:
                self.ctx.act(action)
            except Exception:
                return
            try:
                self.ctx.wait_for_tick()
            except Exception:
                return
        self.log("controller_summary", score=getattr(game, "score", None),
                 ticks=getattr(game, "ticks", None), board_epoch=self.board.epoch,
                 life_losses=self.risks.losses, edge_failures=self.risks.edge_failures,
                 route_style=self.planner.style)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run(game)


def build(ctx):
    return HarnessProgram(ctx)

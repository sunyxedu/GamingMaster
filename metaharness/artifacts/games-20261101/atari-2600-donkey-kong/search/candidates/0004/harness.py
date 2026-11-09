"""Asynchronous hierarchical controller for Atari 2600 Donkey Kong."""

import json
import math
import time

from metaharness.models.jev import Choice, Noul


ACTIONS = {
    "NOOP": "noop",
    "FIRE": "fire",
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "LEFTFIRE": "leftfire",
    "RIGHTFIRE": "rightfire",
    "UPFIRE": "upfire",
    "DOWNFIRE": "downfire",
}

LADDER_PRIOR = [
    (108, 176), (48, 148), (80, 148), (88, 120), (108, 120),
    (48, 92), (68, 92), (108, 64), (76, 40), (32, 40),
]

SAFE_ACTIONS = ("noop", "left", "right", "up", "down",
                "leftfire", "rightfire", "upfire")


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def center(obj):
    if not isinstance(obj, dict):
        return (0.0, 0.0)
    x = number(obj.get("x", obj.get("left", 0)))
    y = number(obj.get("y", obj.get("top", 0)))
    w = number(obj.get("w", obj.get("width", 0)))
    h = number(obj.get("h", obj.get("height", 0)))
    return x + w / 2, y + h / 2


def visible(obj):
    return isinstance(obj, dict) and obj.get("visible", True) is not False


def compact_observation(obs):
    if not isinstance(obs, dict):
        return {}

    player = obs.get("player", {})
    px, py = center(player)

    hazards = []
    for h in obs.get("hazards", []) or []:
        if visible(h):
            hx, hy = center(h)
            hazards.append({
                "x": round(hx, 1),
                "y": round(hy, 1),
                "dx": round(number(h.get("dx")), 2),
                "dy": round(number(h.get("dy")), 2),
                "slot": h.get("slot", -1),
            })

    ladders = []
    for i, ladder in enumerate(obs.get("ladders", []) or []):
        if visible(ladder):
            lx, ly = center(ladder)
            ladders.append({
                "x": round(lx, 1),
                "y": round(ly, 1),
                "slot": i,
            })

    return {
        "stage": obs.get("stage", "unknown"),
        "frame": obs.get("frame", 0),
        "tick": obs.get("tick", 0),
        "player": {
            "x": round(px, 1),
            "y": round(py, 1),
            "dx": round(number(player.get("dx")), 2),
            "dy": round(number(player.get("dy")), 2),
            "visible": visible(player),
        },
        "hazards": hazards,
        "ladders": ladders,
        "hammer": obs.get("hammer"),
        "rivet_rows": obs.get("rivet_rows", []),
        "rivets_removed": obs.get("rivets_removed", 0),
        "bonus": obs.get("current_bonus", 0),
        "score": obs.get("score_total", obs.get("episode_return", 0)),
        "lives": obs.get("lives_remaining", 0),
        "screens_completed": obs.get("screens_completed", 0),
        "terminated": bool(obs.get("terminated", False)),
        "truncated": bool(obs.get("truncated", False)),
        "last_action_error": obs.get("last_action_error"),
    }


def rivet_mask(rows):
    result = []
    if not isinstance(rows, (list, tuple)):
        return result

    for row in rows[:4]:
        try:
            value = int(row)
        except (TypeError, ValueError):
            result.append(0)
            continue
        result.append(value if value in (0, 6, 12, 18) else 0)

    return result


class Controller:

    def __init__(self, ctx):
        self.ctx = ctx
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0
        self.target_x = None
        self.target_y = None
        self.plan = "Use the safest route that makes upward progress."
        self.plan_version = 0
        self.last_astra_tick = -10000
        self.j_batch = None
        self.a_batch = None
        self.j_result = None
        self.j_result_tick = -1
        self.j_result_stage = None
        self.last_jev_tick = -10000
        self.ticks = 0
        self.active_jump = 0
        self.route_direction = 1
        self.last_rivets = None
        self.previous_player = None
        self.recent_deaths = 0

    def log(self, event, **kwargs):
        try:
            self.ctx.log(event, **kwargs)
        except Exception:
            pass

    def start_astra(self, state, reason):
        if self.ctx.done or self.a_batch is not None:
            return

        if self.ctx.budget.remaining()["system2_calls"] <= 0:
            return

        prompt = """
You are the strategic planner for Atari 2600 Donkey Kong, not the
arcade game. The cartridge alternates a ramp/barrel screen and a
rivet/fireball screen. There are no arcade conveyor or pie-factory
boards.

The controller already executes deterministic movement and safety
rules. Your task is to provide a SHORT strategic policy, not an
action sequence. Prioritize completing screens, preserving lives,
and retaining the remaining screen bonus. On the rivet screen,
clear remaining rivets bottom-to-top in a serpentine route and
jump over cleared gaps. On the ramp screen, use the fixed ladder
topology and alternate ramp directions. Take a hammer only when
it is nearly free and does not obstruct climbing.

Return a compact JSON object with:
{"ramp_policy":"...","rivet_policy":"...",
 "risk":"conservative|balanced","replan":"..."}

Do not assume a level from the seed. Base your response on this
observation and distinguish facts from uncertain interpretations.
Keep the complete response under 120 words.

Reason for this planning call: """ + reason + """

Current state:
""" + json.dumps(state, separators=(",", ":"), default=str)

        def work():
            return self.ctx.think(
                prompt,
                timeout_s=100.0,
            )

        self.a_batch = self.ctx.spawn(work)
        self.last_astra_tick = self.ticks

    def collect_astra(self):
        future = self.a_batch
        if future is None or not future.done():
            return

        self.a_batch = None
        try:
            result = future.result()
            raw = getattr(result, "text", "")
            if not isinstance(raw, str):
                return

            parsed = None
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(raw[start:end + 1])
                    except (ValueError, TypeError):
                        pass

            if isinstance(parsed, dict):
                ramp = str(parsed.get("ramp_policy", ""))[:350]
                rivet = str(parsed.get("rivet_policy", ""))[:350]
                risk = str(parsed.get("risk", "balanced"))
                self.plan = (
                    "Ramp: " + ramp + " Rivet: " + rivet +
                    " Risk: " + risk
                )
                self.plan_version += 1
                self.log("astra_plan_updated",
                         version=self.plan_version,
                         risk=risk)
            elif raw.strip():
                self.plan = raw.strip()[:500]
                self.plan_version += 1
                self.log("astra_unstructured_plan")
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    def nearest_ladder(self, state):
        p = state["player"]
        px, py = p["x"], p["y"]
        candidates = []

        for ladder in state["ladders"]:
            lx, ly = ladder["x"], ladder["y"]
            if ly >= py - 5:
                continue
            if abs(ly - py) > 65:
                continue

            score = abs(lx - px) + abs(ly - py) * 0.30
            candidates.append((score, lx, ly))

        if not candidates:
            for lx, ly in LADDER_PRIOR:
                if ly < py - 5:
                    candidates.append(
                        (abs(lx - px) + abs(ly - py) * 0.30, lx, ly)
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], candidates[0][2]

    def hazard_assessment(self, state):
        p = state["player"]
        px, py = p["x"], p["y"]

        nearest = None
        for h in state["hazards"]:
            hx, hy = h["x"], h["y"]
            same_row = abs(hy - py) < 19
            relative = (hx - px) * h["dx"]

            if not same_row:
                continue

            distance = abs(hx - px)
            closing = (
                (hx > px and h["dx"] < 0) or
                (hx < px and h["dx"] > 0) or
                (abs(h["dx"]) < 0.1 and distance < 17)
            )

            if closing or distance < 20:
                danger = distance - (12 if closing else 0)
                if nearest is None or danger < nearest[0]:
                    nearest = (danger, h, closing)

        if nearest is None:
            return {
                "imminent": False,
                "direction": "none",
                "bucket": "clear",
                "hazard": None,
            }

        _, h, closing = nearest
        distance = abs(h["x"] - px)

        direction = "right" if h["x"] > px else "left"
        if distance <= 24 and closing:
            bucket = "imminent"
        elif distance <= 43:
            bucket = "near"
        else:
            bucket = "approaching"

        return {
            "imminent": distance <= 28 and closing,
            "direction": direction,
            "bucket": bucket,
            "hazard": h,
        }

    def rivet_target(self, state):
        rows = rivet_mask(state["rivet_rows"])
        if not rows:
            return None

        p = state["player"]
        px, py = p["x"], p["y"]

        # Each row contains two rivet positions. Select the lowest
        # incomplete row and then the unremoved side nearest Mario.
        for row_index, mask in enumerate(rows):
            if mask == 18:
                continue

            y = 176 - row_index * 28
            options = []

            if mask in (0, 6):
                options.append((48, "left"))
            if mask in (0, 12):
                options.append((108, "right"))

            if options:
                options.sort(key=lambda item: abs(item[0] - px))
                return {
                    "x": options[0][0],
                    "y": y,
                    "side": options[0][1],
                    "row": row_index,
                    "mask": mask,
                }

        return None

    def heuristic(self, state):
        stage = state["stage"]
        p = state["player"]
        px, py = p["x"], p["y"]

        if stage == "idle":
            return "fire"

        if stage in ("game_over", "transition"):
            return "noop"

        danger = self.hazard_assessment(state)

        if danger["imminent"]:
            if danger["direction"] == "right":
                return "leftfire"
            return "rightfire"

        if self.active_jump > 0:
            self.active_jump -= 1
            return self.last_action

        if stage == "ramp":
            ladder = self.nearest_ladder(state)
            if ladder is None:
                return "right" if px < 80 else "left"

            tx, ty = ladder
            self.target_x, self.target_y = tx, ty

            if abs(px - tx) > 5:
                return "right" if px < tx else "left"

            if py - ty > 8:
                return "up"

            return "right" if px < 80 else "left"

        if stage == "rivet":
            target = self.rivet_target(state)
            if target is None:
                return "up"

            tx, ty = target["x"], target["y"]
            self.target_x, self.target_y = tx, ty

            # Crossing a cleared rivet location on foot is unsafe.
            if abs(px - tx) < 22 and abs(py - ty) < 16:
                return "rightfire" if tx < px else "leftfire"

            if abs(py - ty) > 18:
                return "up" if py > ty else "down"

            if abs(px - tx) > 8:
                return "right" if px < tx else "left"

            return "right" if target["side"] == "left" else "left"

        return "noop"

    def make_state(self, obs, state, heuristic):
        p = state["player"]
        danger = self.hazard_assessment(state)

        return {
            "game": "Atari 2600 Donkey Kong NTSC",
            "rules": {
                "fire_starts_idle": True,
                "directional_fire_jumps": True,
                "up_climbs_only_when_aligned": True,
                "cleared_rivet_gaps_are_dangerous": True,
                "goal": "Complete screens and preserve lives.",
            },
            "observation": state,
            "controller": {
                "plan": self.plan,
                "plan_version": self.plan_version,
                "heuristic_action": heuristic,
                "previous_action": self.last_action,
                "target_x": self.target_x,
                "target_y": self.target_y,
                "jump_active": self.active_jump > 0,
                "danger": danger,
                "candidate_actions": [
                    heuristic, "noop", "left", "right",
                    "up", "down", "leftfire", "rightfire",
                ],
            },
        }

    def start_jev(self, state, heuristic):
        if self.ctx.done or self.j_batch is not None:
            return

        remaining = self.ctx.budget.remaining()
        if remaining["system1_calls"] <= 0:
            return

        if self.ticks - self.last_jev_tick < 2:
            return

        danger = self.hazard_assessment(state)
        candidates = list(dict.fromkeys([
            heuristic, "left", "right", "up", "down",
            "leftfire", "rightfire", "noop",
        ]))

        criteria = {}
        descriptions = {
            "left": "Walk left on the current safe surface.",
            "right": "Walk right on the current safe surface.",
            "up": "Climb the aligned usable ladder.",
            "down": "Descend only if a lower ladder is intended.",
            "leftfire": "Jump while moving left to evade or cross a gap.",
            "rightfire": "Jump while moving right to evade or cross a gap.",
            "noop": "Wait briefly when movement is unsafe.",
        }

        for action in candidates:
            criteria[action] = {
                "what": descriptions.get(action, "Follow the computed immediate policy."),
                "not_for": "Do not choose if it contradicts an imminent collision or a known cleared-rivet gap.",
            }

        state_payload = self.make_state(None, state, heuristic)

        questions = {
            "action": Choice(
                "Choose the best immediate action for the CURRENT observation. "
                "Use controller.danger and controller.heuristic_action. "
                "Avoid a hazard that is imminent. Do not walk through a cleared "
                "rivet gap. Prefer upward progress when safe. Choose only an "
                "action from the supplied criteria.",
                criteria,
            ),
            "danger": Noul(
                "Is a visible hazard on the player's current row and likely "
                "to collide with Mario before the next few decisions?",
                {
                    "true": "A hazard is closing on Mario's current path.",
                    "false": "No hazard is likely to collide in the immediate future.",
                },
            ),
        }

        self.j_result = None
        self.j_result_tick = self.ticks
        self.j_result_stage = state["stage"]
        self.last_jev_tick = self.ticks

        def work():
            return self.ctx.decide(
                state_payload,
                questions,
                timeout_s=0.85,
            )

        self.j_batch = self.ctx.spawn(work)

    def collect_jev(self, state):
        future = self.j_batch
        if future is None or not future.done():
            return

        self.j_batch = None

        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return

            answer = answers.get("action")
            if answer is None:
                return

            value = getattr(answer, "value", None)
            if isinstance(value, str) and value in SAFE_ACTIONS:
                self.j_result = value
                self.j_result_tick = self.ticks
                self.j_result_stage = state["stage"]

                probability = getattr(answer, "probabilities", {})
                confidence = getattr(answer, "confidence", None)

                self.log(
                    "jev_action",
                    action=value,
                    confidence=confidence,
                    probabilities=probability,
                    stage=state["stage"],
                )

        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def choose(self, state, fallback):
        # An answer is usable only when the stage is unchanged and it
        # is no more than two emulator ticks old.
        if (
            self.j_result is not None
            and self.j_result_stage == state["stage"]
            and self.ticks - self.j_result_tick <= 2
        ):
            action = self.j_result
            self.j_result = None

            danger = self.hazard_assessment(state)

            # Code retains veto power over lethal decisions.
            if danger["imminent"]:
                return "leftfire" if danger["direction"] == "right" else "rightfire"

            return action

        return fallback

    def hard_safety(self, state, action):
        if action not in SAFE_ACTIONS:
            return "noop"

        if state["stage"] in ("game_over", "transition"):
            return "noop"

        danger = self.hazard_assessment(state)
        if danger["imminent"]:
            self.active_jump = 5
            return (
                "leftfire" if danger["direction"] == "right"
                else "rightfire"
            )

        # Do not continue an old climbing action after an unexpected
        # stage change or life loss.
        if self.last_stage != state["stage"]:
            return self.heuristic(state)

        return action

    def run(self, game):
        self.log("controller_started", tick_s=game.tick_s)

        initial = self.ctx.observe()
        state = compact_observation(initial)

        self.start_astra(state, "Initial strategic policy.")

        # FIRE is a one-time start operation, not a standing jump input.
        if state.get("stage") == "idle":
            self.ctx.act("fire")
            self.last_action = "fire"

        while not self.ctx.done and not game.done:
            self.ctx.wait_for_tick()

            if self.ctx.done or game.done:
                break

            obs = self.ctx.observe()
            state = compact_observation(obs)
            self.ticks += 1

            if state.get("terminated") or state.get("truncated"):
                break

            stage = state.get("stage")
            lives = state.get("lives")
            screens = state.get("screens_completed", 0)

            changed = (
                stage != self.last_stage
                or (self.last_lives is not None and lives != self.last_lives)
                or screens != self.last_screens
            )

            if changed:
                self.j_result = None
                self.target_x = None
                self.target_y = None
                self.active_jump = 0

                self.log(
                    "strategic_change",
                    stage=stage,
                    lives=lives,
                    screens=screens,
                )

                self.start_astra(state, "Stage, life, or completion change.")

            self.collect_astra()
            self.collect_jev(state)

            # Periodic policy review; never block the actor.
            if self.ticks - self.last_astra_tick >= 450:
                self.start_astra(state, "Periodic route and risk review.")

            fallback = self.heuristic(state)

            # Jev is used for tactical judgment; the deterministic
            # controller remains executable even without a response.
            if self.ticks % 2 == 0:
                self.start_jev(state, fallback)

            action = self.choose(state, fallback)
            action = self.hard_safety(state, action)

            if action in ("leftfire", "rightfire"):
                self.active_jump = max(self.active_jump, 4)

            self.ctx.act(action)
            self.last_action = action

            self.last_stage = stage
            self.last_lives = lives
            self.last_screens = screens

            if self.ticks % 100 == 0:
                self.log(
                    "progress",
                    tick=self.ticks,
                    stage=stage,
                    score=state.get("score", 0),
                    lives=lives,
                    bonus=state.get("bonus", 0),
                    rivets_removed=state.get("rivets_removed", 0),
                    screens=screens,
                )

        self.log(
            "controller_finished",
            ticks=self.ticks,
            score=game.score,
            last_action=self.last_action,
        )


class Program:

    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        Controller(self.ctx).run(game)


def build(ctx):
    return Program(ctx)

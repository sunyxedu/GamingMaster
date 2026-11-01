"""0014: outcome-accounted corridor selection with an independent tick actuator.

Public observations only.  Code measures geometry and executes reachable legs;
Jev chooses an actual next corridor; Astra revises the tradeoff rubric, not a
stale joystick command.  Every action keeps FIRE held (the ROM owns its lock).
"""
from __future__ import annotations

import json
import math
import statistics
from collections import deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")
PROFILES = {
    "BALANCED": "Prefer productive coverage. Try a distinct working corridor after repeated silent traverses; otherwise preserve broad coverage.",
    "HARVEST": "Prefer a reachable, open firing corridor concentrating on live low targets, especially when broad traverses are silent. Keep moving and firing.",
    "ESCAPE": "Prefer wide lateral room and fewer reversals after losses or poor control. Never interpret missing bombs as safety.",
}
NAMES = ("SWEEP", "LEFT_WORK", "RIGHT_WORK")


def num(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError, OverflowError):
        return default


def integer(x, default=0):
    return int(num(x, default))


def clamp(x, a, b):
    return max(a, min(b, x))


def obj(x):
    return x if isinstance(x, dict) else {}


def objects(x):
    return [v for v in x if isinstance(v, dict)] if isinstance(x, (list, tuple)) else []


def boolean(x):
    return x.lower() in ("true", "yes", "1") if isinstance(x, str) else bool(x)


def observation(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except (ValueError, TypeError):
            return {}
    return obj(x)


def field(answer, name, default=None):
    return answer.get(name, default) if isinstance(answer, dict) else getattr(answer, name, default)


def supported(answer, value, threshold=0.60):
    probs = obj(field(answer, "probabilities", {}))
    conf = field(answer, "confidence")
    return (num(probs.get(value), -1) >= threshold
            and (conf is None or num(conf, -1) >= 0.25))


def log(ctx, event, **data):
    ctx.log(event, **data)


class Calls:
    """Only finite ctx calls are spawned; no persistent Python worker loop."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.jev = None
        self.astra = None
        self.last_s1 = -10000
        self.last_s2 = -10000
        self.s2_count = 0
        self.latencies = deque(maxlen=24)
        self.session = None
        self.last_requested_slot = None

    def ask_jev(self, controller):
        if (self.jev is not None or getattr(self.ctx, "system1", True) is None
                or self.ctx.done or not controller.ready):
            return
        tag = controller.tag()
        slot = (tag["life_epoch"], tag["wave"], tag["version"], tag["slot"])
        # Ask for the next slot while the current leg is still in progress.
        # The tag names route_id+1, so a response can arrive before the
        # boundary and be consumed by the next commit; waiting until the
        # boundary would make a fast answer too late.
        if (slot == self.last_requested_slot or controller.frame - self.last_s1 < 24):
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) <= 2:
            return
        state = controller.state()
        descriptions = {
            "SWEEP": "Traverse the whole usable lane. Largest coverage and escape room, but includes shield-blocked or empty firing lanes.",
            "LEFT_WORK": "Moving, continuous-fire working corridor on the live left side. More repeated visits to its target lanes; sacrifices opposite-side coverage.",
            "RIGHT_WORK": "Moving, continuous-fire working corridor on the live right side. More repeated visits to its target lanes; sacrifices opposite-side coverage.",
            "UNKNOWN": "Evidence is insufficient or all useful alternatives are unavailable; use the checked fallback.",
        }
        if controller.route_id % 2:
            descriptions = dict(reversed(list(descriptions.items())))
        instruction = (
            "Choose the next executable corridor using plan.rubric and candidate facts. "
            "You are selecting a real spatial intervention, not a generic strategy label. "
            "Unavailable candidates must not be selected. Prefer actual productive/open "
            "target lanes over repeated silent coverage when the rubric permits. "
            "An untried corridor is uncertain, not known safe. Empty bombs are never a "
            "safety certificate. Code computed all comparisons; do not do arithmetic. "
            "A work corridor still makes long lateral legs with FIRE always held. "
            "Your answer applies only after the current leg finishes."
        )
        questions = {
            "corridor": Choice(instruction, descriptions),
            "crosscheck": Choice(instruction, dict(reversed(list(descriptions.items())))),
            "review": Noul(
                "Does evidence show repeated silent completed traverses OR repeated life "
                "losses, making the current strategic rubric worth reviewing? "
                "A single quiet shot, startup, and a missing bomb list do not suffice.",
                {"true": "Repeated explicit failures support a strategic review.",
                 "false": "No repeated failure is established."}),
        }
        self.last_requested_slot = slot
        self.last_s1 = controller.frame

        def request():
            start = self.ctx.clock.now()
            try:
                answers = self.ctx.decide(state, questions, timeout_s=1.0)
                return {"tag": tag, "answers": answers,
                        "latency": max(0.0, self.ctx.clock.now() - start)}
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                return {"tag": tag, "error": type(exc).__name__}

        self.jev = self.ctx.spawn(request)

    def ask_astra(self, controller, initial=False):
        if self.astra is not None or self.s2_count >= 2 or self.ctx.done:
            return
        remaining = self.ctx.budget.remaining()
        if remaining.get("system2_calls", 0) < 2 or remaining.get("wall_s", 0) < 65:
            return
        if not initial:
            if (controller.frame - self.last_s2 < 1800 or not controller.review_needed
                    or len(controller.history) < 4):
                return
        # A conditional rubric is valid after a life change; it is not an old
        # snapshot action.  Keep the output to a single validated label.
        data = {"current_rubric": controller.profile, "score": controller.score,
                "lives": controller.lives, "blind_losses": controller.blind_losses,
                "completed_legs": list(controller.history)[-8:],
                "movement_status": controller.motion_status,
                "candidate_facts": controller.candidate_facts()}
        prompt = (
            "Select a strategic rubric for a real-time Atari Space Invaders harness. "
            "Reply ONLY BALANCED, HARVEST, or ESCAPE, one word, no explanation. "
            "The controller posts at every 4-frame tick; one shot at a time is enforced "
            "by the game. It always holds FIRE and checks visible bombs locally. "
            "Public archive: full-rate broad sweeps scored 280 mean, but no first wave "
            "cleared. Boundary-repair-only descendants scored 209/256. Repeating generic "
            "WIDE advice did not alter their actions. Empty bomb lists preceded deaths. "
            "The current structural test lets Jev choose distinct reachable corridors "
            "from measured score yield, open-lane geometry, coverage, and control evidence. "
            "BALANCED explores a distinct corridor after repeated silent legs; HARVEST "
            "concentrates moving fire on promising live lanes rather than full-width "
            "patrol; ESCAPE favors wide room/fewer reversals after losses. None allows "
            "stationary camping, stopping fire, or changing emulator settings. "
            "Your rubric is sent verbatim in the next Jev state and changes the code "
            "fallback ranking. It is conditional strategy, NOT a command for this old "
            "snapshot. Choose based on evidence, not a seed or unseen information. "
            "Local Jev contract: ctx.decide returns dict[str,System1Answer]; Choice.value "
            "is a literal option with probabilities/confidence; Noul.value is yes "
            "probability. Jev cannot reliably count or compare numbers: code supplies "
            "semantic facts and candidates. Two reordered Choices must agree with each "
            "probability >=0.60; optional confidence must be >=0.25. Stale/life/wave/plan "
            "mismatches are discarded. Code rechecks legality and candidate availability "
            "at the next leg boundary. Do not generate questions, code, or new labels. "
            "Evidence: " + json.dumps(data, allow_nan=False, separators=(",", ":"))
        )
        session = self.session
        tag = {"frame": controller.frame, "wave": controller.wave}
        self.last_s2 = controller.frame
        self.s2_count += 1

        def request():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=60.0)
                return {"text": getattr(result, "text", ""),
                        "session": getattr(result, "session_id", None), "tag": tag}
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                return {"error": type(exc).__name__, "tag": tag}

        self.astra = self.ctx.spawn(request)
        log(self.ctx, "astra_rubric_requested", frame=controller.frame,
            reason="startup_conditional_strategy" if initial else "outcome_review")

    def poll(self, controller):
        if self.astra is not None and self.astra.done():
            future, self.astra = self.astra, None
            try:
                result = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                result = {"error": type(exc).__name__}
            if isinstance(result, dict):
                self.session = result.get("session", self.session)
                text = result.get("text", "")
                word = text.strip().strip('"\'` .\n').upper() if isinstance(text, str) else ""
                if word in PROFILES:
                    controller.apply_profile(word)
                else:
                    log(self.ctx, "astra_rubric_rejected", reason=result.get("error", "invalid_label"))
        if self.jev is not None and self.jev.done():
            future, self.jev = self.jev, None
            try:
                result = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                result = {"error": type(exc).__name__}
            if isinstance(result, dict) and "latency" in result:
                self.latencies.append(num(result["latency"]))
            controller.accept(result)

    def close(self):
        for future in (self.jev, self.astra):
            if future is not None:
                future.cancel()
        self.jev = self.astra = None


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.px, self.py = 90.0, 190.0
        self.score, self.lives, self.wave = 0.0, 3, 1
        self.epoch = 0
        self.version = 0
        self.profile = "BALANCED"
        self.last_action = "fire"
        self.last_obs = None
        self.ready = False
        self.speed = 0.5
        self.speeds = deque(maxlen=20)
        self.formation = deque(maxlen=20)
        self.vx = 0.0
        self.laser_speed = 2.0
        self.left, self.right = 37.0, 119.0
        self.boundary_confirmed = [False, False]
        self.stationary = 0
        self.last_move_frame = 0
        self.last_move_x = self.px
        self.freeze = None
        self.motion_status = "STARTUP_OR_UNOBSERVED"
        self.alive = []
        self.columns = {}
        self.shields = []
        self.tracks = []
        self.last_bomb = -10000
        self.route = None
        self.route_id = 0
        self.direction = -1
        self.reference_direction = -1
        self.advice = None
        self.history = deque(maxlen=16)
        self.stats = {k: {"legs": 0, "gain": 0.0, "frames": 0, "losses": 0} for k in NAMES}
        self.blind_losses = 0
        self.losses = 0
        self.silent = 0
        self.review_needed = False
        self.life_event = False
        self.wave_event = False
        self.last_progress = 0
        self.accepted = self.effective = self.astra_changes = 0
        self.overrides = self.actions = self.model_divergent_actions = 0
        self.last_facts = None
        self.last_decision_source = "fallback"

    def bounds(self):
        # Interior of the measured 37..119 envelope in archived public traces.
        # Expand if an episode actually demonstrates a wider travel envelope.
        return self.left + 3.0, self.right - 3.0

    def _bombs(self, obs):
        old = self.tracks
        tracks = []
        for b in objects(obs.get("bombs"))[:16]:
            if b.get("x") is None or b.get("y") is None:
                continue
            x, y = num(b["x"], -999), num(b["y"], -999)
            if not 0 <= x <= 159 or not 0 <= y <= 209:
                continue
            matches = [t for t in old if abs(t["x"]-x) <= 3 and 0 < self.frame-t["frame"] <= 16
                       and y >= t["y"]]
            v = 1.2
            if matches:
                t = min(matches, key=lambda t: abs(y-t["y"]-1.2*(self.frame-t["frame"])))
                v = clamp((y-t["y"])/(self.frame-t["frame"]), 0.5, 2.8)
            tracks.append({"x": x, "y": y, "v": v, "frame": self.frame})
        if tracks:
            self.last_bomb = self.frame
        tracks.extend(t for t in old if self.frame-t["frame"] <= 16
                      and not any(abs(t["x"]-n["x"]) <= 3 for n in tracks))
        self.tracks = tracks[:20]

    def update(self, raw):
        obs = observation(raw)
        frame = integer(obs.get("frame"), self.frame+4)
        if self.last_obs is not None and frame <= self.frame:
            return False
        old_frame, old_px = self.frame, self.px
        df = max(1, frame-old_frame)
        player = obj(obs.get("player"))
        px = clamp(num(player.get("x"), self.px), 0, 159)
        py = clamp(num(player.get("y"), self.py), 0, 209)
        lives, wave = integer(obs.get("lives"), self.lives), integer(obs.get("wave"), self.wave)
        self.life_event = lives < self.lives
        self.wave_event = wave != self.wave
        dx = px-self.px
        discontinuity = abs(dx) > max(10.0, 1.5*df)
        self.frame = frame
        self.score = num(obs.get("score"), self.score)
        if self.last_obs and self.score > num(self.last_obs.get("score")):
            self.last_progress = frame
        if self.life_event:
            self.losses += self.lives-lives
            blind = frame-self.last_bomb > 90
            self.blind_losses += int(blind)
            evidence = self.freeze or {"frame": old_frame, "x": old_px}
            log(self.ctx, "life_outcome", frame=frame, previous_x=old_px,
                last_controllable=evidence, blind=blind, corridor=(self.route or {}).get("name"))
            self.finish("life_loss", loss=True)
            self.epoch += 1
            self.advice = None
            self.review_needed = self.losses >= 2
            self.tracks = []
            self.freeze = None
            self.stationary = 0
        if self.wave_event:
            self.finish("wave_change")
            self.epoch += 1
            self.advice = None
            self.formation.clear()
            self.tracks = []
            self.silent = 0
            self.freeze = None
        self.lives, self.wave, self.px, self.py = lives, wave, px, py
        shot, oldshot = obj(obs.get("player_shot")), obj((self.last_obs or {}).get("player_shot"))
        d = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
        if self.last_obs and not discontinuity and not self.life_event and not self.wave_event:
            if abs(dx) >= 0.5:
                self.ready = True
                self.last_move_frame, self.last_move_x = frame, px
                if d and 0.1 <= dx*d/df <= 1.5:
                    self.speeds.append(abs(dx)/df)
                    self.speed = clamp(statistics.median(self.speeds), 0.25, 1.0)
                self.stationary = 0
                if self.freeze is not None:
                    log(self.ctx, "control_resumed", frame=frame, frozen_frames=frame-self.freeze["frame"])
                self.freeze = None
            elif d:
                self.stationary += df
            if boolean(shot.get("active")) and boolean(oldshot.get("active")):
                if (shot.get("y") is not None and oldshot.get("y") is not None
                        and abs(num(shot.get("x"))-num(oldshot.get("x"))) <= 2):
                    speed = (num(oldshot["y"])-num(shot["y"]))/df
                    if 0.5 <= speed <= 3.0:
                        self.laser_speed = 0.8*self.laser_speed+0.2*speed
            # Learn a clip only at an extreme, after demonstrated control. An
            # interior freeze can be an explosion/respawn, NOT a new wall.
            if self.ready and self.stationary >= 12:
                if d < 0 and px <= self.left+2:
                    self.left = max(self.left, px)
                    self.boundary_confirmed[0] = True
                elif d > 0 and px >= self.right-2:
                    self.right = min(self.right, px)
                    self.boundary_confirmed[1] = True
                elif self.freeze is None:
                    self.freeze = {"frame": self.last_move_frame, "x": self.last_move_x}
                    log(self.ctx, "interior_control_gap", frame=frame, last_controllable=self.freeze)
        if discontinuity:
            self.route = None
            self.advice = None
            self.epoch += 1
            self.stationary = 0
            self.freeze = None
        if self.ready and not discontinuity and not self.life_event:
            self.left = min(self.left, px)
            self.right = max(self.right, px)
        self.motion_status = ("STARTUP_OR_UNOBSERVED" if not self.ready else
                              "CONTROL_GAP_NOT_A_WALL" if self.freeze else "RESPONDING")
        self.alive = [a for a in objects(obs.get("invaders")) if boolean(a.get("alive"))
                      and a.get("x") is not None and a.get("y") is not None]
        self.columns = {}
        for a in self.alive:
            col = integer(a.get("col"), -1)
            if 0 <= col <= 5:
                if col not in self.columns or num(a["y"]) > num(self.columns[col]["y"]):
                    self.columns[col] = a
        if self.alive:
            offset = statistics.median(num(a["x"])-16*integer(a.get("col")) for a in self.alive)
            self.formation.append((frame, offset))
            while self.formation and frame-self.formation[0][0] > 48:
                self.formation.popleft()
            if len(self.formation) > 1:
                f, x = self.formation[0]
                self.vx = clamp((offset-x)/max(1, frame-f), -0.5, 0.5)
        self.shields = objects(obs.get("shields"))
        self._bombs(obs)
        self.last_obs = obs
        return True

    def shield_block(self, x):
        for s in self.shields:
            # Public decoder: eight horizontal occupancy bits, nine rows over
            # the shield's 18 scan lines. Coordinates are its center.
            col = int(math.floor(x-(num(s.get("x"))-4)))
            mask = s.get("health_mask")
            if 0 <= col < 8 and isinstance(mask, list):
                return sum(bool(integer(row) & (1 << (7-col))) for row in mask[:9])/9.0
        return 0.0

    def candidates(self):
        left, right = self.bounds()
        if right-left < 48:
            left, right = 40.0, 116.0
        lanes = []
        for a in self.columns.values():
            flight = max(0.0, self.py-6-num(a["y"]))/max(0.8, self.laser_speed)
            x = num(a["x"])+clamp(self.vx*flight, -10, 10)
            if left-6 <= x <= right+6:
                lanes.append((clamp(x, left, right), num(a["y"])))
        xs = sorted(x for x, _ in lanes)
        # Side work remains a spatial traverse (at least 46 pixels), not a
        # short relocation around a guessed point target.
        span = min(52.0, right-left)
        if xs:
            split = max(1, (len(xs)+1)//2)
            lc = statistics.mean(xs[:split])
            rc = statistics.mean(xs[-split:])
        else:
            lc, rc = left+span/2, right-span/2
        llo = clamp(lc-span/2, left, right-span)
        rlo = clamp(rc-span/2, left, right-span)
        bands = {"SWEEP": (left, right), "LEFT_WORK": (llo, llo+span),
                 "RIGHT_WORK": (rlo, rlo+span)}
        result = {}
        for name, (lo, hi) in bands.items():
            covered = [(x, y) for x, y in lanes if lo-3 <= x <= hi+3]
            density = sum(1.0-self.shield_block(x)*0.7 for x, _ in covered)/max(1, hi-lo)
            history = self.stats[name]
            rate = history["gain"]*60/max(1, history["frames"])
            unique = name == "SWEEP" or (hi-lo < right-left-8 and covered
                       and (name != "RIGHT_WORK" or abs(rlo-llo) >= 8))
            result[name] = {"lo": lo, "hi": hi, "available": bool(unique),
                            "count": len(covered), "total": len(lanes),
                            "density": density, "rate": rate,
                            "low": any(y >= 120 for _, y in covered),
                            "block": statistics.mean([self.shield_block(x) for x, _ in covered]) if covered else 1.0,
                            "legs": history["legs"], "losses": history["losses"]}
        return result

    def candidate_facts(self):
        candidates = self.candidates()
        best_density = max((c["density"] for c in candidates.values() if c["available"]), default=0)
        result = {}
        for name, c in candidates.items():
            stat = self.stats[name]
            result[name] = {
                "available": c["available"],
                "coverage": "ALL_REACHABLE" if c["count"] == c["total"] and c["total"] else "PARTIAL" if c["count"] else "NONE",
                "escape_room": "WIDE" if c["hi"]-c["lo"] >= 65 else "MODERATE",
                "firing_path": "MOSTLY_OPEN" if c["block"] < 0.20 else "MIXED" if c["block"] < 0.6 else "SHIELD_BLOCKED",
                "live_lane_density": "BEST" if c["density"] >= best_density-0.001 else "LOWER",
                "low_targets": c["low"],
                "outcome": "UNTRIED" if not stat["legs"] else "PRODUCTIVE" if c["rate"] >= 4 else "SLOW" if stat["gain"] else "SILENT",
                "loss_evidence": "OBSERVED_LOSS" if stat["losses"] else "NO_LOSS_OBSERVED_NOT_PROOF_OF_SAFETY",
            }
        return result

    def state(self):
        return {"schema": "outcome_corridor_v1",
                "rules": "Space Invaders: code owns coordinates, time, legal actions and one-beam firing. All candidates move and hold FIRE. Decoder silence is unknown danger, not safe.",
                "plan": {"profile": self.profile, "rubric": PROFILES[self.profile]},
                "evidence": {"movement": self.motion_status,
                             "progress": "REPEATED_SILENT_LEGS" if self.silent >= 2 else "RECENT_PROGRESS" if self.frame-self.last_progress < 180 else "QUIET",
                             "losses": "REPEATED" if self.losses >= 2 else "ONE" if self.losses else "NONE_OBSERVED",
                             "bombs": "VISIBLE_TRACKS" if self.tracks else "UNOBSERVED_OR_EMPTY",
                             "current_corridor": (self.route or {}).get("name", "SWEEP"),
                             "phase": "SPARSE" if len(self.columns) <= 3 else "MULTIPLE_COLUMNS",
                             "last_completed": list(self.history)[-3:]},
                "candidates": self.candidate_facts()}

    def tag(self):
        return {"frame": self.frame, "life_epoch": self.epoch, "wave": self.wave,
                "version": self.version, "slot": self.route_id+1}

    def accept(self, result):
        if not isinstance(result, dict) or result.get("error"):
            if isinstance(result, dict) and result.get("error"):
                log(self.ctx, "jev_failed", error=result["error"])
            return
        tag = obj(result.get("tag"))
        if (tag.get("life_epoch") != self.epoch or tag.get("wave") != self.wave
                or tag.get("version") != self.version
                or tag.get("slot") != self.route_id+1
                or not 0 <= self.frame-num(tag.get("frame"), -10000) <= 72):
            log(self.ctx, "jev_discarded", reason="age_or_decision_epoch")
            return
        answers = obj(result.get("answers"))
        a, b = answers.get("corridor"), answers.get("crosscheck")
        key = field(a, "value")
        review = num(field(answers.get("review"), "value"))
        if review >= 0.80 and (self.silent >= 2 or self.losses >= 2):
            self.review_needed = True
        if (not isinstance(key, str) or key not in NAMES or field(b, "value") != key
                or not supported(a, key) or not supported(b, key)):
            return
        self.advice = {"name": key, "tag": tag}
        self.accepted += 1
        log(self.ctx, "jev_next_corridor", frame=self.frame, corridor=key,
            slot=tag["slot"], latency_s=result.get("latency"))

    def apply_profile(self, word):
        previous = self.profile
        self.profile = word
        self.version += 1
        self.advice = None
        self.review_needed = False
        self.astra_changes += int(word != previous)
        log(self.ctx, "astra_rubric_applied", previous=previous, profile=word,
            changed=word != previous, version=self.version, frame=self.frame,
            effect="next_corridor_ranking_and_jev_rubric")

    def fallback(self, candidates):
        eligible = [k for k in NAMES if candidates[k]["available"]]
        if self.profile == "ESCAPE" or (self.profile == "BALANCED" and self.silent < 2):
            return "SWEEP"
        def rank(k):
            c = candidates[k]
            # Geometry is a prior; observed ALE gain (not disappearing sprites)
            # supplies the outcome evidence. Do not select by raw row values.
            empirical = min(c["rate"], 18)/18 if c["legs"] else 0.35
            return c["density"]*35 + empirical*0.8 - c["losses"]*0.35
        return max(eligible, key=rank) if eligible else "SWEEP"

    def finish(self, reason, loss=False):
        route = self.route
        if route is None:
            return
        elapsed = max(1, self.frame-route["start"])
        distance = abs(self.px-route["from_x"])
        gain = max(0.0, self.score-route["score"])
        # Frozen startup/death animations are not evidence that a different
        # corridor would have produced zero yield during the same interval.
        active_frames = max(1, elapsed-route["frozen_frames"])
        valid = distance >= 12 and active_frames >= 16
        stat = self.stats[route["name"]]
        if valid:
            stat["legs"] += 1
            stat["gain"] += gain
            stat["frames"] += active_frames
            self.silent = self.silent+1 if gain <= 0 else 0
        if loss:
            stat["losses"] += 1
        report = {"corridor": route["name"], "reason": reason,
                  "progress": "SCORE_GAIN" if gain else "SILENT",
                  "control": "RESPONDED" if valid else "INSUFFICIENT_CONTROL",
                  "loss": bool(loss)}
        self.history.append(report)
        log(self.ctx, "corridor_outcome", **report, route_id=route["id"],
            frames=elapsed, active_frames=active_frames, distance=round(distance, 2),
            score_gain=gain, start_x=route["from_x"], end_x=self.px,
            source=route["source"])
        self.route = None

    def commit(self):
        candidates = self.candidates()
        fallback = self.fallback(candidates)
        chosen, source = fallback, "code_fallback"
        advice = self.advice
        if advice:
            tag = advice["tag"]
            if (tag.get("life_epoch") == self.epoch and tag.get("wave") == self.wave
                    and tag.get("version") == self.version and tag.get("slot") == self.route_id+1
                    and 0 <= self.frame-num(tag.get("frame"), -10000) <= 144
                    and candidates[advice["name"]]["available"]):
                chosen, source = advice["name"], "jev"
        self.advice = None
        c = candidates[chosen]
        lo, hi = c["lo"], c["hi"]
        d = -self.direction if self.route_id else (-1 if self.px > (lo+hi)/2 else 1)
        if self.px < lo+8:
            d = 1
        elif self.px > hi-8:
            d = -1
        end = hi if d > 0 else lo
        if abs(end-self.px) < 12:
            d = -d
            end = hi if d > 0 else lo
        self.direction = d
        self.route_id += 1
        self.route = {"id": self.route_id, "name": chosen, "lo": lo, "hi": hi,
                      "end": end, "direction": d, "start": self.frame,
                      "from_x": self.px, "score": self.score, "frozen_frames": 0,
                      "source": source, "version": self.version}
        changed = source == "jev" and chosen != fallback
        self.effective += int(changed)
        self.last_decision_source = source
        log(self.ctx, "corridor_committed", frame=self.frame, route_id=self.route_id,
            corridor=chosen, fallback=fallback, source=source, model_changed_plan=changed,
            lo=round(lo, 2), hi=round(hi, 2), endpoint=round(end, 2),
            direction=d, profile=self.profile, version=self.version)

    def risk(self, d):
        left, right = self.bounds()
        previous = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
        worst = 0.0
        for bomb in self.tracks:
            y = bomb["y"] + bomb["v"]*max(0, self.frame-bomb["frame"])
            for v in (0.7, bomb["v"], 2.2):
                for dt in range(0, 49, 4):
                    by = y+v*dt
                    if not self.py-12 <= by <= self.py+6:
                        continue
                    for delay in (0, 4):
                        x = clamp(self.px+self.speed*(previous*min(delay, dt)+d*max(0, dt-delay)), left, right)
                        gap = abs(x-bomb["x"])
                        if gap < 10:
                            worst = max(worst, 100+10-gap+(48-dt)/48)
                        elif gap < 18:
                            worst = max(worst, (18-gap)/8)
        return worst

    def action(self):
        left, right = self.bounds()
        if self.px <= left+2:
            self.reference_direction = 1
        elif self.px >= right-2:
            self.reference_direction = -1
        if self.route:
            r = self.route
            reached = (r["end"]-self.px)*r["direction"] <= max(2.0, self.speed*4)
            # Ignore interior non-response; it is often a death animation. Do
            # not manufacture repeated zero-distance routes or a false wall.
            if self.freeze is not None:
                r["frozen_frames"] += 4
            expired = (self.frame-r["start"]-r["frozen_frames"] > 300)
            if reached or expired:
                self.finish("reached" if reached else "active_timeout")
        if self.route is None:
            self.commit()
        d = self.route["direction"]
        if d < 0 and self.px <= left+1:
            d = 1
        elif d > 0 and self.px >= right-1:
            d = -1
        risks = {k: self.risk(k) for k in (-1, 0, 1)}
        source = self.route["source"]
        if risks[d] >= 100:
            alternative = min(risks, key=lambda k: (risks[k], k != d, k == 0))
            if risks[alternative] < risks[d]:
                d = alternative
                self.overrides += 1
                source = "visible_bomb"
                log(self.ctx, "visible_bomb_action", frame=self.frame, direction=d)
        if source == "jev" and d != self.reference_direction:
            self.model_divergent_actions += 1
            if self.model_divergent_actions % 20 == 1:
                log(self.ctx, "jev_action_consequence", frame=self.frame, route_id=self.route_id,
                    actual_direction=d, same_state_wide_direction=self.reference_direction)
        action = "left_fire" if d < 0 else "right_fire" if d > 0 else "fire"
        return action, source


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        spec = getattr(game, "action_spec", None)
        legal = set(getattr(spec, "names", ACTIONS) or ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in legal:
            noop = next(iter(legal))
        controller = Controller(ctx)
        calls = Calls(ctx)
        started_astra = False
        try:
            while not ctx.done and not game.done:
                raw = ctx.observe()
                obs = observation(raw)
                if boolean(obs.get("game_over")):
                    break
                if not controller.update(obs):
                    ctx.wait_for_tick()
                    continue
                calls.poll(controller)
                action, source = controller.action()
                action = action if action in legal else ("fire" if "fire" in legal else noop)
                ctx.act(action)
                controller.last_action = action
                controller.actions += 1
                # Post first: neither model scheduling nor logging may delay
                # the actuator's fresh tick command.
                calls.ask_jev(controller)
                calls.ask_astra(controller, initial=not started_astra)
                started_astra = True
                if controller.actions % 90 == 0:
                    log(ctx, "corridor_status", frame=controller.frame, x=controller.px,
                        score=controller.score, lives=controller.lives, direction=controller.direction,
                        corridor=(controller.route or {}).get("name"), source=source,
                        motion=controller.motion_status, silent_legs=controller.silent,
                        jev_accepted=controller.accepted, jev_changed=controller.effective,
                        profile=controller.profile, bounds=list(controller.bounds()))
                ctx.wait_for_tick()
        except (BudgetExceeded, ModelUnavailable):
            raise
        finally:
            calls.close()
            log(ctx, "corridor_finished", frame=controller.frame, score=controller.score,
                actions=controller.actions, completed_legs=len(controller.history),
                jev_accepted=controller.accepted, jev_changed=controller.effective,
                jev_divergent_actions=controller.model_divergent_actions,
                astra_changes=controller.astra_changes, astra_calls=calls.s2_count,
                visible_overrides=controller.overrides, stats=controller.stats,
                jev_latency_median=statistics.median(calls.latencies) if calls.latencies else None)


def build(ctx):
    return Harness(ctx)

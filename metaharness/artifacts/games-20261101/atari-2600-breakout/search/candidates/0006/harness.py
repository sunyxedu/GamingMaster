"""Stateful interception with persistent, confidence-gated model contracts.

Only public observations are used. Models never own a frame-level action.
"""
from __future__ import annotations

from collections import deque
import json
import math
import re

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError, OverflowError):
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sign(x):
    return (x > 0) - (x < 0)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.frame = -1
        self.hx = deque(maxlen=8)
        self.hy = deque(maxlen=8)
        self.sx = 0
        self.sy = 0
        self.epoch = 0
        self.policy_version = 0
        self.last_action = "NOOP"
        self.paddle_steps = deque(maxlen=32)
        self.paddle_velocity = 0.0
        self.last_target = None
        self.last_facts = {}
        self.policy = "STABILITY"
        self.mode = "STABLE"
        self.mode_until = -1
        self.s1 = None
        self.s1_tag = None
        self.s1_last = -1000
        self.s1_count = 0
        self.s2 = None
        self.s2_count = 0
        self.s2_last = -100000
        self.s2_tag = None
        self.losses = 0
        self.losses_at_review = 0
        self.last_score_frame = 0
        self.s1_latency_frames = 18.0
        self.last_logged_mode = None

    def emit(self, event, **fields):
        self.ctx.log(event, **fields)

    @staticmethod
    def reflect(x):
        # Provisional geometry used by the successful archived interceptor.
        phase = (x - 8.0) % 284.0
        return 8.0 + (phase if phase <= 142.0 else 284.0 - phase)

    def reset_flight(self):
        self.hx.clear()
        self.hy.clear()
        self.sx = self.sy = 0
        self.last_target = None
        self.mode = "STABLE"
        self.mode_until = -1
        self.epoch += 1

    def observe_motion(self, obs):
        old = self.previous or {}
        frame = int(number(obs.get("frame"), self.frame + 1))
        gap = max(1, frame - int(number(old.get("frame"), frame - 1)))
        ball = obs.get("ball")
        previous_ball = old.get("ball")
        life_loss = number(obs.get("lives"), 5) < number(old.get("lives"), 5)
        wall_change = obs.get("wall_index", 0) != old.get("wall_index", 0)
        scored = number(obs.get("score")) > number(old.get("score"))
        if scored:
            self.last_score_frame = frame
        if life_loss or wall_change or not isinstance(ball, dict) or not isinstance(previous_ball, dict):
            self.reset_flight()
        if life_loss:
            self.losses += 1
            self.emit("life_loss", frame=frame, score=obs.get("score"),
                      preceding_controller=self.last_facts)
        p = obs.get("paddle") or {}
        op = old.get("paddle") or {}
        self.paddle_velocity = (number(p.get("x"), 99) - number(op.get("x"), number(p.get("x"), 99))) / gap
        if gap <= 2 and self.last_action in ("LEFT", "RIGHT"):
            step = abs(self.paddle_velocity)
            if 0.25 <= step <= 12:
                self.paddle_steps.append(step)
        if isinstance(ball, dict) and isinstance(previous_ball, dict) and not life_loss:
            dx = (number(ball.get("x")) - number(previous_ball.get("x"))) / gap
            dy = (number(ball.get("y")) - number(previous_ball.get("y"))) / gap
            # Collision deltas must never be averaged across a direction change.
            flip_x = dx != 0 and self.sx != 0 and sign(dx) != self.sx
            flip_y = dy != 0 and self.sy != 0 and sign(dy) != self.sy
            if scored or gap > 2 or flip_x:
                self.hx.clear()
            if scored or gap > 2 or flip_y:
                self.hy.clear()
            if flip_y or scored:
                self.epoch += 1
                self.mode_until = -1
            if flip_y and dy < 0 and number(previous_ball.get("y")) > 165:
                self.emit("paddle_return", frame=frame,
                          previous_target=self.last_target,
                          paddle_x=number(p.get("x")))
            if abs(dx) < 15 and abs(dy) < 15:
                self.hx.append((dx * gap, gap))
                self.hy.append((dy * gap, gap))
                if dx:
                    self.sx = sign(dx)
                if dy:
                    self.sy = sign(dy)
        self.frame = frame
        self.previous = obs
        return life_loss or wall_change

    @staticmethod
    def average(history):
        if not history:
            return 0.0
        return sum(v for v, _ in history) / sum(t for _, t in history)

    def facts(self, obs):
        ball = obs.get("ball")
        p = obs.get("paddle") or {}
        px = number(p.get("x"), 99)
        pw = number(p.get("w"), 16)
        step = sorted(self.paddle_steps)[len(self.paddle_steps) // 2] if self.paddle_steps else 3.0
        result = {"phase": "absent", "target": px, "eta": None,
                  "uncertainty": "unsettled", "catch": "unknown", "speed": "unknown",
                  "landing_lane": "unknown", "step": step, "spread": 0.0,
                  "safe_lo": px, "safe_hi": px}
        if not isinstance(ball, dict):
            return result
        bx, by = number(ball.get("x"), 80), number(ball.get("y"), 120)
        bw, bh = number(ball.get("w"), 2), number(ball.get("h"), 4)
        vx, vy = self.average(self.hx), self.average(self.hy)
        result["speed"] = "fast" if abs(vy) >= 2.5 else "ordinary"
        result["phase"] = "descending" if self.sy > 0 else "ascending" if self.sy < 0 else "unsettled"
        result["target"] = 76.0
        if self.sy <= 0 or vy <= 0:
            # Unknown velocity is not proof of ascent. Hold until it is measured.
            if self.sy == 0:
                result["target"] = px
            return result
        distance = max(0.0, number(p.get("y"), 189) - bh - by)
        eta = distance / vy
        landing = self.reflect(bx + vx * eta) + bw / 2.0
        # Quantized one-frame deltas alternate. The multi-frame slope is primary;
        # a shorter window is a disagreement diagnostic, not another target.
        short_x = self.average(list(self.hx)[-3:])
        short_y = self.average(list(self.hy)[-3:])
        alternate = landing
        if short_y > 0:
            alternate = self.reflect(bx + short_x * distance / short_y) + bw / 2.0
        spread = min(12.0, abs(alternate - landing))
        margin = 2.0
        safe_lo = clamp(landing - pw + margin, 8.0, 144.0)
        safe_hi = clamp(landing - margin, 8.0, 144.0)
        target = clamp(landing - pw / 2, 8.0, 144.0)
        travel = abs(target - px) / max(1.0, step)
        catch = "comfortable" if eta > travel + 10 else "tight" if eta > travel + 3 else "urgent"
        result.update(phase="descending", target=target, eta=eta, spread=spread,
                      uncertainty="settled" if len(self.hy) >= 4 and spread <= 4 else "unsettled",
                      catch=catch, landing_lane="left" if landing < 55 else "right" if landing > 105 else "middle",
                      safe_lo=min(safe_lo, safe_hi), safe_hi=max(safe_lo, safe_hi))
        return result

    def action(self, obs, facts, mode):
        if obs.get("terminated") or obs.get("truncated"):
            return "NOOP"
        if obs.get("serve_required"):
            return "FIRE"
        if not isinstance(obs.get("ball"), dict):
            return "NOOP"
        p = obs.get("paddle") or {}
        px = number(p.get("x"), 99)
        target = facts["target"]
        descending = facts["phase"] == "descending"
        # Model modes select a tracking contract, never an absolute paddle side.
        # The contract is re-evaluated on current geometry every native frame.
        if descending and mode == "QUIET" and facts["catch"] == "comfortable":
            if facts["safe_lo"] + 1 <= px <= facts["safe_hi"] - 1:
                target = px
        self.last_target = target
        # Observed displacement gives a modest sticky-action stopping allowance;
        # it does not assume NOOP instantaneously brakes or hardcode acceleration.
        drift = clamp(self.paddle_velocity, -facts["step"], facts["step"]) * 0.4
        error = target - (px + drift)
        deadband = 1.5 if descending and facts["eta"] < 8 else 2.5
        if descending and facts["catch"] == "urgent":
            deadband = 1.0
        if abs(error) <= deadband:
            return "NOOP"
        return "RIGHT" if error > 0 else "LEFT"

    def remaining(self, key):
        return number(self.ctx.budget.remaining().get(key), 0)

    def collect(self):
        if self.s1 is not None and self.s1.done():
            future, tag = self.s1, self.s1_tag
            self.s1 = None
            try:
                answers = future.result()
                a, b = answers.get("mode_a"), answers.get("mode_b")
                value = getattr(a, "value", None)
                pa = number((getattr(a, "probabilities", None) or {}).get(value))
                pb = number((getattr(b, "probabilities", None) or {}).get(value))
                ca, cb = getattr(a, "confidence", None), getattr(b, "confidence", None)
                veto = number(getattr(answers.get("veto"), "value", None), 1.0)
                age = self.frame - tag[0]
                self.s1_latency_frames = 0.8 * self.s1_latency_frames + 0.2 * max(1, age)
                fresh = self.epoch == tag[1] and self.policy_version == tag[2] and age <= 90
                confidence_ok = ca is not None and cb is not None and min(number(ca), number(cb)) >= 0.15
                if fresh and value in ("STABLE", "QUIET") and value == getattr(b, "value", None) and min(pa, pb) >= 0.70 and confidence_ok:
                    # A doubtful safety answer cannot authorize reduced motion.
                    accepted = value if value == "STABLE" or veto <= 0.15 else "STABLE"
                    self.mode = accepted
                    self.mode_until = self.frame + 120
                    self.emit("jev_contract", frame=self.frame, mode=accepted,
                              age_frames=age, probability=min(pa, pb), veto=veto)
                else:
                    self.emit("jev_contract_rejected", frame=self.frame, age_frames=age, fresh=fresh)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except Exception as exc:
                self.emit("jev_fallback", error=type(exc).__name__)
        if self.s2 is not None and self.s2.done():
            future = self.s2
            self.s2 = None
            try:
                text = getattr(future.result(), "text", "")
                match = re.fullmatch(r"\s*(STABILITY|ECONOMY)\s*[.!]?\s*", text.upper()) if isinstance(text, str) else None
                if match and self.frame - self.s2_tag <= 15000:
                    self.policy = match.group(1)
                    self.policy_version += 1
                    self.mode_until = -1
                    self.emit("astra_policy", frame=self.frame, policy=self.policy)
                else:
                    self.emit("astra_policy_rejected", frame=self.frame)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except Exception as exc:
                self.emit("astra_fallback", error=type(exc).__name__)

    def launch_jev(self, obs, facts):
        if getattr(self.ctx, "system1", True) is None or self.s1 is not None:
            return
        if facts["phase"] != "descending" or facts["eta"] is None:
            return
        spacing = max(36, int(self.s1_latency_frames * 2))
        if self.frame - self.s1_last < spacing or facts["eta"] < self.s1_latency_frames + 8:
            return
        if self.remaining("system1_calls") <= 5 or self.remaining("wall_s") < 6:
            return
        state = {
            "game": "Atari 2600 Breakout; preserve the ball before seeking points.",
            "plan": {"policy": self.policy,
                     "meaning": "STABILITY prioritizes centered catches. ECONOMY avoids needless motion only within a safe catch corridor."},
            "flight": {k: facts[k] for k in ("phase", "uncertainty", "catch", "speed", "landing_lane")},
            "progress": "stalled" if self.frame - self.last_score_frame > 900 else "scoring",
            "life_reserve": "last_ball" if number(obs.get("lives"), 5) <= 1 else "multiple_balls",
            "rules": "STABLE tracks the smoothed intercept. QUIET holds only inside the current safe corridor, with ample travel slack. Neither mode changes collision geometry or serves. Unknown facts require STABLE."
        }
        descriptions = {
            "STABLE": "Use centered smoothed interception when facts are uncertain, time is tight, or the plan prioritizes stability.",
            "QUIET": "Avoid needless paddle motion only for a settled, comfortable flight when the ECONOMY plan supports it.",
            "UNKNOWN": "The facts do not support a decision; use deterministic STABLE fallback."
        }
        instruction = "Select a tracking contract using plan.policy and flight's named facts. Do not infer a direction or calculate. Choose QUIET only when its stated preconditions all hold."
        questions = {
            "mode_a": Choice(instruction, descriptions),
            "mode_b": Choice(instruction, dict(reversed(list(descriptions.items())))),
            "veto": Noul("Do any supplied facts make reduced paddle movement inappropriate? Answer yes for unsettled prediction, tight/urgent catch, last_ball, or STABILITY policy.")
        }
        self.s1_last = self.frame
        self.s1_tag = (self.frame, self.epoch, self.policy_version)
        self.s1_count += 1
        self.s1 = self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=3.0))

    def launch_astra(self, obs, facts, boundary):
        if getattr(self.ctx, "system2", True) is None or self.s2 is not None:
            return
        first = self.s2_count == 0 and self.frame >= 180
        review = self.s2_count == 1 and boundary and self.losses > self.losses_at_review and self.frame - self.s2_last >= 3600
        if not (first or review) or self.remaining("system2_calls") <= 1 or self.remaining("wall_s") < 30:
            return
        evidence = {"score": obs.get("score"), "lives": obs.get("lives"),
                    "losses": self.losses, "recent_controller": facts,
                    "current_policy": self.policy}
        prompt = (
            "You design a persistent control contract, not a current Atari action. "
            "An independent 60-Hz controller serves, averages recent collision-free RAM displacement, "
            "reflects the landing, and recomputes safe paddle overlap every frame. "
            "The archived single-frame predictor chattered LEFT/RIGHT and lost all lives. "
            "Jev receives named prediction certainty, catch slack, speed and your policy. "
            "It selects STABLE (centered interception) or QUIET (hold only inside the current safe corridor). "
            "Two reordered choices must agree with selected probability >=0.70, confidence >=0.15; "
            "a separate safety veto must be <=0.15 for QUIET. Code revalidates all safety conditions. "
            "Choose STABILITY if uncertainty, speed or losses justify centered catches; ECONOMY only "
            "if unnecessary movement appears the more important problem. Your answer may arrive much "
            "later and therefore must express a general preference, never a side, coordinate or action. "
            "Reply with exactly STABILITY or ECONOMY, no explanation.\n" + json.dumps(evidence)
        )
        self.s2_last = self.frame
        self.s2_tag = self.frame
        self.s2_count += 1
        self.losses_at_review = self.losses
        self.s2 = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=240.0))

    def run_episode(self, game):
        ctx = self.ctx
        try:
            while not ctx.done and not game.done:
                obs = ctx.observe()
                if not isinstance(obs, dict):
                    ctx.act("NOOP")
                    ctx.wait_for_tick()
                    continue
                frame = int(number(obs.get("frame"), self.frame + 1))
                if frame == self.frame:
                    ctx.wait_for_tick()
                    continue
                boundary = self.observe_motion(obs)
                facts = self.facts(obs)
                self.last_facts = {k: facts[k] for k in ("phase", "eta", "target", "uncertainty", "catch", "spread")}
                self.collect()
                mode = self.mode if self.frame <= self.mode_until else "STABLE"
                if facts["uncertainty"] != "settled" or facts["catch"] != "comfortable" or number(obs.get("lives"), 5) <= 1:
                    mode = "STABLE"
                action = self.action(obs, facts, mode)
                fallback = self.action(obs, facts, "STABLE")
                # Post first: model scheduling is outside the deadline-critical path.
                ctx.act(action if action in ACTIONS else "NOOP")
                self.last_action = action
                if action != fallback:
                    self.emit("jev_changes_action", frame=self.frame, action=action,
                              fallback=fallback, contract=mode)
                if self.frame % 120 == 0:
                    self.emit("controller_status", frame=self.frame, mode=mode,
                              policy=self.policy, **self.last_facts)
                self.launch_jev(obs, facts)
                self.launch_astra(obs, facts, boundary)
                ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver):
            return


def build(ctx):
    return Harness(ctx)

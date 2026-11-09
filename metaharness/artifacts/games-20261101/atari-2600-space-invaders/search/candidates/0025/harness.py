"""0025: calibrated evidence control with a post-life route reversal.

The productive mechanism from 0016/0023 remains: a fresh combined-FIRE sweep
runs every tick while short asynchronous Jev reviews classify ambiguous sensor
evidence.  The only new control transition is conservative and event driven:
a confirmed life decrement reverses the next sweep heading, so a respawn does
not immediately replay the route phase that preceded the hit.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "left", "right", "left_fire", "right_fire")


def num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    return int(num(value, default))


def bounded(value, low, high):
    return max(low, min(high, value))


def mapping(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def truth(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def observation(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return mapping(value)


def answer_field(answer, name, default=None):
    if isinstance(answer, dict):
        return answer.get(name, default)
    return getattr(answer, name, default)


def agreed(answers, key):
    """Accept a high-stakes review only when its two orderings agree."""
    first = answers.get(key + "a")
    second = answers.get(key + "b")
    value = answer_field(first, "value")
    allowed = ("INCOMING", "OUTGOING", "UNRESOLVED")
    if not isinstance(value, str) or value not in allowed:
        return None
    if answer_field(second, "value") != value:
        return None
    for result in (first, second):
        probabilities = mapping(answer_field(result, "probabilities", {}))
        if num(probabilities.get(value), 0.0) < 0.72:
            return None
        confidence = answer_field(result, "confidence")
        if confidence is not None and num(confidence, -1.0) < 0.30:
            return None
    return value


class Mailbox:
    """One bounded background inference call; never owns a polling loop."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.future = None
        self.tag = None

    def launch(self, function, tag):
        if self.future is not None or self.ctx.done:
            return False

        def work():
            try:
                return {"result": function()}
            except Exception as exc:  # ctx records fatal provider conditions itself
                return {"error": type(exc).__name__}

        try:
            self.future = self.ctx.spawn(work)
            self.tag = tag
            return True
        except Exception:
            self.future = None
            self.tag = None
            return False

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, tag = self.future, self.tag
        self.future = None
        self.tag = None
        try:
            result = future.result()
        except Exception as exc:
            result = {"error": type(exc).__name__}
        return mapping(result), tag

    def close(self):
        if self.future is not None:
            self.future.cancel()
        self.future = None
        self.tag = None


class EvidenceController:
    """Parent sweep plus one confirmed-life route-diversification transition."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.previous = None
        self.px, self.py = 90.0, 190.0
        self.lives, self.wave, self.score = 3, 1, 0.0
        self.epoch = 0

        # heading is an abstract command: +1 is the action named RIGHT.
        # It is deliberately separate from screen x direction.
        self.heading = 1
        self.left, self.right = 38.0, 116.0
        self.speed = 0.55
        self.action_previous = "noop"

        # The parent treated command sign as screen-x sign.  The canonical
        # build can present the horizontal axis mirrored.  Learn it from
        # actual, non-teleporting player displacement and use it only where
        # geometry is required.
        self.polarity = 1
        self.polarity_votes = deque(maxlen=9)
        self.polarity_logged = False

        self.positions = deque(maxlen=48)
        self.launches = deque(maxlen=32)
        self.tracks = {}
        self.chips = {}
        self.serial = 0
        self.mask_previous = None
        self.recent_alive = set()

        self.rubric = "CORROBORATE"
        self.plan_version = 0
        self.last_query = -10000
        self.last_weak_escape = -10000
        self.escape_until = -1
        self.last_motion_frame = 0
        self.motion_gap_start = None
        # A route diversification marker, not a new timer or movement loop.
        self.last_respawn_reverse_frame = -10000

        self.count = {
            "actions": 0,
            "down_in_shot_field": 0,
            "ambiguous_reviews": 0,
            "jev_promotions": 0,
            "jev_suppressions": 0,
            "jev_changed_actions": 0,
            "strong_dodges": 0,
            "chip_escapes": 0,
            "life_losses": 0,
            "respawn_reversals": 0,
            "identity_revivals": 0,
            "polarity_samples": 0,
            "polarity_corrections": 0,
            "astra_updates": 0,
        }

    def log(self, event, **fields):
        try:
            self.ctx.log(event, frame=self.frame, **fields)
        except Exception:
            pass

    def next_id(self, prefix):
        self.serial += 1
        return prefix + str(self.serial)

    @staticmethod
    def command_sign(action):
        if "left" in str(action):
            return -1
        if "right" in str(action):
            return 1
        return 0

    def screen_sign(self, action):
        command = self.command_sign(action)
        return command * (self.polarity if self.polarity else 1)

    def learn_polarity(self, dx, action, dt):
        command = self.command_sign(action)
        if not command or abs(dx) < 0.45 or abs(dx) > max(16.0, 4.5 * dt):
            return
        vote = 1 if dx * command > 0 else -1
        self.polarity_votes.append(vote)
        self.count["polarity_samples"] += 1
        positives = sum(1 for item in self.polarity_votes if item > 0)
        negatives = len(self.polarity_votes) - positives
        if len(self.polarity_votes) < 2:
            return
        learned = 1 if positives >= negatives else -1
        if learned != self.polarity:
            old = self.polarity
            self.polarity = learned
            self.count["polarity_corrections"] += 1
            self.log("joystick_polarity_calibrated", old=old, new=learned,
                     samples=len(self.polarity_votes), last_dx=round(dx, 2),
                     command=action)
        elif not self.polarity_logged:
            self.polarity_logged = True
            self.log("joystick_polarity_confirmed", polarity=self.polarity,
                     samples=len(self.polarity_votes))

    def shield_cells(self, obs):
        cells = {}
        shields = records(obs.get("shields"))[:3]
        for shield_index, shield in enumerate(shields):
            mask = shield.get("health_mask")
            if not isinstance(mask, list):
                continue
            x0 = num(shield.get("x")) - 4
            for row, bits in enumerate(mask[:9]):
                value = integer(bits)
                for col in range(8):
                    if value & (1 << (7 - col)):
                        cells[(shield_index, row, col)] = x0 + col
        return cells

    def note_shield_chips(self, obs, discontinuity):
        current = self.shield_cells(obs)
        old, self.mask_previous = self.mask_previous, current
        if discontinuity or old is None or len(records(obs.get("shields"))) != 3:
            return
        removed = sorted(set(old) - set(current))
        # A large redraw is not a ballistic cue.
        if not removed or len(removed) > 24 or (len(old) > 12 and not current):
            return
        by_shield = {}
        for key in removed:
            shield_index, row, col = key
            top = min((r for s, r, c in old if s == shield_index and c == col), default=9)
            by_shield.setdefault(shield_index, []).append((old[key], row <= top + 2))
        for shield_index, changes in by_shield.items():
            xs = [x for x, unused in changes]
            x = sum(xs) / len(xs)
            top_entry = any(is_top for unused, is_top in changes)
            own = any(abs(x - launch_x) <= 4.5 and 0 <= self.frame - launch_frame <= 48
                      for launch_x, launch_frame in self.launches)
            remaining = sum(1 for xx in current.values() if abs(xx - x) <= 1.5)
            key = self.next_id("C")
            self.chips[key] = {
                "id": key,
                "x": x,
                "f": self.frame,
                "top": top_entry,
                "own": own,
                "thin": remaining <= 6,
                "verdict": None,
                "verdict_f": -10000,
                "used": False,
                "reviewed": False,
            }
            if top_entry and not own:
                self.log("unattributed_chip", cue=key, x=round(x, 1),
                         thin=remaining <= 6, polarity=self.polarity)

    def track_projectiles(self, obs):
        points = []
        for bomb in records(obs.get("bombs"))[:12]:
            if bomb.get("x") is not None and bomb.get("y") is not None:
                points.append((num(bomb.get("x")), num(bomb.get("y")), "bomb"))
        shot = mapping(obs.get("player_shot"))
        if truth(shot.get("active")) and shot.get("x") is not None and shot.get("y") is not None:
            sx, sy = num(shot.get("x")), num(shot.get("y"))
            if not any(abs(sx - x) < 2 and abs(sy - y) < 3 for x, y, unused in points):
                points.append((sx, sy, "player_shot"))

        used = set()
        for x, y, source in points:
            if not (0 <= x <= 159 and 0 <= y <= 209):
                continue
            candidates = []
            for key, track in self.tracks.items():
                dt = self.frame - track["f"]
                if key in used or not (0 < dt <= 12) or abs(x - track["x"]) > 2.5:
                    continue
                dy = y - track["y"]
                if abs(dy) > 3.5 * dt + 3:
                    continue
                # A newly spawned muzzle beam cannot be an upper continuation.
                if y >= 168 and track["y"] < 150 and dy > 0:
                    continue
                prediction = track["y"] + track["vy"] * dt
                candidates.append((abs(y - prediction) + 3 * abs(x - track["x"]), key))
            if candidates:
                _, key = min(candidates)
                track = self.tracks[key]
                dt = self.frame - track["f"]
                dy = y - track["y"]
                if abs(dy) >= 1:
                    velocity = bounded(dy / max(1, dt), -3.5, 3.5)
                    old_sign = 1 if track["vy"] > 0.2 else -1 if track["vy"] < -0.2 else 0
                    new_sign = 1 if velocity > 0.2 else -1 if velocity < -0.2 else 0
                    if old_sign and new_sign and old_sign != new_sign:
                        track["rev"] += 1
                        track["verdict"] = None
                        track["reviewed"] = False
                        track["down"] = track["up"] = 0
                        track["own"] = False
                    track["vy"] = velocity
                    track["down"] = track["down"] + 1 if new_sign > 0 else 0
                    track["up"] = track["up"] + 1 if new_sign < 0 else 0
                track.update(x=x, y=y, f=self.frame, source=source)
                track["samples"] += 1
            else:
                key = self.next_id("T")
                near_muzzle = y >= self.py - 25 and abs(x - (self.px + 1)) <= 5
                own_launch = source == "player_shot" and near_muzzle and "fire" in self.action_previous
                track = {
                    "id": key, "x": x, "y": y, "f": self.frame, "born": self.frame,
                    "source": source, "first_source": source, "vy": 0.0,
                    "samples": 1, "up": 0, "down": 0, "own": own_launch,
                    "rev": 0, "verdict": None, "verdict_f": -10000,
                    "reviewed": False, "logged": False,
                }
                self.tracks[key] = track
                if own_launch:
                    self.launches.append((x, self.frame))
            used.add(key)
            if track["down"] >= 2 and track["first_source"] == "player_shot" and not track["logged"]:
                track["logged"] = True
                self.count["down_in_shot_field"] += 1
                self.log("descending_shot_field", track=key, x=round(x, 1),
                         y=round(y, 1), vy=round(track["vy"], 2),
                         polarity=self.polarity)

        self.tracks = {key: item for key, item in self.tracks.items()
                       if self.frame - item["f"] <= 24}
        self.chips = {key: item for key, item in self.chips.items()
                      if self.frame - item["f"] <= 120}

    def update(self, raw):
        obs = observation(raw)
        frame = integer(obs.get("frame"), self.frame + 4)
        player = mapping(obs.get("player"))
        x = bounded(num(player.get("x"), self.px), 0, 159)
        y = bounded(num(player.get("y"), self.py), 150, 209)
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        dt = max(1, frame - self.frame)
        lost = lives < self.lives
        jump = self.previous is not None and abs(x - self.px) > max(12.0, 1.5 * dt)
        discontinuity = self.previous is not None and (lost or wave != self.wave or jump)

        if lost:
            old_heading = self.heading
            # The only new policy transition: begin the fresh life on the
            # opposite sweep phase.  Ordinary x jumps and motion gaps do not
            # trigger this, because they may be decoder/death-animation noise.
            self.heading = -old_heading if old_heading else 1
            self.last_respawn_reverse_frame = frame
            self.count["life_losses"] += max(1, self.lives - lives)
            self.count["respawn_reversals"] += 1
            self.log("respawn_sweep_reversal", old_heading=old_heading,
                     new_heading=self.heading, previous_x=round(self.px, 1),
                     lives=lives)
            self.log("life_evidence", lives=lives, previous_x=round(self.px, 1),
                     last_mobile_frame=self.last_motion_frame,
                     gap_start=self.motion_gap_start,
                     polarity=self.polarity,
                     warning="life flag may lag impact; route phase reversed once")

        self.frame = frame
        if discontinuity:
            self.epoch += 1
            self.tracks.clear()
            self.chips.clear()
            self.launches.clear()
            self.escape_until = -1
            self.motion_gap_start = None
        elif self.previous is not None:
            dx = x - self.px
            self.learn_polarity(dx, self.action_previous, dt)
            command = self.command_sign(self.action_previous)
            expected_screen = command * (self.polarity if self.polarity else 1)
            if command and expected_screen * dx > 0.35:
                self.speed = bounded(0.86 * self.speed + 0.14 * abs(dx) / dt, 0.25, 1.5)
                self.last_motion_frame = frame
                self.motion_gap_start = None
            elif command and abs(dx) < 0.5 and 43 < x < 111:
                if self.motion_gap_start is None:
                    self.motion_gap_start = frame - dt

        self.px, self.py = x, y
        self.lives, self.wave = lives, wave
        self.positions.append((frame, x))
        self.track_projectiles(obs)
        self.note_shield_chips(obs, discontinuity)

        live = {(integer(item.get("row")), integer(item.get("col")))
                for item in records(obs.get("invaders")) if truth(item.get("alive"))}
        if self.previous is not None and wave == integer(self.previous.get("wave"), wave):
            revived = live - self.recent_alive
            if revived:
                self.count["identity_revivals"] += len(revived)
                if len(revived) >= 3:
                    self.log("lattice_identity_conflict", revived=len(revived),
                             consequence="no aim or kill calibration")
        self.recent_alive = live
        self.score = num(obs.get("score"), self.score)
        self.previous = obs
        return obs

    def correlated_chip(self, track):
        return any(chip["top"] and not chip["own"]
                   and abs(chip["x"] - track["x"]) <= 5
                   and abs(chip["f"] - track["f"]) <= 24
                   for chip in self.chips.values())

    def hostile(self, track, use_model=True):
        if track["up"] >= 1:
            return False
        if track["source"] == "bomb" or track["down"] >= 2:
            return True
        if not use_model or track.get("verdict") != "INCOMING":
            return False
        if track["down"] < 1 or track["own"]:
            return False
        if self.frame - track.get("verdict_f", -10000) > 24:
            return False
        return self.rubric == "MOTION" or self.correlated_chip(track)

    def sweep(self):
        if self.px <= self.left:
            return 1
        if self.px >= self.right:
            return -1
        return self.heading

    def path_risk(self, candidate, use_model=True):
        """Predict screen x with calibrated command-to-screen polarity."""
        prior = self.command_sign(self.action_previous)
        polarity = self.polarity if self.polarity else 1
        worst = 0.0
        for track in self.tracks.values():
            if not self.hostile(track, use_model):
                continue
            age = max(0, self.frame - track["f"])
            velocity = track["vy"] if track["vy"] > 0.2 else 1.2
            for future in range(0, 65, 2):
                by = track["y"] + velocity * (age + future)
                if not self.py - 13 <= by <= self.py + 8:
                    continue
                for sticky_delay in (0, 4):
                    command_distance = (prior * min(future, sticky_delay)
                                        + candidate * max(0, future - sticky_delay))
                    predicted_x = bounded(self.px + polarity * self.speed * command_distance,
                                          37, 119)
                    gap = abs(predicted_x - track["x"])
                    if gap <= 9:
                        worst = max(worst, 100 + (9 - gap) + (64 - future) / 64)
                    elif gap < 15:
                        worst = max(worst, (15 - gap) / 6)
        return worst

    def decision(self, use_model=True):
        baseline = self.sweep()
        risks = {candidate: self.path_risk(candidate, use_model)
                 for candidate in (-1, 0, 1)}
        if risks[baseline] >= 100:
            escape = min(risks, key=lambda value: (risks[value], value == 0, value != baseline))
            if risks[escape] + 0.5 < risks[baseline]:
                return escape, "projectile", None
        if self.frame < self.escape_until and risks[baseline] < 100:
            return baseline, "escape_commit", None

        # Preserve the parent’s deliberately weak shield fallback.  Respawn
        # reversal is applied in update(), never inferred from this evidence.
        if self.frame - self.last_weak_escape >= 48:
            candidates = []
            for key, chip in self.chips.items():
                if chip["used"] or chip["own"] or not chip["top"]:
                    continue
                if self.frame - chip["f"] > 96:
                    continue
                verdict = chip.get("verdict") if use_model else None
                if verdict == "OUTGOING":
                    continue
                if not chip["thin"] and verdict != "INCOMING":
                    continue
                gap = chip["x"] - self.px
                if abs(gap) > 23 or baseline * gap < -2:
                    continue
                away = -1 if gap > 1 else 1 if gap < -1 else (1 if self.px < 77 else -1)
                if (away < 0 and self.px <= 42) or (away > 0 and self.px >= 112):
                    continue
                if risks[away] >= 100:
                    continue
                candidates.append((abs(gap), key, away))
            if candidates:
                _, key, away = min(candidates)
                return away, "chip", key
        return baseline, "sweep", None

    def action(self):
        fallback, unused_source, unused_cue = self.decision(False)
        chosen, source, cue = self.decision(True)
        if chosen != fallback:
            self.count["jev_changed_actions"] += 1
            self.log("jev_changed_action", baseline=fallback, selected=chosen,
                     source=source, cue=cue, polarity=self.polarity)
        if source == "chip" and cue in self.chips:
            self.chips[cue]["used"] = True
            self.last_weak_escape = self.frame
            self.escape_until = self.frame + 16
            self.count["chip_escapes"] += 1
            self.log("chip_escape", cue=cue, direction=chosen,
                     model=self.chips[cue].get("verdict"))
        elif source == "projectile" and chosen != self.sweep():
            self.count["strong_dodges"] += 1
            self.log("projectile_escape", direction=chosen, polarity=self.polarity)
        if chosen:
            self.heading = chosen
        self.count["actions"] += 1
        return "left_fire" if chosen < 0 else "right_fire" if chosen > 0 else "fire"

    def packet(self):
        candidates, tags = {}, {}
        for key, track in self.tracks.items():
            if track["reviewed"] or self.frame - track["f"] > 8:
                continue
            if track["own"] or track["up"] or track["down"] >= 2 or track["source"] == "bomb":
                continue
            if track["y"] < 115 or (track["down"] < 1 and not self.correlated_chip(track)):
                continue
            candidates[key] = {
                "kind": "projectile",
                "decoder_label": track["source"],
                "motion": "one_downward_increment" if track["down"] else "not_yet_measured",
                "own_launch": "not_matched",
                "same_lane_top_chip": self.correlated_chip(track),
                "position": "below_shields" if track["y"] >= 174 else "above_or_at_shields",
                "continuity": "short_matched_track",
                "model_role": "classify evidence only; never choose movement",
            }
            tags[key] = {"kind": "track", "rev": track["rev"], "f": track["f"]}
            if len(candidates) >= 2:
                break

        for key, chip in reversed(list(self.chips.items())):
            if len(candidates) >= 3:
                break
            if chip["reviewed"] or chip["used"] or not chip["top"]:
                continue
            if self.frame - chip["f"] > 32:
                continue
            ascending = any(track["up"] and abs(track["x"] - chip["x"]) <= 5
                            and abs(track["f"] - chip["f"]) <= 24
                            for track in self.tracks.values())
            descending = any(track["down"] and abs(track["x"] - chip["x"]) <= 5
                             and abs(track["f"] - chip["f"]) <= 24
                             for track in self.tracks.values())
            candidates[key] = {
                "kind": "shield_chip",
                "erosion": "top_entry",
                "own_launch": "matched" if chip["own"] else "not_matched",
                "ascending_same_lane": ascending,
                "descending_same_lane": descending,
                "remaining_cover": "thin" if chip["thin"] else "substantial",
                "interpretation_limit": "damage can consume a projectile; do not infer a surviving bomb",
            }
            tags[key] = {"kind": "chip", "f": chip["f"]}
        if not candidates:
            return None

        state = {
            "schema": "space_invaders_temporal_evidence_v3_respawn_reverse",
            "control": {
                "joystick_polarity": "screen_x_follows_command" if self.polarity > 0 else "screen_x_opposes_command",
                "motion": "responsive" if self.motion_gap_start is None else "motion_gap",
                "post_life_direction": "reversed" if self.frame - self.last_respawn_reverse_frame <= 64 else "unchanged",
                "model_must_not_choose_action": True,
            },
            "candidates": candidates,
            "rules": {
                "up": "an outgoing laser rises toward smaller y",
                "down": "an incoming bomb descends toward larger y",
                "labels": "decoder labels are fallible; empty bombs is not proof of safety",
                "no_creation": "classify only supplied evidence, never invent an unseen projectile",
            },
            "plan": {
                "rubric": self.rubric,
                "meaning": "MOTION allows one continuous descending increment without an own launch; CORROBORATE additionally requires same-lane top shield damage",
            },
        }
        options = {
            "INCOMING": "Evidence favors enemy-origin descending fire or enemy-origin top shield damage, without an outgoing-shot contradiction.",
            "OUTGOING": "An own launch or ascending same-lane trajectory explains the signal; do not dodge a friendly laser.",
            "UNRESOLVED": "Evidence is insufficient, discontinuous, or contradictory; do not manufacture certainty.",
        }
        questions = {}
        for key in candidates:
            instruction = (
                "Classify candidates." + key + " from the supplied temporal, launch, and shield facts. "
                "A shield chip is origin evidence only, not a surviving bomb. A decoder name or empty bomb list is insufficient. "
                "For a projectile, respect plan.rubric. Top-entry plus matching descent supports INCOMING; "
                "matching own launch or ascending motion supports OUTGOING; otherwise use UNRESOLVED. "
                "Do not calculate coordinates and do not select movement."
            )
            ordered = list(options.items())
            if self.serial % 2:
                ordered.reverse()
            questions[key + "a"] = Choice(instruction, dict(ordered))
            questions[key + "b"] = Choice(instruction, dict(reversed(ordered)))
        tag = {"frame": self.frame, "epoch": self.epoch, "plan": self.plan_version, "items": tags}
        return state, questions, tag

    def mark_submitted(self, tag):
        self.last_query = self.frame
        for key, item in mapping(tag.get("items")).items():
            target = self.tracks.get(key) if item.get("kind") == "track" else self.chips.get(key)
            if target is not None:
                target["reviewed"] = True
        self.count["ambiguous_reviews"] += 1

    def accept(self, packet):
        if packet is None:
            return
        result, tag = packet
        if result.get("error"):
            self.log("jev_failure", error=result["error"])
            return
        if not isinstance(tag, dict) or tag.get("epoch") != self.epoch or tag.get("plan") != self.plan_version:
            return
        if not 0 <= self.frame - integer(tag.get("frame"), -10000) <= 36:
            self.log("jev_discard", reason="age")
            return
        answers = mapping(result.get("result"))
        for key, item in mapping(tag.get("items")).items():
            verdict = agreed(answers, key)
            if verdict is None:
                continue
            target = self.tracks.get(key) if item.get("kind") == "track" else self.chips.get(key)
            if target is None:
                continue
            if item.get("kind") == "track":
                if target["rev"] != integer(item.get("rev"), -1) or self.frame - target["f"] > 8:
                    continue
                if verdict == "INCOMING" and (target["up"] or target["own"] or target["down"] < 1):
                    continue
            elif verdict == "INCOMING":
                if (target["own"] or not target["top"] or
                        not any(track["down"] and abs(track["x"] - target["x"]) <= 5
                                and abs(track["f"] - target["f"]) <= 24
                                for track in self.tracks.values())):
                    continue
            target["verdict"] = verdict
            target["verdict_f"] = self.frame
            if verdict == "INCOMING":
                self.count["jev_promotions"] += 1
            elif verdict == "OUTGOING":
                self.count["jev_suppressions"] += 1
            self.log("jev_evidence_verdict", cue=key, verdict=verdict,
                     age=self.frame - integer(tag.get("frame"), self.frame))

    def astra_prompt(self):
        evidence = {
            "score": self.score,
            "lives": self.lives,
            "downward_tracks": self.count["down_in_shot_field"],
            "blind_losses": self.count["life_losses"],
            "identity_revivals": self.count["identity_revivals"],
            "joystick_polarity": self.polarity,
            "polarity_samples": self.count["polarity_samples"],
        }
        return (
            "Choose one evidence rubric for a real-time Atari 2600 Space Invaders sensor reviewer. "
            "Reply with exactly one token: MOTION or CORROBORATE. Code holds combined FIRE and sweeps every tick. "
            "Jev only classifies supplied temporal evidence with paired INCOMING/OUTGOING/UNRESOLVED Choices; "
            "it never selects a joystick action. MOTION permits a single continuous downward increment without own-shot evidence; "
            "CORROBORATE additionally requires same-lane top shield damage. Neither option may invent an unseen projectile. "
            "The code has separately calibrated whether command labels increase or decrease screen x; do not propose code or prose. "
            "Current evidence: " + json.dumps(evidence, separators=(",", ":"), allow_nan=False)
        )

    def apply_astra(self, packet):
        if packet is None:
            return
        result, unused_tag = packet
        if result.get("error"):
            self.log("astra_failure", error=result["error"])
            return
        model_result = result.get("result")
        text = getattr(model_result, "text", None)
        if not isinstance(text, str):
            return
        choice = text.strip().strip("\"'` .\n").upper()
        if choice not in ("MOTION", "CORROBORATE"):
            self.log("astra_rejected", reason="not_one_allowed_token")
            return
        if choice != self.rubric:
            old = self.rubric
            self.rubric = choice
            self.plan_version += 1
            self.count["astra_updates"] += 1
            for track in self.tracks.values():
                track["reviewed"] = False
                track["verdict"] = None
            self.log("astra_rubric_changed", old=old, new=choice,
                     plan_version=self.plan_version)
        else:
            self.log("astra_rubric_confirmed", rubric=choice)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        controller = EvidenceController(ctx)
        jev = Mailbox(ctx)
        astra = Mailbox(ctx)
        spec = getattr(game, "action_spec", None)
        legal = set(getattr(spec, "names", ACTIONS))
        astra_sent = False
        last_frame = None
        try:
            while not ctx.done and not getattr(game, "done", False):
                raw = observation(ctx.observe())
                if truth(raw.get("game_over")):
                    break
                frame = integer(raw.get("frame"), controller.frame + 4)
                if frame == last_frame:
                    ctx.wait_for_tick()
                    continue
                last_frame = frame
                controller.update(raw)
                controller.apply_astra(astra.poll())
                controller.accept(jev.poll())

                action = controller.action()
                if action not in legal:
                    action = "fire" if "fire" in legal else getattr(spec, "noop", "noop")
                ctx.act(action)
                controller.action_previous = action

                remaining = mapping(ctx.budget.remaining())
                if (jev.future is None and controller.frame - controller.last_query >= 16
                        and num(remaining.get("system1_calls")) > 1
                        and num(remaining.get("wall_s")) > 2):
                    packet = controller.packet()
                    if packet is not None:
                        state, questions, tag = packet
                        if jev.launch(lambda s=state, q=questions: ctx.decide(s, q, timeout_s=1.0), tag):
                            controller.mark_submitted(tag)

                if (not astra_sent and num(remaining.get("system2_calls")) > 0
                        and num(remaining.get("wall_s")) > 60):
                    prompt = controller.astra_prompt()
                    astra_sent = astra.launch(
                        lambda p=prompt: ctx.think(p, timeout_s=55.0),
                        {"frame": controller.frame},
                    )

                if controller.count["actions"] % 150 == 0:
                    controller.log("evidence_summary", score=controller.score,
                                   lives=controller.lives, counts=dict(controller.count),
                                   polarity=controller.polarity, rubric=controller.rubric)
                ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver):
            pass
        finally:
            jev.close()
            astra.close()
            controller.log("evidence_finished", score=controller.score,
                           lives=controller.lives, counts=dict(controller.count),
                           polarity=controller.polarity, rubric=controller.rubric)


def build(ctx):
    return Harness(ctx)

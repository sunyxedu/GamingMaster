"""Hybrid tier-2 controller for Atari 2600 Space Invaders.

The emulator is allowed to run every tick.  Code owns the irreversible pieces
(target geometry, the one-shot lock, collision margins, and a small lateral
patrol); Jev judges only a compact tactical state, and Astra writes a slow
wave-level policy in the background.  A deterministic controller is always
available when either model is late or returns unusable data.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice, Noul


# The canonical decoder reports the left edge of an invader sprite and the
# player/laser lane is about eleven pixels to its right.  This is deliberately
# kept as a named calibration rather than asking a model to do pixel arithmetic.
INVADER_TO_PLAYER_OFFSET = 11
PLAYER_MIN_X = 35
PLAYER_MAX_X = 143
ALIGN_TOLERANCE = 4


def _int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _obs_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _player(obs: dict[str, Any]) -> tuple[int, int]:
    p = obs.get("player")
    if not isinstance(p, dict):
        return 90, 190
    return _int(p.get("x"), 90), _int(p.get("y"), 190)


def _invaders(obs: dict[str, Any]) -> list[dict[str, Any]]:
    raw = obs.get("invaders")
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _alive(invader: dict[str, Any]) -> bool:
    value = invader.get("alive", False)
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _bombs(obs: dict[str, Any]) -> list[dict[str, Any]]:
    raw = obs.get("bombs")
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _shot_active(obs: dict[str, Any]) -> bool:
    shot = obs.get("player_shot")
    return isinstance(shot, dict) and bool(shot.get("active", False))


def _wave(obs: dict[str, Any]) -> int | None:
    value = obs.get("wave")
    if value is None:
        return None
    parsed = _int(value, -1)
    return None if parsed < 0 else parsed


def _shield_pixels(shield: dict[str, Any]) -> int:
    if "pixels" in shield:
        return max(0, _int(shield.get("pixels"), 0))
    mask = shield.get("health_mask")
    if not isinstance(mask, list):
        return 0
    total = 0
    for row in mask:
        number = _int(row, 0)
        total += max(0, number & 255).bit_count()
    return total


def _answer_value(answer: Any) -> Any:
    if isinstance(answer, dict):
        if "value" in answer:
            return answer.get("value")
        if "choice" in answer:
            return answer.get("choice")
    return getattr(answer, "value", None)


def _answer_prob(answer: Any, value: Any) -> float:
    probabilities = answer.get("probabilities", {}) if isinstance(answer, dict) else getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    wanted = str(value)
    for key, probability in probabilities.items():
        if str(key).upper() == wanted.upper():
            try:
                return float(probability)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _choice(answer: Any) -> str:
    value = _answer_value(answer)
    return value.upper() if isinstance(value, str) else ""


def _noul_probability(answer: Any) -> float | None:
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0 or number > 1.0:
        return None
    return number


def _safe_json_object(text: Any) -> dict[str, Any]:
    """Extract the first small JSON object from arbitrary Astra output."""
    if not isinstance(text, str):
        return {}
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.plan = {
            "sweep": "LEFT_TO_RIGHT",
            "risk": "CAUTIOUS",
            "bonus": "NO",
            "priority": "SURVIVE",
        }
        self.plan_version = 0
        self.astra_future = None
        self.astra_session = None
        self.astra_started = False
        self.astra_revision_started = False

        self.jev_future = None
        self.jev_meta: dict[str, Any] | None = None
        self.tactical: dict[str, Any] | None = None

        self.target_col: int | None = None
        self.target_preference = "PLAN"
        self.previous_obs: dict[str, Any] = {}
        self.last_lives: int | None = None
        self.last_wave: int | None = None
        self.last_score: int = 0
        self.last_posted_action = "noop"
        self.recovery_ticks = 0

        self.patrol_direction = "right"
        self.patrol_ticks = 0
        self.patrol_phase = "away"
        self.fire_frame: int | None = None

    # ---------- model scheduling ----------

    def _astra_prompt(self, game: Any, obs: dict[str, Any], reason: str) -> str:
        compact = self._compact_state(obs)
        try:
            description = game.describe()
        except Exception:
            description = "Atari 2600 Space Invaders, canonical mode 0 difficulty 0."
        return (
            "You are the slow wave strategist for a real-time Atari 2600 Space Invaders "
            "controller.  The game advances while you think, so do not prescribe a "
            "sequence of timed actions.  Code handles exact coordinates, collision "
            "margins, legality, the one-player-shot lock, and every tick.  Choose only "
            "a durable policy for the fast tactical controller.\n\n"
            f"Game description:\n{description}\n\n"
            f"Reason for this plan/revision: {reason}\n"
            f"Current derived state:\n{json.dumps(compact, sort_keys=True)}\n\n"
            "Return one JSON object (no markdown) with only these fields: "
            "sweep (LEFT_TO_RIGHT or RIGHT_TO_LEFT), risk (CAUTIOUS, BALANCED, "
            "or AGGRESSIVE), bonus (YES or NO for opportunistic UFO shots), and "
            "priority (SURVIVE or SCORE).  Prefer survival and an outer-column sweep "
            "unless the evidence in the state supports changing them."
        )

    def _launch_astra(self, game: Any, obs: dict[str, Any], reason: str, revision: bool = False) -> None:
        if self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if _int(remaining.get("system2_calls"), 0) <= 0:
                return
        except Exception:
            pass
        prompt = self._astra_prompt(game, obs, reason)
        session = self.astra_session

        def call() -> Any:
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=75.0)
            except Exception as exc:  # the main loop retains its code fallback
                return {"_error": type(exc).__name__}

        try:
            self.astra_future = self.ctx.spawn(call)
            self.astra_started = True
            if revision:
                self.astra_revision_started = True
        except Exception:
            self.astra_future = None

    def _poll_astra(self) -> None:
        future = self.astra_future
        if future is None:
            return
        try:
            if not future.done():
                return
            result = future.result()
        except Exception:
            self.astra_future = None
            return
        self.astra_future = None
        text = getattr(result, "text", None)
        session = getattr(result, "session_id", None)
        if isinstance(result, dict):
            text = result.get("text", text)
            session = result.get("session_id", session)
        if isinstance(session, str) and session:
            self.astra_session = session
        proposal = _safe_json_object(text)
        changed = False
        sweep = _text(proposal.get("sweep")).upper()
        risk = _text(proposal.get("risk")).upper()
        bonus = _text(proposal.get("bonus")).upper()
        priority = _text(proposal.get("priority")).upper()
        if sweep in {"LEFT_TO_RIGHT", "RIGHT_TO_LEFT"}:
            if self.plan["sweep"] != sweep:
                self.plan["sweep"] = sweep
                changed = True
        if risk in {"CAUTIOUS", "BALANCED", "AGGRESSIVE"}:
            if self.plan["risk"] != risk:
                self.plan["risk"] = risk
                changed = True
        if bonus in {"YES", "NO"}:
            if self.plan["bonus"] != bonus:
                self.plan["bonus"] = bonus
                changed = True
        if priority in {"SURVIVE", "SCORE"}:
            if self.plan["priority"] != priority:
                self.plan["priority"] = priority
                changed = True
        if changed:
            self.plan_version += 1
            self.ctx.log("astra_policy_updated", plan=dict(self.plan), version=self.plan_version)

    def _compact_state(self, obs: dict[str, Any]) -> dict[str, Any]:
        target = self._target_info(obs)
        danger = self._danger(obs)
        alive = [x for x in _invaders(obs) if _alive(x)]
        rows = [_int(x.get("row"), 0) for x in alive]
        lowest_y = max((_int(x.get("y"), 0) for x in alive), default=0)
        px, _ = _player(obs)
        if px < 65:
            lane = "LEFT"
        elif px > 110:
            lane = "RIGHT"
        else:
            lane = "CENTER"
        return {
            "schema": "space_invaders_tactical_v2",
            "frame": _int(obs.get("frame"), 0),
            "wave": _wave(obs),
            "lives": _int(obs.get("lives"), 0),
            "phase": "OPENING" if len(alive) >= 30 else ("MIDWAVE" if len(alive) >= 15 else "LATEWAVE"),
            "player": {"lane": lane, "x_bucket": "LOW" if px < 65 else ("HIGH" if px > 110 else "MID")},
            "shot": "ACTIVE" if _shot_active(obs) else "READY",
            "target": {
                "column": target.get("col"),
                "side": target.get("side", "NONE"),
                "row_band": target.get("row_band", "NONE"),
                "alignment": target.get("alignment", "NONE"),
                "shield": target.get("shield", "UNKNOWN"),
            },
            "threat": {
                "bomb": danger["bomb_side"],
                "urgency": danger["urgency"],
                "escape": danger["direction"],
                "formation": danger["formation"],
            },
            "formation": {
                "alive": len(alive),
                "lowest_band": "BOTTOM" if max(rows, default=0) >= 5 else ("MIDDLE" if max(rows, default=0) >= 3 else "TOP"),
                "lowest_y_band": "LOW" if lowest_y >= 135 else ("MID" if lowest_y >= 95 else "HIGH"),
                "left_outer_alive": any(_int(x.get("col"), -1) == 0 for x in alive),
                "right_outer_alive": any(_int(x.get("col"), -1) == 5 for x in alive),
            },
            "cover": {
                "left": _shield_quality(obs, 0),
                "center": _shield_quality(obs, 1),
                "right": _shield_quality(obs, 2),
            },
            "plan": dict(self.plan),
            "recent_action": self.last_posted_action,
            "recovery": self.recovery_ticks > 0,
        }

    def _jev_questions(self, state: dict[str, Any]) -> dict[str, Any]:
        # Rotate insertion order to avoid making the first option a permanent
        # default.  The descriptions still state the exact semantic test.
        frame = _int(state.get("frame"), 0)
        if (frame // 8) % 2:
            move_names = ["RIGHT", "HOLD", "LEFT"]
            defense_names = ["DODGE_RIGHT", "NO_DODGE", "DODGE_LEFT"]
        else:
            move_names = ["LEFT", "HOLD", "RIGHT"]
            defense_names = ["DODGE_LEFT", "NO_DODGE", "DODGE_RIGHT"]
        move_criteria = {
            name: {
                "LEFT": "Move left for this tick; choose only if it is consistent with the target alignment or the stated escape lane.",
                "RIGHT": "Move right for this tick; choose only if it is consistent with the target alignment or the stated escape lane.",
                "HOLD": "Release horizontal movement for this tick; use when aligned, patrolling, or movement would worsen the stated threat.",
            }[name]
            for name in move_names
        }
        defense_criteria = {
            name: {
                "DODGE_LEFT": "A bomb threat in `threat` makes left the safer immediate escape; this is not for a distant or absent bomb.",
                "DODGE_RIGHT": "A bomb threat in `threat` makes right the safer immediate escape; this is not for a distant or absent bomb.",
                "NO_DODGE": "There is no immediate bomb escape, or the listed escape is not supported by the current threat fields.",
            }[name]
            for name in defense_names
        }
        target_criteria = {
            "KEEP_CURRENT": "Keep the current target column while its living invader and shot geometry remain valid.",
            "LEFT_OUTER": "Prefer the leftmost living outer column for the next deliberate shot, not a dead column.",
            "RIGHT_OUTER": "Prefer the rightmost living outer column for the next deliberate shot, not a dead column.",
            "LOWEST_THREAT": "Prefer a living column whose lowest invader is the most immediate bottom threat.",
        }
        return {
            "defense": Choice(
                "Using only `threat.bomb`, `threat.urgency`, and `threat.escape`, classify the near-term defensive choice. Do not invent a bomb from unrelated fields.",
                defense_criteria,
            ),
            "move": Choice(
                "Using `player`, `target.alignment`, `target.side`, and `threat`, choose the horizontal intent for the next tick. This is a tactical judgment, not a long plan.",
                move_criteria,
            ),
            "fire": Noul(
                "Should the controller fire on this tick? Say yes only when `shot` is READY, the target alignment is ALIGNED, there is no immediate bomb/formation emergency, and the shot is useful; otherwise say no.",
                {
                    "true": "Fire now: the one-shot lock is ready and the aligned target is safe and useful.",
                    "false": "Do not fire now: the shot is active, the target is not aligned, or an immediate threat makes waiting safer.",
                },
            ),
            "target": Choice(
                "Choose the next target policy. Code will reject a dead or unsafe column and will not switch during an active shot.",
                target_criteria,
            ),
        }

    def _launch_jev(self, obs: dict[str, Any]) -> None:
        if self.jev_future is not None:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if _int(remaining.get("system1_calls"), 0) <= 0:
                return
        except Exception:
            pass
        state = self._compact_state(obs)
        questions = self._jev_questions(state)
        meta = {
            "frame": _int(obs.get("frame"), 0),
            "lives": _int(obs.get("lives"), 0),
            "wave": _wave(obs),
            "target_col": self.target_col,
            "state": state,
        }

        def call() -> Any:
            try:
                return self.ctx.decide(state, questions, timeout_s=0.45)
            except Exception as exc:  # late/failed judgments use the code policy
                return {"_error": type(exc).__name__}

        try:
            self.jev_future = self.ctx.spawn(call)
            self.jev_meta = meta
        except Exception:
            self.jev_future = None
            self.jev_meta = None

    def _poll_jev(self) -> None:
        future = self.jev_future
        if future is None:
            return
        try:
            if not future.done():
                return
            result = future.result()
        except Exception:
            self.jev_future = None
            self.jev_meta = None
            return
        meta = self.jev_meta or {}
        self.jev_future = None
        self.jev_meta = None
        if not isinstance(result, dict) or result.get("_error"):
            return
        # Keep raw typed answers only briefly.  The current observation is
        # rechecked before any answer can influence an action.
        self.tactical = {
            "frame": _int(meta.get("frame"), 0),
            "lives": _int(meta.get("lives"), 0),
            "wave": meta.get("wave"),
            "target_col": meta.get("target_col"),
            "defense": result.get("defense"),
            "move": result.get("move"),
            "fire": result.get("fire"),
            "target": result.get("target"),
        }

    # ---------- symbolic controller ----------

    def _sweep_order(self, preference: str | None = None) -> list[int]:
        direction = preference or self.plan.get("sweep", "LEFT_TO_RIGHT")
        if direction in {"RIGHT_TO_LEFT", "RIGHT_OUTER", "RIGHT"}:
            return [5, 4, 3, 2, 1, 0]
        return [0, 1, 2, 3, 4, 5]

    def _alive_columns(self, obs: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        result: dict[int, list[dict[str, Any]]] = {}
        for invader in _invaders(obs):
            if not _alive(invader):
                continue
            col = _int(invader.get("col"), -1)
            if 0 <= col <= 5:
                result.setdefault(col, []).append(invader)
        return result

    def _select_column(self, obs: dict[str, Any], preference: str | None = None) -> int | None:
        by_col = self._alive_columns(obs)
        if not by_col:
            return None
        order = self._sweep_order(preference)
        if preference == "LOWEST_THREAT":
            # The comparison is exact code, not a Jev arithmetic task.
            px, _ = _player(obs)
            return max(
                by_col,
                key=lambda col: (
                    max(_int(x.get("row"), 0) for x in by_col[col]),
                    -abs(px - (_int(by_col[col][0].get("x"), 0) + INVADER_TO_PLAYER_OFFSET)),
                    -order.index(col),
                ),
            )
        for col in order:
            if col in by_col:
                return col
        return next(iter(by_col))

    def _target_info(self, obs: dict[str, Any]) -> dict[str, Any]:
        by_col = self._alive_columns(obs)
        if self.target_col not in by_col:
            preference = self.target_preference
            if preference == "KEEP_CURRENT":
                preference = None
            self.target_col = self._select_column(obs, preference)
            # A Jev preference is a one-transition hint, not a permanent
            # override that can defeat the Astra sweep on every later shot.
            self.target_preference = "PLAN"
        if self.target_col is None or self.target_col not in by_col:
            return {"col": None, "side": "NONE", "row_band": "NONE", "alignment": "NONE", "shield": "UNKNOWN"}
        candidates = by_col[self.target_col]
        # A beam in a column reaches the lowest surviving invader first.
        target = max(candidates, key=lambda x: (_int(x.get("row"), 0), _int(x.get("y"), 0)))
        px, _ = _player(obs)
        target_x = _int(target.get("x"), px - INVADER_TO_PLAYER_OFFSET)
        desired = _clamp(target_x + INVADER_TO_PLAYER_OFFSET, PLAYER_MIN_X, PLAYER_MAX_X)
        error = desired - px
        row = _int(target.get("row"), 0)
        if self.target_col <= 1:
            side = "LEFT"
        elif self.target_col >= 4:
            side = "RIGHT"
        else:
            side = "CENTER"
        shield = _target_shield_quality(obs, desired)
        return {
            "col": self.target_col,
            "row": row,
            "row_band": "BOTTOM" if row >= 5 else ("MIDDLE" if row >= 3 else "TOP"),
            "side": side,
            "x": target_x,
            "desired_x": desired,
            "error": error,
            "alignment": "ALIGNED" if abs(error) <= ALIGN_TOLERANCE else ("APPROACHING" if abs(error) <= 18 else "FAR"),
            "shield": shield,
        }

    def _danger(self, obs: dict[str, Any]) -> dict[str, Any]:
        px, py = _player(obs)
        risk = self.plan.get("risk", "CAUTIOUS")
        vertical_limit = {"CAUTIOUS": 125, "BALANCED": 105, "AGGRESSIVE": 90}.get(risk, 125)
        urgent_limit = {"CAUTIOUS": 55, "BALANCED": 45, "AGGRESSIVE": 35}.get(risk, 55)
        best_rank: tuple[int, int] | None = None
        best_direction = ""
        best_urgency = "WATCH"
        for bomb in _bombs(obs):
            bx = _int(bomb.get("x"), px)
            by = _int(bomb.get("y"), -100)
            vertical = py - by
            gap = abs(bx - px)
            if vertical <= 0 or vertical > vertical_limit or gap > 28:
                continue
            if bx < px - 7:
                direction = "right"
            elif bx > px + 7:
                direction = "left"
            else:
                # Directly overhead: preserve an escape route and avoid
                # oscillating at an edge.
                if px <= 58:
                    direction = "right"
                elif px >= 120:
                    direction = "left"
                else:
                    direction = self.patrol_direction if self.patrol_direction in {"left", "right"} else "right"
            if direction == "left" and px <= PLAYER_MIN_X + 2:
                direction = "right"
            elif direction == "right" and px >= PLAYER_MAX_X - 2:
                direction = "left"
            urgency = "URGENT" if vertical <= urgent_limit else "WATCH"
            rank = (0 if urgency == "URGENT" else 1, vertical + gap)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_direction = direction
                best_urgency = urgency
        alive = [x for x in _invaders(obs) if _alive(x)]
        lowest_y = max((_int(x.get("y"), 0) for x in alive), default=0)
        formation = "URGENT" if lowest_y >= 150 else ("WATCH" if lowest_y >= 135 else "NONE")
        if best_rank is None:
            return {"urgency": formation, "direction": "", "bomb_side": "NONE", "formation": formation}
        urgency = "URGENT" if best_urgency == "URGENT" else "WATCH"
        direction = best_direction
        bomb_side = "LEFT" if direction == "right" else ("RIGHT" if direction == "left" else "CENTER")
        if formation == "URGENT":
            urgency = "URGENT"
        return {"urgency": urgency, "direction": direction, "bomb_side": bomb_side, "formation": formation}

    def _patrol_action(self, obs: dict[str, Any], target: dict[str, Any]) -> str | None:
        if not _shot_active(obs) or self.fire_frame is None:
            return None
        frame = _int(obs.get("frame"), self.fire_frame)
        elapsed = max(0, frame - self.fire_frame)
        # Change lateral intent every four emulator framesets while a beam is
        # in flight.  This creates a small escape oscillation without losing
        # the target lane permanently.
        phase = (elapsed // 16) % 2
        desired = _int(target.get("desired_x"), _player(obs)[0])
        px, _ = _player(obs)
        away = "right" if desired <= px else "left"
        if px <= PLAYER_MIN_X + 3:
            away = "right"
        elif px >= PLAYER_MAX_X - 3:
            away = "left"
        direction = away if phase == 0 else ("left" if away == "right" else "right")
        return direction

    def _code_action(self, obs: dict[str, Any]) -> str:
        danger = self._danger(obs)
        target = self._target_info(obs)
        if danger["direction"] and danger["urgency"] in {"URGENT", "WATCH"}:
            return danger["direction"]
        if self.recovery_ticks > 0 and danger["formation"] == "NONE":
            # Move out of the respawn center before the next shot; target
            # geometry will take over on subsequent ticks.
            if target.get("desired_x") is not None:
                px, _ = _player(obs)
                if abs(_int(target.get("desired_x"), px) - px) > ALIGN_TOLERANCE:
                    return "right" if _int(target.get("desired_x"), px) > px else "left"
        patrol = self._patrol_action(obs, target)
        if patrol is not None and danger["formation"] != "URGENT":
            return patrol
        if target.get("col") is None:
            return "noop"
        px, _ = _player(obs)
        desired = _int(target.get("desired_x"), px)
        error = desired - px
        if abs(error) > ALIGN_TOLERANCE:
            return "right" if error > 0 else "left"
        if not _shot_active(obs) and danger["formation"] != "URGENT":
            self.fire_frame = _int(obs.get("frame"), 0)
            self.patrol_direction = "right" if desired <= px else "left"
            return "fire"
        return "noop"

    def _model_action(self, obs: dict[str, Any]) -> str | None:
        tactical = self.tactical
        if not tactical:
            return None
        frame = _int(obs.get("frame"), 0)
        old_frame = _int(tactical.get("frame"), frame)
        lives = _int(obs.get("lives"), 0)
        wave = _wave(obs)
        if frame < old_frame or frame - old_frame > 40:
            return None
        if lives != tactical.get("lives") or wave != tactical.get("wave"):
            return None
        danger = self._danger(obs)
        if danger["direction"] and danger["urgency"] in {"URGENT", "WATCH"}:
            return danger["direction"]
        shot = _shot_active(obs)

        # Jev's target choice changes only the next target lock, never a beam
        # already in flight.  It is a preference, not an unchecked coordinate.
        target_choice = _choice(tactical.get("target"))
        target_confident = _answer_prob(tactical.get("target"), target_choice) >= 0.43
        if target_choice in {"LEFT_OUTER", "RIGHT_OUTER", "LOWEST_THREAT"} and target_confident and not shot:
            self.target_preference = {
                "LEFT_OUTER": "LEFT_TO_RIGHT",
                "RIGHT_OUTER": "RIGHT_TO_LEFT",
                "LOWEST_THREAT": "LOWEST_THREAT",
            }[target_choice]
            if self.target_col not in self._alive_columns(obs):
                self.target_col = None

        target = self._target_info(obs)
        if target.get("col") is None:
            return self._code_action(obs)
        px, _ = _player(obs)
        desired = _int(target.get("desired_x"), px)
        error = desired - px
        move_answer = tactical.get("move")
        move = _choice(move_answer)
        move_ok = move in {"LEFT", "RIGHT", "HOLD"} and _answer_prob(move_answer, move) >= 0.43
        fire_probability = _noul_probability(tactical.get("fire"))

        if shot:
            # While a beam is in flight code's patrol and fire lock are safer
            # than a stale semantic action.
            return self._patrol_action(obs, target) or self._code_action(obs)

        if abs(error) > ALIGN_TOLERANCE:
            wanted = "RIGHT" if error > 0 else "LEFT"
            if move_ok and move == wanted:
                return move.lower()
            return self._code_action(obs)

        if move_ok and move in {"LEFT", "RIGHT"} and abs(error) <= 8:
            if fire_probability is not None and fire_probability >= 0.58:
                self.fire_frame = frame
                self.patrol_direction = move.lower()
                return move.lower() + "_fire"
            return move.lower()
        if fire_probability is not None and fire_probability >= 0.58:
            self.fire_frame = frame
            self.patrol_direction = "right" if desired <= px else "left"
            return "fire"
        if fire_probability is not None and fire_probability <= 0.34:
            return "noop"
        return self._code_action(obs)

    def _choose_action(self, obs: dict[str, Any]) -> str:
        action = self._model_action(obs)
        if action not in {"noop", "fire", "right", "left", "right_fire", "left_fire"}:
            action = self._code_action(obs)
        if action in {"fire", "right_fire", "left_fire"} and _shot_active(obs):
            # Enforce the ROM's one-shot rule in code even if Jev is stale.
            action = action.replace("_fire", "") or "noop"
        if action not in {"noop", "fire", "right", "left", "right_fire", "left_fire"}:
            action = "noop"
        self.last_posted_action = action
        return action

    def _observe_events(self, obs: dict[str, Any], game: Any) -> None:
        lives = _int(obs.get("lives"), 0)
        wave = _wave(obs)
        if self.last_lives is not None and lives < self.last_lives:
            self.recovery_ticks = 10
            self.fire_frame = None
            self.tactical = None
            if not self.astra_revision_started:
                self._launch_astra(game, obs, "a life was lost; revise the risk and escape policy", revision=True)
        if self.last_wave is not None and wave is not None and wave > self.last_wave:
            self.target_col = None
            self.target_preference = "PLAN"
            self.recovery_ticks = 8
            self._launch_astra(game, obs, "a new wave was inferred; choose the opening sweep again", revision=True)
        self.last_lives = lives
        self.last_wave = wave
        self.last_score = _int(obs.get("score"), self.last_score)
        if self.recovery_ticks > 0:
            self.recovery_ticks -= 1
        self.previous_obs = obs

    # ---------- contract entry point ----------

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        try:
            obs = _obs_dict(game.observe())
        except Exception:
            return
        self.last_lives = _int(obs.get("lives"), 0)
        self.last_wave = _wave(obs)
        self._launch_astra(game, obs, "initial wave policy")
        self._launch_jev(obs)
        try:
            ctx.act(self._code_action(obs))
        except (BudgetExceeded, EpisodeOver):
            return
        except Exception:
            return

        while not ctx.done and not bool(getattr(game, "done", False)):
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                break
            except Exception:
                break
            if ctx.done or bool(getattr(game, "done", False)):
                break
            try:
                obs = _obs_dict(game.observe())
            except Exception:
                break
            if bool(obs.get("game_over", False)):
                break
            self._poll_astra()
            self._poll_jev()
            self._observe_events(obs, game)
            if self.jev_future is None:
                self._launch_jev(obs)
            action = self._choose_action(obs)
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                break
            except Exception:
                # An invalid external action must never be allowed to turn a
                # recoverable model response into a harness error.
                try:
                    ctx.act("noop")
                except Exception:
                    break


def _shield_quality(obs: dict[str, Any], index: int) -> str:
    shields = obs.get("shields")
    if not isinstance(shields, list) or index >= len(shields) or not isinstance(shields[index], dict):
        return "UNKNOWN"
    pixels = _shield_pixels(shields[index])
    return "GOOD" if pixels >= 85 else ("DAMAGED" if pixels >= 35 else "OPEN")


def _target_shield_quality(obs: dict[str, Any], x: int) -> str:
    shields = obs.get("shields")
    if not isinstance(shields, list):
        return "UNKNOWN"
    for shield in shields:
        if not isinstance(shield, dict):
            continue
        sx = _int(shield.get("x"), -100)
        if sx - 2 <= x <= sx + 17:
            return "GOOD" if _shield_pixels(shield) >= 85 else ("DAMAGED" if _shield_pixels(shield) >= 35 else "OPEN")
    return "CLEAR"


def build(ctx: Any) -> Harness:
    return Harness(ctx)

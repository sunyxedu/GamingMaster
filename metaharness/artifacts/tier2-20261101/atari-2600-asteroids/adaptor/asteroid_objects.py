"""Read-only approximate object extraction; never changes emulator state.

Coordinate formulas follow the OCAtari Asteroids RAM documentation.
See THIRD_PARTY.md for attribution and limitations.
"""
from collections import deque

ROCK_SLOTS = (3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19)


def decode_x(value: int) -> int:
    value = int(value)
    if value == 0:
        return 64
    if value == 1:
        return 4
    low, add, sub = value & 15, 8 * (value >> 7), (value >> 4) & 7
    if low % 2 == 0:
        return 97 + 15 * (low // 2 - 1) + add - sub
    return 10 + 15 * ((low - 1) // 2 - 1) + add - sub


def displayed_score(ram) -> int:
    def bcd(v):
        return 10 * (int(v) >> 4) + (int(v) & 15)
    return 1000 * bcd(ram[61]) + 10 * bcd(ram[62])


class ObjectDecoder:
    def __init__(self):
        self.history = {}

    def _motion(self, key, x, y, state, frame):
        history = self.history.get(key, deque(maxlen=4))
        discontinuity = False
        if history:
            pf, px, py, ps = history[-1]
            discontinuity = (frame != pf + 1 or state != ps or
                             abs(x - px) > 12 or abs(y - py) > 16)
            if discontinuity:
                history.clear()
        velocity = None
        if history:
            pf, px, py, _ = history[0]
            dt = frame - pf
            if dt > 0:
                velocity = [round((x - px) / dt, 3), round((y - py) / dt, 3)]
        history.append((frame, x, y, state))
        self.history[key] = history
        return {"velocity_px_per_frame": velocity,
                "velocity_window_frames": 0 if velocity is None else dt,
                "motion_discontinuity": discontinuity}

    def decode(self, values, frame):
        r = [int(v) for v in values]
        active = set()
        orientation = r[60] & 15
        player = {"present": r[74] != 224, "orientation_code": orientation}
        if player["present"]:
            x = decode_x(r[73]) + (-1 if orientation == 4 else 1 if orientation == 12 else 0)
            y = 100 + 2 * (r[74] - 41)
            player.update(x=x, y=y, width=6, height=10, raw_xy=[r[73], r[74]])
            player.update(self._motion("player", decode_x(r[73]), y, r[60] >> 4, frame))
            active.add("player")
        else:
            player.update(x=None, y=None, width=None, height=None, raw_xy=[r[73], r[74]],
                          velocity_px_per_frame=None, velocity_window_frames=0, motion_discontinuity=True)
        rocks = []
        for slot in ROCK_SLOTS:
            if r[slot + 18] == 0 or r[slot] & 128:
                continue
            x, y, code = decode_x(r[slot + 18]), 24 + 2 * r[slot], r[slot + 36] & 127
            size, w, h = ("large", 16, 28) if code < 32 else (("medium", 8, 15) if code < 48 else ("small", 4, 8))
            key = f"rock:{slot}"
            rock = {"source_slot": slot, "x": x, "y": y, "width": w, "height": h,
                    "size": size, "size_code": code, "raw_xy": [r[slot + 18], r[slot]]}
            rock.update(self._motion(key, x, y, size, frame))
            rocks.append(rock)
            active.add(key)
        missiles = []
        for slot in (83, 84):
            if r[slot] == 0 or r[slot + 3] & 128:
                continue
            x, y = decode_x(r[slot]) + 1, 17 + 2 * r[slot + 3]
            key = f"missile:{slot}"
            missile = {"source_slot": slot, "x": x, "y": y, "width": 1, "height": 2,
                       "direction_code": r[slot + 6], "raw_xy": [r[slot], r[slot + 3]]}
            missile.update(self._motion(key, x, y, r[slot + 6], frame))
            missiles.append(missile)
            active.add(key)
        for key in list(self.history):
            if key not in active:
                del self.history[key]
        return {"player": player, "asteroids": rocks, "player_missiles": missiles,
                "enemy_craft": None, "enemy_projectiles": None}

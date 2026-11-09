from __future__ import annotations


def build(ctx):
    return Probe(ctx)


class Probe:
    def __init__(self, ctx):
        self.ctx = ctx
        self.target = 0
        self.last_y = None
        self.still = 0
        self.last_stage = None
        self.transition = 0
        self.last_lives = 2
        self.jump_action = None
        self.escape = 0
        self.jump_cooldown = 0

    @staticmethod
    def num(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def jump_left(self, obs):
        raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
        return isinstance(raw, list) and len(raw) > 5 and self.num(raw[5]) != 0

    def threat(self, obs):
        p = obs.get("player", {}) if isinstance(obs, dict) else {}
        px, py = self.num(p.get("x"), 45), self.num(p.get("y"), 176)
        best = None
        for h in obs.get("hazards", []) if isinstance(obs, dict) and isinstance(obs.get("hazards", []), list) else []:
            if not isinstance(h, dict) or h.get("visible") is False:
                continue
            hx, hy = self.num(h.get("x")), self.num(h.get("y"))
            dx = self.num(h.get("dx"))
            dy = self.num(h.get("dy"))
            dist = abs(hx - px)
            same = abs(hy - py) <= 16
            closing = ((hx < px and dx > 0) or (hx > px and dx < 0) or
                       (dist <= 18 and abs(dx) <= 1))
            if same and closing and dist <= 58:
                if best is None or dist < best[0]:
                    best = (dist, hx < px)
        return best

    def run_episode(self, game):
        xs = (107, 47, 107, 47, 107, 31)
        self.ctx.act("fire")
        n = 0
        while not self.ctx.done and not game.done and n < 900:
            self.ctx.wait_for_tick()
            if self.ctx.done or game.done:
                break
            obs = game.observe()
            p = obs.get("player", {}) if isinstance(obs, dict) else {}
            x, y = self.num(p.get("x"), 45), self.num(p.get("y"), 176)
            stage = obs.get("stage")
            lives = self.num(obs.get("lives_remaining"), 0)
            if stage != self.last_stage:
                if stage == "transition": self.transition = 0
                if stage == "ramp":
                    self.target = 0
                    self.still = 0
                    self.last_y = None
                    self.jump_action = None
                    self.escape = 0
                self.last_stage = stage
            if lives != self.last_lives:
                self.last_lives = lives
                self.target = 0
                self.still = 0
                self.last_y = None
                self.jump_action = None
                self.escape = 0
            if stage == "ramp" and self.jump_cooldown > 0:
                self.jump_cooldown -= 1
            if stage == "transition":
                self.transition += 1
                action = "fire" if lives > 0 and self.transition == 30 else "noop"
            elif stage != "ramp":
                action = "fire" if stage == "idle" else "noop"
            elif self.jump_left(obs):
                action = self.jump_action or "rightfire"
            elif self.escape > 0:
                action = "right" if self.jump_action == "rightfire" else "left"
                self.escape -= 1
            else:
                self.jump_action = None
                t = self.threat(obs) if self.jump_cooldown == 0 else None
                if t is not None:
                    _, hazard_left = t
                    # The jump is committed in the direction Mario is already
                    # travelling; this makes the collision a crossing jump,
                    # rather than steering into a wall after the escape.
                    tx = xs[min(self.target, len(xs)-1)]
                    if hazard_left:
                        action = "leftfire" if x > tx else "rightfire"
                    else:
                        action = "rightfire" if x < tx else "leftfire"
                    self.jump_action = action
                    self.escape = 14
                    self.jump_cooldown = 36
                else:
                    tx = xs[min(self.target, len(xs)-1)]
                    if abs(x - tx) > 1:
                        action = "right" if x < tx else "left"
                        self.still = 0
                    else:
                        action = "up"
                        if self.last_y is not None and y == self.last_y:
                            self.still += 1
                        else:
                            self.still = 0
                        if self.still >= 8 and self.target + 1 < len(xs):
                            self.target += 1
                            self.still = 0
                            tx = xs[self.target]
                            action = "right" if x < tx else "left" if x > tx else "up"
            if n % 5 == 0 or stage != "ramp" or (action.endswith("fire") and not self.jump_left(obs)):
                hs=[]
                for h in obs.get("hazards",[]) if isinstance(obs,dict) else []:
                    if isinstance(h,dict) and h.get("visible",True): hs.append((h.get("x"),h.get("y"),h.get("dx"),h.get("dy")))
                self.ctx.log("probe", n=n, action=action, target=self.target, stage=stage,
                             x=x, y=y, dx=p.get("dx"), dy=p.get("dy"), jump=(obs.get("raw_ram", [0]*6)[5] if isinstance(obs.get("raw_ram"),list) else None),
                             lives=lives, screens=obs.get("screens_completed"), rivets=obs.get("rivets_removed"), hazards=hs, escape=self.escape)
            self.ctx.act(action)
            self.last_y = y
            n += 1

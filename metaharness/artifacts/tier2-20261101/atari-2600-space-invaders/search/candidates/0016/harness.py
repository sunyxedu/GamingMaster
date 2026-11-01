"""0016: label-independent projectile tracking and asynchronous evidence review.

Only public observations enter this module.  FIRE remains held; the ROM owns
its one-laser lock.  Models review sensor evidence, not joystick geometry.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "left", "right", "left_fire", "right_fire")


def num(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError, OverflowError):
        return default


def integer(x, default=0):
    return int(num(x, default))


def bounded(x, low, high):
    return max(low, min(high, x))


def mapping(x):
    return x if isinstance(x, dict) else {}


def observation(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except (ValueError, TypeError):
            return {}
    return mapping(x)


def records(x):
    return [v for v in x if isinstance(v, dict)] if isinstance(x, list) else []


def truth(x):
    return x.lower() in ("true", "yes", "1") if isinstance(x, str) else bool(x)


def field(answer, key, default=None):
    return answer.get(key, default) if isinstance(answer, dict) else getattr(answer, key, default)


def agreed(answers, key, choices):
    a, b = answers.get(key + "a"), answers.get(key + "b")
    value = field(a, "value")
    if not isinstance(value, str) or value not in choices or field(b, "value") != value:
        return None
    for answer in (a, b):
        if num(mapping(field(answer, "probabilities", {})).get(value)) < 0.72:
            return None
        confidence = field(answer, "confidence")
        if confidence is not None and num(confidence, -1) < 0.30:
            return None
    return value


class Mailbox:
    """One bounded model call per mailbox, with no background Python loops."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.future = None
        self.tag = None

    def launch(self, fn, tag):
        if self.future is not None or self.ctx.done:
            return False
        def work():
            try:
                return {"result": fn()}
            except Exception as exc:
                # ModelUnavailable is also recorded by ctx and stops the episode.
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
        self.future = self.tag = None
        try:
            result = future.result()
        except Exception as exc:
            result = {"error": type(exc).__name__}
        return mapping(result), tag

    def close(self):
        if self.future is not None:
            self.future.cancel()
        self.future = self.tag = None


class EvidenceController:
    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.px, self.py = 90.0, 190.0
        self.lives, self.wave, self.score = 3, 1, 0.0
        self.epoch = 0
        self.direction = 1
        # A working sweep interval, not a claim inferred from a stalled sprite.
        self.left, self.right = 38.0, 116.0
        self.speed = 0.5
        self.action_previous = "noop"
        self.previous = None
        self.positions = deque(maxlen=40)
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
        self.last_review_key = None
        self.last_motion_frame = 0
        self.motion_gap_start = None
        self.last_score_frame = 0
        self.last_jev_effect_frame = -10000
        self.count = {"actions": 0, "down_in_shot_field": 0, "ambiguous_reviews": 0,
                      "jev_promotions": 0, "jev_suppressions": 0, "jev_changed_actions": 0,
                      "strong_dodges": 0, "chip_escapes": 0, "life_losses": 0,
                      "identity_revivals": 0, "astra_updates": 0}

    def log(self, event, **fields):
        try:
            self.ctx.log(event, frame=self.frame, **fields)
        except Exception:
            pass

    def next_id(self, prefix):
        self.serial += 1
        return prefix + str(self.serial)

    def masks(self, obs):
        cells = {}
        for si, shield in enumerate(records(obs.get("shields"))[:3]):
            raw = shield.get("health_mask")
            if not isinstance(raw, list):
                continue
            x0 = num(shield.get("x")) - 4
            for row, bits in enumerate(raw[:9]):
                for col in range(8):
                    if integer(bits) & (1 << (7-col)):
                        cells[(si, row, col)] = x0 + col
        return cells

    def note_chips(self, obs, discontinuity):
        current = self.masks(obs)
        old, self.mask_previous = self.mask_previous, current
        if discontinuity or old is None or len(records(obs.get("shields"))) != 3:
            return
        removed = sorted(old.keys() - current.keys())
        # Bulk clearing/animation is not a projectile trajectory.
        if not removed or len(removed) > 24 or (len(old) > 12 and not current):
            return
        groups = {}
        for key in removed:
            si, row, col = key
            top = min((r for s, r, c in old if s == si and c == col), default=9)
            groups.setdefault(si, []).append((old[key], row <= top + 2))
        for si, changes in groups.items():
            xs = [x for x, _ in changes]
            x = sum(xs) / len(xs)
            top_entry = any(top for _, top in changes)
            own = any(abs(x-lx) <= 4.5 and 0 <= self.frame-lf <= 48 for lx, lf in self.launches)
            cover = sum(1 for xx in current.values() if abs(xx-x) <= 1.5)
            key = self.next_id("C")
            self.chips[key] = {"id": key, "x": x, "f": self.frame,
                               "top": top_entry, "own": own, "thin": cover <= 6,
                               "verdict": None, "used": False, "reviewed": False}
            if top_entry and not own:
                self.log("unattributed_chip", cue=key, x=round(x, 1), thin=cover <= 6)

    def track_projectiles(self, obs):
        points = []
        for bomb in records(obs.get("bombs"))[:12]:
            if bomb.get("x") is not None and bomb.get("y") is not None:
                points.append((num(bomb["x"]), num(bomb["y"]), "bomb"))
        shot = mapping(obs.get("player_shot"))
        if truth(shot.get("active")) and shot.get("x") is not None and shot.get("y") is not None:
            sx, sy = num(shot["x"]), num(shot["y"])
            if not any(abs(sx-x)<2 and abs(sy-y)<3 for x,y,_ in points):
                points.append((sx, sy, "player_shot"))
        used = set()
        for x, y, source in points:
            if not (0 <= x <= 159 and 0 <= y <= 209):
                continue
            matches = []
            for key, tr in self.tracks.items():
                dt = self.frame-tr["f"]
                if key in used or not (0 < dt <= 12) or abs(x-tr["x"]) > 2.5:
                    continue
                dy = y-tr["y"]
                if abs(dy) > 3.5*dt+3:
                    continue
                # A new beam near the muzzle must not join an old upper beam.
                if y >= 168 and tr["y"] < 150 and dy > 0:
                    continue
                prediction = tr["y"] + tr["vy"]*dt
                matches.append((abs(y-prediction) + 3*abs(x-tr["x"]), key))
            if matches:
                _, key = min(matches)
                tr = self.tracks[key]
                dt = self.frame-tr["f"]
                dy = y-tr["y"]
                if abs(dy) >= 1:
                    velocity = bounded(dy/max(1,dt), -3.5, 3.5)
                    old_sign = 1 if tr["vy"] > 0.2 else -1 if tr["vy"] < -0.2 else 0
                    new_sign = 1 if velocity > 0.2 else -1 if velocity < -0.2 else 0
                    if old_sign and new_sign and old_sign != new_sign:
                        # Identity conflict invalidates any cached semantic answer.
                        tr["rev"] += 1
                        tr["verdict"] = None
                        tr["reviewed"] = False
                        tr["down"] = tr["up"] = 0
                        tr["own"] = False
                    tr["vy"] = velocity
                    tr["down"] = tr["down"]+1 if new_sign > 0 else 0
                    tr["up"] = tr["up"]+1 if new_sign < 0 else 0
                tr.update(x=x, y=y, f=self.frame, source=source)
                tr["samples"] += 1
            else:
                key = self.next_id("T")
                near_muzzle = y >= self.py-25 and abs(x-(self.px+1)) <= 5
                own_launch = source == "player_shot" and near_muzzle and "fire" in self.action_previous
                tr = {"id": key, "x": x, "y": y, "f": self.frame, "born": self.frame,
                      "source": source, "first_source": source, "vy": 0.0,
                      "samples": 1, "up": 0, "down": 0, "own": own_launch,
                      "rev": 0, "verdict": None, "reviewed": False, "logged": False}
                self.tracks[key] = tr
                if own_launch:
                    self.launches.append((x, self.frame))
            used.add(key)
            if tr["down"] >= 2 and tr["first_source"] == "player_shot" and not tr["logged"]:
                tr["logged"] = True
                self.count["down_in_shot_field"] += 1
                self.log("descending_shot_field", track=key, x=round(x,1), y=round(y,1), vy=round(tr["vy"],2))
        self.tracks = {k:t for k,t in self.tracks.items() if self.frame-t["f"] <= 24}
        self.chips = {k:c for k,c in self.chips.items() if self.frame-c["f"] <= 120}

    def update(self, raw):
        obs = observation(raw)
        frame = integer(obs.get("frame"), self.frame+4)
        p = mapping(obs.get("player"))
        x = bounded(num(p.get("x"), self.px), 0, 159)
        y = bounded(num(p.get("y"), self.py), 150, 209)
        lives, wave = integer(obs.get("lives"), self.lives), integer(obs.get("wave"), self.wave)
        df = max(1, frame-self.frame)
        lost = lives < self.lives
        jump = self.previous is not None and abs(x-self.px) > max(12, 1.5*df)
        reset = self.previous is not None and (lost or wave != self.wave or jump)
        self.frame = frame
        if lost:
            self.count["life_losses"] += self.lives-lives
            self.log("life_evidence", lives=lives, last_mobile_frame=self.last_motion_frame,
                     gap_start=self.motion_gap_start, previous_x=round(self.px,1),
                     warning="life flag may lag impact; no death-lane boundary learning")
        if reset:
            self.epoch += 1
            self.tracks.clear()
            self.chips.clear()
            self.launches.clear()
            self.escape_until = -1
            self.motion_gap_start = None
        elif self.previous is not None:
            dx = x-self.px
            if 0.5 <= abs(dx) <= 1.3*df:
                direction = -1 if "left" in self.action_previous else 1 if "right" in self.action_previous else 0
                if direction*dx > 0:
                    self.speed = bounded(0.85*self.speed+0.15*abs(dx)/df, 0.25, 1.0)
                self.last_motion_frame = frame
                self.motion_gap_start = None
            elif abs(dx) < 0.5 and 43 < x < 111 and ("left" in self.action_previous or "right" in self.action_previous):
                if self.motion_gap_start is None:
                    self.motion_gap_start = frame-df
        self.px, self.py, self.lives, self.wave = x, y, lives, wave
        self.positions.append((frame,x))
        self.track_projectiles(obs)
        self.note_chips(obs, reset)
        live = {(integer(a.get("row")),integer(a.get("col"))) for a in records(obs.get("invaders")) if truth(a.get("alive"))}
        if self.previous is not None and wave == integer(self.previous.get("wave"),wave):
            revived = live-self.recent_alive
            if revived:
                self.count["identity_revivals"] += len(revived)
                if len(revived) >= 3:
                    self.log("lattice_identity_conflict", revived=len(revived), consequence="no aim or kill calibration")
        self.recent_alive = live
        score = num(obs.get("score"), self.score)
        if score > self.score:
            self.last_score_frame = frame
        self.score = score
        self.previous = obs
        return obs

    def correlated_chip(self, tr):
        return any(c["top"] and not c["own"] and abs(c["x"]-tr["x"])<=5
                   and abs(c["f"]-tr["f"])<=24 for c in self.chips.values())

    def hostile(self, tr, use_model=True):
        if tr["up"] >= 1:
            return False
        if tr["source"] == "bomb" or tr["down"] >= 2:
            return True
        if not use_model or tr.get("verdict") != "INCOMING":
            return False
        # Jev may resolve ONE supported descending increment, not invent motion.
        if tr["down"] < 1 or tr["own"] or self.frame-tr.get("verdict_f",-10000)>24:
            return False
        return self.rubric == "MOTION" or self.correlated_chip(tr)

    def sweep(self):
        if self.px <= self.left:
            return 1
        if self.px >= self.right:
            return -1
        return self.direction

    def path_risk(self, direction, use_model=True):
        prior = -1 if "left" in self.action_previous else 1 if "right" in self.action_previous else 0
        worst = 0.0
        for tr in self.tracks.values():
            if not self.hostile(tr, use_model):
                continue
            age = max(0, self.frame-tr["f"])
            vy = tr["vy"] if tr["vy"] > 0.2 else 1.2
            for dt in range(0,65,2):
                by = tr["y"]+vy*(age+dt)
                if not self.py-13 <= by <= self.py+8:
                    continue
                for delay in (0,4):
                    px = bounded(self.px+self.speed*(prior*min(dt,delay)+direction*max(0,dt-delay)),37,119)
                    gap = abs(px-tr["x"])
                    if gap <= 9:
                        worst = max(worst, 100+(9-gap)+(64-dt)/64)
                    elif gap < 15:
                        worst = max(worst, (15-gap)/6)
        return worst

    def decision(self, use_model=True):
        base = self.sweep()
        risks = {d:self.path_risk(d,use_model) for d in (-1,0,1)}
        if risks[base] >= 100:
            escape = min(risks, key=lambda d:(risks[d],d==0,d!=base))
            if risks[escape]+0.5 < risks[base]:
                return escape,"projectile",None
        if self.frame < self.escape_until and risks[base] < 100:
            return base,"escape_commit",None
        # A chip is NOT a surviving bomb.  It can justify one bounded detour,
        # never a permanent wall or a long-lived ballistic track.
        if self.frame-self.last_weak_escape >= 48:
            candidates = []
            for c in self.chips.values():
                if c["used"] or c["own"] or not c["top"] or self.frame-c["f"]>96:
                    continue
                verdict = c.get("verdict") if use_model else None
                if verdict == "OUTGOING":
                    continue
                promoted = verdict == "INCOMING"
                if not c["thin"] and not promoted:
                    continue
                gap = c["x"]-self.px
                if abs(gap)>23 or base*gap < -2:
                    continue
                away = -1 if gap > 1 else 1 if gap < -1 else (1 if self.px<77 else -1)
                if (away<0 and self.px<=42) or (away>0 and self.px>=112) or risks[away]>=100:
                    continue
                candidates.append((abs(gap),c["id"],away))
            if candidates:
                _,key,away = min(candidates)
                return away,"chip",key
        return base,"sweep",None

    def action(self):
        fallback,_,_ = self.decision(False)
        chosen,source,cue = self.decision(True)
        if chosen != fallback:
            self.count["jev_changed_actions"] += 1
            self.last_jev_effect_frame = self.frame
            self.log("jev_changed_action", baseline=fallback, selected=chosen, source=source, cue=cue,
                     rubric=self.rubric, plan_version=self.plan_version)
        if source == "chip" and cue in self.chips:
            self.chips[cue]["used"] = True
            self.last_weak_escape = self.frame
            self.escape_until = self.frame+16
            self.count["chip_escapes"] += 1
            self.log("chip_escape", cue=cue, direction=chosen, model=self.chips[cue].get("verdict"))
        elif source == "projectile" and chosen != self.sweep():
            self.count["strong_dodges"] += 1
            self.log("projectile_escape", direction=chosen)
        if chosen:
            self.direction = chosen
        self.count["actions"] += 1
        return "left_fire" if chosen<0 else "right_fire" if chosen>0 else "fire"

    def packet(self):
        candidates, tags = {}, {}
        for key,tr in self.tracks.items():
            if tr["reviewed"] or self.frame-tr["f"]>8 or tr["own"] or tr["up"] or tr["down"]>=2 or tr["source"]=="bomb":
                continue
            if tr["y"]<115 or (tr["down"]<1 and not self.correlated_chip(tr)):
                continue
            candidates[key] = {"kind":"projectile", "decoder_label":tr["source"],
                               "motion":"one_downward_increment" if tr["down"] else "not_yet_measured",
                               "own_launch":"not_matched", "same_lane_top_chip":self.correlated_chip(tr),
                               "position":"below_shields" if tr["y"]>=174 else "above_or_at_shields",
                               "continuity":"short_matched_track", "model_can":"classify evidence; not choose a dodge"}
            tags[key] = {"kind":"track", "rev":tr["rev"], "f":tr["f"]}
            if len(candidates)>=2:
                break
        for key,c in reversed(list(self.chips.items())):
            if len(candidates)>=3:
                break
            if c["reviewed"] or c["used"] or not c["top"] or self.frame-c["f"]>32:
                continue
            near_up = any(t["up"] and abs(t["x"]-c["x"])<=5 and abs(t["f"]-c["f"])<=24 for t in self.tracks.values())
            near_down = any(t["down"] and abs(t["x"]-c["x"])<=5 and abs(t["f"]-c["f"])<=24 for t in self.tracks.values())
            candidates[key] = {"kind":"shield_chip", "erosion":"top_entry", "own_launch": "matched" if c["own"] else "not_matched",
                               "ascending_same_lane":near_up, "descending_same_lane":near_down,
                               "remaining_cover":"thin" if c["thin"] else "substantial",
                               "interpretation_limit":"damage can consume a projectile; do not infer a surviving bomb"}
            tags[key] = {"kind":"chip", "f":c["f"]}
        if not candidates:
            return None
        state = {"schema":"space_invaders_temporal_evidence_v1", "candidates":candidates,
                 "rules":{"up":"an outgoing laser rises toward smaller y", "down":"an incoming bomb descends toward larger y",
                          "labels":"decoder labels are fallible; empty bombs is not proof of safety",
                          "no_creation":"classify only supplied evidence, never invent an unseen projectile"},
                 "plan":{"rubric":self.rubric, "meaning":"MOTION allows a single continuous downward increment without an own launch; CORROBORATE additionally requires same-lane top shield damage for an ambiguous projectile"},
                 "life_state":"motion_gap" if self.motion_gap_start is not None else "responsive"}
        qs = {}
        options = {"INCOMING":"Evidence favors enemy-origin descending fire or enemy-origin top shield damage, without outgoing-shot contradiction.",
                   "OUTGOING":"An own launch or ascending same-lane trajectory explains the signal; do not dodge a friendly laser.",
                   "UNRESOLVED":"Insufficient, discontinuous, or contradictory evidence; do not manufacture certainty."}
        for key,candidate in candidates.items():
            instruction = ("Classify candidates."+key+" using its supplied qualitative temporal and launch facts, plus rules and plan. "
                           "A shield chip is origin evidence only, not a surviving bomb. A mere decoder name or absence of bombs is insufficient. "
                           "For a projectile, respect plan.rubric. For a chip, matching own launch/ascending evidence supports OUTGOING; "
                           "top-entry plus matching descent supports INCOMING; unexplained erosion alone is UNRESOLVED. "
                           "Do not calculate coordinates or select movement.")
            items = list(options.items())
            if self.serial%2:
                items.reverse()
            qs[key+"a"] = Choice(instruction,dict(items))
            qs[key+"b"] = Choice(instruction,dict(reversed(items)))
        tag = {"frame":self.frame,"epoch":self.epoch,"plan":self.plan_version,"items":tags}
        return state,qs,tag

    def mark_submitted(self, tag):
        self.last_query = self.frame
        for key,item in tag["items"].items():
            obj = self.tracks.get(key) if item["kind"]=="track" else self.chips.get(key)
            if obj is not None:
                obj["reviewed"] = True
        self.count["ambiguous_reviews"] += 1

    def accept(self, packet):
        if packet is None:
            return
        result,tag = packet
        if result.get("error"):
            self.log("jev_failure", error=result["error"])
            return
        if not isinstance(tag,dict) or tag.get("epoch")!=self.epoch or tag.get("plan")!=self.plan_version:
            return
        if not 0<=self.frame-integer(tag.get("frame"),-10000)<=36:
            self.log("jev_discard", reason="age")
            return
        answers = mapping(result.get("result"))
        for key,item in mapping(tag.get("items")).items():
            verdict = agreed(answers,key,("INCOMING","OUTGOING","UNRESOLVED"))
            if verdict is None:
                continue
            obj = self.tracks.get(key) if item["kind"]=="track" else self.chips.get(key)
            if obj is None:
                continue
            if item["kind"]=="track":
                if obj["rev"]!=item["rev"] or self.frame-obj["f"]>8:
                    continue
                if verdict=="INCOMING" and (obj["up"] or obj["own"] or obj["down"]<1):
                    continue
            elif verdict=="INCOMING":
                if obj["own"] or not obj["top"] or not any(t["down"] and abs(t["x"]-obj["x"])<=5 and abs(t["f"]-obj["f"])<=24 for t in self.tracks.values()):
                    continue
            obj["verdict"],obj["verdict_f"] = verdict,self.frame
            if verdict=="INCOMING":
                self.count["jev_promotions"]+=1
            if verdict=="OUTGOING":
                self.count["jev_suppressions"]+=1
            self.log("jev_evidence_verdict", cue=key, verdict=verdict, age=self.frame-tag["frame"])

    def astra_prompt(self):
        evidence = {"score":self.score,"lives":self.lives,"observed_descending_shot_labels":self.count["down_in_shot_field"],
                    "blind_loss_count":self.count["life_losses"],"lattice_revivals":self.count["identity_revivals"]}
        return ("Choose the evidence rubric for a real-time Space Invaders sensor reviewer. Reply with ONE token: MOTION or CORROBORATE. "
                "Code holds FIRE and sweeps every tick; it independently avoids explicit bombs and two successive downward projectile increments. "
                "Jev classifies ambiguous temporal evidence, not raw actions, with paired Choice questions INCOMING/OUTGOING/UNRESOLVED, "
                "probability >=0.72, fresh track identity, and a live collision recheck. A single downward increment can be a decoder identity swap. "
                "MOTION allows Jev to promote one continuous descending increment if no own launch or upward evidence exists. "
                "CORROBORATE additionally requires recent same-lane top-entry shield damage. Neither option turns unexplained damage into a known surviving bomb. "
                "Archive evidence: 0008 lost all lives with empty bomb samples; 0012 seed7 lost lives at frames1128/1744 despite three chip escapes. "
                "The archive does NOT establish mislabeled enemy projectiles. Shield-only threat inference is unvalidated. "
                "The choice changes future Jev state and the compiled promotion guard; no prose or code. Current episode evidence: "
                +json.dumps(evidence,separators=(",",":"),allow_nan=False))

    def apply_astra(self, packet):
        if packet is None:
            return
        result,tag = packet
        if result.get("error"):
            self.log("astra_failure",error=result["error"])
            return
        text = getattr(result.get("result"),"text",None)
        if not isinstance(text,str):
            return
        choice = text.strip().strip('"\'` .\n').upper()
        if choice not in ("MOTION","CORROBORATE"):
            self.log("astra_rejected",reason="not_one_allowed_token")
            return
        if choice != self.rubric:
            old = self.rubric
            self.rubric = choice
            self.plan_version += 1
            self.count["astra_updates"]+=1
            for t in self.tracks.values():
                t["reviewed"] = False
                t["verdict"] = None
            self.log("astra_rubric_changed",old=old,new=choice,plan_version=self.plan_version)
        else:
            self.log("astra_rubric_confirmed",rubric=choice)


class Harness:
    def __init__(self,ctx):
        self.ctx=ctx

    def run_episode(self,game):
        ctx=self.ctx
        ctl=EvidenceController(ctx)
        jev,astra=Mailbox(ctx),Mailbox(ctx)
        spec=getattr(game,"action_spec",None)
        legal=set(getattr(spec,"names",ACTIONS))
        astra_sent=False
        last_frame=None
        try:
            while not ctx.done and not getattr(game,"done",False):
                raw=observation(ctx.observe())
                if truth(raw.get("game_over")):
                    break
                frame=integer(raw.get("frame"),ctl.frame+4)
                if frame==last_frame:
                    ctx.wait_for_tick()
                    continue
                last_frame=frame
                ctl.update(raw)
                ctl.apply_astra(astra.poll())
                ctl.accept(jev.poll())
                action=ctl.action()
                if action not in legal:
                    action="fire" if "fire" in legal else getattr(spec,"noop","noop")
                ctx.act(action)
                ctl.action_previous=action
                remaining=mapping(ctx.budget.remaining())
                if (jev.future is None and ctl.frame-ctl.last_query>=16
                        and num(remaining.get("system1_calls"))>1 and num(remaining.get("wall_s"))>2):
                    packet=ctl.packet()
                    if packet is not None:
                        state,questions,tag=packet
                        if jev.launch(lambda s=state,q=questions:ctx.decide(s,q,timeout_s=1.0),tag):
                            ctl.mark_submitted(tag)
                # Start once, immediately, rather than waiting for the first loss
                # and then spending another 30-50 seconds on a verbose answer.
                if not astra_sent and num(remaining.get("system2_calls"))>0 and num(remaining.get("wall_s"))>60:
                    prompt=ctl.astra_prompt()
                    astra_sent=astra.launch(lambda p=prompt:ctx.think(p,timeout_s=55.0),{"frame":ctl.frame})
                if ctl.count["actions"]%150==0:
                    ctl.log("evidence_summary",score=ctl.score,lives=ctl.lives,counts=dict(ctl.count),rubric=ctl.rubric)
                ctx.wait_for_tick()
        except (BudgetExceeded,EpisodeOver):
            pass
        finally:
            jev.close()
            astra.close()
            ctl.log("evidence_finished",score=ctl.score,lives=ctl.lives,counts=dict(ctl.count),rubric=ctl.rubric)


def build(ctx):
    return Harness(ctx)

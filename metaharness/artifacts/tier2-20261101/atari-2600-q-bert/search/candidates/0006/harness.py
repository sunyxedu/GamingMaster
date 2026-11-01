"""Round-local perception repair and transaction-based Q*bert control.

Only public observations are used. Raw palette *values* are never assigned
permanent meanings: the cartridge reuses them in a different role next round.
"""
from __future__ import annotations

import json
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul

DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
GRAPH = {}
ROWS = {}
for r in range(6):
    for c in range(r + 1):
        n = r * (r + 1) // 2 + c
        ROWS[n] = r
        edges = {}
        if c < r:
            edges["UP"] = n - r
        if c:
            edges["LEFT"] = n - r - 1
        if r < 5:
            edges["DOWN"] = n + r + 1
            edges["RIGHT"] = n + r + 2
        GRAPH[n] = edges
DIST = {}
for root in GRAPH:
    dist = {root: 0}
    q = deque([root])
    while q:
        a = q.popleft()
        for b in GRAPH[a].values():
            if b not in dist:
                dist[b] = dist[a] + 1
                q.append(b)
    DIST[root] = dist


def integer(x, default=0):
    try:
        return int(x)
    except (ValueError, TypeError, OverflowError):
        return default


def probability(answer, key):
    p = getattr(answer, "probabilities", {})
    if not isinstance(p, dict):
        return 0.0
    try:
        return float(p.get(key, 0.0))
    except (ValueError, TypeError):
        return 0.0


def motion(obs):
    value = obs.get("qbert_motion")
    return value if isinstance(value, dict) else {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tick = 0
        self.epoch = 0
        self.colors = None
        self.start_color = None
        self.color_path = []
        self.flash = False
        self.flash_score = 0.0
        self.flash_tick = -999
        self.stable = 0
        self.rounds = 0
        self.visits = [0] * 21
        self.recent = deque(maxlen=12)
        self.changes = deque(maxlen=8)
        self.last_score = 0.0
        self.last_progress = 0
        self.lives = None
        self.need = [1] * 21
        self.hop = None
        self.last_node = None
        self.last_landing_tick = -1
        self.jev = None
        self.jev_meta = None
        self.next_jev = 0
        self.advice = None
        self.astra = None
        self.next_astra = 0
        self.astra_calls = 0
        self.policy_version = 0
        self.policy = {"coverage": "nearest", "hazard": "predictive", "evidence": "palette_epoch"}
        self.astra_reason = "initial calibration audit"
        self.enemy_memory = {}

    def event(self, name, **fields):
        self.ctx.log(name, **fields)

    def update_board(self, obs):
        self.tick = integer(obs.get("tick", obs.get("ticks")))
        score = float(obs.get("score", self.last_score))
        raw = obs.get("cube_colors")
        if not isinstance(raw, list) or len(raw) != 21:
            raw = self.colors or [None] * 21
        raw = list(raw)
        changes = [] if self.colors is None else [i for i in range(21) if raw[i] != self.colors[i]]
        uniform = len(set(raw)) == 1 and raw[0] is not None
        self.stable = self.stable + 1 if raw == self.colors else 0
        if self.start_color is None and uniform:
            self.start_color = raw[0]
            self.color_path = [raw[0]]
        if len(changes) >= 18 and uniform:
            if not self.flash:
                self.flash = True
                self.flash_score = self.last_score
                self.flash_tick = self.tick
                self.hop = None
                self.epoch += 1
                self.advice = None
                self.event("palette_flash_begin", tick=self.tick, score=score,
                           reported_rounds=obs.get("completed_rounds"))
            self.stable = 0
        if self.flash:
            # Award animation changes all cubes every one or two ticks. Require
            # several unchanged ticks and the player returned to the apex.
            if uniform and self.stable >= 3 and obs.get("qbert_node") == 0:
                self.flash = False
                self.rounds += 1
                self.start_color = raw[0]
                self.color_path = [raw[0]]
                self.visits = [0] * 21
                self.recent.clear()
                self.hop = None
                self.epoch += 1
                self.last_progress = self.tick
                self.astra_reason = "whole-board flash and fresh palette roles"
                self.event("round_epoch_rebased", tick=self.tick, inferred_rounds=self.rounds,
                           award=score-self.flash_score, start_color=self.start_color,
                           reported_rounds=obs.get("completed_rounds"))
            self.need = [1] * 21
        else:
            level = min(5, self.rounds // 4 + 1)
            goal = 1 if level in (1, 3) else 2
            # Learn ordered colors from sparse, local transitions. A concurrent
            # Sam reversal does not establish a new forward color.
            qnode = obs.get("qbert_node")
            for n in changes:
                near = n == qnode or (self.hop is not None and n == self.hop[1])
                if near and len(changes) <= 2 and raw[n] not in self.color_path:
                    if self.colors[n] == self.color_path[-1] and len(self.color_path) <= goal:
                        self.color_path.append(raw[n])
                        self.event("palette_role_learned", tick=self.tick, path=list(self.color_path))
                if near:
                    self.changes.append({"tick": self.tick, "node": n,
                                         "from": self.colors[n], "to": raw[n],
                                         "reward": score-self.last_score})
            self.need = []
            for color in raw:
                if color in self.color_path:
                    phase = self.color_path.index(color)
                    self.need.append(max(0, goal-phase))
                else:
                    self.need.append(goal)
            if changes or score != self.last_score:
                self.last_progress = self.tick
        new_lives = integer(obs.get("lives"), 4)
        if self.lives is not None and new_lives < self.lives:
            self.epoch += 1
            self.hop = None
            self.advice = None
            self.astra_reason = "life loss: review local hazard mode"
            self.event("life_loss", tick=self.tick, node=obs.get("qbert_node"),
                       lives=new_lives, inferred_rounds=self.rounds)
        self.lives = new_lives
        self.colors = raw
        self.last_score = score

    def anchor(self, obs):
        """A nearest cube during an arc is NOT a confirmed landing."""
        m = motion(obs)
        if m.get("falling") or m.get("phase") == "not_visible":
            return None
        x, y = m.get("ram_x"), m.get("ram_y")
        details = obs.get("cube_details", [])
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            for cube in details:
                n = cube.get("node")
                if n in GRAPH and abs(x-(cube.get("x", -999)+9)) <= 1 and abs(y-(25+28*ROWS[n])) <= 1:
                    return n
        if not obs.get("qbert_in_hop") and m.get("phase") == "standing":
            n = obs.get("qbert_node")
            return n if n in GRAPH else None
        return None

    def hazards(self, obs):
        current = []
        for h in obs.get("hazards", []):
            if not isinstance(h, dict) or not h.get("lethal"):
                continue
            n = h.get("estimated_node", h.get("node"))
            if n not in GRAPH:
                continue
            kind = h.get("kind", "enemy")
            previous = self.enemy_memory.get(kind)
            entry = {"kind": kind, "node": n, "tick": self.tick, "previous": previous["node"] if previous else n}
            self.enemy_memory[kind] = entry
            current.append(entry)
        for kind, h in list(self.enemy_memory.items()):
            if self.tick-h["tick"] > 8:
                del self.enemy_memory[kind]
            elif not any(e["kind"] == kind for e in current):
                current.append(h)
        return current

    def candidates(self, obs, node):
        threats = self.hazards(obs)
        safe = obs.get("safe_moves", [])
        legal = obs.get("legal_moves", [])
        result = []
        for action, dest in GRAPH[node].items():
            if action not in legal:
                continue
            occupied = any(h["node"] == dest for h in threats)
            predicted = False
            adjacent = False
            for h in threats:
                en = h["node"]
                adjacent |= DIST[dest][en] == 1
                if h["kind"] in ("purple_ball", "red_ball"):
                    predicted |= dest in [v for a,v in GRAPH[en].items() if a in ("DOWN", "RIGHT")]
                elif h["kind"] == "coily":
                    predicted |= dest in GRAPH[en].values() and DIST[node][en] <= 2
            escape = sum(not any(h["node"] == k for h in threats) for k in GRAPH[dest].values())
            target_dist = min((DIST[dest][i]+(1 if i == node else 0) for i in GRAPH if self.need[i]>0), default=0)
            risk = 100 if occupied or action not in safe else 10 if predicted and self.policy["hazard"] == "predictive" else 2 if adjacent else 0
            progress = self.need[dest] > 0
            revisits = sum(1 for n in self.recent if n == dest)
            preference = (risk, target_dist, not progress, revisits, self.visits[dest], ROWS[dest], dest)
            if self.policy["coverage"] == "frontier":
                preference = (risk, not progress, revisits, target_dist, self.visits[dest], ROWS[dest], dest)
            result.append({"action": action, "dest": dest, "risk": risk,
                           "danger": "occupied" if risk==100 else "predicted_intersection" if risk==10 else "adjacent_enemy" if risk else "clear",
                           "progress": "useful_landing" if progress else "transit",
                           "escape": "multiple" if escape >= 2 else "single",
                           "repeat": "recently_repeated" if revisits >= 2 else "not_repeated",
                           "rank": preference})
        result.sort(key=lambda c:c["rank"])
        return result

    def launch_jev(self, obs, node, options):
        if self.jev is not None or self.tick < self.next_jev or not options:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) < 2:
            return
        self.next_jev = self.tick+7
        choices = {c["action"]: {k:v for k,v in c.items() if k != "rank"} for c in options if c["risk"]<100}
        if not choices:
            return
        choices["unknown"] = "Insufficient evidence; let deterministic control proceed."
        state = {"rules": "Survive; complete the pyramid. Code computed geometry and threat buckets. Never chase bonuses.",
                 "policy": self.policy, "round_evidence": {"inferred_completed": self.rounds,
                 "reported_completed": obs.get("completed_rounds"), "palette_roles": list(self.color_path),
                 "recent_local_changes": list(self.changes)[-3:]},
                 "position": node, "fallback": options[0]["action"], "alternatives": choices,
                 "decoder_disagreement": any((v>0) != (integer(o)>0) for v,o in zip(self.need,obs.get("needed_landings",[])))}
        instruction = "Choose the safest useful alternative from alternatives. Occupied is forbidden. Predicted intersection is worse than adjacency, which is worse than clear. With equal danger prefer useful progress and avoid recently repeated branches. Unknown means defer to code."
        questions = {"route_a": Choice(instruction, choices),
                     "route_b": Choice(instruction, dict(reversed(list(choices.items())))),
                     "perception_conflict": Noul("Do round_evidence and decoder_disagreement indicate the fixed decoder color labels are unreliable for this new round? Local changed cubes following a whole-board award flash support round-local color roles.")}
        meta = (self.tick, self.epoch, node, self.policy_version)
        self.jev_meta = meta
        self.jev = self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=1.5))

    def collect(self):
        if self.jev is not None and self.jev.done():
            future, meta = self.jev, self.jev_meta
            self.jev = None
            try:
                ans = future.result()
                a, b = ans.get("route_a"), ans.get("route_b")
                val = getattr(a, "value", None)
                if val in DIRECTIONS and val == getattr(b, "value", None) and min(probability(a,val),probability(b,val)) >= .62:
                    self.advice = (meta, val)
                conflict = getattr(ans.get("perception_conflict"), "value", 0)
                if isinstance(conflict,(int,float)) and conflict >= .8:
                    self.astra_reason = "Jev confirms round-local perception conflict"
            except Exception as exc:
                self.event("jev_fallback", error=type(exc).__name__)
        if self.astra is not None and self.astra.done():
            future = self.astra
            self.astra = None
            try:
                text = future.result().text
                data = json.loads(text[text.index("{"):text.rindex("}")+1])
                changed = False
                if isinstance(data,dict):
                    for key, domain in (("coverage",("nearest","frontier")),("hazard",("predictive","occupied_only"))):
                        if data.get(key) in domain and self.policy[key] != data[key]:
                            self.policy[key] = data[key]
                            changed = True
                if changed:
                    self.policy_version += 1
                    self.advice = None
                self.event("astra_policy", accepted=isinstance(data,dict), changed=changed, policy=self.policy)
            except Exception as exc:
                self.event("astra_fallback", error=type(exc).__name__)

    def launch_astra(self, obs):
        if self.astra is not None or self.tick < self.next_astra or self.astra_calls >= 4:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 1:
            return
        reason = self.astra_reason
        if not reason:
            return
        self.astra_reason = ""
        self.next_astra = self.tick+450
        self.astra_calls += 1
        evidence = {"reason": reason, "tick": self.tick, "lives": self.lives,
                    "inferred_rounds": self.rounds, "reported_rounds": obs.get("completed_rounds"),
                    "palette_path": self.color_path, "recent_changes": list(self.changes),
                    "need": self.need, "policy": self.policy}
        prompt = ("Audit this Atari 2600 Q*bert harness, not individual joystick moves. "
                  "Public archive: 21 cubes became complete, then 31 x 100 reward during whole-board flashing, "
                  "then palette roles reversed while reported completed_rounds stayed zero. Fixed labels caused an endless top-cube loop. "
                  "Code now discovers per-round palette roles, quarantines flashing, and pulses one legal hop at a time. "
                  "Choose coverage nearest (shortest unfinished path) or frontier (favor useful adjacent landing and avoid repeated branch); "
                  "choose hazard predictive (avoid potential next enemy node) or occupied_only (less conservative when forecasts cause loops). "
                  "Your choices are passed explicitly as policy to Jev's two independent, reordered Choices over code-generated moves. "
                  "Jev also judges decoder conflict with a Noul. Code rejects low-probability, stale, unsafe results. "
                  "Return ONLY JSON with keys coverage and hazard, under 30 tokens. No raw actions or code. Evidence: "+json.dumps(evidence))
        self.astra = self.ctx.spawn(lambda:self.ctx.think(prompt, timeout_s=110.0))

    def act_on(self, obs):
        if self.flash or motion(obs).get("falling"):
            self.hop = None
            return "NOOP"
        node = self.anchor(obs)
        if self.hop is not None:
            source, dest, action, start, departed = self.hop
            age = self.tick-start
            if node == dest and age >= 2:
                self.visits[dest] += 1
                self.recent.append(dest)
                self.last_landing_tick = self.tick
                self.event("hop_landed", tick=self.tick, source=source, dest=dest, age=age)
                self.hop = None
            elif age > 18 or (node is not None and node not in (source,dest)):
                self.event("hop_resync", tick=self.tick, expected=dest, observed=node)
                self.hop = None
            else:
                departed = departed or node != source
                self.hop = (source,dest,action,start,departed)
                # Release after launch; a held edge must never start a second hop.
                return action if not departed and age < 3 and action in obs.get("safe_moves",[]) else "NOOP"
        if node is None:
            return "NOOP"
        options = self.candidates(obs,node)
        if not options:
            return "NOOP"
        selected = options[0]
        if selected["risk"] >= 100:
            return "NOOP"
        if self.advice is not None:
            meta, proposed = self.advice
            age = self.tick-meta[0]
            if 0 <= age <= 10 and meta[1:] == (self.epoch,node,self.policy_version):
                match = next((c for c in options if c["action"]==proposed and c["risk"]<=selected["risk"]),None)
                if match is not None:
                    if match["action"] != selected["action"]:
                        self.event("jev_changed_hop", tick=self.tick, fallback=selected["action"], action=proposed, age=age)
                    selected=match
            self.advice=None
        a,d = selected["action"],selected["dest"]
        self.hop = (node,d,a,self.tick,False)
        # Pipeline the NEXT landing's semantic judgment during this hop.
        # The result is conditional on reaching d; live legality and risk are
        # checked again at arrival, not copied from this prediction.
        projected = dict(obs)
        projected["qbert_node"] = d
        projected["legal_moves"] = list(GRAPH[d])
        occupied = {h["node"] for h in self.hazards(obs)}
        projected["safe_moves"] = [edge for edge,nxt in GRAPH[d].items() if nxt not in occupied]
        self.launch_jev(projected,d,self.candidates(projected,d))
        return a

    def run_episode(self, game):
        while not self.ctx.done and not game.done:
            obs = self.ctx.observe()
            if not isinstance(obs,dict):
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.update_board(obs)
            self.collect()
            action = self.act_on(obs)
            self.ctx.act(action)
            self.launch_astra(obs)
            if self.tick % 30 == 0:
                self.event("control_state", tick=self.tick, score=obs.get("score"), node=obs.get("qbert_node"),
                           inferred_rounds=self.rounds, flash=self.flash, need=self.need,
                           colors=self.colors, anchor=self.anchor(obs), action=action,
                           motion=motion(obs), hazards=obs.get("hazards",[]))
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)

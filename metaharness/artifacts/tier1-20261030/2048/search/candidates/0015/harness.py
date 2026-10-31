"""Spawn-agnostic, risk-sensitive 2048 controller.

The planner treats the random spawn as an adversary when the board is crowded,
and as an expectation while there is room.  Only the first action of a search
is committed; the realized observation is therefore always the authority.
"""
from __future__ import annotations

import json, re, time
from functools import lru_cache

DIRECTIONS = ("up", "down", "left", "right")
LINES = {
    "left": tuple(tuple(4*r+c for c in range(4)) for r in range(4)),
    "right": tuple(tuple(4*r+c for c in range(3,-1,-1)) for r in range(4)),
    "up": tuple(tuple(4*r+c for r in range(4)) for c in range(4)),
    "down": tuple(tuple(4*r+c for r in range(3,-1,-1)) for c in range(4)),
}
SNAKES = {
    "ul": (0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12),
    "ur": (3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15),
    "dl": (12,13,14,15,11,10,9,8,4,5,6,7,3,2,1,0),
    "dr": (15,14,13,12,8,9,10,11,7,6,5,4,0,1,2,3),
}

def obs(v):
    if isinstance(v, str):
        try: v = json.loads(v)
        except Exception: return {}
    return v if isinstance(v, dict) else {}

def flat(o):
    try: return tuple(int(x) for row in o["board"] for x in row)
    except Exception: return (0,) * 16

@lru_cache(maxsize=65536)
def slide(row):
    a = [x for x in row if x]; out=[]; gain=0; i=0
    while i < len(a):
        if i+1 < len(a) and a[i] == a[i+1]:
            out.append(a[i]*2); gain += a[i]*2; i += 2
        else: out.append(a[i]); i += 1
    return tuple(out + [0]*(4-len(out))), gain

@lru_cache(maxsize=131072)
def transition(board, action):
    out=list(board); gain=0
    for line in LINES[action]:
        vals,g=slide(tuple(board[i] for i in line)); gain += g
        for i,v in zip(line, vals): out[i]=v
    return tuple(out), gain

def legal(board):
    return [a for a in DIRECTIONS if transition(board,a)[0] != board]

def transition_error(before, after, action):
    """Check the realized step before trusting the next search root."""
    b, a = flat(before), flat(after); moved, gain = transition(b, action)
    if moved == b: return "local action was a no-op"
    diffs = [i for i in range(16) if moved[i] != a[i]]
    if len(diffs) != 1 or moved[diffs[0]] != 0 or a[diffs[0]] not in (2,4):
        return "unexpected spawn/board delta"
    if after.get("score", 0) - before.get("score", 0) != gain:
        return "merge score mismatch"
    if after.get("moves", 0) != before.get("moves", 0) + 1:
        return "move counter mismatch"
    return None

def spawned_children(board):
    empty=[i for i,v in enumerate(board) if not v]
    if not empty: return ((board,1.0),)
    p=1.0/len(empty)
    return tuple((tuple(board[:i]+(v,)+board[i+1:]), p*q)
                 for i in empty for v,q in ((2,.9),(4,.1)))

def monotone(board, anchor):
    order=SNAKES.get(anchor, SNAKES["dr"])
    ranks=[(board[i].bit_length()-1 if board[i] else 0) for i in order]
    # Penalise reversals; the first (anchor) end should contain the largest tile.
    bad=0
    for a,b in zip(ranks,ranks[1:]):
        if a < b: bad += (b-a)*(b-a)
    return -bad

def corner_ok(board, anchor):
    idx=SNAKES.get(anchor,SNAKES["dr"])[0]
    return board[idx] == max(board) if max(board) else True

class Harness:
    def __init__(self, ctx):
        self.ctx=ctx; self.anchor="dr"; self.mode="balanced"; self.future=None
        self.session=None; self.calls=0; self.last_review=-100; self.last_illegal=0
        self.history=[]; self.last_board=None

    def heuristic(self, board, gain, move, anchor=None):
        anchor=anchor or self.anchor
        empty=board.count(0); mx=max(board)
        mono=monotone(board,anchor)
        smooth=0; pairs=0
        for line in LINES["left"]+LINES["up"]:
            vals=[board[i].bit_length()-1 for i in line if board[i]]
            for a,b in zip(vals,vals[1:]): smooth += abs(a-b); pairs += (a==b)
        corner=mx if corner_ok(board,anchor) else -2*mx
        pressure = 1.0 if empty <= 3 else (0.35 if empty <= 6 else 0.0)
        shape_weight = max(0.0, (200-move)/40.0)
        if self.mode == "space": empty_w=180
        elif self.mode == "consolidate": empty_w=55
        else: empty_w=105
        return (gain + mx + empty_w*empty + 30*pairs - 2.2*smooth
                + 3.0*mono + corner*(1.0+pressure) + shape_weight*mono)

    def choose_anchor(self, board):
        # Hysteresis: retain a working corner; change only when the largest tile
        # has left it and another corner is clearly better.
        corners={"ul":0,"ur":3,"dl":12,"dr":15}
        mx=max(board)
        if board[corners.get(self.anchor,15)] == mx or mx == 0: return
        scores={a:(board[i]==mx)*1000 + monotone(board,a) for a,i in corners.items()}
        best=max(scores,key=scores.get)
        if scores[best] > scores.get(self.anchor,-10**9)+20: self.anchor=best

    def poll(self, current):
        if self.future is None or not self.future.done(): return
        f,self.future=self.future,None
        try:
            r=f.result(); self.session=getattr(r,"session_id",self.session)
            txt=getattr(r,"text","") or ""; m=re.search(r"\{[^{}]*\}",txt)
            d=json.loads(m.group()) if m else {}
            if d.get("anchor") in SNAKES: self.anchor=d["anchor"]
            if d.get("mode") in ("balanced","space","consolidate"): self.mode=d["mode"]
            self.ctx.log("review_applied", move=current.get("moves",0), anchor=self.anchor, mode=self.mode)
        except Exception: pass

    def review(self, current, reason):
        self.poll(current); rem=self.ctx.budget.remaining()
        if (self.future is not None or self.ctx.done or self.calls>=5 or
            rem.get("system2_calls",0)<=1 or rem.get("wall_s",0)<180 or
            getattr(self.ctx,"system2",True) is None): return
        self.calls += 1; self.last_review=current.get("moves",0)
        prompt=("You supervise a verified 2048 controller. Choose a persistent corner anchor "
                 "(ul,ur,dl,dr) and mode (balanced, space, consolidate). The local search "
                 "already computes exact slides and averages all legal 2/4 spawn cells; "
                 "when there are <=3 empties it hedges against the worst spawn. Do not emit "
                 "a move sequence. Reply only JSON {\"anchor\":\"dr\",\"mode\":\"balanced\"}.\n"+
                 json.dumps({"observation":current,"progress":self.ctx.progress(),"reason":reason,
                             "anchor":self.anchor,"mode":self.mode,"recent":self.history[-8:]},separators=(",",":")))
        def ask():
            try: return self.ctx.think(prompt,session_id=self.session,timeout_s=180.0)
            except Exception: return None
        try: self.future=self.ctx.spawn(ask)
        except Exception: self.future=None

    @lru_cache(maxsize=20000)
    def rollout(self, board, depth, move, anchor, mode):
        acts=legal(board)
        if depth<=0 or not acts: return self.heuristic(board,0,move,anchor)
        vals=[]
        for a in acts:
            moved,g=transition(board,a); children=spawned_children(moved)
            childvals=[(self.rollout(ch,depth-1,move+1,anchor,mode),p) for ch,p in children]
            mean=sum(v*p for v,p in childvals)
            worst=min(v for v,p in childvals)
            empt=moved.count(0); risk=(0.55 if empt<=3 else 0.12)
            vals.append(g + (1-risk)*mean + risk*worst)
        return max(vals)

    def search(self, board, move, acts):
        self.choose_anchor(board)
        scored=[]
        for a in acts:
            moved,g=transition(board,a); children=spawned_children(moved)
            # One continuation ply keeps the all-spawn root evaluation cheap
            # enough to run for the full 200-move episode.
            vals=[self.heuristic(ch,g,move+1)+0.65*self.rollout(ch,1,move+1,self.anchor,self.mode) for ch,p in children]
            mean=sum(v*p for v,(ch,p) in zip(vals,children)); worst=min(vals)
            risk=0.60 if moved.count(0)<=3 else (0.25 if moved.count(0)<=6 else 0.08)
            scored.append((g+(1-risk)*mean+risk*worst,a))
        return max(scored)[1] if scored else (acts[0] if acts else "left")

    def run_episode(self, game):
        ctx=self.ctx; cur=obs(ctx.observe())
        if "board" not in cur: return
        while not ctx.done and not cur.get("game_over",False):
            self.poll(cur); board=flat(cur); acts=[a for a in cur.get("legal_actions",DIRECTIONS) if a in DIRECTIONS]
            if not acts: break
            try: action=self.search(board,int(cur.get("moves",0)),acts)
            except Exception: action=acts[0]
            before=cur
            try: nxt=ctx.act(action); cur=obs(nxt if nxt is not None else ctx.observe())
            except Exception as exc:
                if ctx.done: break
                self.review(cur,"rejected action "+type(exc).__name__); continue
            prog=ctx.progress(); nb=flat(cur)
            err=transition_error(before,cur,action)
            if err:
                self.review(cur, "transition check failed: "+err)
                self.mode="space"
            if len(self.history)>=8: self.history.pop(0)
            self.history.append({"move":cur.get("moves",0),"action":action,"empty":nb.count(0),
                                 "score_delta":cur.get("score",0)-before.get("score",0)})
            reasons=[]
            if err: reasons.append(err)
            if prog.get("illegal_actions",0)>self.last_illegal: reasons.append("rejected action")
            self.last_illegal=prog.get("illegal_actions",0)
            if prog.get("since_obs_change",0)>=1: reasons.append("unchanged observation")
            if prog.get("since_score_change",0)>=10: reasons.append("scoreless run")
            if nb.count(0)<=2: self.mode="space"; reasons.append("space pressure")
            if reasons and ("rejected action" in reasons or cur.get("moves",0)-self.last_review>=24): self.review(cur,"; ".join(reasons))

def build(ctx): return Harness(ctx)

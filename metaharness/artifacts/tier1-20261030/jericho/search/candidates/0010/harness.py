"""Cycle-aware, bounded-context Astra controller for Jericho turn games."""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable

LIFE = re.compile(r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I)
BAD = re.compile(r"not a verb|don't know the word|do not know the word|don't understand|do not understand|not a sentence|not recogni[sz]|can't see any such|cannot see any such|no such (?:thing|object)|can't go|cannot go|not holding|not carrying|don't have|do not have|which .{0,100}(?:mean|refer)|be more specific|yes or no|\by/n\b|nothing happens|doesn't seem to work|is locked|is closed", re.I|re.S)
DANGER = re.compile(r"pitch[- ]black|in the darkness|grue|lamp.{0,80}(?:dim|flicker|out|dying|fading)|attacks? you|strikes? you|wounded|bleeding|flood|stolen|disarmed|you are dead|you have died|game over|congratulations|you win", re.I|re.S)
MENU = re.compile(r"q\s*=\s*resume game|character input|press\s+space|help menu", re.I)

def od(x: Any) -> dict[str, Any]:
    if isinstance(x, dict): return dict(x)
    if isinstance(x, str):
        try:
            y = json.loads(x)
            if isinstance(y, dict): return y
        except Exception: pass
    return {"text": "" if x is None else str(x)}

def tx(o: dict[str, Any]) -> str: return str(o.get("text", ""))

def clean(v: Any) -> str | None:
    if not isinstance(v, str): return None
    s = v.replace("\x00", " ").strip()
    s = re.sub(r"^```(?:json|text|command)?\s*|\s*```$", "", s, flags=re.I).strip()
    s = re.sub(r"^(?:[-*>]|\d+[.)])\s*", "", s).strip()
    s = re.sub(r"^(?:command|action|input|cmd)\s*[:=|]\s*", "", s, flags=re.I).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`": s = s[1:-1].strip()
    if not s or len(s) > 200 or "\n" in s or "\r" in s or LIFE.match(s): return None
    if s.casefold() in {"command", "action", "none", "null", "noop", "q"}: return None
    if any(c in s for c in "{}[];") or re.search(r"\b(?:then|and then)\b", s, re.I): return None
    return s.rstrip(".").strip() or None

def decode(raw: Any) -> tuple[list[str], str]:
    if not isinstance(raw, str): return [], ""
    s = raw.strip()[:30000]; obj = None
    for c in (s, re.sub(r"^```[^\n]*\n?|\n?```$", "", s).strip()):
        for load in (json.loads, ast.literal_eval):
            try: obj = load(c); break
            except Exception: pass
        if obj is not None: break
    vals: list[Any] = []
    def collect(x: Any, d=0):
        if d > 5 or len(vals) >= 10: return
        if isinstance(x, str): vals.append(x)
        elif isinstance(x, list):
            for z in x: collect(z, d+1)
        elif isinstance(x, dict):
            for k in ("commands", "actions", "plan", "steps", "command", "action", "input", "cmd"):
                if k in x: collect(x[k], d+1)
    if obj is not None: collect(obj)
    if not vals:
        tagged = [ln for ln in s.splitlines() if re.match(r"^\s*(?:command|action|input|cmd)\s*[:=]", ln, re.I)]
        vals = tagged or ([s] if "\n" not in s else [])
    out=[]
    for v in vals:
        c=clean(v)
        if c: out.append(c)
        if len(out)>=8: break
    mem = ""
    if isinstance(obj, dict):
        m=obj.get("beliefs", obj.get("memory", ""))
        if isinstance(m,(dict,list)): m=json.dumps(m, ensure_ascii=False)
        if isinstance(m,str): mem=m[:2600]
    return out, mem

class Harness:
    def __init__(self, ctx):
        self.ctx=ctx; self.rows=[]; self.beliefs=""; self.failed=[]
        self.states=Counter(); self.pair=Counter(); self.invalid=0; self.failcalls=0

    def prompt(self, game, obs, rem):
        recent=[{"step":r.get("step"),"command":r.get("command"),"score":r.get("score"),"delta":r.get("delta"),"text":str(r.get("text",""))[:850]} for r in self.rows[-10:]]
        status={k:obs.get(k) for k in ("game_id","score","max_score","score_delta","native_moves","step_count")}
        status.update({"calls_left":rem.get("system2_calls"),"wall_left":round(float(rem.get("wall_s",0) or 0),1),"cycle_count":max(self.states.values(),default=0)})
        return ("You are Astra controlling a turn-based offline Jericho interactive-fiction story. Maximize FINAL native score. "
                "Each command is one parser turn; reasoning is free. Return ONLY JSON {\"commands\":[\"...\"],\"beliefs\":\"...\"}. "
                "Give 3-8 commands forming a supported short route. The first must fit CURRENT OBSERVATION; later commands must be safe consequences. "
                "Use ordinary parser text <=200 chars, no lifecycle or compound commands. Preserve clarification prompts. "
                "If the state has repeated with no score change, BREAK THE LOOP: choose an untried salient clue/object action or a different reachable exit, and do not repeat look/east/west cycles. "
                "Stop after parser failure, hazard, ambiguity, or changed objective. Beliefs are hypotheses, cite evidence and keep them concise.\n\n"
                f"GAME: {str(game.describe())[:1600]}\nSTATUS: {json.dumps(status)}\nCURRENT TEXT: {tx(obs)[:4200]}\n"
                f"OPENING TEXT: {str(self.rows[0].get('text',''))[:2200] if self.rows else tx(obs)[:2200]}\n"
                f"RECENT EXACT TRANSCRIPT: {json.dumps(recent, ensure_ascii=False)}\nFAILED ACTIONS: {json.dumps(self.failed[-8:], ensure_ascii=False)}\n"
                f"BELIEFS (unverified): {self.beliefs[:2600]}\nReason silently and emit compact JSON.")

    def run_episode(self, game) -> None:
        ctx=self.ctx
        if getattr(ctx,"system2",True) is None: return
        try: obs=od(game.observe())
        except Exception: obs={"text":""}
        self.rows.append({"step":obs.get("step_count",0),"command":None,"score":obs.get("score"),"delta":0,"text":tx(obs)})
        pending=[]; last_score=obs.get("score")
        while not ctx.done and not any(bool(obs.get(k)) for k in ("terminated","truncated","won","lost")):
            if not pending:
                rem=ctx.budget.remaining(); wall=float(rem.get("wall_s",0) or 0)
                if rem.get("system2_calls",0)<=0 or wall<8: return
                try:
                    result=ctx.think(self.prompt(game,obs,rem), session_id=None, timeout_s=max(1.0,min(110.0,wall-4.0)))
                    pending, mem=decode(getattr(result,"text",""))
                    if mem: self.beliefs=mem; ctx.memory["beliefs"]=mem
                    self.failcalls=0
                except (BudgetExceeded,EpisodeOver,ModelUnavailable): return
                except (TimeoutError,RuntimeError) as e:
                    ctx.log("think_failed",error=type(e).__name__); self.failcalls+=1; pending=[]
                    if ctx.done or self.failcalls>=3: return
                except Exception as e:
                    ctx.log("think_parse_failed",error=type(e).__name__); self.failcalls+=1; pending=[]
                    if self.failcalls>=3: return
                if not pending:
                    self.invalid+=1
                    if self.invalid==1: pending=["look"]
                    elif self.invalid>=3: return
                else:
                    self.invalid=0; ctx.log("plan",commands=pending[:])
                if not pending:
                    continue
            command=clean(pending.pop(0))
            if not command: continue
            try: out=ctx.act(command); obs=od(out if out is not None else game.observe())
            except Exception as e:
                ctx.log("act_failed",command=command,error=type(e).__name__); self.failed.append({"command":command,"error":type(e).__name__}); pending=[]
                if ctx.done: return
                continue
            response=tx(obs); delta=obs.get("score_delta")
            if not isinstance(delta,(int,float)) and isinstance(obs.get("score"),(int,float)) and isinstance(last_score,(int,float)): delta=obs.get("score")-last_score
            row={"step":obs.get("step_count",len(self.rows)),"command":command,"score":obs.get("score"),"delta":delta,"text":response}; self.rows.append(row); ctx.memory["transcript"]=self.rows[-80:]
            sig=(re.sub(r"\s+"," ",response).strip()[:1000],str(obs.get("score")))
            self.states[sig]+=1; self.pair[(command.casefold(),sig[0],sig[1])]+=1
            interrupted=bool(BAD.search(response) or DANGER.search(response) or MENU.search(response))
            if isinstance(delta,(int,float)) and delta<0: interrupted=True
            # A broad state recurrence catches alternating commands that evade pair checks.
            if self.states[sig]>=3 or self.pair[(command.casefold(),sig[0],sig[1])]>=2: interrupted=True
            if interrupted:
                pending=[]; self.failed.append({"command":command,"response":response[:350],"delta":delta,"cycle":self.states[sig]})
            elif response==tx(self.rows[-2]) and command.casefold() in {"look","l","inventory","i"}:
                pending=[]
            last_score=obs.get("score")
            ctx.log("step",command=command,score=obs.get("score"),delta=delta,cycle=self.states[sig])
            if ctx.done: return

def build(ctx): return Harness(ctx)

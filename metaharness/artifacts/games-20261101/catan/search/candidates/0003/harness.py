"""Plan-and-check Catan harness: Astra chooses a strategic goal; code executes only live legal IDs."""
from __future__ import annotations
import json,re
RES=("WOOD","BRICK","SHEEP","WHEAT","ORE")
PIPS={2:1,3:2,4:3,5:4,6:5,8:5,9:4,10:3,11:2,12:1}
GOALS={"settlement","city","development","road","army","balanced"}

def legal(o): return o.get("legal_actions",[]) if isinstance(o,dict) else []
def par(a): return a.get("parameters") if isinstance(a,dict) else None
def groups(o):
 d={}
 for a in legal(o): d.setdefault(a.get("type"),[]).append(a)
 return d
def setup_scores(o):
 b=o.get("board",{}); hs={h.get("id"):h for h in b.get("hexes",[])}; out={}
 for v in b.get("vertices",[]):
  kinds=set(); p=0
  for hid in v.get("adjacent_hexes",[]):
   h=hs.get(hid,{})
   if h.get("terrain") in RES and isinstance(h.get("number"),int): kinds.add(h["terrain"]); p+=PIPS.get(h["number"],0)
  out[v.get("id")]=p+1.35*len(kinds)+1.0*("WHEAT" in kinds)+.75*("ORE" in kinds)+1.15*("WOOD" in kinds)+1.15*("BRICK" in kinds)+.6*("SHEEP" in kinds)+.8*({"WOOD","BRICK"}<=kinds)
 return out
def site_quality(o,s,vid):
 vs={v.get("id"):v for v in o.get("board",{}).get("vertices",[])}; v=vs.get(vid,{})
 blocked={x.get("id") for x in vs.values() if x.get("building")}
 return max((s.get(n,0) for n in v.get("neighbors",[]) if n not in blocked),default=0)
def resources(o):
 r=o.get("self",{}).get("resources",{}) if isinstance(o,dict) else {}
 return {str(k).upper():int(v) for k,v in (r or {}).items()}
def choose(o,plan):
 aa=legal(o)
 if not aa:return None
 g=groups(o); r=resources(o); b=o.get("board",{}); hs={h.get("id"):h for h in b.get("hexes",[])}; s=setup_scores(o); goal=plan.get("goal","balanced")
 first=lambda t:g.get(t,[None])[0]
 phase=str(o.get("pending_decision",o.get("phase",""))).lower()
 if "initial_settlement" in phase and g.get("build_settlement"): return max(g["build_settlement"],key=lambda a:(s.get(par(a),0),str(a.get("id"))))
 if "initial_road" in phase and g.get("build_road"):
  mine=o.get("self",{}).get("id"); owned={v.get("id") for v in b.get("vertices",[]) if v.get("owner")==mine}
  def rk(a):
   p=par(a)
   if not isinstance(p,(list,tuple)) or len(p)!=2:return (0,0)
   q=p[1] if p[0] in owned else p[0]
   return (site_quality(o,s,q),-int(q) if isinstance(q,int) else 0)
  return max(g["build_road"],key=rk)
 if g.get("discard_resource"): return max(g["discard_resource"],key=lambda a:(r.get(str(par(a)).upper(),0),str(par(a))))
 if g.get("move_robber"):
  ops=o.get("opponents",[])
  def rk(a):
   p=par(a) or {}; hid=p.get("hex",p.get("hex_id",p.get("coordinate"))) if isinstance(p,dict) else None; h=hs.get(hid,{}); vic=p.get("victim") if isinstance(p,dict) else None; op=next((x for x in ops if x.get("id")==vic),{})
   return (PIPS.get(h.get("number"),0)*(1+.15*op.get("public_vp",0)),op.get("resource_count",0))
  return max(g["move_robber"],key=rk)
 for t in ("play_year_of_plenty","play_monopoly","play_knight_card"):
  if g.get(t): return g[t][0]
 if goal in ("settlement","balanced") and g.get("build_settlement"): return max(g["build_settlement"],key=lambda a:(s.get(par(a),0),site_quality(o,s,par(a))))
 if goal in ("city","balanced") and g.get("build_city"):
  def ck(a):
   v=next((v for v in b.get("vertices",[]) if v.get("id")==par(a)),{}); return (sum(PIPS.get(hs.get(x,{}).get("number"),0) for x in v.get("adjacent_hexes",[])),str(a.get("id")))
  return max(g["build_city"],key=ck)
 if goal in ("road","settlement","balanced") and g.get("build_road"):
  def roadk(a):
   p=par(a)
   if not isinstance(p,(list,tuple)) or len(p)!=2:return (0,str(a.get("id")))
   return (max(site_quality(o,s,p[0]),site_quality(o,s,p[1])),str(a.get("id")))
  return max(g["build_road"],key=roadk)
 if goal in ("development","army","balanced") and g.get("buy_development_card"): return g["buy_development_card"][0]
 if g.get("maritime_trade"):
  for a in g["maritime_trade"]:
   p=par(a)
   if isinstance(p,(list,tuple)) and p and str(p[-1]).upper() in RES and r.get(str(p[-1]).upper(),0)==0:return a
 return first("roll") or first("end_turn") or aa[0]
def parse_plan(text):
 obj=None
 try: obj=json.loads((text or "").strip().strip("`"))
 except Exception:
  m=re.search(r"\{.*?\}",text or "",re.S)
  if m:
   try: obj=json.loads(m.group(0))
   except Exception: pass
 if not isinstance(obj,dict):obj={}
 goal=str(obj.get("goal","balanced")).lower(); goal=goal if goal in GOALS else "balanced"
 pri=obj.get("resource_priority",[]); pri=pri if isinstance(pri,list) else []
 return {"goal":goal,"resource_priority":[str(x).upper() for x in pri if str(x).upper() in RES][:5],"reason":str(obj.get("reason",""))[:240]}
class Harness:
 def __init__(self,ctx): self.ctx=ctx; self.session=None; self.plan={"goal":"balanced"}; self.replans=0; self.last_replan=-99
 def ask_plan(self,o,evidence="initial state"):
  c=self.ctx
  if c.system2 is None or c.done or c.budget.remaining().get("system2_calls",0)<=0:return
  compact={k:o.get(k) for k in ("phase","pending_decision","turn_number","self","opponents","awards","legal_actions")}; compact["board"]=o.get("board",{})
  prompt=("Plan Catan. Return ONLY JSON: goal is one of settlement,city,development,road,army,balanced; resource_priority is a list; reason is short. Do not return an action ID. Reconsider the route when evidence shows stagnation. Evidence="+evidence+" State="+json.dumps(compact,separators=(",",":"))[:18000])
  try:
   z=c.think(prompt,session_id=self.session,timeout_s=240.0); self.session=z.session_id; self.plan=parse_plan(z.text); self.replans+=1
  except (TimeoutError,RuntimeError): pass
 def run_episode(self,game):
  c=self.ctx; o=game.observe(); self.ask_plan(o)
  while not c.done and not game.done:
   p=c.progress(); step=int(p.get("acts",0)); stalled=p.get("since_obs_change",0)>=24 or p.get("illegal_actions",0)>0
   if stalled and step-self.last_replan>=8 and self.replans<10: self.last_replan=step; self.ask_plan(o,"ctx.progress="+json.dumps(p,separators=(",",":")))
   a=choose(o,self.plan)
   if a is None:break
   aid=a.get("id") if isinstance(a,dict) else a; before=p
   try:nxt=c.act(aid)
   except Exception:
    if c.done:break
    o=game.observe(); continue
   after=c.progress()
   if after.get("illegal_actions",0)>before.get("illegal_actions",0): o=game.observe(); continue
   o=nxt if isinstance(nxt,dict) else game.observe()
def build(ctx): return Harness(ctx)

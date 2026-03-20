#!/usr/bin/env python3
"""
Convert RL self-play replay JSON to an HTML visualization.

Usage:
    python scripts/view_rl_replay.py replay_iter50.json -o replay.html
"""

import argparse
import json

_DIR_DR = [1, 0, -1, 0]
_DIR_DC = [0, 1, 0, -1]


def generate_html(replay, output_path):
    steps = replay["steps"]
    H, W = 25, 25
    real_h = replay["map"]["real_h"]
    real_w = replay["map"]["real_w"]

    # Build frames from stored snapshots (no reconstruction needed)
    states_flat = []
    armies_flat = []

    # Initial frame: use first step's state (before any moves would be turn 0)
    # We use step 0's snapshot as frame 0 (state after turn 0)
    for step in steps:
        st = []
        ar = []
        for r in range(H):
            for c in range(W):
                st.append(step["state"][r][c])
                ar.append(max(step["army"][r][c], 0))
        states_flat.append(st)
        armies_flat.append(ar)

    T = len(states_flat)
    moves_data = []

    for step in steps:
        p0a = step["p0_action"]
        p1a = step["p1_action"]

        def action_coords(action):
            if action is None:
                return [-1, -1], [-1, -1], -1, 0
            src = action["source_idx"]
            sr, sc = divmod(src, W)
            d = action["direction"]
            er, ec = sr + _DIR_DR[d], sc + _DIR_DC[d]
            return [sr, sc], [er, ec], d, action["split"]

        p0s, p0e, p0d, p0spl = action_coords(p0a)
        p1s, p1e, p1d, p1spl = action_coords(p1a)

        moves_data.append({
            "p0s": p0s, "p0e": p0e, "p0d": p0d, "p0spl": p0spl,
            "p0pass": p0a is None,
            "p0illegal": step["illegal"][0],
            "p1s": p1s, "p1e": p1e, "p1d": p1d, "p1spl": p1spl,
            "p1pass": p1a is None,
            "p1illegal": step["illegal"][1],
            "r0": round(step["p0_reward"], 4),
            "r1": round(step["p1_reward"], 4),
            "p0_land": step["p0_stats"]["land"],
            "p0_army": step["p0_stats"]["army"],
            "p1_land": step["p1_stats"]["land"],
            "p1_army": step["p1_stats"]["army"],
        })

    ep_len = replay["ep_len"]
    winner = replay["winner"]
    total_r = replay["total_reward_p0"]
    n_illegal_p0 = sum(1 for s in steps if s["illegal"][0])
    n_illegal_p1 = sum(1 for s in steps if s["illegal"][1])
    n_pass_p0 = sum(1 for s in steps if s["p0_action"] is None)
    n_pass_p1 = sum(1 for s in steps if s["p1_action"] is None)

    outcome = f"Winner: P{winner}" if winner is not None else "Draw"

    payload = json.dumps({
        "H": H, "W": W, "T": T,
        "realH": real_h, "realW": real_w,
        "states": states_flat, "armies": armies_flat,
        "moves": moves_data,
        "stats": {
            "ep_len": ep_len, "outcome": outcome,
            "total_r": round(total_r, 3),
            "illegal_p0": n_illegal_p0, "illegal_p1": n_illegal_p1,
            "pass_p0": n_pass_p0, "pass_p1": n_pass_p1,
            "iteration": replay.get("iteration", "?"),
        },
    }, separators=(",", ":"))

    html = RL_HTML_TEMPLATE.replace("PAYLOAD_PLACEHOLDER", payload)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved {output_path}")


RL_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>RL Self-Play Replay</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#eee;font-family:'Menlo','Consolas',monospace;
     display:flex;flex-direction:column;align-items:center;padding:16px;min-height:100vh}
h2{margin-bottom:8px;font-size:15px;color:#aaa}
canvas{display:block;image-rendering:pixelated}
#hud{display:flex;gap:16px;margin-top:10px;font-size:13px;flex-wrap:wrap;justify-content:center}
#hud .box{background:#222;border-radius:6px;padding:6px 12px}
#slider-row{margin-top:8px;display:flex;align-items:center;gap:8px;width:100%;max-width:900px}
#slider-row input[type=range]{flex:1}
#turn-label{min-width:120px;text-align:center;font-size:13px}
#controls{margin-top:10px;font-size:12px;color:#777;text-align:center;line-height:1.7}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;justify-content:center;font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:4px}
.legend i{display:inline-block;width:28px;height:4px;border-radius:2px}
#summary{margin-top:8px;font-size:13px;color:#aaa}
</style></head><body>
<h2>RL Self-Play Replay</h2>
<canvas id="cv"></canvas>
<div id="slider-row">
  <button id="bprev">&#9664;</button>
  <input type="range" id="slider" min="0" value="0">
  <button id="bnext">&#9654;</button>
  <span id="turn-label">t=0</span>
</div>
<div id="hud">
  <div class="box" id="p0-info"></div>
  <div class="box" id="p1-info"></div>
  <div class="box" id="reward-info"></div>
</div>
<div id="summary"></div>
<div class="legend">
  <span><i style="background:#4CAF50"></i> P0 territory</span>
  <span><i style="background:#f44336"></i> P1 territory</span>
  <span><i style="background:#00e5ff"></i> P0 move</span>
  <span><i style="background:#ff6600"></i> P1 move</span>
  <span><i style="background:#ffd700"></i> City</span>
  <span><i style="background:#555"></i> Mountain</span>
</div>
<div id="controls">
  &larr; &rarr; step | Space: play/pause | [ ] speed | Home/End: jump
</div>

<script type="application/json" id="payload">PAYLOAD_PLACEHOLDER</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("payload").textContent);
const H=D.H,W=D.W,T=D.T,rH=D.realH,rW=D.realW;
const CELL=Math.min(Math.floor((window.innerWidth-60)/W),
                    Math.floor((window.innerHeight-320)/H),32);
const cv=document.getElementById("cv");
cv.width=W*CELL; cv.height=H*CELL;
const ctx=cv.getContext("2d");

const COLORS={"-1":"#0e0e1a","0":"#2d2d44","1":"#4a4a6a","2":"#c8c8c8",
"3":"#ffd700","4":"#555555","5":"#4CAF50","6":"#2E7D32","7":"#1B5E20",
"8":"#f44336","9":"#c62828","10":"#b71c1c"};
const DARR=["\u2193","\u2192","\u2191","\u2190"];
const SPL=["all","50%"];

let turn=0,playing=false,speed=300;
const slider=document.getElementById("slider");
slider.max=T-1;

function drawArrow(x1,y1,x2,y2,color,lw){
  ctx.strokeStyle=color;ctx.lineWidth=lw;
  ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  const ang=Math.atan2(y2-y1,x2-x1);
  const hl=CELL*0.3;
  ctx.beginPath();
  ctx.moveTo(x2,y2);
  ctx.lineTo(x2-hl*Math.cos(ang-0.4),y2-hl*Math.sin(ang-0.4));
  ctx.moveTo(x2,y2);
  ctx.lineTo(x2-hl*Math.cos(ang+0.4),y2-hl*Math.sin(ang+0.4));
  ctx.stroke();
}

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  for(let r=0;r<H;r++) for(let c=0;c<W;c++){
    const idx=r*W+c;
    const s=D.states[turn][idx];
    if(r>=rH||c>=rW){
      ctx.fillStyle="#0e0e1a";
    } else {
      ctx.fillStyle=COLORS[String(s)]||"#000";
    }
    ctx.fillRect(c*CELL,r*CELL,CELL,CELL);
    const a=D.armies[turn][idx];
    if(a>0 && r<rH && c<rW && s!==4){
      ctx.fillStyle=(s>=5&&s<=10)?"#fff":"#000";
      const fs=a<100?Math.round(CELL*0.36):a<1000?Math.round(CELL*0.28):Math.round(CELL*0.22);
      ctx.font="bold "+fs+"px monospace";
      ctx.textAlign="center";ctx.textBaseline="middle";
      ctx.fillText(String(a),c*CELL+CELL/2,r*CELL+CELL/2);
    }
  }
  ctx.strokeStyle="#33333366";ctx.lineWidth=0.5;
  for(let r=0;r<=H;r++){ctx.beginPath();ctx.moveTo(0,r*CELL);ctx.lineTo(W*CELL,r*CELL);ctx.stroke();}
  for(let c=0;c<=W;c++){ctx.beginPath();ctx.moveTo(c*CELL,0);ctx.lineTo(c*CELL,H*CELL);ctx.stroke();}
  ctx.strokeStyle="#ffffff44";ctx.lineWidth=2;
  ctx.strokeRect(0,0,rW*CELL,rH*CELL);

  const m=D.moves[turn];
  if(m){
    if(!m.p0pass && m.p0s[0]>=0){
      const col=m.p0illegal?"#ff660088":"#00e5ff";
      const x1=m.p0s[1]*CELL+CELL/2, y1=m.p0s[0]*CELL+CELL/2;
      const x2=m.p0e[1]*CELL+CELL/2, y2=m.p0e[0]*CELL+CELL/2;
      drawArrow(x1,y1,x2,y2,col,3);
    }
    if(!m.p1pass && m.p1s[0]>=0){
      const col=m.p1illegal?"#ff660088":"#ff6600";
      const x1=m.p1s[1]*CELL+CELL/2-2, y1=m.p1s[0]*CELL+CELL/2-2;
      const x2=m.p1e[1]*CELL+CELL/2-2, y2=m.p1e[0]*CELL+CELL/2-2;
      drawArrow(x1,y1,x2,y2,col,2.5);
    }
  }

  slider.value=turn;
  document.getElementById("turn-label").textContent="t="+turn+" / "+(T-1);

  const p0i=document.getElementById("p0-info");
  const p1i=document.getElementById("p1-info");
  const ri=document.getElementById("reward-info");

  if(m){
    const fmt=(s,e,d,spl,pass_,illegal)=>{
      if(pass_) return "<b>PASS</b>";
      let txt="("+s[0]+","+s[1]+") "+DARR[d]+" "+SPL[spl];
      if(illegal) txt+=" <span style='color:#f44'>[ILLEGAL]</span>";
      return txt;
    };
    p0i.innerHTML="<b style='color:#00e5ff'>P0:</b> "+fmt(m.p0s,m.p0e,m.p0d,m.p0spl,m.p0pass,m.p0illegal)
      +" | land:"+m.p0_land+" army:"+m.p0_army;
    p1i.innerHTML="<b style='color:#ff6600'>P1:</b> "+fmt(m.p1s,m.p1e,m.p1d,m.p1spl,m.p1pass,m.p1illegal)
      +" | land:"+m.p1_land+" army:"+m.p1_army;
    ri.innerHTML="r0="+m.r0.toFixed(4)+" r1="+m.r1.toFixed(4);
  } else {
    p0i.textContent="P0: --";
    p1i.textContent="P1: --";
    ri.textContent="--";
  }

  const st=D.stats;
  document.getElementById("summary").innerHTML=
    "Iter "+st.iteration+" | "+st.outcome+" | len="+st.ep_len
    +" | r_total="+st.total_r
    +" | P0 illegal:"+st.illegal_p0+" pass:"+st.pass_p0
    +" | P1 illegal:"+st.illegal_p1+" pass:"+st.pass_p1;
}

function setTurn(t){turn=Math.max(0,Math.min(T-1,t));draw();}
let timer=null;
function togglePlay(){
  playing=!playing;
  if(playing){timer=setInterval(()=>{if(turn<T-1)setTurn(turn+1);else{playing=false;clearInterval(timer);}},speed);}
  else if(timer){clearInterval(timer);timer=null;}
}
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft") setTurn(turn-1);
  else if(e.key==="ArrowRight") setTurn(turn+1);
  else if(e.key===" "){e.preventDefault();togglePlay();}
  else if(e.key==="["){speed=Math.min(2000,speed+50);if(playing){clearInterval(timer);timer=setInterval(()=>{if(turn<T-1)setTurn(turn+1);else{playing=false;clearInterval(timer);}},speed);}}
  else if(e.key==="]"){speed=Math.max(20,speed-50);if(playing){clearInterval(timer);timer=setInterval(()=>{if(turn<T-1)setTurn(turn+1);else{playing=false;clearInterval(timer);}},speed);}}
  else if(e.key==="Home") setTurn(0);
  else if(e.key==="End") setTurn(T-1);
});
slider.addEventListener("input",()=>setTurn(parseInt(slider.value)));
document.getElementById("bprev").addEventListener("click",()=>setTurn(turn-1));
document.getElementById("bnext").addEventListener("click",()=>setTurn(turn+1));
draw();
</script></body></html>"""


def main():
    p = argparse.ArgumentParser(description="View RL self-play replay")
    p.add_argument("replay_json", help="Path to replay_iter*.json")
    p.add_argument("-o", "--output", default=None, help="Output HTML path")
    args = p.parse_args()

    with open(args.replay_json) as f:
        replay = json.load(f)

    output = args.output or args.replay_json.replace(".json", ".html")
    generate_html(replay, output)


if __name__ == "__main__":
    main()

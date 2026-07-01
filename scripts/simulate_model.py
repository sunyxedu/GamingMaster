#!/usr/bin/env python3
"""
Simulate a trained model on a replay and compare predictions to ground truth.

Usage:
    python3 scripts/simulate_model.py \
        --checkpoint checkpoints/checkpoint_epoch11.pt \
        --replay data/my_replays/training/MeltedToast/0v1x6p-8g.json \
        [--max_steps 50] [--device cpu]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.network import GeneralsNetwork

DIR_NAMES = ["down", "right", "up", "left"]
DIR_ARROWS = ["\u2193", "\u2192", "\u2191", "\u2190"]
SPLIT_NAMES = ["all", "50%", "pass"]

STATE_CHARS = {
    -1: " ", 0: "\u00b7", 1: "#", 2: "_", 3: "C", 4: "M",
    5: "o", 6: "O", 7: "G", 8: "x", 9: "X", 10: "E",
}


def load_model(checkpoint_path, device):
    from model.rl_utils import load_model_state

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
    model = GeneralsNetwork(cfg)
    state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
    load_model_state(model, ckpt[state_key], ctx=checkpoint_path)
    model.to(device)
    model.eval()
    if "epoch" in ckpt:
        print(f"Loaded IL checkpoint: epoch {ckpt['epoch']}, loss {ckpt.get('loss', 0):.4f}")
    else:
        print(f"Loaded RL checkpoint: iteration {ckpt.get('iteration', '?')}")
    print(f"Model: d={cfg.d_model}, heads={cfg.n_heads}, params={sum(p.numel() for p in model.parameters()):,}")
    return model, cfg


def load_replay(replay_path, max_steps=None):
    with open(replay_path) as f:
        data = json.load(f)

    # Detect data format:
    # - convert_simulator_to_gio format: has legal_source_mask_seq, src_y is [T, K]
    #   target[t] = action decided after seeing state[t] (already aligned)
    # - old format: src_y is scalar, no legal_source_mask_seq
    #   target[t] = action at sim turn t (needs +1 shift to align with model)
    is_new_format = "legal_source_mask_seq" in data
    if not is_new_format:
        print("NOTE: Old data format detected — shifting targets by +1 to align with model")

    grid = torch.tensor(data["grid_seq"], dtype=torch.float32)
    T_raw = grid.shape[0]

    if not is_new_format:
        # Old format: drop last observation (no future action for it)
        T = T_raw - 1
        grid = grid[:T]
    else:
        T = T_raw

    if max_steps and T > max_steps:
        T = max_steps
        grid = grid[:T]

    globals_seq = torch.tensor(
        data.get("globals_seq", [[0.0] * 4] * T_raw), dtype=torch.float32
    )[:T]

    H, W = 25, 25
    legal_mask = torch.tensor(
        data.get("legal_source_mask_seq", [[[[True] * W] * H]] * T_raw), dtype=torch.bool
    )[:T]

    K = data.get("max_actions_per_tick", 4)
    raw_count = data.get("count", [0] * T_raw)
    raw_src_y = data.get("src_y", [[0] * K] * T_raw)
    raw_src_x = data.get("src_x", [[0] * K] * T_raw)
    raw_dir_id = data.get("dir_id", [[0] * K] * T_raw)
    raw_ratio_id = data.get("ratio_id", [[0] * K] * T_raw)

    if not is_new_format:
        # Shift targets by +1: target[t] = raw action at t+1
        count = raw_count[1:T + 1]
        src_y = raw_src_y[1:T + 1]
        src_x = raw_src_x[1:T + 1]
        dir_id = raw_dir_id[1:T + 1]
        ratio_id = raw_ratio_id[1:T + 1]
    else:
        count = raw_count[:T]
        src_y = raw_src_y[:T]
        src_x = raw_src_x[:T]
        dir_id = raw_dir_id[:T]
        ratio_id = raw_ratio_id[:T]

    return {
        "grid": grid,
        "globals": globals_seq,
        "legal_mask": legal_mask,
        "count": count,
        "src_y": src_y,
        "src_x": src_x,
        "dir_id": dir_id,
        "ratio_id": ratio_id,
        "T": T,
    }


def render_grid_ascii(grid_channels, src_pred=None, src_gt=None):
    """Render one timestep's grid as ASCII. grid_channels: [C, H, W]."""
    H, W = 25, 25
    state_map = [[-1] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            for ch in range(11):
                if grid_channels[ch][r][c] > 0.5:
                    state_map[r][c] = ch
                    break

    lines = []
    for r in range(H):
        row = ""
        for c in range(W):
            ch = STATE_CHARS.get(state_map[r][c], "?")
            if src_pred and (r, c) == src_pred:
                ch = "P"
            if src_gt and (r, c) == src_gt:
                ch = "T" if ch != "P" else "B"  # B = both
            row += ch
        # Trim trailing padding
        row = row.rstrip()
        if row:
            lines.append(row)
    # Remove empty trailing lines
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


DIR_DELTAS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # down, right, up, left


def run_inference(model, replay, device, grid_w):
    """Run model on full sequence and return per-step predictions."""
    T = replay["T"]
    W = grid_w

    grid_seq = replay["grid"].unsqueeze(0).to(device)
    globals_seq = replay["globals"].unsqueeze(0).to(device)
    legal_mask = replay["legal_mask"].unsqueeze(0).to(device)

    # Condition direction/split on the GT source (teacher forcing): the
    # reported dir/split accuracies compare against the human's move, which
    # is only meaningful when the heads saw the human's source embedding —
    # otherwise they are conditioned on an internally sampled source while
    # the display shows the argmax source.
    src_y_gt = torch.full((1, T), -1, dtype=torch.long)
    src_x_gt = torch.full((1, T), -1, dtype=torch.long)
    for t in range(T):
        if replay["count"][t] > 0:
            sy = replay["src_y"][t][0] if isinstance(replay["src_y"][t], list) else replay["src_y"][t]
            sx = replay["src_x"][t][0] if isinstance(replay["src_x"][t], list) else replay["src_x"][t]
            src_y_gt[0, t] = int(sy)
            src_x_gt[0, t] = int(sx)

    with torch.no_grad():
        pred = model(grid_seq, globals_seq, legal_mask=legal_mask,
                     source_y=src_y_gt.to(device), source_x=src_x_gt.to(device))

    src_logits = pred["source_logits"][0]
    dir_logits = pred["direction_logits"][0]
    spl_logits = pred["split_logits"][0]
    values = pred["value"][0]

    lm = legal_mask[0].reshape(T, -1)
    src_logits_masked = src_logits.clone()
    src_logits_masked[~lm] = float("-inf")
    bad = src_logits_masked.isinf().all(dim=-1)
    if bad.any():
        src_logits_masked[bad, 0] = 0.0

    pass_logits = pred["pass_logits"][0]

    src_probs = F.softmax(src_logits_masked, dim=-1)
    pred_src_idx = src_probs.argmax(dim=-1)
    pred_dir = dir_logits.argmax(dim=-1)
    pred_spl = spl_logits.argmax(dim=-1)
    pred_pass = pass_logits > 0  # True = pass (no move)

    steps = []
    for t in range(T):
        gt_count = replay["count"][t]
        if isinstance(replay["src_y"][t], list):
            gt_sy, gt_sx = replay["src_y"][t][0], replay["src_x"][t][0]
            gt_dir, gt_ratio = replay["dir_id"][t][0], replay["ratio_id"][t][0]
        else:
            gt_sy, gt_sx = replay["src_y"][t], replay["src_x"][t]
            gt_dir, gt_ratio = replay["dir_id"][t], replay["ratio_id"][t]

        p_src = pred_src_idx[t].item()
        p_sy, p_sx = p_src // W, p_src % W
        p_dir = pred_dir[t].item()
        p_spl = pred_spl[t].item()
        val = values[t].item()
        has_move = gt_count > 0
        gt_src_prob = src_probs[t, gt_sy * W + gt_sx].item() if has_move else 0.0

        p_pass = pred_pass[t].item()  # from dedicated pass head

        steps.append({
            "p_sy": p_sy, "p_sx": p_sx, "p_dir": p_dir, "p_spl": p_spl,
            "p_pass": p_pass,
            "gt_sy": gt_sy, "gt_sx": gt_sx, "gt_dir": gt_dir, "gt_ratio": gt_ratio,
            "has_move": has_move, "val": val, "gt_src_prob": gt_src_prob,
            "src_ok": has_move and (not p_pass) and p_sy == gt_sy and p_sx == gt_sx,
            "dir_ok": has_move and (not p_pass) and p_dir == gt_dir,
            "spl_ok": has_move and (not p_pass) and p_spl == gt_ratio,
        })
    return steps


def print_results(steps, only_moves):
    """Print text summary to stdout."""
    total_moves = src_correct = dir_correct = spl_correct = all_correct = 0
    for t, s in enumerate(steps):
        if only_moves and not s["has_move"]:
            continue
        if s["has_move"]:
            total_moves += 1
            src_correct += int(s["src_ok"])
            dir_correct += int(s["dir_ok"])
            spl_correct += int(s["spl_ok"])
            all_correct += int(s["src_ok"] and s["dir_ok"] and s["spl_ok"])
            status = ""
            if s["p_pass"]:
                status = " [model pass, should move!]"
            elif s["src_ok"] and s["dir_ok"] and s["spl_ok"]:
                status = " [MATCH]"
            elif s["src_ok"]:
                status = " [src ok]"
            pred_str = "PASS" if s["p_pass"] else f"({s['p_sy']:2d},{s['p_sx']:2d}) {DIR_ARROWS[s['p_dir']]} {SPLIT_NAMES[s['p_spl']]:>3s}"
            print(
                f"t={t:3d}  "
                f"pred={pred_str:>20s}  "
                f"gt=({s['gt_sy']:2d},{s['gt_sx']:2d}) {DIR_ARROWS[s['gt_dir']]} {SPLIT_NAMES[s['gt_ratio']]:>3s}  "
                f"val={s['val']:+.3f}  "
                f"p(gt_src)={s['gt_src_prob']:.3f}"
                f"{status}"
            )
        else:
            pred_str = "PASS" if s["p_pass"] else f"({s['p_sy']:2d},{s['p_sx']:2d}) {DIR_ARROWS[s['p_dir']]}"
            tag = " [ok]" if s["p_pass"] else " [model moves, should pass!]"
            print(f"t={t:3d}  pred={pred_str:>20s}  [no move]  val={s['val']:+.3f}{tag}")

    print("=" * 70)
    if total_moves > 0:
        print(f"Total moves: {total_moves}")
        print(f"Source accuracy:    {src_correct}/{total_moves} = {100*src_correct/total_moves:.1f}%")
        print(f"Direction accuracy: {dir_correct}/{total_moves} = {100*dir_correct/total_moves:.1f}%")
        print(f"Split accuracy:     {spl_correct}/{total_moves} = {100*spl_correct/total_moves:.1f}%")
        print(f"Full move match:    {all_correct}/{total_moves} = {100*all_correct/total_moves:.1f}%")
    else:
        print("No moves in replay.")


def generate_html(replay, steps, output_path):
    """Generate interactive HTML visualization."""
    import math
    T = replay["T"]
    grid = replay["grid"]  # [T, C, H, W]

    # Extract state IDs and army values from grid channels
    states_flat = []
    armies_flat = []
    for t in range(T):
        st = []
        ar = []
        for r in range(25):
            for c in range(25):
                sid = -1
                for ch in range(11):
                    if grid[t, ch, r, c] > 0.5:
                        sid = ch
                        break
                st.append(sid)
                log_army = grid[t, 11, r, c].item()
                army = int(round(math.expm1(log_army * 10.0))) if log_army > 0 else 0
                ar.append(army)
        states_flat.append(st)
        armies_flat.append(ar)

    # Build per-step move data for JS
    moves_data = []
    for t, s in enumerate(steps):
        dr, dc = DIR_DELTAS[s["p_dir"]]
        p_ey, p_ex = s["p_sy"] + dr, s["p_sx"] + dc
        if s["has_move"]:
            gdr, gdc = DIR_DELTAS[s["gt_dir"]]
            g_ey, g_ex = s["gt_sy"] + gdr, s["gt_sx"] + gdc
        else:
            g_ey, g_ex = -1, -1
        moves_data.append({
            "ps": [s["p_sy"], s["p_sx"]], "pe": [p_ey, p_ex],
            "pd": s["p_dir"], "pspl": s["p_spl"], "pp": s["p_pass"],
            "gs": [s["gt_sy"], s["gt_sx"]], "ge": [g_ey, g_ex],
            "gd": s["gt_dir"], "gspl": s["gt_ratio"],
            "hm": s["has_move"], "val": round(s["val"], 3),
            "gp": round(s["gt_src_prob"], 4),
            "sok": s["src_ok"], "dok": s["dir_ok"], "spok": s["spl_ok"],
        })

    n_model_moves = sum(1 for s in steps if not s["p_pass"])
    n_passes = sum(1 for s in steps if s["p_pass"])

    payload = json.dumps({
        "H": 25, "W": 25, "T": T,
        "states": states_flat, "armies": armies_flat,
        "moves": moves_data,
        "stats": {"total": n_model_moves, "passes": n_passes},
    }, separators=(",", ":"))

    html = HTML_VIS_TEMPLATE.replace("PAYLOAD_PLACEHOLDER", payload)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML visualization saved to {output_path}")


HTML_VIS_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Model Prediction Viewer</title>
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
.match{color:#4CAF50} .miss{color:#f44336} .partial{color:#ffd700}
</style></head><body>
<h2>Model Prediction Viewer</h2>
<canvas id="cv"></canvas>
<div id="slider-row">
  <button id="bprev">&#9664;</button>
  <input type="range" id="slider" min="0" value="0">
  <button id="bnext">&#9654;</button>
  <span id="turn-label">t=0</span>
</div>
<div id="hud">
  <div class="box" id="pred-info"></div>
  <div class="box" id="val-info"></div>
</div>
<div id="summary"></div>
<div class="legend">
  <span><i style="background:#00e5ff"></i> Model's move</span>
  <span><i style="background:#ffe000"></i> Human's move</span>
</div>
<div id="controls">
  &larr; &rarr; step | Space: play/pause | [ ] speed | Home/End: jump
</div>

<script type="application/json" id="payload">PAYLOAD_PLACEHOLDER</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("payload").textContent);
const H=D.H,W=D.W,T=D.T;
const CELL=Math.min(Math.floor((window.innerWidth-60)/W),
                    Math.floor((window.innerHeight-280)/H),32);
const cv=document.getElementById("cv");
cv.width=W*CELL; cv.height=H*CELL;
const ctx=cv.getContext("2d");

const COLORS={"-1":"#1a1a2e","0":"#2d2d44","1":"#4a4a6a","2":"#e8e8e8",
"3":"#ffd700","4":"#555555","5":"#4CAF50","6":"#2E7D32","7":"#1B5E20",
"8":"#f44336","9":"#c62828","10":"#b71c1c"};
const DIR=["down","right","up","left"];
const DARR=["\u2193","\u2192","\u2191","\u2190"];
const SPL=["all","50%","pass"];

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

function drawDot(cx,cy,r,color){
  ctx.fillStyle=color;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.fill();
}

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  // grid
  for(let r=0;r<H;r++) for(let c=0;c<W;c++){
    const idx=r*W+c;
    ctx.fillStyle=COLORS[String(D.states[turn][idx])]||"#000";
    ctx.fillRect(c*CELL,r*CELL,CELL,CELL);
    const a=D.armies[turn][idx];
    if(a>0){
      const s=D.states[turn][idx];
      ctx.fillStyle=(s>=5&&s<=10)?"#fff":"#000";
      const fs=a<100?Math.round(CELL*0.38):a<1000?Math.round(CELL*0.3):Math.round(CELL*0.24);
      ctx.font="bold "+fs+"px monospace";
      ctx.textAlign="center";ctx.textBaseline="middle";
      ctx.fillText(String(a),c*CELL+CELL/2,r*CELL+CELL/2);
    }
  }
  // grid lines
  ctx.strokeStyle="#33333366";ctx.lineWidth=0.5;
  for(let r=0;r<=H;r++){ctx.beginPath();ctx.moveTo(0,r*CELL);ctx.lineTo(W*CELL,r*CELL);ctx.stroke();}
  for(let c=0;c<=W;c++){ctx.beginPath();ctx.moveTo(c*CELL,0);ctx.lineTo(c*CELL,H*CELL);ctx.stroke();}

  const m=D.moves[turn];

  // Human arrow (yellow)
  if(m.hm && m.gs[0]>=0){
    const x1=m.gs[1]*CELL+CELL/2, y1=m.gs[0]*CELL+CELL/2;
    const x2=m.ge[1]*CELL+CELL/2, y2=m.ge[0]*CELL+CELL/2;
    drawArrow(x1,y1,x2,y2,"#ffe000",3);
    drawDot(x1,y1,CELL*0.15,"#ffe000");
  }

  // Model arrow (cyan) — only if model predicts a move (not pass)
  if(!m.pp && m.ps[0]>=0){
    let ox=0,oy=0;
    if(m.hm && m.ps[0]===m.gs[0] && m.ps[1]===m.gs[1]){ox=3;oy=-3;}
    const x1=m.ps[1]*CELL+CELL/2+ox, y1=m.ps[0]*CELL+CELL/2+oy;
    const x2=m.pe[1]*CELL+CELL/2+ox, y2=m.pe[0]*CELL+CELL/2+oy;
    drawArrow(x1,y1,x2,y2,"#00e5ff",2.5);
    drawDot(m.ps[1]*CELL+CELL/2,m.ps[0]*CELL+CELL/2,CELL*0.12,"#00e5ff");
  }

  // HUD
  slider.value=turn;
  document.getElementById("turn-label").textContent="t="+turn+" / "+(T-1);

  const pi=document.getElementById("pred-info");
  if(m.pp){
    pi.innerHTML="<b style='color:#00e5ff'>PASS</b> (no move)";
  } else {
    pi.innerHTML="<b style='color:#00e5ff'>Move:</b> ("+m.ps[0]+","+m.ps[1]+") "+DARR[m.pd]+" "+SPL[m.pspl];
  }

  document.getElementById("val-info").textContent="value="+m.val.toFixed(3);

  document.getElementById("summary").innerHTML=
    D.stats.total+" moves | "+D.stats.passes+" passes | "+T+" steps";
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
    p = argparse.ArgumentParser(description="Simulate model on a replay")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--replay", type=str, required=True)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--show_grid", action="store_true", help="Show ASCII grid at each step")
    p.add_argument("--only_moves", action="store_true", help="Only show steps with actual moves")
    p.add_argument("--html", type=str, default=None, help="Output HTML visualization to this path")
    args = p.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    replay = load_replay(args.replay, args.max_steps)
    T = replay["T"]

    print(f"Replay: {args.replay} ({T} steps)")
    print("=" * 70)

    steps = run_inference(model, replay, device, cfg.grid_w)
    print_results(steps, args.only_moves)

    if args.html:
        generate_html(replay, steps, args.html)



if __name__ == "__main__":
    main()

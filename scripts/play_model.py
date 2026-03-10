#!/usr/bin/env python3
"""
Simulate a trained model playing on a real replay map.

Same as simulate_model.py, but the model's moves are ACTUALLY EXECUTED on the
game state.  The opponent still plays their original replay moves.  If the
model's predicted move is invalid, that turn is skipped.

Usage:
    python3 scripts/play_model.py \
        --checkpoint checkpoints/checkpoint_epoch11.pt \
        --replay data/my_replays/replays/0v1x6p-8g.json \
        --player MeltedToast \
        [--max_steps 200] [--html out.html] [--device cpu]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.processing.simulator import ReplaySimulator
from model.config import ModelConfig
from model.network import GeneralsNetwork

DIR_NAMES = ["down", "right", "up", "left"]
DIR_ARROWS = ["\u2193", "\u2192", "\u2191", "\u2190"]
SPLIT_NAMES = ["all", "50%", "pass"]
DIR_DR = [1, 0, -1, 0]
DIR_DC = [0, 1, 0, -1]


# ── Model loading ──────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
    model = GeneralsNetwork(cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    print(f"Model: d={cfg.d_model}, heads={cfg.n_heads}, params={sum(p.numel() for p in model.parameters()):,}")
    return model, cfg


# ── Observation encoding (matches convert_simulator_to_gio) ────────────

def state_to_grid(state, army):
    """Convert simulator state+army to 12-channel grid tensor [C, H, W]."""
    H, W = 25, 25
    grid = np.zeros((12, H, W), dtype=np.float32)
    for r in range(len(state)):
        for c in range(len(state[0])):
            val = state[r][c]
            if 0 <= val <= 10:
                grid[val, r, c] = 1.0
            a = army[r][c]
            if a > 0:
                grid[11, r, c] = np.log1p(a) / 10.0
    return grid


def compute_globals(sim, player_id):
    """Compute [4] globals vector from simulator state."""
    my = sim.players[player_id]
    op = sim.players[1 - player_id]
    my_army = my.army
    my_land = len(my.land)
    op_army = op.army
    op_land = len(op.land)
    return [
        np.log1p(my_army) / 10.0,
        np.log1p(my_land) / 10.0,
        np.log1p(op_army) / 10.0,
        np.log1p(op_land) / 10.0,
    ]


def compute_legal_mask(state, army):
    """Legal source mask [1, H, W]: owned tiles with army > 1."""
    H, W = 25, 25
    mask = np.zeros((1, H, W), dtype=bool)
    for r in range(len(state)):
        for c in range(len(state[0])):
            if state[r][c] in (5, 6, 7) and army[r][c] > 1:
                mask[0, r, c] = True
    return mask


# ── Move validation ────────────────────────────────────────────────────

def decode_model_move(out, sim, player_id):
    """Decode model output into a validated simulator move, or None."""
    if out["pass_logits"].item() > 0:  # PassHead says pass
        return None
    split = out["action_split"].item()

    src_idx = out["action_source"].item()
    direction = out["action_direction"].item()

    src_r = src_idx // 25
    src_c = src_idx % 25

    # Out of map bounds
    if src_r >= sim.map_height or src_c >= sim.map_width:
        return None
    # Not owned or insufficient army
    tile = sim.map[src_r][src_c]
    if tile.occupied != player_id or tile.army <= 1:
        return None

    dst_r = src_r + DIR_DR[direction]
    dst_c = src_c + DIR_DC[direction]

    # Destination out of bounds
    if not (0 <= dst_r < sim.map_height and 0 <= dst_c < sim.map_width):
        return None
    # Destination is mountain
    if sim.map[dst_r][dst_c].terrain == 3:
        return None

    return {
        "start": src_r * sim.map_width + src_c,
        "end": dst_r * sim.map_width + dst_c,
        "index": player_id,
        "is50": split == 1,
    }


# ── Main simulation loop ──────────────────────────────────────────────

def compute_direction_mask(state, army, src_r, src_c, sim):
    """Return a bool list of length 4: True if direction is legal (down, right, up, left)."""
    mask = [False, False, False, False]
    for d in range(4):
        dst_r = src_r + DIR_DR[d]
        dst_c = src_c + DIR_DC[d]
        if not (0 <= dst_r < sim.map_height and 0 <= dst_c < sim.map_width):
            continue
        if sim.map[dst_r][dst_c].terrain == 3:  # mountain
            continue
        mask[d] = True
    return mask


def play_on_replay(model, cfg, device, replay_path, player_name, max_steps=None, use_mask=False):
    with open(replay_path) as f:
        replay_data = json.load(f)

    sim = ReplaySimulator(replay_data, player_name)
    sim.init_map()

    player_id = sim.player_id
    opponent_id = 1 - player_id
    max_turn = sim.moves[-1]["turn"] + 2 if sim.moves else 2
    if max_steps:
        max_turn = min(max_turn, max_steps)

    move_idx = 0
    hidden = None
    st_cache = None
    frames = []
    pending_model_move = None
    winner = None

    for turn in range(0, max_turn + 2):
        # ── 1. Collect opponent's move for execution ──
        opponent_move = None
        while move_idx < len(sim.moves) and sim.moves[move_idx]["turn"] + 1 == turn:
            move = sim.moves[move_idx]
            if move["index"] == opponent_id:
                opponent_move = move
            move_idx += 1

        # ── 2. Execute pending model move + opponent move ──
        sim.process_moves(pending_model_move, opponent_move)

        # ── 3. Increment armies ──
        sim.increment_armies(turn)

        # ── 4. Update vision, get observation ──
        sim.update_vision()
        state, army = sim.get_state_encoding(player_id)

        # ── 5. Check if max_steps reached ──
        if max_steps and turn >= max_steps:
            break

        # ── 6. Feed to model ──
        grid = state_to_grid(state, army)
        globals_vec = compute_globals(sim, player_id)
        legal_mask = compute_legal_mask(state, army)

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(device)
        globals_t = torch.tensor(globals_vec, dtype=torch.float32).unsqueeze(0).to(device)
        legal_t = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model.step(grid_t, globals_t,
                             legal_mask=legal_t if use_mask else None,
                             hidden=hidden, st_cache=st_cache)

            # Direction masking: re-sample direction with invalid dirs masked out
            if use_mask:
                src_idx_tmp = out["action_source"].item()
                sr, sc = src_idx_tmp // 25, src_idx_tmp % 25
                dir_mask = compute_direction_mask(state, army, sr, sc, sim)
                if any(dir_mask):
                    dir_logits = out["direction_logits"].clone()
                    for d in range(4):
                        if not dir_mask[d]:
                            dir_logits[0, d] = float("-inf")
                    dir_probs = F.softmax(dir_logits, dim=-1)
                    out["action_direction"] = torch.multinomial(dir_probs, 1).squeeze(-1)

        hidden = out.get("hidden")
        st_cache = out.get("st_cache")

        # ── 7. Decode model's move → pending for next turn ──
        pending_model_move = decode_model_move(out, sim, player_id)

        value = out["value"].item()
        src_idx = out["action_source"].item()
        direction = out["action_direction"].item()
        split_val = out["action_split"].item()
        p_sy, p_sx = src_idx // 25, src_idx % 25

        did_pass = out["pass_logits"].item() > 0
        invalid = (not did_pass) and (pending_model_move is None)

        # ── Human reference for display (peek ahead: game_turn == turn) ──
        human_src = [-1, -1]
        human_dst = [-1, -1]
        human_is_50 = -1
        peek = move_idx
        while peek < len(sim.moves) and sim.moves[peek]["turn"] == turn:
            mv = sim.moves[peek]
            if mv["index"] == player_id:
                sr = mv["start"] // sim.map_width
                sc = mv["start"] % sim.map_width
                er = mv["end"] // sim.map_width
                ec = mv["end"] % sim.map_width
                human_src = [sr, sc]
                human_dst = [er, ec]
                human_is_50 = 1 if mv["is50"] else 0
            peek += 1

        # Compute human direction for display
        if human_src[0] >= 0:
            from data_processing.parsers.generals_json import compute_direction
            human_dir = compute_direction(human_src, human_dst)
            human_has_move = True
        else:
            human_dir = 0
            human_has_move = False

        frame = {
            "state": state,
            "army": army,
            # Model's prediction
            "p_sy": p_sy, "p_sx": p_sx, "p_dir": direction, "p_spl": split_val,
            "did_pass": did_pass, "invalid": invalid,
            "executed": pending_model_move is not None,
            "value": value,
            # Human's original move (for reference)
            "gt_sy": human_src[0], "gt_sx": human_src[1],
            "gt_dir": human_dir,
            "gt_ratio": max(human_is_50, 0),
            "has_move": human_has_move,
            # Match check (note: game state may differ from original)
            "src_ok": human_has_move and p_sy == human_src[0] and p_sx == human_src[1],
            "dir_ok": human_has_move and direction == human_dir,
            "spl_ok": human_has_move and split_val == max(human_is_50, 0),
        }
        frames.append(frame)

        # ── Live output ──
        if did_pass:
            tag = "PASS"
        elif invalid:
            tag = "INVALID(skip)"
        elif pending_model_move is not None:
            tag = "MOVE"
        else:
            tag = "?"
        h_str = ""
        if human_has_move:
            match_tag = ""
            if frame["src_ok"] and frame["dir_ok"] and frame["spl_ok"]:
                match_tag = " [MATCH]"
            elif frame["src_ok"]:
                match_tag = " [src ok]"
            h_str = (f"  human=({frame['gt_sy']:2d},{frame['gt_sx']:2d}) "
                     f"{DIR_ARROWS[frame['gt_dir']]} {SPLIT_NAMES[frame['gt_ratio']]:>3s}{match_tag}")
        else:
            h_str = "  human=[no move]"
        print(
            f"t={turn:3d}  "
            f"model=({p_sy:2d},{p_sx:2d}) {DIR_ARROWS[direction]} "
            f"{SPLIT_NAMES[split_val]:>3s}  "
            f"[{tag:>13s}]  "
            f"val={value:+.3f}"
            f"{h_str}",
            flush=True,
        )

        # ── 8. Check winner ──
        winner = sim.check_winner()
        if winner is not None:
            break

    return frames, winner, sim


# ── Text output ────────────────────────────────────────────────────────

def print_results(frames, player_name):
    total_moves = n_executed = n_pass = n_invalid = 0
    human_moves = src_match = dir_match = full_match = 0

    for t, f in enumerate(frames):
        if f["executed"]:
            n_executed += 1
            tag = "MOVE"
        elif f["did_pass"]:
            n_pass += 1
            tag = "PASS"
        elif f["invalid"]:
            n_invalid += 1
            tag = "INVALID(skip)"
        else:
            tag = "?"

        # Human comparison
        h_str = ""
        if f["has_move"]:
            human_moves += 1
            if f["src_ok"]:
                src_match += 1
            if f["dir_ok"]:
                dir_match += 1
            if f["src_ok"] and f["dir_ok"] and f["spl_ok"]:
                full_match += 1
            match_tag = ""
            if f["src_ok"] and f["dir_ok"] and f["spl_ok"]:
                match_tag = " [MATCH]"
            elif f["src_ok"]:
                match_tag = " [src ok]"
            h_str = (f"  human=({f['gt_sy']:2d},{f['gt_sx']:2d}) "
                     f"{DIR_ARROWS[f['gt_dir']]} {SPLIT_NAMES[f['gt_ratio']]:>3s}{match_tag}")
        else:
            h_str = "  human=[no move]"

        print(
            f"t={t:3d}  "
            f"model=({f['p_sy']:2d},{f['p_sx']:2d}) {DIR_ARROWS[f['p_dir']]} "
            f"{SPLIT_NAMES[f['p_spl']]:>3s}  "
            f"[{tag:>13s}]  "
            f"val={f['value']:+.3f}"
            f"{h_str}"
        )

    print("=" * 80)
    print(f"Player: {player_name}")
    print(f"Turns: {len(frames)}  |  Executed: {n_executed}  Pass: {n_pass}  Invalid: {n_invalid}")
    if human_moves > 0:
        print(f"vs Human — src: {src_match}/{human_moves} ({100*src_match/human_moves:.1f}%)  "
              f"dir: {dir_match}/{human_moves} ({100*dir_match/human_moves:.1f}%)  "
              f"full: {full_match}/{human_moves} ({100*full_match/human_moves:.1f}%)")


# ── HTML visualization ────────────────────────────────────────────────

def generate_html(frames, winner, player_name, output_path):
    T = len(frames)
    states_flat = []
    armies_flat = []
    moves_data = []

    for t, f in enumerate(frames):
        st = []
        ar = []
        for r in range(25):
            for c in range(25):
                st.append(f["state"][r][c] if r < len(f["state"]) and c < len(f["state"][0]) else -1)
                a = f["army"][r][c] if r < len(f["army"]) and c < len(f["army"][0]) else -1
                ar.append(a if a > 0 else 0)
        states_flat.append(st)
        armies_flat.append(ar)

        # Model move endpoints
        if f["executed"]:
            dr, dc = DIR_DR[f["p_dir"]], DIR_DC[f["p_dir"]]
            p_ey, p_ex = f["p_sy"] + dr, f["p_sx"] + dc
        else:
            p_ey, p_ex = f["p_sy"], f["p_sx"]

        # Human move endpoints
        if f["has_move"]:
            gdr, gdc = DIR_DR[f["gt_dir"]], DIR_DC[f["gt_dir"]]
            g_ey, g_ex = f["gt_sy"] + gdr, f["gt_sx"] + gdc
        else:
            g_ey, g_ex = -1, -1

        moves_data.append({
            "ps": [f["p_sy"], f["p_sx"]], "pe": [p_ey, p_ex],
            "pd": f["p_dir"], "pspl": f["p_spl"],
            "gs": [f["gt_sy"], f["gt_sx"]], "ge": [g_ey, g_ex],
            "gd": f["gt_dir"], "gspl": f["gt_ratio"],
            "hm": f["has_move"],
            "val": round(f["value"], 3),
            "executed": f["executed"],
            "passed": f["did_pass"],
            "invalid": f["invalid"],
            "sok": f["src_ok"], "dok": f["dir_ok"], "spok": f["spl_ok"],
        })

    human_total = sum(1 for f in frames if f["has_move"])
    src_acc = sum(1 for f in frames if f["src_ok"]) / max(human_total, 1) * 100
    dir_acc = sum(1 for f in frames if f["dir_ok"]) / max(human_total, 1) * 100
    full_acc = sum(1 for f in frames if f["src_ok"] and f["dir_ok"] and f["spl_ok"]) / max(human_total, 1) * 100
    n_exec = sum(1 for f in frames if f["executed"])
    n_pass = sum(1 for f in frames if f["did_pass"])
    n_inv = sum(1 for f in frames if f["invalid"])

    if winner is not None:
        outcome = f"Winner: player {winner}"
    else:
        outcome = "No winner (game ongoing)"

    payload = json.dumps({
        "H": 25, "W": 25, "T": T,
        "states": states_flat, "armies": armies_flat,
        "moves": moves_data,
        "stats": {
            "total": human_total, "src": round(src_acc, 1),
            "dir": round(dir_acc, 1), "full": round(full_acc, 1),
            "executed": n_exec, "passed": n_pass, "invalid": n_inv,
            "outcome": outcome, "player": player_name,
        },
    }, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("PAYLOAD_PLACEHOLDER", payload)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML visualization saved to {output_path}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Model Self-Play Viewer</title>
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
<h2>Model Self-Play (model replaces human player)</h2>
<canvas id="cv"></canvas>
<div id="slider-row">
  <button id="bprev">&#9664;</button>
  <input type="range" id="slider" min="0" value="0">
  <button id="bnext">&#9654;</button>
  <span id="turn-label">t=0</span>
</div>
<div id="hud">
  <div class="box" id="model-info"></div>
  <div class="box" id="human-info"></div>
  <div class="box" id="val-info"></div>
</div>
<div id="summary"></div>
<div class="legend">
  <span><i style="background:#00e5ff"></i> Model's move (executed)</span>
  <span><i style="background:#ffe000"></i> Human's original move</span>
  <span><i style="background:#76ff03"></i> Both match</span>
  <span><i style="background:#ff6600"></i> Model move (invalid/skipped)</span>
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

function drawDot(cx,cy,r,color){
  ctx.fillStyle=color;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.fill();
}

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
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
  ctx.strokeStyle="#33333366";ctx.lineWidth=0.5;
  for(let r=0;r<=H;r++){ctx.beginPath();ctx.moveTo(0,r*CELL);ctx.lineTo(W*CELL,r*CELL);ctx.stroke();}
  for(let c=0;c<=W;c++){ctx.beginPath();ctx.moveTo(c*CELL,0);ctx.lineTo(c*CELL,H*CELL);ctx.stroke();}

  const m=D.moves[turn];
  const bothSrc=m.hm&&m.sok;
  const bothAll=m.hm&&m.sok&&m.dok&&m.spok;

  // Human arrow (yellow)
  if(m.hm && m.gs[0]>=0){
    const col=bothAll?"#76ff03":"#ffe000";
    const x1=m.gs[1]*CELL+CELL/2, y1=m.gs[0]*CELL+CELL/2;
    const x2=m.ge[1]*CELL+CELL/2, y2=m.ge[0]*CELL+CELL/2;
    drawArrow(x1,y1,x2,y2,col,3);
    if(!bothSrc) drawDot(x1,y1,CELL*0.15,"#ffe000");
  }

  // Model arrow
  if(!m.passed && m.ps[0]>=0){
    let ox=0,oy=0;
    if(m.hm && m.ps[0]===m.gs[0] && m.ps[1]===m.gs[1] && !bothAll){
      ox=3;oy=-3;
    }
    const col=bothAll?"#76ff03":m.executed?"#00e5ff":"#ff6600";
    if(!bothAll){
      const x1=m.ps[1]*CELL+CELL/2+ox, y1=m.ps[0]*CELL+CELL/2+oy;
      const x2=m.pe[1]*CELL+CELL/2+ox, y2=m.pe[0]*CELL+CELL/2+oy;
      drawArrow(x1,y1,x2,y2,col,2.5);
      drawDot(m.ps[1]*CELL+CELL/2,m.ps[0]*CELL+CELL/2,CELL*0.12,col);
    }
  }

  slider.value=turn;
  document.getElementById("turn-label").textContent="t="+turn+" / "+(T-1);

  const mi=document.getElementById("model-info");
  if(m.passed){
    mi.innerHTML="<b style='color:#ffd700'>Model: PASS</b>";
  } else if(m.invalid){
    mi.innerHTML="<b style='color:#ff6600'>Model:</b> ("+m.ps[0]+","+m.ps[1]+") "+DARR[m.pd]+" "+SPL[m.pspl]+" <span class='miss'>[INVALID]</span>";
  } else {
    mi.innerHTML="<b style='color:#00e5ff'>Model:</b> ("+m.ps[0]+","+m.ps[1]+") "+DARR[m.pd]+" "+SPL[m.pspl];
  }

  const hi=document.getElementById("human-info");
  if(m.hm){
    let cls=bothAll?"match":m.sok?"partial":"miss";
    let tag=bothAll?"MATCH":m.sok?(m.dok?"src+dir":"src only"):"miss";
    hi.innerHTML="<b style='color:#ffe000'>Human:</b> ("+m.gs[0]+","+m.gs[1]+") "+DARR[m.gd]+" "+SPL[m.gspl]
      +" <span class='"+cls+"'>["+tag+"]</span>";
  } else {
    hi.innerHTML="<b style='color:#ffe000'>Human:</b> <i>no move</i>";
  }

  document.getElementById("val-info").textContent="value="+m.val.toFixed(3);

  const st=D.stats;
  document.getElementById("summary").innerHTML=
    st.outcome+" | "+st.player+" | "
    +"exec:"+st.executed+" pass:"+st.passed+" invalid:"+st.invalid+" | "
    +"vs Human — src:"+st.src+"% dir:"+st.dir+"% full:"+st.full+"% ("+st.total+" human moves)";
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


# ── Main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Model plays on a real replay map")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--replay", type=str, required=True, help="Raw replay JSON (from generals.io)")
    p.add_argument("--player", type=str, required=True, help="Player name to replace with model")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--html", type=str, default=None, help="Output HTML visualization")
    p.add_argument("--mask", action="store_true", default=False,
                   help="Enable legal masking for source and direction during inference")
    args = p.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)

    print(f"Replay: {args.replay}")
    print(f"Replacing player: {args.player}")
    print("=" * 80)

    frames, winner, sim = play_on_replay(
        model, cfg, device, args.replay, args.player, args.max_steps,
        use_mask=args.mask,
    )

    print_results(frames, args.player)

    if winner is not None:
        w_name = sim.players[winner].name
        print(f"\nWinner: {w_name} (player {winner})")
        if winner == sim.player_id:
            print("Model WINS!")
        else:
            print("Model LOSES.")
    else:
        print("\nNo winner (game ended without capture).")

    if args.html:
        generate_html(frames, winner, args.player, args.html)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end pipeline: download a generals.io replay → simulate → interactive HTML viewer.

Usage:
    # From a replay URL (player auto-detected from URL query param)
    python3 scripts/replay_to_visual.py \\
        --url "https://generals.io/replays/asS2S5O_y?p=%7EHemlix%7E"

    # From a replay ID
    python3 scripts/replay_to_visual.py --id asS2S5O_y --player "~Hemlix~"

    # From a local JSON file
    python3 scripts/replay_to_visual.py --file replay.json

    # God-view (no fog, both players visible)
    python3 scripts/replay_to_visual.py --id asS2S5O_y --god

    # Also export PNGs
    python3 scripts/replay_to_visual.py --id asS2S5O_y --png

Output:
    <out>/replay.html   – self-contained interactive viewer (← → keys)
    <out>/frames/*.png  – (only with --png)
"""

import argparse
import asyncio
import json
import os
import ssl
import sys
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from data_processing.download.replay_fetcher import ReplayFetcher  # noqa: E402
from data_processing.processing.simulator import ReplaySimulator  # noqa: E402

# ---------------------------------------------------------------------------
# Colour palette (shared between Python PNG renderer and JS template)
# ---------------------------------------------------------------------------

COLORS = {
    -1: "#1a1a2e", 0: "#2d2d44", 1: "#4a4a6a", 2: "#e8e8e8",
    3: "#ffd700", 4: "#555555", 5: "#4CAF50", 6: "#2E7D32",
    7: "#1B5E20", 8: "#f44336", 9: "#c62828", 10: "#b71c1c",
}

STATE_NAMES = {
    -1: "Padding", 0: "Fog (blank)", 1: "Fog (obstacle)",
    2: "Blank", 3: "Neutral City", 4: "Mountain",
    5: "Your Land", 6: "Your City", 7: "Your General",
    8: "Enemy Land", 9: "Enemy City", 10: "Enemy General",
}

GOD_COLORS = {
    "blank": "#e8e8e8", "mountain": "#555555", "city_n": "#ffd700",
    "p0": "#4CAF50", "p0_city": "#2E7D32", "p0_gen": "#1B5E20",
    "p1": "#f44336", "p1_city": "#c62828", "p1_gen": "#b71c1c",
}

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

async def download_replay(replay_id: str, cache_dir: str) -> Dict[str, Any]:
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"{replay_id}.json")
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)

    fetcher = ReplayFetcher(server="na")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ctx)
    async with aiohttp.ClientSession(connector=connector) as session:
        data = await fetcher.fetch_replay(session, replay_id)
    if data is None:
        raise RuntimeError(f"Failed to download replay {replay_id}")
    with open(cached, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ---------------------------------------------------------------------------
# Simulate & collect frame data for the HTML viewer
# ---------------------------------------------------------------------------

def _collect_fog_frames(
    replay_data: Dict[str, Any], player: str,
) -> Dict[str, Any]:
    """Run fog-of-war simulation and return compact frame data."""
    sim = ReplaySimulator(replay_data, player, strict=False)
    result = sim.simulate()
    H = replay_data["mapHeight"]
    W = replay_data["mapWidth"]
    T = len(result["map"])

    states: List[List[int]] = []
    armies: List[List[int]] = []
    moves: List[List[int]] = []
    stats: List[List[int]] = []

    for t in range(T):
        flat_s = []
        flat_a = []
        for r in range(H):
            for c in range(W):
                flat_s.append(result["map"][t][r][c])
                flat_a.append(max(result["armies"][t][r][c], 0))
        states.append(flat_s)
        armies.append(flat_a)
        ms = result["move_start"][t]
        me = result["move_end"][t]
        moves.append([ms[0], ms[1], me[0], me[1]])
        stats.append([
            result["info_myArmy"][t], result["info_myLand"][t],
            result["info_opArmy"][t], result["info_opLand"][t],
        ])

    others = [n for n in replay_data["usernames"] if n != player]
    opp = others[0] if others else "?"

    return {
        "mode": "fog",
        "H": H, "W": W, "T": T,
        "player": player, "opponent": opp,
        "states": states, "armies": armies,
        "moves": moves, "stats": stats,
    }


def _collect_god_frames(replay_data: Dict[str, Any]) -> Dict[str, Any]:
    """Step-by-step god-mode simulation returning compact frame data."""
    sim = ReplaySimulator(replay_data, replay_data["usernames"][0], strict=False)
    sim.init_map()
    H, W = sim.map_height, sim.map_width
    max_turn = sim.moves[-1]["turn"] if sim.moves else 0
    move_idx = 0

    armies: List[List[int]] = []
    owners: List[List[int]] = []
    terrains: List[List[int]] = []
    pstats: List[List[int]] = []

    for turn in range(0, max_turn + 2):
        turn_moves_arr = [None, None]
        cnt = 0
        while move_idx < len(sim.moves) and sim.moves[move_idx]["turn"] + 1 == turn:
            m = sim.moves[move_idx]
            if cnt < 2:
                turn_moves_arr[cnt] = m
            cnt += 1
            move_idx += 1
        sim.process_moves(turn_moves_arr[0], turn_moves_arr[1])
        sim.increment_armies(turn)
        sim.update_vision()

        flat_a: List[int] = []
        flat_o: List[int] = []
        flat_t: List[int] = []
        for r in range(H):
            for c in range(W):
                tile = sim.map[r][c]
                flat_a.append(tile.army)
                flat_o.append(tile.occupied)
                flat_t.append(tile.terrain)
        armies.append(flat_a)
        owners.append(flat_o)
        terrains.append(flat_t)

        row: List[int] = []
        for p in sim.players:
            row.extend([p.army, len(p.land)])
        pstats.append(row)

        if sim.check_winner() is not None:
            break

    return {
        "mode": "god",
        "H": H, "W": W, "T": len(armies),
        "players": replay_data["usernames"],
        "armies": armies, "owners": owners, "terrains": terrains,
        "pstats": pstats,
    }


# ---------------------------------------------------------------------------
# HTML template (self-contained interactive viewer)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Generals.io Replay – TITLE_PLACEHOLDER</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#eee;font-family:'Menlo','Consolas',monospace;
     display:flex;flex-direction:column;align-items:center;padding:16px;min-height:100vh}
h2{margin-bottom:8px;font-size:15px;color:#aaa}
#wrap{position:relative;display:inline-block}
canvas{display:block;image-rendering:pixelated}
#hud{display:flex;justify-content:space-between;width:100%;margin-top:8px;font-size:13px}
#hud .stats{background:#222;border-radius:6px;padding:6px 12px}
#controls{margin-top:10px;font-size:12px;color:#777;text-align:center;line-height:1.7}
#slider-row{margin-top:8px;display:flex;align-items:center;gap:8px;width:100%}
#slider-row input[type=range]{flex:1}
#turn-label{min-width:120px;text-align:center;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;justify-content:center}
.legend span{display:inline-flex;align-items:center;gap:3px;font-size:11px}
.legend i{display:inline-block;width:14px;height:14px;border-radius:2px;border:1px solid #555}
</style></head><body>
<h2 id="title"></h2>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="slider-row">
  <button id="bprev" title="Previous turn">◀</button>
  <input type="range" id="slider" min="0" value="0">
  <button id="bnext" title="Next turn">▶</button>
  <span id="turn-label">Turn 0</span>
</div>
<div id="hud"><div class="stats" id="stats-left"></div>
<div class="stats" id="stats-right"></div></div>
<div id="controls">
  ← → &nbsp;step &nbsp;|&nbsp; Space: play/pause &nbsp;|&nbsp;
  [ ] &nbsp;speed &nbsp;|&nbsp; Home/End: jump
</div>
<div class="legend" id="legend"></div>

<script type="application/json" id="payload">PAYLOAD_PLACEHOLDER</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("payload").textContent);
const H=D.H,W=D.W,T=D.T;
const CELL=Math.min(Math.floor((window.innerWidth-60)/W),
                    Math.floor((window.innerHeight-220)/H),36);
const cv=document.getElementById("cv");
cv.width=W*CELL; cv.height=H*CELL;
const ctx=cv.getContext("2d");

const COLORS={"-1":"#1a1a2e","0":"#2d2d44","1":"#4a4a6a","2":"#e8e8e8",
"3":"#ffd700","4":"#555555","5":"#4CAF50","6":"#2E7D32","7":"#1B5E20",
"8":"#f44336","9":"#c62828","10":"#b71c1c"};

const GOD_C={"blank":"#e8e8e8","mt":"#555555","cn":"#ffd700",
"p0":"#4CAF50","p0c":"#2E7D32","p0g":"#1B5E20",
"p1":"#f44336","p1c":"#c62828","p1g":"#b71c1c"};

function fogColor(s){return COLORS[String(s)]||"#000"}
function godColor(terrain,owner){
  if(terrain===3) return GOD_C.mt;
  if(owner===-1) return terrain===1?GOD_C.cn:GOD_C.blank;
  const p="p"+owner;
  if(terrain===2) return GOD_C[p+"g"]||GOD_C[p];
  if(terrain===1) return GOD_C[p+"c"]||GOD_C[p];
  return GOD_C[p]||"#888";
}

let turn=0, playing=false, speed=200;
const slider=document.getElementById("slider");
slider.max=T-1;

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  const isFog=(D.mode==="fog");
  for(let r=0;r<H;r++) for(let c=0;c<W;c++){
    const idx=r*W+c;
    let col;
    if(isFog){col=fogColor(D.states[turn][idx]);}
    else{col=godColor(D.terrains[turn][idx],D.owners[turn][idx]);}
    ctx.fillStyle=col;
    ctx.fillRect(c*CELL,r*CELL,CELL,CELL);
    // army text
    const a=D.armies[turn][idx];
    if(a>0){
      let tc="#000";
      if(isFog){const s=D.states[turn][idx]; if(s>=5&&s<=10) tc="#fff";}
      else{if(D.owners[turn][idx]!==-1) tc="#fff";}
      ctx.fillStyle=tc;
      const fs=a<100?Math.round(CELL*0.42):a<1000?Math.round(CELL*0.34):Math.round(CELL*0.26);
      ctx.font="bold "+fs+"px monospace";
      ctx.textAlign="center";ctx.textBaseline="middle";
      ctx.fillText(String(a),c*CELL+CELL/2,r*CELL+CELL/2);
    }
  }
  // grid lines
  ctx.strokeStyle="#33333366";ctx.lineWidth=0.5;
  for(let r=0;r<=H;r++){ctx.beginPath();ctx.moveTo(0,r*CELL);
    ctx.lineTo(W*CELL,r*CELL);ctx.stroke();}
  for(let c=0;c<=W;c++){ctx.beginPath();ctx.moveTo(c*CELL,0);
    ctx.lineTo(c*CELL,H*CELL);ctx.stroke();}

  // move arrow (fog mode)
  if(isFog && D.moves[turn]){
    const m=D.moves[turn];
    if(m[0]>=0){
      const x1=m[1]*CELL+CELL/2, y1=m[0]*CELL+CELL/2;
      const x2=m[3]*CELL+CELL/2, y2=m[2]*CELL+CELL/2;
      ctx.strokeStyle="#ffe000";ctx.lineWidth=3;
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
      // arrowhead
      const ang=Math.atan2(y2-y1,x2-x1);
      const hl=CELL*0.35;
      ctx.beginPath();
      ctx.moveTo(x2,y2);
      ctx.lineTo(x2-hl*Math.cos(ang-0.45),y2-hl*Math.sin(ang-0.45));
      ctx.moveTo(x2,y2);
      ctx.lineTo(x2-hl*Math.cos(ang+0.45),y2-hl*Math.sin(ang+0.45));
      ctx.stroke();
    }
  }

  // HUD
  slider.value=turn;
  document.getElementById("turn-label").textContent="Turn "+turn+" / "+(T-1);
  const sl=document.getElementById("stats-left");
  const sr=document.getElementById("stats-right");
  if(isFog){
    const s=D.stats[turn];
    sl.textContent=D.player+"  Army:"+s[0]+"  Land:"+s[1];
    sr.textContent=D.opponent+"  Army:"+s[2]+"  Land:"+s[3];
  }else{
    const ps=D.pstats[turn];
    const names=D.players;
    sl.textContent=names[0]+"  Army:"+ps[0]+"  Land:"+ps[1];
    sr.textContent=(names[1]||"")+"  Army:"+(ps[2]||0)+"  Land:"+(ps[3]||0);
  }
}

function setTurn(t){turn=Math.max(0,Math.min(T-1,t));draw();}
let timer=null;
function togglePlay(){
  playing=!playing;
  if(playing){timer=setInterval(()=>{if(turn<T-1)setTurn(turn+1);else{playing=false;clearInterval(timer);}},speed);}
  else if(timer){clearInterval(timer);timer=null;}
}
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft"||e.key==="Left") setTurn(turn-1);
  else if(e.key==="ArrowRight"||e.key==="Right") setTurn(turn+1);
  else if(e.key===" "){e.preventDefault();togglePlay();}
  else if(e.key==="["){speed=Math.min(2000,speed+50);if(playing){clearInterval(timer);timer=setInterval(()=>{if(turn<T-1)setTurn(turn+1);else{playing=false;clearInterval(timer);}},speed);}}
  else if(e.key==="]"){speed=Math.max(20,speed-50);if(playing){clearInterval(timer);timer=setInterval(()=>{if(turn<T-1)setTurn(turn+1);else{playing=false;clearInterval(timer);}},speed);}}
  else if(e.key==="Home") setTurn(0);
  else if(e.key==="End") setTurn(T-1);
});
slider.addEventListener("input",()=>setTurn(parseInt(slider.value)));
document.getElementById("bprev").addEventListener("click",()=>setTurn(turn-1));
document.getElementById("bnext").addEventListener("click",()=>setTurn(turn+1));

// legend
const legendEl=document.getElementById("legend");
const LEGEND_ITEMS=D.mode==="fog"?[
  ["#1B5E20","Your General"],["#2E7D32","Your City"],["#4CAF50","Your Land"],
  ["#b71c1c","Enemy General"],["#c62828","Enemy City"],["#f44336","Enemy Land"],
  ["#ffd700","Neutral City"],["#555555","Mountain"],["#e8e8e8","Blank"],
  ["#4a4a6a","Fog (obstacle)"],["#2d2d44","Fog (blank)"]
]:[
  ["#1B5E20","P0 General"],["#2E7D32","P0 City"],["#4CAF50","P0 Land"],
  ["#b71c1c","P1 General"],["#c62828","P1 City"],["#f44336","P1 Land"],
  ["#ffd700","Neutral City"],["#555555","Mountain"],["#e8e8e8","Blank"]
];
LEGEND_ITEMS.forEach(([col,lab])=>{
  const s=document.createElement("span");
  s.innerHTML='<i style="background:'+col+'"></i>'+lab;
  legendEl.appendChild(s);
});

document.getElementById("title").textContent=TITLE_JS;
draw();
</script></body></html>"""


def _build_html(
    frame_data: Dict[str, Any],
    title: str,
) -> str:
    """Inject frame data into the HTML template."""
    payload = json.dumps(frame_data, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("PAYLOAD_PLACEHOLDER", payload)
    html = html.replace("TITLE_PLACEHOLDER", html_escape(title))
    html = html.replace("TITLE_JS", json.dumps(title))
    return html


# ---------------------------------------------------------------------------
# Optional PNG export (headless matplotlib)
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    return int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255


def export_pngs(
    frame_data: Dict[str, Any],
    out_dir: str,
    step: int = 1,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    H, W, T = frame_data["H"], frame_data["W"], frame_data["T"]
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    rendered = 0
    for t in range(0, T, step):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        grid = np.zeros((H, W, 3))
        for r in range(H):
            for c in range(W):
                idx = r * W + c
                s = frame_data["states"][t][idx] if "states" in frame_data else 0
                grid[r, c] = _hex_to_rgb(COLORS.get(s, "#000"))
        ax.imshow(grid, aspect="equal")
        for r in range(H):
            for c in range(W):
                a = frame_data["armies"][t][r * W + c]
                if a > 0:
                    s = frame_data.get("states", [[]])[t][r * W + c] if "states" in frame_data else -1
                    tc = "white" if 5 <= s <= 10 else "black"
                    ax.text(c, r, str(a), ha="center", va="center",
                            fontsize=6, color=tc, fontweight="bold")
        ax.set_title(f"Turn {t}/{T - 1}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        path = os.path.join(frames_dir, f"turn_{t:04d}.png")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        rendered += 1
    print(f"  Saved {rendered} PNGs to {frames_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_pipeline(args):
    # ── resolve replay data ──
    replay_data = None
    replay_id = None

    if args.file:
        with open(args.file) as f:
            replay_data = json.load(f)
        replay_id = replay_data.get("id", Path(args.file).stem)
    else:
        if args.url:
            parsed = urlparse(args.url)
            parts = parsed.path.strip("/").split("/")
            replay_id = parts[-1] if parts else None
            qs = parse_qs(parsed.query)
            if "p" in qs and not args.player:
                args.player = unquote(qs["p"][0])
        elif args.replay_id:
            replay_id = args.replay_id

        if not replay_id:
            print("Error: provide --url, --id, or --file"); sys.exit(1)

        cache = os.path.join(args.out, ".cache")
        print(f"Downloading replay {replay_id} …")
        replay_data = asyncio.run(download_replay(replay_id, cache))

    print(f"Replay  : {replay_data.get('id', '?')}")
    print(f"Map     : {replay_data['mapWidth']}×{replay_data['mapHeight']}")
    print(f"Players : {replay_data['usernames']}")
    print(f"Moves   : {len(replay_data['moves'])}")

    player = args.player or replay_data["usernames"][0]
    if player not in replay_data["usernames"]:
        print(f"Warning: '{player}' not found, using {replay_data['usernames'][0]}")
        player = replay_data["usernames"][0]

    # ── simulate ──
    os.makedirs(args.out, exist_ok=True)
    god = args.god

    if god:
        print("Simulating (god-view) …")
        frame_data = _collect_god_frames(replay_data)
    else:
        print(f"Simulating ({player}'s fog-of-war) …")
        frame_data = _collect_fog_frames(replay_data, player)

    print(f"  {frame_data['T']} turns collected.")

    # ── build HTML ──
    title = (
        f"{replay_id}  |  {' vs '.join(replay_data['usernames'])}  |  "
        f"{replay_data['mapWidth']}×{replay_data['mapHeight']}"
    )
    if not god:
        title += f"  |  POV: {player}"

    html_path = os.path.join(args.out, "replay.html")
    print(f"Generating interactive HTML → {html_path}")
    html = _build_html(frame_data, title)
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Done ({len(html) / 1024 / 1024:.1f} MB). Open in your browser!")

    # ── optional PNG export ──
    if args.png:
        print("Exporting PNGs …")
        export_pngs(frame_data, args.out, step=args.png_step)


def main():
    p = argparse.ArgumentParser(
        description="Download → Simulate → Interactive HTML viewer for generals.io",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--url", help="Full replay URL")
    src.add_argument("--id", dest="replay_id", help="Replay ID")
    src.add_argument("--file", help="Local replay JSON path")

    p.add_argument("--player", help="Player name (default: first)")
    p.add_argument("--out", default="viz_output", help="Output directory")
    p.add_argument("--god", action="store_true", help="God-view (no fog)")
    p.add_argument("--png", action="store_true", help="Also export PNGs")
    p.add_argument("--png-step", type=int, default=1, help="PNG export step")
    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

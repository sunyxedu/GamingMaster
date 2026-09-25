"""DIAL step-0 zero-training headroom probe (othello): does explicit lookahead
with the network's OWN value head beat the raw single-forward policy of a
trained model?  Go/no-go gate for distilling depth-indexed lookahead into the
recurrent iterations: if 1-ply lookahead adds < ~30 Elo, don't build it.

    uv run python -m scripts.dial_probe --ckpt checkpoints/multi/multi_final.pt \\
        --val_dir data/othello/val --json dial_probe.json

Players (deterministic):
  raw@r   greedy legal-masked argmax of the place logits at depth r
  la1@r   1-ply lookahead: every legal move's successor is scored by the
          network value of the next frame (exact W/D/L if terminal), taken
          for the mover; argmax, ties broken by the policy logit
  la2@r   depth-2 minimax with network values at the leaves
Frames stream through ``GameNetwork.step`` exactly as a training sample
(every ply from the side-to-move perspective, forced pass frames included);
sign conventions: training/othello_probe.py module docstring.

Reports
  matches   paired games from the first --opening_plies plies of distinct
            validation games, each opening played twice with seats swapped:
            W/D/L, score, Elo of A over B with 95% CIs (normal on the paired
            score; bootstrap over opening pairs)
  endgame   positions with <= --empties empties vs an exact alpha-beta
            solver, for every r in --depths: sign accuracy of V (decisive
            positions), MSE of V against the exact W/D/L outcome in {-1,0,1}
            (the training value target's scale), how often RAW / LA1 pick an
            optimal move, their mean exact disc regret and W/D/L-losing
            blunder rate
  human     top-1 agreement of RAW and LA1 with the human move, value MSE vs
            the game result, for every r (every --human_stride-th frame of
            each game, so the positions spread over many games)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from data_processing.games.dataset import GameShardDataset
from training.eval_multi_game import load_multi
from training.othello_probe import (
    PASS,
    NetEvaluator,
    analyse_positions,
    game_score,
    iter_val_games,
    match_stats,
    n_empties,
    parse_player,
    play_games,
    solve_positions,
)
from training.train_il_game import pick_device


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="train_multi_game checkpoint with othello")
    p.add_argument("--val_dir", required=True, help="othello validation shards")
    p.add_argument("--pairs", type=int, default=200, help="opening pairs per match")
    p.add_argument("--opening_plies", type=int, default=6)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="recurrent depths r for the endgame / human analyses")
    p.add_argument("--empties", type=int, default=10, help="endgame positions: max empties")
    p.add_argument("--max_positions", type=int, default=1000,
                   help="cap on endgame positions and (separately) on human positions")
    p.add_argument("--human_stride", type=int, default=4,
                   help="human agreement: every k-th frame per game (spreads the positions)")
    p.add_argument("--matches", nargs="*", default=["la1:raw", "la2:raw"],
                   metavar="A:B", help="player A vs player B; raw | la<d>, optional @r")
    p.add_argument("--match_depth", type=int, default=None,
                   help="r of players given without @r (default: the model's default depth)")
    p.add_argument("--gate_elo", type=float, default=30.0,
                   help="go/no-go threshold for la1 over raw")
    p.add_argument("--parallel", type=int, default=64,
                   help="games streamed in lockstep (lower it for attention-mixer "
                        "checkpoints: their history cache grows with the game)")
    p.add_argument("--max_batch", type=int, default=1024, help="rows per network call")
    p.add_argument("--solver_workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--skip_matches", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    p.add_argument("--raw", action="store_true", help="raw weights instead of EMA")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--seed", type=int, default=0, help="validation-game order + bootstrap")
    p.add_argument("--device", default=None)
    p.add_argument("--json", default=None, help="write the full results here")
    return p.parse_args(argv)


def _fmt_ci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:+.0f}, {ci[1]:+.0f}]"


def collect_openings(ds, order, plies: int, want: int, skipped: List[str]) -> List[tuple]:
    """First ``plies`` actions of distinct validation games (in ``order``)."""
    seen, out = set(), []
    for g in iter_val_games(ds, order, skipped):
        if len(g.frames) <= plies:
            continue
        op = tuple(f.action for f in g.frames[:plies])
        if op in seen:
            continue
        seen.add(op)
        out.append(op)
        if len(out) >= want:
            break
    return out


def select_positions(ds, order, empties: int, quota: int, skipped: List[str],
                     human_stride: int = 4):
    """Validation games (in ``order``) until both quotas are met: every
    ``human_stride``-th frame of each game if it is a decision frame (human
    agreement; the phase offset varies per game, so all game phases are
    covered while positions spread over many games), and every decision
    frame with <= ``empties`` empties (endgame)."""
    human_left, end_left = quota, quota
    stride = max(1, human_stride)
    sel = []
    for g in iter_val_games(ds, order, skipped):
        hum, end = set(), set()
        for t, f in enumerate(g.frames):
            if f.action == PASS:
                continue
            if human_left > 0 and (t + g.index) % stride == 0:
                hum.add(t)
                human_left -= 1
            if end_left > 0 and n_empties(f.black, f.white) <= empties:
                end.add(t)
                end_left -= 1
        if hum or end:
            sel.append((g, hum, end))
        if human_left <= 0 and end_left <= 0:
            break
    return sel


def main(argv: Optional[List[str]] = None) -> Dict:
    a = parse_args(argv)
    device = torch.device(a.device) if a.device else pick_device()
    torch.manual_seed(a.seed)
    model = load_multi(a.ckpt, device, raw=a.raw)
    if "othello" not in model.games:
        raise SystemExit(f"{a.ckpt}: no othello network (games: {model.games})")
    net = model["othello"]
    if type(net.action_decoder).__name__ != "PlacementDecoder" or net.cfg.piece_channels != 3:
        raise SystemExit("othello network is not BoardEncoder(3 planes) + PlacementDecoder")
    r0 = a.match_depth or net.default_depth
    ev = NetEvaluator(net, device, max_batch=a.max_batch, bf16=a.bf16)
    ds = GameShardDataset(a.val_dir, game="othello")
    order = np.random.RandomState(a.seed).permutation(len(ds)).tolist()
    skipped: List[str] = []
    results: Dict = {"ckpt": a.ckpt, "args": vars(a), "default_depth": net.default_depth,
                     "core_type": net.core_type, "temporal_mixer": net.cfg.temporal_mixer,
                     "val_games": len(ds)}
    print(f"[init] core={net.core_type} mixer={net.cfg.temporal_mixer} "
          f"default r={net.default_depth} device={device.type} val games={len(ds)}", flush=True)

    if not a.skip_matches and a.matches:
        openings = collect_openings(ds, order, a.opening_plies, a.pairs, skipped)
        if len(openings) < a.pairs:
            print(f"[warn] only {len(openings)} distinct {a.opening_plies}-ply openings "
                  f"(asked for {a.pairs} pairs)", flush=True)
        results["openings"] = len(openings)
        results["matches"] = []
        for spec in a.matches:
            sa, _, sb = spec.partition(":")
            pa, pb = parse_player(sa, r0), parse_player(sb or "raw", r0)
            t0 = time.time()
            games = []
            for op in openings:
                games += [(op, pa, pb), (op, pb, pa)]      # A black, then A white
            res = play_games(ev, games, parallel=a.parallel)
            pairs = [(game_score(res[2 * i], True), game_score(res[2 * i + 1], False))
                     for i in range(len(openings))]
            st = match_stats(pairs, seed=a.seed)
            st.update(a=pa.name, b=pb.name, seconds=round(time.time() - t0, 1),
                      per_game=[{"opening": i // 2, "a_black": i % 2 == 0, **r}
                                for i, r in enumerate(res)])
            results["matches"].append(st)
            print(f"[match] {pa.name} vs {pb.name}: {st['games']} games  "
                  f"W {st['W']} D {st['D']} L {st['L']}  score {st['score']:.3f}  "
                  f"Elo {st['elo']:+.1f}  95% CI normal {_fmt_ci(st['elo_ci95_normal'])} "
                  f"boot {_fmt_ci(st['elo_ci95_bootstrap'])}  ({st['seconds']:.0f}s)",
                  flush=True)
        gate = next((m for m in results["matches"]
                     if m["a"].startswith("la1@") and m["b"].startswith("raw@")
                     and m["a"].split("@")[1] == m["b"].split("@")[1]), None)
        if gate is not None:
            ok = gate["elo"] >= a.gate_elo
            results["gate"] = {"match": f"{gate['a']} vs {gate['b']}", "elo": gate["elo"],
                               "threshold": a.gate_elo, "go": ok,
                               "ci95_normal": gate["elo_ci95_normal"]}
            print(f"[gate] {gate['a']} over {gate['b']}: {gate['elo']:+.1f} Elo "
                  f"(threshold {a.gate_elo:+.0f}) -> {'GO' if ok else 'NO-GO'}", flush=True)

    if not a.skip_analysis:
        t0 = time.time()
        sel = select_positions(ds, order, a.empties, a.max_positions, skipped,
                               a.human_stride)
        keys, pos = [], []
        for gi, (g, _, end) in enumerate(sel):
            for t in sorted(end):
                keys.append((gi, t))
                pos.append(g.frames[t].own_opp())
        exact = dict(zip(keys, solve_positions(pos, a.solver_workers)))
        t_solve = time.time() - t0
        ana = analyse_positions(ev, sel, a.depths, exact, parallel=a.parallel)
        results["endgame"] = {"empties": a.empties, "n_positions": len(pos),
                              "solver_seconds": round(t_solve, 1),
                              "per_depth": {str(r): v for r, v in ana["endgame"].items()}}
        results["human"] = {"n_positions": sum(len(h) for _, h, _ in sel),
                            "per_depth": {str(r): v for r, v in ana["human"].items()}}
        results["analysis_seconds"] = round(time.time() - t0, 1)
        print(f"[endgame] <= {a.empties} empties, {len(pos)} positions "
              f"(solver {t_solve:.0f}s); V vs exact W/D/L, move = optimal-move rate", flush=True)
        print("     r  sign_acc  mse_wdl  raw_best  la1_best  raw_regret  la1_regret  "
              "raw_blunder  la1_blunder")
        for r, m in ana["endgame"].items():
            print(f"  {r:4d}  {m['value_sign_acc']:8.3f}  {m['value_mse_wdl']:7.3f}  "
                  f"{m['raw_best']:8.3f}  {m['la1_best']:8.3f}  {m['raw_regret']:10.2f}  "
                  f"{m['la1_regret']:10.2f}  {m['raw_wdl_blunder']:11.3f}  "
                  f"{m['la1_wdl_blunder']:11.3f}")
        print(f"[human] {results['human']['n_positions']} positions: top-1 agreement")
        print("     r  raw_top1  la1_top1  value_mse_vs_result")
        for r, m in ana["human"].items():
            print(f"  {r:4d}  {m['raw_top1']:8.3f}  {m['la1_top1']:8.3f}  "
                  f"{m['value_mse_vs_result']:19.3f}")
    results["skipped_val_games"] = skipped
    results["net_calls"], results["net_rows"] = ev.calls, ev.rows
    if skipped:
        print(f"[warn] {len(skipped)} validation games skipped (rules mismatch), e.g. {skipped[0]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"[done] wrote {a.json}")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

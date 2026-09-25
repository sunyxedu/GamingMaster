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
          for the mover; argmax, ties broken by the policy logit.  An exact
          win outranks every estimate and an exact loss ranks below every
          estimate (the value head is unbounded and overshoots +-1)
  la2@r   depth-2 minimax with network values at the leaves
Frames stream through ``GameNetwork.step`` exactly as a training sample
(every ply from the side-to-move perspective, forced pass frames included);
sign conventions: training/othello_probe.py module docstring.

Reports
  matches   paired games from distinct validation opening POSITIONS after
            --opening_plies plies (raised in steps of 2, up to
            --max_opening_plies, until --pairs distinct ones exist), each
            played twice with seats swapped: W/D/L, score, Elo of A over B
            with 95% CIs (normal on the paired score; bootstrap over opening
            pairs), and how often a search move differed from RAW's in a tree
            that reached an exact terminal (the exact-terminal confound;
            --solve_empties N removes it by making both players play exactly
            optimal moves at <= N empties)
  gate      three-way verdict on la1@r over raw@r (same r): GO if the Elo
            point estimate >= --gate_elo and the 95% CI excludes 0, NO-GO if
            the CI's upper end is below --gate_elo, INCONCLUSIVE otherwise
            (or with fewer than --gate_min_pairs pairs): then play more pairs
  endgame   up to --endgame_per_game positions per game with <= --empties
            empties and >= 2 legal moves vs an exact alpha-beta solver, for
            every r in --depths: sign accuracy of V (decisive positions), MSE
            of V against the exact W/D/L outcome in {-1,0,1} (the training
            value target's scale), how often RAW / LA1 pick an optimal move,
            their mean exact disc regret and W/D/L-losing blunder rate — on
            all positions, on the contested ones (not every move equally
            good) and the critical ones (some move changes the W/D/L), with
            game-clustered bootstrap CIs on LA1 - RAW
  human     top-1 agreement of RAW and LA1 with the human move, value MSE vs
            the game result, for every r (candidates: every --human_stride-th
            decision frame with >= 2 legal moves, at most --human_per_game per
            game, so the positions spread over many games)

--bf16 (CUDA only) runs the trunk in bf16 with the value head and the decoder
kept in fp32 (a bf16 value would create artificial lookahead ties).
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
    gate_verdict,
    iter_val_games,
    legal_moves,
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
    p.add_argument("--pairs", type=int, default=500,
                   help="opening pairs per match (500 pairs: 95%% CI about +-22 Elo)")
    p.add_argument("--opening_plies", type=int, default=6,
                   help="book plies; raised in steps of 2 until --pairs distinct positions exist")
    p.add_argument("--max_opening_plies", type=int, default=16)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="recurrent depths r for the endgame / human analyses")
    p.add_argument("--empties", type=int, default=10, help="endgame positions: max empties")
    p.add_argument("--max_positions", type=int, default=1000,
                   help="cap on endgame positions and (separately) on human positions")
    p.add_argument("--endgame_per_game", type=int, default=2,
                   help="endgame positions sampled per game (spreads them over games)")
    p.add_argument("--human_stride", type=int, default=4,
                   help="human agreement: candidate frames every k-th frame per game")
    p.add_argument("--human_per_game", type=int, default=4,
                   help="human agreement: positions sampled per game from the candidates")
    p.add_argument("--matches", nargs="*", default=["la1:raw", "la2:raw"],
                   metavar="A:B", help="player A vs player B; raw | la<d>, optional @r")
    p.add_argument("--match_depth", type=int, default=None,
                   help="r of players given without @r (default: the model's default depth)")
    p.add_argument("--gate_elo", type=float, default=30.0,
                   help="go/no-go threshold for la1 over raw")
    p.add_argument("--gate_min_pairs", type=int, default=100,
                   help="fewer opening pairs than this -> INCONCLUSIVE")
    p.add_argument("--solve_empties", type=int, default=0,
                   help="ablation: both players play exactly optimal moves at <= N empties "
                        "(removes lookahead's exact-terminal advantage over raw)")
    p.add_argument("--parallel", type=int, default=64,
                   help="games streamed in lockstep (lower it for attention-mixer "
                        "checkpoints: their history cache grows with the game)")
    p.add_argument("--max_batch", type=int, default=1024, help="rows per network call")
    p.add_argument("--solver_workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--skip_matches", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    p.add_argument("--raw", action="store_true", help="raw weights instead of EMA")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast for the trunk, value head / decoder in fp32 (CUDA only)")
    p.add_argument("--seed", type=int, default=0, help="validation-game order + bootstrap")
    p.add_argument("--device", default=None)
    p.add_argument("--json", default=None, help="write the full results here")
    return p.parse_args(argv)


def collect_openings(ds, order, plies: int, want: int, skipped: Dict[int, str],
                     max_plies: Optional[int] = None):
    """Opening action sequences leading to distinct positions (board and side
    to move) after ``plies`` half-moves, taken from validation games in
    ``order`` — transposed move orders reach the same board and would replay
    near-identical games.  If fewer than ``want`` exist, the smallest
    ``plies + 2k <= max_plies`` that yields ``want`` is used (else the one
    with the most openings).  Returns (openings, plies used)."""
    levels = list(range(plies, max(plies, max_plies or plies) + 1, 2))
    seen: Dict[int, set] = {p: set() for p in levels}
    found: Dict[int, List[tuple]] = {p: [] for p in levels}
    for g in iter_val_games(ds, order, skipped):
        for p in levels:
            if len(g.frames) <= p or len(found[p]) >= want:
                continue
            f = g.frames[p]                            # the position after p plies
            key = (f.black, f.white, f.mover)
            if key not in seen[p]:
                seen[p].add(key)
                found[p].append(tuple(x.action for x in g.frames[:p]))
        if len(found[plies]) >= want:
            break
    for p in levels:
        if len(found[p]) >= want:
            return found[p], p
    p = max(levels, key=lambda q: len(found[q]))       # ties -> fewest plies
    return found[p], p


def select_positions(ds, order, empties: int, quota: int, skipped: Dict[int, str],
                     human_stride: int = 4, human_per_game: int = 4,
                     endgame_per_game: int = 2, seed: int = 0):
    """Validation games (in ``order``) until both quotas are met.  Only
    decision frames with >= 2 legal moves count (with one legal move every
    player agrees and is optimal).  Human agreement: candidates every
    ``human_stride``-th frame (phase offset varies per game), at most
    ``human_per_game`` sampled per game; endgame: frames with <= ``empties``
    empties, at most ``endgame_per_game`` sampled per game — so the positions
    spread over many games instead of clustering in a few."""
    rng = np.random.RandomState(seed)

    def pick(cands, cap, left):
        k = min(len(cands), max(0, cap), max(0, left))
        return set(rng.choice(cands, size=k, replace=False).tolist()) if k else set()

    human_left, end_left = quota, quota
    stride = max(1, human_stride)
    sel = []
    for g in iter_val_games(ds, order, skipped):
        hum_c, end_c = [], []
        for t, f in enumerate(g.frames):
            if f.action == PASS or legal_moves(*f.own_opp()).bit_count() < 2:
                continue
            if (t + g.index) % stride == 0:
                hum_c.append(t)
            if n_empties(f.black, f.white) <= empties:
                end_c.append(t)
        hum = pick(hum_c, human_per_game, human_left)
        end = pick(end_c, endgame_per_game, end_left)
        human_left -= len(hum)
        end_left -= len(end)
        if hum or end:
            sel.append((g, hum, end))
        if human_left <= 0 and end_left <= 0:
            break
    return sel


def _fmt_ci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:+.0f}, {ci[1]:+.0f}]"


def _fmt_diff(d: Dict, scale: float = 1.0, digits: int = 3) -> str:
    if not d.get("n"):
        return "n/a"
    ci = d["ci95"]
    tail = "" if ci is None else f" [{ci[0] * scale:+.{digits}f}, {ci[1] * scale:+.{digits}f}]"
    return f"{d['mean'] * scale:+.{digits}f}{tail}"


def _search_totals(res: List[Dict], a_seats: List[str]) -> Dict[str, Dict[str, int]]:
    """Sum the per-game search counters for player A and player B."""
    tot = {"a": {}, "b": {}}
    for r, a_col in zip(res, a_seats):
        for who, col in (("a", a_col), ("b", "white" if a_col == "black" else "black")):
            for k, v in r["stats"][col].items():
                tot[who][k] = tot[who].get(k, 0) + v
    return tot


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
    bf16 = a.bf16 and device.type == "cuda"
    if a.bf16 and not bf16:
        print(f"[warn] --bf16 ignored on {device.type} (CUDA only)", flush=True)
    ev = NetEvaluator(net, device, max_batch=a.max_batch, bf16=bf16)
    ds = GameShardDataset(a.val_dir, game="othello")
    order = np.random.RandomState(a.seed).permutation(len(ds)).tolist()
    skipped: Dict[int, str] = {}
    results: Dict = {"ckpt": a.ckpt, "args": vars(a), "default_depth": net.default_depth,
                     "core_type": net.core_type, "temporal_mixer": net.cfg.temporal_mixer,
                     "val_games": len(ds), "bf16": bf16}
    print(f"[init] core={net.core_type} mixer={net.cfg.temporal_mixer} "
          f"default r={net.default_depth} device={device.type} val games={len(ds)}", flush=True)

    if not a.skip_matches and a.matches:
        openings, plies = collect_openings(ds, order, a.opening_plies, a.pairs, skipped,
                                           a.max_opening_plies)
        if plies != a.opening_plies:
            print(f"[info] fewer than {a.pairs} distinct {a.opening_plies}-ply opening "
                  f"positions: using {plies}-ply openings", flush=True)
        if len(openings) < a.pairs:
            print(f"[warn] only {len(openings)} distinct {plies}-ply opening positions "
                  f"(asked for {a.pairs} pairs; raise --max_opening_plies)", flush=True)
        results["openings"] = len(openings)
        results["opening_plies"] = plies
        results["matches"] = []
        for spec in a.matches:
            sa, _, sb = spec.partition(":")
            pa, pb = parse_player(sa, r0), parse_player(sb or "raw", r0)
            t0 = time.time()
            games = []
            for op in openings:
                games += [(op, pa, pb), (op, pb, pa)]      # A black, then A white
            res = play_games(ev, games, parallel=a.parallel, solve_empties=a.solve_empties)
            pairs = [(game_score(res[2 * i], True), game_score(res[2 * i + 1], False))
                     for i in range(len(openings))]
            st = match_stats(pairs, seed=a.seed)
            search_tot = _search_totals(res, ["black" if i % 2 == 0 else "white"
                                              for i in range(len(res))])
            st.update(a=pa.name, b=pb.name, seconds=round(time.time() - t0, 1),
                      opening_plies=plies, solve_empties=a.solve_empties, search=search_tot,
                      per_game=[{"opening": i // 2, "a_black": i % 2 == 0, **r}
                                for i, r in enumerate(res)])
            results["matches"].append(st)
            print(f"[match] {pa.name} vs {pb.name}: {st['games']} games  "
                  f"W {st['W']} D {st['D']} L {st['L']}  score {st['score']:.3f}  "
                  f"Elo {st['elo']:+.1f}  95% CI normal {_fmt_ci(st['elo_ci95_normal'])} "
                  f"boot {_fmt_ci(st['elo_ci95_bootstrap'])}  ({st['seconds']:.0f}s)",
                  flush=True)
            for who, pl in (("a", pa), ("b", pb)):
                t = search_tot[who]
                if t.get("searched") or t.get("solved"):
                    print(f"        {pl.name}: {t['searched']} searched decisions, "
                          f"{t['terminal']} reached an exact terminal, "
                          f"{t['terminal_changed']} of those differ from raw"
                          + (f"; {t['solved']} solver moves" if a.solve_empties else ""),
                          flush=True)
        gate = next((m for m in results["matches"]
                     if m["a"].startswith("la1@") and m["b"].startswith("raw@")
                     and m["a"].split("@")[1] == m["b"].split("@")[1]), None)
        if gate is not None:
            gv = gate_verdict(gate, a.gate_elo, a.gate_min_pairs)
            gv.update(match=f"{gate['a']} vs {gate['b']}", opening_plies=plies,
                      pairs_requested=a.pairs, solve_empties=a.solve_empties,
                      ci95_normal=gate["elo_ci95_normal"],
                      ci95_bootstrap=gate["elo_ci95_bootstrap"])
            results["gate"] = gv
            short = (f"; only {gv['pairs']} of {a.pairs} requested pairs"
                     if gv["pairs"] < a.pairs else "")
            print(f"[gate] {gate['a']} over {gate['b']}: {gate['elo']:+.1f} Elo, 95% CI "
                  f"{_fmt_ci(gv['ci95'])} over {gv['pairs']} pairs (threshold "
                  f"{a.gate_elo:+.0f}{short}) -> {gv['verdict']}", flush=True)

    if not a.skip_analysis:
        t0 = time.time()
        sel = select_positions(ds, order, a.empties, a.max_positions, skipped,
                               a.human_stride, a.human_per_game, a.endgame_per_game, a.seed)
        keys, pos = [], []
        for gi, (g, _, end) in enumerate(sel):
            for t in sorted(end):
                keys.append((gi, t))
                pos.append(g.frames[t].own_opp())
        exact = dict(zip(keys, solve_positions(pos, a.solver_workers)))
        t_solve = time.time() - t0
        ana = analyse_positions(ev, sel, a.depths, exact, parallel=a.parallel, seed=a.seed)
        results["endgame"] = {"empties": a.empties, "n_positions": len(pos),
                              "n_games": sum(1 for _, _, e in sel if e),
                              "solver_seconds": round(t_solve, 1),
                              "per_depth": {str(r): v for r, v in ana["endgame"].items()}}
        results["human"] = {"n_positions": sum(len(h) for _, h, _ in sel),
                            "n_games": sum(1 for _, h, _ in sel if h),
                            "per_depth": {str(r): v for r, v in ana["human"].items()}}
        results["analysis_seconds"] = round(time.time() - t0, 1)
        print(f"[endgame] <= {a.empties} empties, {len(pos)} positions from "
              f"{results['endgame']['n_games']} games (solver {t_solve:.0f}s); V vs exact "
              f"W/D/L; best = optimal-move rate, regret in discs, blunder = W/D/L lost",
              flush=True)
        print("  all positions:\n     r  sign_acc  mse_wdl  raw_best  la1_best  raw_regret  "
              "la1_regret  raw_blunder  la1_blunder")
        for r, m in ana["endgame"].items():
            print(f"  {r:4d}  {m['value_sign_acc']:8.3f}  {m['value_mse_wdl']:7.3f}  "
                  f"{m['raw_best']:8.3f}  {m['la1_best']:8.3f}  {m['raw_regret']:10.2f}  "
                  f"{m['la1_regret']:10.2f}  {m['raw_wdl_blunder']:11.3f}  "
                  f"{m['la1_wdl_blunder']:11.3f}")
        print("  contested (best / regret) and critical (blunder) positions, la1 - raw "
              "with game-clustered 95% CI:")
        for r, m in ana["endgame"].items():
            d = m["la1_minus_raw"]
            by_e = "  ".join(f"{k} empties n={v['n']} raw {v['raw']:.3f} la1 {v['la1']:.3f}"
                             for k, v in m["blunder_critical_by_empties"].items())
            print(f"  r={r}\n"
                  f"    contested n={m['n_contested']}: best raw {m['raw_best_contested']:.3f} "
                  f"la1 {m['la1_best_contested']:.3f} diff {_fmt_diff(d['best_contested'])}; "
                  f"regret raw {m['raw_regret_contested']:.2f} "
                  f"la1 {m['la1_regret_contested']:.2f} "
                  f"diff {_fmt_diff(d['regret_contested'], digits=2)}\n"
                  f"    critical n={m['n_critical']}: blunder raw "
                  f"{m['raw_blunder_critical']:.3f} la1 {m['la1_blunder_critical']:.3f} "
                  f"diff {_fmt_diff(d['blunder_critical'])}; {by_e}")
        print(f"[human] {results['human']['n_positions']} positions from "
              f"{results['human']['n_games']} games: top-1 agreement")
        print("     r  raw_top1  la1_top1  value_mse_vs_result  la1-raw top1 (95% CI)")
        for r, m in ana["human"].items():
            print(f"  {r:4d}  {m['raw_top1']:8.3f}  {m['la1_top1']:8.3f}  "
                  f"{m['value_mse_vs_result']:19.3f}  {_fmt_diff(m['la1_minus_raw_top1'])}")
    results["skipped_val_games"] = [f"game {i}: {e}" for i, e in sorted(skipped.items())]
    results["net_calls"], results["net_rows"] = ev.calls, ev.rows
    if skipped:
        print(f"[warn] {len(skipped)} validation games skipped (rules mismatch), e.g. "
              f"{results['skipped_val_games'][0]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"[done] wrote {a.json}")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

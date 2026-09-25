"""Per-decision inference latency of the game network (batch 1, streaming).

Measures ``net.step()`` — Stage-1 incremental update with its temporal state
plus the Stage-2 core at a given depth — the way a live agent calls it: one
frame at a time, carrying the temporal cache.  Reports p50 / p90 / p99 in ms
over many frames after warm-up, for a grid of cores, inference depths,
MoE on/off, precision and execution mode:

    eager                    plain PyTorch
    compile                  torch.compile(mode="max-autotune-no-cudagraphs")
                             (also enabled by the legacy --compile flag)
    compile-reduce-overhead  torch.compile(mode="reduce-overhead"): inductor
                             kernels replayed from CUDA graphs
    cuda-graph               manual torch.cuda.CUDAGraph capture of one step
                             (model.cuda_graph.GraphedStep): static input
                             buffers, static recurrent state updated in place

The compiled modes feed the step index as a device tensor and start from the
materialised initial state, so every frame hits the same graph.  All modes but
eager need the recurrent temporal mixer (fixed-size state; the attention
mixer's KV cache grows every frame); the two graph modes need CUDA.

    uv run python scripts/bench_latency.py --game generals --device cuda \\
        --cores recurrent_depth transformer --depths 1 2 4 8 16 --moe 0 8 --bf16 \\
        --mode eager cuda-graph compile-reduce-overhead

Prints one JSON line per configuration (easy to aggregate).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from typing import Callable, Dict, List

import torch

from model.games import build_game_network

MODES = ("eager", "compile", "compile-reduce-overhead", "cuda-graph")
GRAPH_MODES = ("compile-reduce-overhead", "cuda-graph")


def _obs_like(net, game: str, device) -> Dict[str, torch.Tensor]:
    """One realistic single-frame observation [1, ...] for ``game``, in
    ``GameNetwork.step``'s obs format."""
    cfg = net.cfg
    if game == "generals":
        return {"grid_seq": torch.rand(1, cfg.grid_channels, cfg.grid_h, cfg.grid_w,
                                       device=device),
                "globals_seq": torch.randn(1, cfg.global_features, device=device)}
    raise SystemExit(f"--game {game}: only generals has a built-in observation stub; "
                     f"use --game generals or extend _obs_like")


def _stepper(net, mode: str, obs: Dict[str, torch.Tensor], depth: int, bf16: bool,
             device) -> Callable[[int], None]:
    """frame index t → run one decision (the carried state lives in here)."""
    from model.game_network import GameNetwork

    if mode == "cuda-graph":
        from model.cuda_graph import GraphedStep
        g = GraphedStep(net, obs, act_steps=depth,
                        autocast_dtype=torch.bfloat16 if bf16 else None)
        return lambda t: g(obs, t)

    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)
    if mode == "eager":
        state = {"cache": None}

        def run(t: int) -> None:
            with torch.inference_mode(), autocast:
                out = GameNetwork.step(net, obs, st_cache=state["cache"], step_idx=t,
                                       act_steps=depth)
            state["cache"] = out["st_cache"]
        return run

    if mode in ("compile", "compile-reduce-overhead"):
        # Tensor step index + fixed-size state from t=0: one graph for every
        # frame (dynamic=False specialises on a Python int step_idx, which
        # would recompile per frame until the cache limit drops to eager).
        graphs = mode == "compile-reduce-overhead"
        net.prepare_inference(device)
        step = torch.compile(GameNetwork.step, dynamic=False,
                             mode="reduce-overhead" if graphs else "max-autotune-no-cudagraphs")
        t_buf = torch.zeros((), dtype=torch.long, device=device)
        with torch.inference_mode():   # same tensor kind as later caches: no recompile
            state = {"cache": net.initial_st_cache(1)}

        def run(t: int) -> None:
            if graphs:
                torch.compiler.cudagraph_mark_step_begin()
            t_buf.fill_(t)
            with torch.inference_mode(), autocast:
                out = step(net, obs, st_cache=state["cache"], step_idx=t_buf,
                           act_steps=depth)
                # cudagraph outputs are overwritten by the next replay
                state["cache"] = ([c.clone() for c in out["st_cache"]] if graphs
                                  else out["st_cache"])
        return run
    raise ValueError(f"unknown mode {mode!r}")


def bench_one(game: str, core: str, depth: int, moe: int, bf16: bool, mode: str,
              frames: int, warmup: int, device, base: Dict) -> Dict:
    torch.manual_seed(0)
    kw = dict(base, core_type=core, moe_experts=moe)
    net = build_game_network(game, **kw).to(device).eval()
    obs = _obs_like(net, game, device)
    run = _stepper(net, mode, obs, depth, bf16, device)
    times: List[float] = []
    sync = (torch.cuda.synchronize if device.type == "cuda"
            else (torch.mps.synchronize if device.type == "mps" else (lambda: None)))
    for t in range(warmup + frames):
        sync()
        t0 = time.perf_counter()
        run(t)
        sync()
        if t >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    q = statistics.quantiles(times, n=100)
    n_params = sum(p.numel() for p in net.parameters())
    return {"game": game, "core": core, "depth": depth, "moe": moe, "bf16": bf16,
            "mode": mode, "compile": mode.startswith("compile"),
            "params_M": round(n_params / 1e6, 2),
            "p50_ms": round(q[49], 3), "p90_ms": round(q[89], 3), "p99_ms": round(q[98], 3),
            "device": str(device), "frames": frames}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--game", default="generals")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cores", nargs="+", default=["recurrent_depth", "transformer"])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--moe", type=int, nargs="+", default=[0, 8])
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--mode", nargs="+", choices=MODES, default=["eager"],
                   help="execution modes to measure (default: eager)")
    p.add_argument("--compile", action="store_true",
                   help="legacy: also measure mode 'compile'")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    import ast
    base = {"temporal_mixer": "recurrent", "dropout": 0.0}
    for kv in a.override:
        k, _, v = kv.partition("=")
        try:
            base[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            base[k] = v
    modes = list(dict.fromkeys(a.mode + (["compile"] if a.compile else [])))
    device = torch.device(a.device)
    static_modes = [m for m in modes if m != "eager"]
    if static_modes and base["temporal_mixer"] != "recurrent":
        raise SystemExit(
            f"--mode {' '.join(static_modes)} needs temporal_mixer=recurrent: the "
            f"{base['temporal_mixer']!r} mixer's KV cache grows by one frame per "
            "step, so the step has no fixed shapes to compile / capture in a CUDA "
            "graph (measure it with --mode eager)")
    graph_modes = [m for m in modes if m in GRAPH_MODES]
    if graph_modes and device.type != "cuda":
        raise SystemExit(f"--mode {' '.join(graph_modes)} needs --device cuda")
    for core in a.cores:
        depths = a.depths if core == "recurrent_depth" else [1]
        for moe in a.moe:
            for depth in depths:
                for mode in modes:
                    try:
                        r = bench_one(a.game, core, depth, moe, a.bf16, mode,
                                      a.frames, a.warmup, device, base)
                    except Exception as e:  # keep the sweep going; record the failure
                        r = {"core": core, "depth": depth, "moe": moe, "mode": mode,
                             "compile": mode.startswith("compile"),
                             "error": f"{type(e).__name__}: {e}"[:300]}
                    print(json.dumps(r), flush=True)
                    with contextlib.suppress(Exception):
                        torch._dynamo.reset()      # compiled graphs of this config


if __name__ == "__main__":
    main()

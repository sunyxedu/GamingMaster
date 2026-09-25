"""Per-decision inference latency of the game network (batch 1, streaming).

Measures ``net.step()`` — Stage-1 incremental update with its temporal state
plus the Stage-2 core at a given depth — the way a live agent calls it: one
frame at a time, carrying the temporal cache.  Reports p50 / p90 / p99 in ms
over many frames after warm-up, for a grid of cores, inference depths,
MoE on/off, precision and torch.compile.

    uv run python scripts/bench_latency.py --game generals --device cuda \\
        --cores recurrent_depth transformer --depths 1 2 4 8 16 --moe 0 8 --bf16 --compile

Prints one JSON line per configuration (easy to aggregate).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict, List

import torch

from model.games import build_game_network


def _obs_like(net, game: str, device) -> Dict[str, torch.Tensor]:
    """One realistic single-frame observation [1, ...] for ``game``."""
    cfg = net.cfg
    if game == "generals":
        return {"grid": torch.rand(1, cfg.grid_channels, cfg.grid_h, cfg.grid_w, device=device),
                "globals": torch.randn(1, cfg.global_features, device=device)}
    raise SystemExit(f"--game {game}: only generals has a built-in observation stub; "
                     f"use --game generals or extend _obs_like")


def bench_one(game: str, core: str, depth: int, moe: int, bf16: bool, compile_: bool,
              frames: int, warmup: int, device, base: Dict) -> Dict:
    torch.manual_seed(0)
    kw = dict(base, core_type=core, moe_experts=moe)
    net = build_game_network(game, **kw).to(device).eval()
    obs = _obs_like(net, game, device)
    step = net.step
    if compile_:
        step = torch.compile(net.step, mode="max-autotune-no-cudagraphs", dynamic=False)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)
    times: List[float] = []
    cache = None
    sync = (torch.cuda.synchronize if device.type == "cuda"
            else (torch.mps.synchronize if device.type == "mps" else (lambda: None)))
    with torch.inference_mode(), autocast:
        for t in range(warmup + frames):
            sync()
            t0 = time.perf_counter()
            out = step(obs["grid"], obs["globals"], st_cache=cache, step_idx=t,
                       act_steps=depth)
            sync()
            if t >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
            cache = out["st_cache"]
    q = statistics.quantiles(times, n=100)
    n_params = sum(p.numel() for p in net.parameters())
    return {"game": game, "core": core, "depth": depth, "moe": moe, "bf16": bf16,
            "compile": compile_, "params_M": round(n_params / 1e6, 2),
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
    p.add_argument("--compile", action="store_true", help="also measure torch.compile")
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
    device = torch.device(a.device)
    for core in a.cores:
        depths = a.depths if core == "recurrent_depth" else [1]
        for moe in a.moe:
            for depth in depths:
                for comp in ([False, True] if a.compile else [False]):
                    try:
                        r = bench_one(a.game, core, depth, moe, a.bf16, comp,
                                      a.frames, a.warmup, device, base)
                    except Exception as e:  # keep the sweep going; record the failure
                        r = {"core": core, "depth": depth, "moe": moe, "compile": comp,
                             "error": f"{type(e).__name__}: {e}"[:300]}
                    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()

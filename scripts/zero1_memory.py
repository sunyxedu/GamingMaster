"""Per-rank memory of the optimizer state and EMA with and without
``--shard_optimizer`` (ZeRO-1, training/zero.py) for a multi-game model,
counted on the meta device (nothing is allocated, no GPU needed), and the
expected per-rank peak at other expert counts, extrapolated from one
measured peak.

    uv run python scripts/zero1_memory.py            # the Stage-1 core, 12 ranks, 32 + 64 experts
    uv run python scripts/zero1_memory.py --experts 32 64 96 --world 12 --measured_peak 11.9

Default model: the 10 games of the Stage-1 run on the shared core d=384,
8-layer transformer, qk_norm, MoE top-2 + 1 shared expert (--games,
--override change it; per-game adapter widths from shard metadata are not
applied, a <0.1% difference).  The partition is exactly the trainer's:
training.zero.partition_by_numel over multi_game_param_groups' order.

Peak model, bytes per rank, for E routed experts per layer:

    peak(E) = A + 4·N_E (weights) + 4·N_E (gradients) + S_E
              + 2·(X_E − X_m)        with --bf16: the bf16 casts of the routed
                                     expert weights each micro-batch keeps for
                                     its backward
              + 8·(L_E − L_m)        the MoE backward's transient full-size
                                     copies of its largest expert tensor
                                     (select_backward's zero-fill + the
                                     accumulation buffer, fp32)

with N the parameters, X the routed-expert parameters, L the largest
routed-expert tensor, S = 12·N unsharded (two AdamATan2 moments + the EMA
shadow, fp32) or 12·(largest rank partition) sharded, and A (activations,
workspace, allocator slack, ...) solved from the measured peak of the
unsharded run at m = --measured_experts.  Activations do not grow with E
(top-k routing: the same routed compute per token).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from model.multi_game import build_multi_game  # noqa: E402
from training.zero import (  # noqa: E402
    partition_by_numel,
    rank_loads,
    state_bytes_per_param,
)

#: the Stage-1 run: its 10 games and its core (the expert count varies)
PRODUCTION_GAMES = ("generals", "othello", "chess", "go", "mahjong", "shogi", "pokemon",
                    "lux", "clash_royale", "gomoku")
PRODUCTION_CORE = dict(d_model=384, core_type="transformer", core_layers=8, qk_norm=True,
                       moe_top_k=2, moe_shared_experts=1)
#: the measured peak of the production run (12 x 1g.18gb, unsharded, bf16)
MEASURED_PEAK_GIB = 11.9
MEASURED_EXPERTS = 32
GIB = 2 ** 30
FP32 = 4
ROUTED = ("w_gate_up", "w_down")


@dataclass
class ModelCount:
    experts: int
    numels: List[int]           # trainable parameters, in the optimizer's order
    expert_params: int          # routed-expert parameters
    largest_expert: int         # largest routed-expert tensor

    @property
    def params(self) -> int:
        return sum(self.numels)


@dataclass
class MemoryRow:
    """Bytes per rank (the sharded state: the largest / smallest partition)."""

    experts: int
    params: int
    expert_params: int
    weights: int
    grads: int
    state_unsharded: int
    state_sharded_max: int
    state_sharded_min: int
    peak_unsharded: float
    peak_sharded: float


def count_model(games: Sequence[str], core: Dict, experts: int) -> ModelCount:
    """The trainable parameter sizes of the multi-game model, built on the
    meta device, in the trainer's optimizer order."""
    from training.train_multi_game import multi_game_param_groups

    with torch.device("meta"):
        model = build_multi_game(list(games), {**core, "moe_experts": experts})
    groups = multi_game_param_groups(model, 0.1)
    numels = [p.numel() for g in groups for p in g["params"]]
    routed = [p.numel() for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] in ROUTED]
    return ModelCount(experts, numels, sum(routed), max(routed, default=0))


def memory_plan(games: Sequence[str] = PRODUCTION_GAMES, core: Dict = PRODUCTION_CORE,
                experts: Sequence[int] = (32, 64), world: int = 12,
                measured_peak_gib: float = MEASURED_PEAK_GIB,
                measured_experts: int = MEASURED_EXPERTS, bf16: bool = True,
                ema: bool = True) -> Tuple[List[MemoryRow], float]:
    """(one row per expert count, A in bytes) — see the module docstring."""
    counts = {e: count_model(games, core, e) for e in sorted({*experts, measured_experts})}
    per = state_bytes_per_param(ema)
    base = counts[measured_experts]
    a = measured_peak_gib * GIB - (2 * FP32 + per) * base.params
    rows = []
    for e in experts:
        c = counts[e]
        loads = rank_loads(c.numels, partition_by_numel(c.numels, world), world)
        extra = (2 * (c.expert_params - base.expert_params) * int(bf16)
                 + 2 * FP32 * (c.largest_expert - base.largest_expert))
        full = a + 2 * FP32 * c.params + extra
        rows.append(MemoryRow(
            experts=e, params=c.params, expert_params=c.expert_params,
            weights=FP32 * c.params, grads=FP32 * c.params,
            state_unsharded=per * c.params, state_sharded_max=per * max(loads),
            state_sharded_min=per * min(loads),
            peak_unsharded=full + per * c.params, peak_sharded=full + per * max(loads)))
    return rows, a


def format_plan(rows: List[MemoryRow], a: float, world: int, measured_peak_gib: float,
                measured_experts: int, n_games: int) -> str:
    g = lambda b: f"{b / GIB:.2f}"   # noqa: E731
    lines = [f"{n_games} games, {world} ranks; baseline: measured peak {measured_peak_gib:.2f} "
             f"GiB per rank at {measured_experts} experts, unsharded -> A = {g(a)} GiB "
             f"(activations + everything else)",
             f"{'experts':>7} {'params':>8} {'routed':>8} {'weights':>8} {'grads':>6} "
             f"{'m+v+EMA/rank: unsharded':>24} {'sharded':>8} "
             f"{'peak/rank: unsharded':>21} {'sharded':>8}   (GiB)"]
    for r in rows:
        lines.append(f"{r.experts:>7} {r.params / 1e6:>7.1f}M {r.expert_params / 1e6:>7.1f}M "
                     f"{g(r.weights):>8} {g(r.grads):>6} {g(r.state_unsharded):>24} "
                     f"{g(r.state_sharded_max):>8} {g(r.peak_unsharded):>21} "
                     f"{g(r.peak_sharded):>8}")
    return "\n".join(lines)


def main(argv=None) -> None:
    from training.train_il_game import parse_overrides

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--games", nargs="+", default=list(PRODUCTION_GAMES))
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="core overrides on top of the Stage-1 core")
    p.add_argument("--experts", type=int, nargs="+", default=[32, 64])
    p.add_argument("--world", type=int, default=12)
    p.add_argument("--measured_peak", type=float, default=MEASURED_PEAK_GIB,
                   help="measured peak allocated GiB per rank of the unsharded run")
    p.add_argument("--measured_experts", type=int, default=MEASURED_EXPERTS)
    p.add_argument("--no_bf16", action="store_true", help="the measured run was fp32")
    p.add_argument("--no_ema", action="store_true", help="--ema_rate 0")
    a = p.parse_args(argv)
    core = {**PRODUCTION_CORE, **parse_overrides(a.override)}
    rows, base = memory_plan(a.games, core, a.experts, a.world, a.measured_peak,
                             a.measured_experts, bf16=not a.no_bf16, ema=not a.no_ema)
    print(format_plan(rows, base, a.world, a.measured_peak, a.measured_experts, len(a.games)))


if __name__ == "__main__":
    main()

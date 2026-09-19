"""``python -m metaharness`` — run the meta-harness loop or evaluate one harness.

Examples
--------
# tier 2, snake, offline dry run (no network): scripted proposer, heuristic Jev
python -m metaharness loop --tier 2 --game snake --system1 heuristic --system2 scripted \
    --proposer scripted --clock virtual --iters 1

# tier 2, real models: Jev for System-1, Astra (codex) for System-2 and for proposing
python -m metaharness loop --tier 2 --game snake --system1 jev --system2 astra --proposer astra

# evaluate one harness file on 3 seeds
python -m metaharness eval --tier 2 --game snake --harness metaharness/baselines/tier2_jev_only.py \
    --system1 jev --system2 none --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import DEFAULT_RUNS_DIR, load_dotenv
from .core import Archive, LoopConfig, MetaHarnessLoop
from .core.contract import Budget
from .core.runner import evaluate_harness
from .games.registry import GAMES, make_game
from .models import make_system1, make_system2  # noqa: F401


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _system2(name: str, effort: str | None):
    """``none`` | a ``[system2.*]`` profile from models.toml (``--effort`` overrides the profile)."""
    if name == "none":
        return None
    if name == "scripted":                      # offline stand-in with game-aware canned replies
        from .baselines._scripted import scripted_system2
        return scripted_system2()
    return make_system2(name, reasoning_effort=effort)


def _system1(name: str):
    if name == "none":
        return None
    if name == "heuristic":
        from .baselines._scripted import heuristic_snake_policy
        return make_system1("heuristic", fn=heuristic_snake_policy)
    return make_system1(name)


def _proposer(name: str, effort: str | None, tier_mod):
    if name == "scripted":
        from .baselines._scripted import scripted_proposer
        return scripted_proposer(tier_mod)
    return _system2(name, effort)


def _tier(n: int):
    if n == 1:
        from .tiers import tier1 as t
    elif n == 2:
        from .tiers import tier2 as t
    else:
        raise SystemExit("tier 3 is the trained network in model/ — not handled by metaharness")
    return t


def cmd_loop(a: argparse.Namespace) -> int:
    load_dotenv()
    tier = _tier(a.tier)
    budget = Budget(max_system2_calls=a.s2_budget, max_system1_calls=a.s1_budget, max_wall_s=a.wall_budget)
    game_kwargs = json.loads(a.game_kwargs) if a.game_kwargs else {}
    exp = a.name or f"tier{a.tier}-{a.game}-{time.strftime('%Y%m%d-%H%M%S')}"
    archive = Archive(Path(a.runs_dir) / exp)
    for p in tier.baseline_paths():
        if not any(c.dir.name.endswith(p.stem) for c in archive.candidates()):
            archive.add_seed(p, label=p.stem, note=(p.read_text().split('"""')[1].strip() + "\n") if '"""' in p.read_text() else "")
    cfg = LoopConfig(n_iterations=a.iters, seeds_train=_parse_seeds(a.seeds), seeds_heldout=_parse_seeds(a.heldout),
                     clock_mode=a.clock, budget=budget, proposer_agentic=not a.proposer_text,
                     proposer_timeout_s=a.proposer_timeout, harness_timeout_s=a.harness_timeout,
                     latency_override=json.loads(a.latency_override) if a.latency_override else None,
                     heldout_game=a.heldout_game or None,
                     heldout_game_kwargs=json.loads(a.heldout_game_kwargs) if a.heldout_game_kwargs else {})
    loop = MetaHarnessLoop(
        archive=archive, game_factory=GAMES[a.game][1] if a.game in GAMES else (lambda **k: make_game(a.game, **k)),
        game_kwargs=game_kwargs, task_md=tier.task_md(a.game, budget, a.clock, game_kwargs),
        contract_doc=tier.contract_doc(), proposer=_proposer(a.proposer, a.effort, tier),
        system2=_system2(a.system2, a.effort), system1=_system1(a.system1), config=cfg,
    )
    print(f"archive: {archive.root}")
    best = loop.run()
    print(archive.write_leaderboard())
    if best:
        print(f"best: {best.dir}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    load_dotenv()
    _tier(a.tier)
    budget = Budget(max_system2_calls=a.s2_budget, max_system1_calls=a.s1_budget, max_wall_s=a.wall_budget)
    game_kwargs = json.loads(a.game_kwargs) if a.game_kwargs else {}
    out = Path(a.out) if a.out else Path(a.runs_dir) / f"eval-{Path(a.harness).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    res = evaluate_harness(a.harness, lambda **k: make_game(a.game, **k), _parse_seeds(a.seeds),
                           system2=_system2(a.system2, a.effort), system1=_system1(a.system1), out_dir=out,
                           clock_mode=a.clock, budget_per_episode=budget, harness_timeout_s=a.harness_timeout,
                           latency_override=json.loads(a.latency_override) if a.latency_override else None,
                           game_kwargs=game_kwargs)
    print(json.dumps({k: v for k, v in res.to_json().items() if k != "episodes"}, indent=2, default=str))
    for e in res.episodes:
        print(f"seed {e.seed}: score={e.score} steps={e.steps} stale={e.stale_ticks}/{e.stale_ticks + e.decided_ticks} "
              f"s1={e.system1_calls} s2={e.system2_calls} err={'yes' if e.error else 'no'}")
    print(f"out: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="metaharness")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--tier", type=int, required=True, choices=[1, 2])
        sp.add_argument("--game", required=True)
        sp.add_argument("--game-kwargs", default="", help="JSON kwargs for the game constructor")
        sp.add_argument("--system2", default="astra",
                        help="System-2 profile from metaharness/models.toml (astra, sol, claude, ...) | scripted | none")
        sp.add_argument("--system1", default="jev",
                        help="System-1 profile from metaharness/models.toml (jev) | heuristic | none")
        sp.add_argument("--effort", default=None, help="override the profile's reasoning effort (low|medium|high|xhigh|max)")
        sp.add_argument("--clock", default="wall", choices=["wall", "virtual"])
        sp.add_argument("--latency-override", default="", help='JSON e.g. {"system2": 3.0, "system1": 0.3} (virtual clock)')
        sp.add_argument("--seeds", default="0,1,2")
        sp.add_argument("--s2-budget", type=int, default=40)
        sp.add_argument("--s1-budget", type=int, default=4000)
        sp.add_argument("--wall-budget", type=float, default=900.0)
        sp.add_argument("--harness-timeout", type=float, default=600.0)
        sp.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    lp = sub.add_parser("loop", help="run the meta-harness search")
    common(lp)
    lp.add_argument("--proposer", default="astra", help="System-2 profile for the proposer | scripted")
    lp.add_argument("--proposer-text", action="store_true", help="proposer answers in text (no tools)")
    lp.add_argument("--proposer-timeout", type=float, default=1800.0)
    lp.add_argument("--iters", type=int, default=3)
    lp.add_argument("--heldout", default="100,101,102", help="held-out seeds (same game)")
    lp.add_argument("--heldout-game", default="", help="held-out GAME for the generality test (e.g. lanerunner / arc3)")
    lp.add_argument("--heldout-game-kwargs", default="", help="JSON kwargs for the held-out game")
    lp.add_argument("--name", default="")
    lp.set_defaults(fn=cmd_loop)

    ep = sub.add_parser("eval", help="evaluate one harness file")
    common(ep)
    ep.add_argument("--harness", required=True)
    ep.add_argument("--out", default="")
    ep.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

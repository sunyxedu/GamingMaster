"""``python -m metaharness`` — the Meta-Harness command line.

The general entry point takes only a game name (plus your notes / credentials in .env):

    python -m metaharness run --game "Minecraft" --notes "Java 1.21 server at localhost:25565"
    python -m metaharness run --game "ARC-AGI-3" --tier 1

It runs the phases study -> adapt -> search under runs/<slug>/ and resumes where it
left off. Lower-level commands:

    # search only, on an offline test bed (no network): scripted proposer, heuristic Jev
    python -m metaharness loop --tier 2 --game snake --system1 heuristic --system2 scripted \\
        --proposer scripted --clock virtual --iters 1

    # search on a generated adaptor
    python -m metaharness loop --adaptor runs/minecraft/adaptor --tier 2 --study runs/minecraft/study

    # evaluate one harness file on 3 seeds
    python -m metaharness eval --tier 2 --game snake --harness metaharness/baselines/tier2_jev_only.py \\
        --system1 jev --system2 none --seeds 0,1,2

    # validate an adaptor directory against the contract
    python -m metaharness validate-adaptor runs/minecraft/adaptor
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .config import DEFAULT_RUNS_DIR, load_dotenv
from .core import Archive, LoopConfig, MetaHarnessLoop
from .core.contract import Budget
from .core.runner import evaluate_harness
from .games.registry import GAMES, make_game, register_adaptor
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


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "game"


def _resolve_game(a: argparse.Namespace) -> str:
    """``--adaptor DIR`` registers a generated adaptor and returns its game name; else ``--game``."""
    if getattr(a, "adaptor", ""):
        return register_adaptor(Path(a.adaptor))
    if not a.game:
        raise SystemExit("--game or --adaptor is required")
    return a.game


def _study_text(a: argparse.Namespace) -> str:
    if getattr(a, "study", ""):
        from .phases.study import study_text
        return study_text(Path(a.study))
    return ""


def _run_search(a: argparse.Namespace, game: str, tier_n: int, study: str, exp_dir: Path):
    tier = _tier(tier_n)
    budget = Budget(max_system2_calls=a.s2_budget, max_system1_calls=a.s1_budget, max_wall_s=a.wall_budget)
    game_kwargs = json.loads(a.game_kwargs) if a.game_kwargs else {}
    archive = Archive(exp_dir)
    for p in tier.baseline_paths():
        if not any(c.dir.name.endswith(p.stem) for c in archive.candidates()):
            archive.add_seed(p, label=p.stem, note=(p.read_text().split('"""')[1].strip() + "\n") if '"""' in p.read_text() else "")
    system1 = _system1(a.system1) if tier_n == 2 else (None if a.system1 in ("none", "jev", "heuristic") else _system1(a.system1))
    cfg = LoopConfig(n_iterations=a.iters, seeds_train=_parse_seeds(a.seeds), seeds_heldout=_parse_seeds(a.heldout),
                     clock_mode=a.clock, budget=budget, proposer_agentic=not a.proposer_text,
                     proposer_timeout_s=a.proposer_timeout, harness_timeout_s=a.harness_timeout,
                     latency_override=json.loads(a.latency_override) if a.latency_override else None,
                     heldout_game=a.heldout_game or None, memory_mode=a.memory_mode,
                     heldout_game_kwargs=json.loads(a.heldout_game_kwargs) if a.heldout_game_kwargs else {},
                     preflight=not a.no_preflight)
    loop = MetaHarnessLoop(
        archive=archive, game_factory=GAMES[game][1] if game in GAMES else (lambda **k: make_game(game, **k)),
        game_kwargs=game_kwargs, task_md=tier.task_md(game, budget, a.clock, game_kwargs, study=study),
        contract_doc=tier.contract_doc(), proposer=_proposer(a.proposer, a.effort, tier),
        system2=_system2(a.system2, a.effort), system1=system1, config=cfg,
    )
    print(f"archive: {archive.root}")
    best = loop.run()
    print(archive.write_leaderboard())
    if best:
        print(f"best: {best.dir}")
    return best


def cmd_loop(a: argparse.Namespace) -> int:
    load_dotenv()
    game = _resolve_game(a)
    exp = a.name or f"tier{a.tier}-{_slug(game)}-{time.strftime('%Y%m%d-%H%M%S')}"
    _run_search(a, game, a.tier, _study_text(a), Path(a.runs_dir) / exp)
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    """The general pipeline: study -> adapt -> search, resumable, under runs/<slug>/."""
    load_dotenv()
    from .phases import run_adapt, run_study
    run_dir = Path(a.runs_dir) / (a.name or _slug(a.game))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    agent = _system2(a.agent, a.effort)
    if agent is None:
        raise SystemExit("--agent must be a System-2 profile (e.g. astra)")
    log(f"run: game={a.game!r} dir={run_dir} agent={agent.name}")
    study = run_study(agent, run_dir, a.game, notes=a.notes, tier_hint=a.tier, timeout_s=a.phase_timeout, log=log)
    tier_n = int(study["tier"])
    if a.stop_after == "study":
        return 0
    rep = run_adapt(agent, run_dir, a.game, study, timeout_s=a.phase_timeout, log=log)
    if not rep.ok:
        log("run: adaptor not ready — " + (rep.error or "see adaptor/validation.md")[:500])
        needs = run_dir / "adaptor" / "NEEDS.md"
        if needs.exists():
            print("\n" + needs.read_text())
        return 2
    if a.stop_after == "adapt":
        return 0
    game = register_adaptor(run_dir / "adaptor")
    from .phases.study import study_text
    log(f"search: tier {tier_n} on adaptor game {game!r}")
    if not a.system1 or a.system1 == "auto":
        a.system1 = "jev" if tier_n == 2 else "none"
    prior = study_text(run_dir / "study")
    adaptor_md = run_dir / "adaptor" / "ADAPTOR.md"
    if adaptor_md.exists():                     # interface documentation, not hidden rules
        prior += "\n\n### adaptor/ADAPTOR.md\n" + adaptor_md.read_text()[:8000]
    best = _run_search(a, game, tier_n, prior, run_dir / "search")
    log("run: done" + (f" — best {best.dir.name}" if best else " — no eligible candidate"))
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    load_dotenv()
    _tier(a.tier)
    game = _resolve_game(a)
    budget = Budget(max_system2_calls=a.s2_budget, max_system1_calls=a.s1_budget, max_wall_s=a.wall_budget)
    game_kwargs = json.loads(a.game_kwargs) if a.game_kwargs else {}
    out = Path(a.out) if a.out else Path(a.runs_dir) / f"eval-{Path(a.harness).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    res = evaluate_harness(a.harness, lambda **k: make_game(game, **k), _parse_seeds(a.seeds),
                           system2=_system2(a.system2, a.effort), system1=_system1(a.system1), out_dir=out,
                           clock_mode=a.clock, budget_per_episode=budget, harness_timeout_s=a.harness_timeout,
                           latency_override=json.loads(a.latency_override) if a.latency_override else None,
                           game_kwargs=game_kwargs, memory_mode=a.memory_mode)
    print(json.dumps({k: v for k, v in res.to_json().items() if k != "episodes"}, indent=2, default=str))
    for e in res.episodes:
        print(f"seed {e.seed}: score={e.score} steps={e.steps} stale={e.stale_ticks}/{e.stale_ticks + e.decided_ticks} "
              f"s1={e.system1_calls} s2={e.system2_calls} err={'yes' if e.error else 'no'}")
    print(f"out: {out}")
    return 1 if res.fatal or any(e.error for e in res.episodes) else 0


def cmd_validate_adaptor(a: argparse.Namespace) -> int:
    load_dotenv()
    from .games.adaptor import validate_adaptor
    rep = validate_adaptor(Path(a.path), kwargs=json.loads(a.game_kwargs) if a.game_kwargs else None)
    print(rep.to_markdown())
    return 0 if rep.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="metaharness")
    sub = p.add_subparsers(dest="cmd", required=True)

    def search_opts(sp, tier_required: bool):
        if tier_required:
            sp.add_argument("--tier", type=int, required=True, choices=[1, 2])
        sp.add_argument("--game-kwargs", default="", help="JSON kwargs for the game constructor")
        sp.add_argument("--system2", default="astra",
                        help="System-2 profile from metaharness/models.toml (astra, sol, claude, ...) | scripted | none")
        sp.add_argument("--system1", default="jev",
                        help="System-1 profile from metaharness/models.toml (jev) | heuristic | none")
        sp.add_argument("--effort", default=None, help="override the profile's reasoning effort (low|medium|high|xhigh|max)")
        sp.add_argument("--clock", default="wall", choices=["wall", "virtual"])
        sp.add_argument("--memory-mode", default="episode", choices=["episode", "shared"],
                        help="episode: independent cold starts; shared: adaptation across ordered seeds")
        sp.add_argument("--latency-override", default="", help='JSON e.g. {"system2": 3.0, "system1": 0.3} (virtual clock)')
        sp.add_argument("--seeds", default="0,1,2")
        sp.add_argument("--s2-budget", type=int, default=40)
        sp.add_argument("--s1-budget", type=int, default=4000)
        sp.add_argument("--wall-budget", type=float, default=900.0)
        sp.add_argument("--harness-timeout", type=float, default=600.0)
        sp.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    def loop_opts(sp):
        sp.add_argument("--proposer", default="astra", help="System-2 profile for the proposer | scripted")
        sp.add_argument("--proposer-text", action="store_true", help="proposer answers in text (no tools)")
        sp.add_argument("--proposer-timeout", type=float, default=1800.0)
        sp.add_argument("--iters", type=int, default=3)
        sp.add_argument("--heldout", default="100,101,102", help="held-out seeds (same game)")
        sp.add_argument("--heldout-game", default="", help="held-out GAME for the generality test (registered name)")
        sp.add_argument("--heldout-game-kwargs", default="", help="JSON kwargs for the held-out game")
        sp.add_argument("--no-preflight", action="store_true", help="skip the stand-in dry run + repair round")
        sp.add_argument("--name", default="")

    rp = sub.add_parser("run", help="general pipeline: study -> adapt -> search from a game name")
    rp.add_argument("--game", required=True, help='game name, e.g. "Minecraft", "ARC-AGI-3", "Snake"')
    rp.add_argument("--notes", default="", help="platform / account / constraints for the study & adapt agents")
    rp.add_argument("--tier", type=int, default=None, choices=[1, 2], help="override the tier the study decides")
    rp.add_argument("--agent", default="astra", help="System-2 profile for study/adapt agents")
    rp.add_argument("--phase-timeout", type=float, default=3600.0)
    rp.add_argument("--stop-after", default="", choices=["", "study", "adapt"])
    search_opts(rp, tier_required=False)
    loop_opts(rp)
    rp.set_defaults(fn=cmd_run, system1="auto")

    lp = sub.add_parser("loop", help="run the harness search on a registered game or adaptor")
    lp.add_argument("--game", default="", help="registered game name (test beds: snake, lanerunner, rulepuzzle, arc3)")
    lp.add_argument("--adaptor", default="", help="generated adaptor directory (runs/<game>/adaptor)")
    lp.add_argument("--study", default="", help="study directory whose documents become prior knowledge")
    search_opts(lp, tier_required=True)
    loop_opts(lp)
    lp.set_defaults(fn=cmd_loop)

    ep = sub.add_parser("eval", help="evaluate one harness file")
    ep.add_argument("--game", default="")
    ep.add_argument("--adaptor", default="")
    search_opts(ep, tier_required=True)
    ep.add_argument("--harness", required=True)
    ep.add_argument("--out", default="")
    ep.set_defaults(fn=cmd_eval)

    vp = sub.add_parser("validate-adaptor", help="run the generic adaptor smoke test")
    vp.add_argument("path")
    vp.add_argument("--game-kwargs", default="")
    vp.set_defaults(fn=cmd_validate_adaptor)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

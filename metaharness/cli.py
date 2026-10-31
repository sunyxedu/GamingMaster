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
import signal
import sys
import threading
import time
from pathlib import Path

from .config import DEFAULT_RUNS_DIR, load_dotenv
from .core import Archive, GameSpec, LoopConfig, MetaHarnessLoop
from .core.calibrate import calibrate
from .core.contract import Budget
from .core.loop import SeedRefused, _slug as _game_slug
from .core.runner import DEFAULT_GRACE_S, evaluate_harness
from .games import registry as _registry
from .games.registry import GAMES, make_game, register_adaptor
from .models import ModelUnavailable, make_system1, make_system2  # noqa: F401


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _system2(name: str, effort: str | None):
    """``none`` | a ``[system2.*]`` profile from models.toml (``--effort`` overrides the profile)."""
    if name == "none" or name.startswith("silent"):
        return None if name == "none" else _silent(name)
    if name == "scripted":                      # offline stand-in with game-aware canned replies
        from .baselines._scripted import scripted_system2
        return scripted_system2()
    return make_system2(name, reasoning_effort=effort)


def _silent(name: str):
    """``silent`` or ``silent:<seconds>``: every call fails after that latency (the no-models ablation)."""
    from .models.mock import SilentModel
    return SilentModel(float(name.partition(":")[2] or 0))


def _system1(name: str):
    if name == "none" or name.startswith("silent"):
        return None if name == "none" else _silent(name)
    if name == "heuristic":
        from .baselines._scripted import heuristic_snake_policy
        return make_system1("heuristic", fn=heuristic_snake_policy)
    return make_system1(name)


def _system1_for(name: str | None, tier_n: int):
    """``auto``: Jev for tier 2, no System-1 for tier 1 (a tier-1 task never has one)."""
    if not name or name == "auto":
        name = "jev" if tier_n == 2 else "none"
    if tier_n == 1 and name in ("jev", "heuristic"):
        return None
    return _system1(name)


def _adaptor_spec(game: str):
    lookup = getattr(_registry, "adaptor_spec", None)
    return lookup(game) if lookup else None


def _kwargs(text: str, flag: str) -> dict:
    try:
        value = json.loads(text) if text else {}
    except ValueError as e:
        raise SystemExit(f"{flag} is not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise SystemExit(f"{flag} must be a JSON object")
    return value


def _game_setup(a: argparse.Namespace, game: str) -> tuple[dict, dict]:
    """(kwargs, declared seeds) for a game: an adaptor's EVAL_KWARGS overridden by --game-kwargs
    (as validate_adaptor merges them) and its SEEDS split, if it declares one."""
    spec = _adaptor_spec(game)
    kwargs = {**(getattr(spec, "eval_kwargs", None) or {}), **_kwargs(a.game_kwargs, "--game-kwargs")}
    return kwargs, dict(getattr(spec, "seeds", None) or {})


def _resolve_heldout_game(a: argparse.Namespace) -> tuple[str | None, dict, list[int] | None]:
    """``--heldout-game``: a registered game name or a generated adaptor directory, checked before
    the search starts -> (key, kwargs, declared held-out seeds). An adaptor's EVAL_KWARGS and SEEDS
    apply to it as they do to the training game; --heldout-game-kwargs override the kwargs."""
    name = a.heldout_game
    user = _kwargs(a.heldout_game_kwargs, "--heldout-game-kwargs")
    if not name:
        return None, user, None
    if Path(name).is_dir():
        key = register_adaptor(Path(name), kwargs=user or None)
        spec = _adaptor_spec(key)
        seeds = (getattr(spec, "seeds", None) or {}).get("heldout")
        return key, {**(getattr(spec, "eval_kwargs", None) or {}), **user}, (list(seeds) if seeds else None)
    _registry._populate()
    if name not in GAMES:
        raise SystemExit(f"--heldout-game {name!r} is neither a registered game ({sorted(GAMES)}) "
                         "nor an adaptor directory")
    return name, user, None


def _resolve_games(a: argparse.Namespace) -> list[tuple[str, str | None]]:
    """The training games: ``--adaptor DIR`` (repeatable) and ``--game NAME[,NAME...]`` ->
    [(registry key, adaptor directory or None), ...] in that order."""
    out: list[tuple[str, str | None]] = []
    dirs = getattr(a, "adaptor", None) or []
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        if d:
            key = register_adaptor(Path(d), kwargs=_kwargs(a.game_kwargs, "--game-kwargs") or None)
            out.append((key, str(Path(d).resolve())))
    for name in [n.strip() for n in (getattr(a, "game", "") or "").split(",") if n.strip()]:
        out.append((name, None))
    if not out:
        raise SystemExit("--game or --adaptor is required")
    return out


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
    """Run-directory name: the name's letters, digits and combining marks, lowercased, '-' between
    them. Unicode-aware — '原神' and '王者荣耀' no longer share runs/game/ — and an ASCII name keeps the
    slug it always had (anything but [a-z0-9], '_' included, separates)."""
    import unicodedata
    t = unicodedata.normalize("NFKC", s).lower()
    t = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else "-" for c in t)
    return re.sub(r"-+", "-", t).strip("-") or "game"


def _resolve_game(a: argparse.Namespace) -> str:
    """The first training game (``eval`` plays one game)."""
    return _resolve_games(a)[0][0]


def _study_text(a: argparse.Namespace, archive_root: Path | None, games: list[tuple[str, str | None]]) -> str:
    """Prior knowledge for the proposer from ``--study`` and/or the adaptors (ADAPTOR.md): the digest
    of ``prior_documents``, which also copies the full documents into ``<archive_root>/prior/``
    (``prior/<game>/`` each when several games have documents)."""
    from .phases.study import prior_documents, study_text
    study = Path(a.study) if getattr(a, "study", "") else None
    adaptors = [(key, Path(d)) for key, d in games if d]
    if archive_root is None:                    # no archive to copy into: the study excerpt alone
        return study_text(study) if study else ""
    if len(adaptors) <= 1:
        if study is None and not adaptors:
            return ""
        return prior_documents(study, adaptors[0][1] if adaptors else None, archive_root)
    parts = []
    for key, d in adaptors:                      # a run directory keeps the study next to its adaptor
        sdir = d.parent / "study" if (d.parent / "study").is_dir() else None
        parts.append(f"## Prior documents for `{key}`\n"
                     + prior_documents(sdir, d, archive_root, label=_game_slug(key)))
    return "\n\n".join(parts)


def _episode_budget(a: argparse.Namespace) -> Budget:
    """One per-episode deadline: the larger of --wall-budget and --harness-timeout (the ARC run lost
    an iteration to a harness that respected --harness-timeout 1800 while a 900 s wall budget cut
    every long episode, 2026-09-21). The runner enforces exactly this and ctx.budget reports it."""
    wall = max(float(a.wall_budget), float(a.harness_timeout))
    return Budget(max_system2_calls=a.s2_budget, max_system1_calls=a.s1_budget, max_wall_s=wall)


def _game_specs(a: argparse.Namespace, games: list[tuple[str, str | None]]) -> list[GameSpec]:
    """A GameSpec per training game: its kwargs (EVAL_KWARGS + --game-kwargs) and its seed blocks:
    --seeds / --val-seeds / --heldout, else the adaptor's declared split (with a validation block
    carved from the end of a declared training list of 4+ seeds), else 0,1,2 / 3,4 / 100,101,102."""
    specs = []
    for key, adaptor_dir in games:
        kwargs, declared = _game_setup(a, key)
        train = _parse_seeds(a.seeds) if a.seeds is not None else list(declared.get("train") or [0, 1, 2])
        held = _parse_seeds(a.heldout) if a.heldout is not None else list(declared.get("heldout") or [100, 101, 102])
        val = _parse_seeds(a.val_seeds) if getattr(a, "val_seeds", "") else list(declared.get("validation") or [])
        if not val and a.seeds is None and declared.get("train") and len(train) >= 4:
            n_val = max(1, len(train) // 4)
            val, train = train[-n_val:], train[:-n_val]
        elif not val and a.seeds is None and not declared.get("train"):
            val = [3, 4]                                   # nothing declared, nothing asked: a default block
        factory = GAMES[key][1] if key in GAMES else (lambda k=key, **kw: make_game(k, **kw))
        specs.append(GameSpec(key, factory, kwargs, train, val, held, adaptor_dir))
    return specs


def _run_search(a: argparse.Namespace, games: list[tuple[str, str | None]], tier_n: int, study: str, exp_dir: Path):
    tier = _tier(tier_n)
    budget = _episode_budget(a)
    specs = _game_specs(a, games)
    first = specs[0]
    heldout_game, heldout_kwargs, heldout_seeds = _resolve_heldout_game(a)
    heldout_adaptor = str(Path(a.heldout_game).resolve()) if a.heldout_game and Path(a.heldout_game).is_dir() else None
    archive = Archive(exp_dir)
    for p in tier.baseline_paths():
        if not any(c.dir.name.endswith(p.stem) for c in archive.candidates()):
            archive.add_seed(p, label=p.stem, note=(p.read_text().split('"""')[1].strip() + "\n") if '"""' in p.read_text() else "")
    system1 = _system1_for(a.system1, tier_n)
    system2 = _system2(a.system2, a.effort)
    sources = [s for s in (getattr(_adaptor_spec(g.name), "source_sha256", None) for g in specs) if s]
    declared = _game_setup(a, first.name)[1]
    dev_arg = getattr(a, "dev_seeds", None)
    seeds_dev = _parse_seeds(dev_arg) if dev_arg else list(declared.get("dev") or [])
    cfg = LoopConfig(n_iterations=a.iters, seeds_train=first.seeds_train, seeds_heldout=first.seeds_heldout,
                     seeds_val=first.seeds_val or [],
                     seeds_dev=seeds_dev, proposer_dev_evals=getattr(a, "dev_evals", 2) if seeds_dev else 0,
                     proposer_tools=not getattr(a, "no_proposer_tools", False),
                     clock_mode=a.clock, budget=budget, proposer_agentic=not a.proposer_text,
                     proposer_timeout_s=a.proposer_timeout, harness_timeout_s=budget.max_wall_s,
                     latency_override=json.loads(a.latency_override) if a.latency_override else None,
                     heldout_game=heldout_game, memory_mode=a.memory_mode, heldout_game_kwargs=heldout_kwargs,
                     preflight=not a.no_preflight, grace_s=a.grace, qualify_seeds=not a.no_seed_check,
                     game_source=";".join(sources) or None, heldout_game_seeds=heldout_seeds,
                     heldout_game_adaptor=heldout_adaptor,
                     proposals_per_iteration=getattr(a, "proposals", 1), parallel=a.parallel,
                     finalists=getattr(a, "finalists", 3), reflect=not getattr(a, "no_reflect", False),
                     patience=getattr(a, "patience", 0),
                     more_games=specs[1:],
                     eval_spec={"system1": a.system1, "system2": a.system2, "effort": a.effort, "tier": tier_n,
                                "proposer": a.proposer, "proposer_effort": getattr(a, "proposer_effort", None)})
    # A calibration already measured for this archive is part of its task text: always reuse it (also
    # with --no-calibrate), measure only for a new archive.
    cal_path = archive.root / "calibration.json"
    calibration = json.loads(cal_path.read_text()) if cal_path.exists() else None
    if calibration is None and not a.no_calibrate and not (archive.root / "manifest.json").exists():
        try:
            calibration = calibrate(archive.root, system2=system2, system1=system1, game_factory=first.factory,
                                    game_kwargs=first.kwargs, seed=first.seeds_train[0])
        except ModelUnavailable as e:
            print(f"calibration: model unavailable ({e}); rerun the same command once it is back", file=sys.stderr)
            a.interrupted = f"model unavailable: {e}"
            return None
    names = [g.name for g in specs]
    kws = [g.kwargs for g in specs]
    task = tier.task_md(names if len(names) > 1 else names[0], budget, a.clock, kws if len(kws) > 1 else kws[0],
                        study=study, generality=heldout_game is not None or len(names) > 1,
                        deadline_s=budget.max_wall_s, memory_mode=a.memory_mode, calibration=calibration)
    loop = MetaHarnessLoop(
        archive=archive, game_factory=first.factory, game_kwargs=first.kwargs, task_md=task,
        contract_doc=tier.contract_doc(), proposer=_proposer(a.proposer, getattr(a, "proposer_effort", None) or a.effort, tier),
        system2=system2, system1=system1, config=cfg, game_name=first.name, game_adaptor=first.adaptor_dir,
    )
    print(f"archive: {archive.root}")
    try:
        best = loop.run()
    except SeedRefused as e:
        raise SystemExit(f"seed check failed: {e}") from None
    print(archive.write_leaderboard())
    if best:
        print(f"best: {best.dir}")
    a.interrupted = loop.interrupted
    return best


def cmd_loop(a: argparse.Namespace) -> int:
    load_dotenv()
    from .games.registry import adaptor_spec
    games = _resolve_games(a)
    for game, _ in games:
        spec = adaptor_spec(game)
        if spec is not None and spec.tier != f"tier{a.tier}":
            print(f"warning: adaptor {game!r} is {spec.tier} ({'real-time' if spec.realtime else 'turn-based'}) "
                  f"but the search runs --tier {a.tier}", file=sys.stderr)
    label = "+".join(_slug(g) for g, _ in games)
    exp_dir = Path(a.runs_dir) / (a.name or f"tier{a.tier}-{label}-{time.strftime('%Y%m%d-%H%M%S')}")
    _run_search(a, games, a.tier, _study_text(a, exp_dir, games), exp_dir)
    if getattr(a, "interrupted", None):
        print(f"interrupted: rerun with --name {exp_dir.name} --runs-dir {exp_dir.parent} (and the same "
              "settings) to resume", file=sys.stderr)
        return 3
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    """The general pipeline: study -> adapt -> search, resumable, under runs/<slug>/."""
    import ast
    import unicodedata
    load_dotenv()
    from .games.registry import adaptor_spec
    from .phases import prior_documents, run_adapt, run_study
    try:
        game_kwargs = json.loads(a.game_kwargs) if a.game_kwargs else None
    except ValueError as e:
        raise SystemExit(f"--game-kwargs is not valid JSON: {e}") from None
    if game_kwargs is not None and not isinstance(game_kwargs, dict):
        raise SystemExit("--game-kwargs must be a JSON object")
    run_dir = Path(a.runs_dir) / (a.name or _slug(a.game))
    run_dir.mkdir(parents=True, exist_ok=True)
    # The directory belongs to the game it was created for: another name with the same slug (or the
    # same --name) must not silently reuse that game's study and adaptor.
    meta, made_for = run_dir / "run.json", None
    if meta.exists():
        try:
            made_for = json.loads(meta.read_text(encoding="utf-8")).get("game")
        except (ValueError, AttributeError):
            raise SystemExit(f"{meta} is not a JSON object with the run's game; fix or delete it") from None
    elif (run_dir / "pipeline.log").exists():   # created before run.json existed: the log names the game
        m = re.search(r"\] run: game=('(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\")",
                      (run_dir / "pipeline.log").read_text(encoding="utf-8", errors="replace"))
        try:
            made_for = ast.literal_eval(m.group(1)) if m else None
        except (ValueError, SyntaxError):
            made_for = None

    def same_game(x: str) -> str:               # case / width / spacing never make another game
        return " ".join(unicodedata.normalize("NFKC", x).casefold().split())

    if made_for is not None and same_game(str(made_for)) != same_game(a.game):
        raise SystemExit(f"{run_dir} was created for --game {made_for!r}, not {a.game!r}: its study and adaptor "
                         f"belong to that game. Pass --name DIR for a separate run directory (or --game {made_for!r}).")
    if not meta.exists():
        meta.write_text(json.dumps({"game": a.game if made_for is None else made_for,
                                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
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
    if a.stop_after == "study":
        return 0
    rep = run_adapt(agent, run_dir, a.game, study, notes=a.notes, kwargs=game_kwargs, timeout_s=a.phase_timeout,
                    review=not (run_dir / "search").exists(), log=log)   # never under a running search
    if not rep.ok:
        log("run: adaptor not ready — " + (rep.error or "see adaptor/validation.md")[:500])
        needs = run_dir / "adaptor" / "NEEDS.md"
        if needs.exists():
            print("\n" + needs.read_text())
        return 2
    if a.stop_after == "adapt":
        return 0
    game = register_adaptor(run_dir / "adaptor", kwargs=game_kwargs)
    spec = adaptor_spec(game)
    tier_n = int(spec.tier.removeprefix("tier"))   # the adaptor's kind decides what the search can drive
    if tier_n != int(study["tier"]):
        log(f"search: the adaptor is tier {tier_n} ({'real-time' if spec.realtime else 'turn-based'}) but "
            f"study.json says tier {study['tier']}; searching tier {tier_n}")
    log(f"search: tier {tier_n} on adaptor game {game!r}")
    if not a.system1 or a.system1 == "auto":
        a.system1 = "jev" if tier_n == 2 else "none"
    prior = prior_documents(run_dir / "study", run_dir / "adaptor", run_dir / "search")
    best = _run_search(a, [(game, str((run_dir / "adaptor").resolve()))], tier_n, prior, run_dir / "search")
    if getattr(a, "interrupted", None):
        log(f"run: search interrupted ({a.interrupted[:200]}); rerun the same command to resume")
        return 3
    log("run: done" + (f" — best {best.dir.name}" if best else " — no eligible candidate"))
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    load_dotenv()
    _tier(a.tier)
    game = _resolve_game(a)
    budget = _episode_budget(a)
    game_kwargs, declared = _game_setup(a, game)
    seeds = _parse_seeds(a.seeds) if a.seeds is not None else list(declared.get("train") or [0, 1, 2])
    out = Path(a.out) if a.out else Path(a.runs_dir) / f"eval-{Path(a.harness).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    res = evaluate_harness(a.harness, lambda **k: make_game(game, **k), seeds,
                           system2=_system2(a.system2, a.effort), system1=_system1_for(a.system1, a.tier),
                           out_dir=out, clock_mode=a.clock, budget_per_episode=budget,
                           harness_timeout_s=budget.max_wall_s,
                           latency_override=json.loads(a.latency_override) if a.latency_override else None,
                           game_kwargs=game_kwargs, memory_mode=a.memory_mode, grace_s=a.grace, game_name=game)
    print(json.dumps({k: v for k, v in res.to_json().items() if k != "episodes"}, indent=2, default=str))
    for e in res.episodes:
        print(f"seed {e.seed}: score={e.score} steps={e.steps} stale={e.stale_ticks}/{e.stale_ticks + e.decided_ticks} "
              f"s1={e.system1_calls} s2={e.system2_calls} ended_by={e.ended_by} err={'yes' if e.error else 'no'}")
    print(f"out: {out}")
    return 1 if res.fatal or any(e.error for e in res.episodes) else 0


def cmd_try(a: argparse.Namespace) -> int:
    """Play one harness for one episode on a training seed of the archive it sits in, with that archive's
    game, clock, budget and System-1 (System-2 only with --astra): a proposer's experiment before the
    evaluation. Validation and held-out seeds are refused."""
    load_dotenv()
    harness = Path(a.harness).resolve()
    root = next((p for p in harness.parents if (p / "manifest.json").exists()), None)
    m = json.loads((root / "manifest.json").read_text()) if root else {}
    run, train = m.get("run") or {}, m.get("seeds_train")
    if not run or "games" in m or not isinstance(train, list):
        raise SystemExit(f"{harness}: try needs a single-game search archive made by `metaharness run` or `loop`")
    seed = train[0] if a.seed is None else a.seed
    if seed not in train:
        raise SystemExit(f"seed {seed} is not a training seed of this archive {train}: validation and held-out "
                         "seeds are the evaluator's")
    kwargs = m.get("game_kwargs") or {}
    key = register_adaptor(Path(run["adaptor"]), kwargs=kwargs or None) if run.get("adaptor") else run["game"]
    budget = {**(m.get("budget") or {})}
    budget["max_wall_s"] = wall = min(float(budget.get("max_wall_s") or 900.0), a.minutes * 60.0)
    out = harness.parent / "tries" / f"seed{seed}-{time.strftime('%H%M%S')}"
    tier_n = int(run.get("tier") or 2)
    res = evaluate_harness(str(harness), lambda **k: make_game(key, **k), [seed],
                           system2=_system2((run.get("system2") or "none") if a.astra else "none", run.get("effort")),
                           system1=_system1_for(run.get("system1") or "none", tier_n), out_dir=out,
                           clock_mode=m.get("clock_mode") or "wall", budget_per_episode=Budget(**budget),
                           harness_timeout_s=wall, latency_override=m.get("latency_override"), game_kwargs=kwargs,
                           memory_mode=m.get("memory_mode") or "episode", grace_s=float(m.get("grace_s") or DEFAULT_GRACE_S),
                           game_name=run.get("game") or key)
    for e in res.episodes:
        print(f"seed {e.seed}: score={e.score} steps={e.steps} stale={e.stale_ticks}/{e.stale_ticks + e.decided_ticks} "
              f"s1={e.system1_calls} s2={e.system2_calls} ended_by={e.ended_by}" + (f"\nerror: {e.error}" if e.error else ""))
    if res.fatal:
        print(f"fatal: {res.fatal}")
    print(f"trace: {out}")
    return 1 if res.fatal or any(e.error for e in res.episodes) else 0


def _archive_setup(a: argparse.Namespace) -> tuple[dict, Path, str, dict]:
    """(manifest, archive root, registered game key, game kwargs) of the archive a proposer works in:
    the game and the profiles come from manifest.json, so the proposer's tools evaluate exactly what
    the search evaluates."""
    load_dotenv()
    root = Path(a.archive).resolve()
    mp = root / "manifest.json"
    if not mp.exists():
        raise SystemExit(f"{mp} not found: run this from the search archive (or pass --archive DIR)")
    manifest = json.loads(mp.read_text())
    run = manifest.get("run") or {}
    if not run.get("game"):
        raise SystemExit("this archive's manifest names no game (older protocol); the tools need a current archive")
    game = register_adaptor(Path(run["adaptor"])) if run.get("adaptor") else run["game"]
    return manifest, root, game, dict(manifest.get("game_kwargs") or {})


def cmd_preflight(a: argparse.Namespace) -> int:
    """Proposer tool: the evaluator's pre-flight of a harness, with the archive's game and budgets."""
    from .core.runner import preflight_harness
    manifest, root, game, kwargs = _archive_setup(a)
    b = manifest.get("budget") or {}
    budget = Budget(max_system2_calls=b.get("max_system2_calls", 40), max_system1_calls=b.get("max_system1_calls", 4000),
                    max_wall_s=b.get("max_wall_s", 900.0))
    seeds = manifest.get("seeds_train") or [0]
    rep = preflight_harness(Path(a.harness), lambda **k: make_game(game, **k), game_kwargs=kwargs,
                            tier2=manifest.get("models", {}).get("system1") is not None, budget=budget, seed=seeds[0])
    print(json.dumps(rep, indent=2, default=str))
    return 0 if rep["ok"] else 1


def cmd_dev_eval(a: argparse.Namespace) -> int:
    """Proposer tool: a real-model evaluation of a harness on the archive's dev seeds (never scored)."""
    manifest, root, game, kwargs = _archive_setup(a)
    run = manifest["run"]
    dev = manifest.get("seeds_dev") or []
    seeds = _parse_seeds(a.seeds) if a.seeds else dev
    if not seeds:
        raise SystemExit("this archive has no dev seeds (--dev-seeds of the search); nothing to evaluate on")
    bad = set(seeds) - set(dev)
    if bad:
        raise SystemExit(f"seeds {sorted(bad)} are not dev seeds of this archive (dev seeds: {dev})")
    b = manifest["budget"]
    budget = Budget(max_system2_calls=b["max_system2_calls"], max_system1_calls=b["max_system1_calls"],
                    max_wall_s=b["max_wall_s"])
    harness = Path(a.harness).resolve()
    out = Path(a.out) if a.out else harness.parent / "dev" / time.strftime("%Y%m%d-%H%M%S")
    tier_n = int(run.get("tier") or 1)
    res = evaluate_harness(harness, lambda **k: make_game(game, **k), seeds,
                           system2=_system2(run["system2"], run.get("effort")),
                           system1=_system1_for(run.get("system1"), tier_n), out_dir=out,
                           clock_mode=manifest.get("clock_mode", "wall"), budget_per_episode=budget,
                           harness_timeout_s=manifest.get("harness_timeout_s", budget.max_wall_s),
                           latency_override=manifest.get("latency_override"), game_kwargs=kwargs,
                           memory_mode=manifest.get("memory_mode", "episode"), grace_s=manifest.get("grace_s", DEFAULT_GRACE_S),
                           game_name=game, parallel=len(seeds))
    for e in res.episodes:
        print(f"seed {e.seed}: score={e.score} steps={e.steps} s2={e.system2_calls} failed_s2={e.failed_s2} "
              f"ended_by={e.ended_by} err={'yes' if e.error else 'no'}")
    print(f"mean score: {res.mean_score}  fatal: {res.fatal}\nout: {out} (eval.json, digest.md, trace_seed*.jsonl)")
    return 1 if res.fatal or any(e.error for e in res.episodes) else 0


def cmd_validate_adaptor(a: argparse.Namespace) -> int:
    load_dotenv()
    from .games.adaptor import validate_adaptor
    rep = validate_adaptor(Path(a.path), kwargs=json.loads(a.game_kwargs) if a.game_kwargs else None,
                           expected_tier=getattr(a, "tier", None))   # if the subcommand defines --tier
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
        sp.add_argument("--system1", default="auto",
                        help="System-1 profile from metaharness/models.toml (jev) | heuristic | none | "
                             "auto (jev for tier 2, none for tier 1)")
        sp.add_argument("--effort", default=None, help="override the profile's reasoning effort (low|medium|high|xhigh|max)")
        sp.add_argument("--clock", default="wall", choices=["wall", "virtual"])
        sp.add_argument("--memory-mode", default="episode", choices=["episode", "shared"],
                        help="episode: independent cold starts; shared: adaptation across ordered seeds")
        sp.add_argument("--latency-override", default="", help='JSON e.g. {"system2": 3.0, "system1": 0.3} (virtual clock)')
        sp.add_argument("--seeds", default=None,
                        help="training seeds (default: the adaptor's declared split, else 0,1,2)")
        sp.add_argument("--s2-budget", type=int, default=40)
        sp.add_argument("--s1-budget", type=int, default=4000)
        sp.add_argument("--wall-budget", type=float, default=900.0,
                        help="per-episode wall-clock deadline in s (the larger of this and --harness-timeout)")
        sp.add_argument("--harness-timeout", type=float, default=600.0,
                        help="see --wall-budget: the episode deadline is the larger of the two")
        sp.add_argument("--grace", type=float, default=DEFAULT_GRACE_S,
                        help="seconds a harness has to return after the deadline (and spawned work after "
                             "run_episode returns) before it counts as an error")
        sp.add_argument("--parallel", type=int, default=1,
                        help="concurrent episodes (one child process each; a shared-memory evaluation keeps a "
                             "game's seeds in one) and concurrent proposers")
        sp.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    def loop_opts(sp):
        sp.add_argument("--proposer", default="astra", help="System-2 profile for the proposer | scripted")
        sp.add_argument("--proposer-effort", default=None, help="reasoning effort of the proposer (default: --effort)")
        sp.add_argument("--proposer-text", action="store_true", help="proposer answers in text (no tools)")
        sp.add_argument("--proposer-timeout", type=float, default=1800.0)
        sp.add_argument("--iters", type=int, default=3, help="generations of proposals this archive should have "
                                                               "(at most, with --patience)")
        sp.add_argument("--patience", type=int, default=0,
                        help="stop proposing once the best candidate has not improved for this many generations")
        sp.add_argument("--proposals", type=int, default=1,
                        help="candidates proposed per generation (each with its own parent / mode directive)")
        sp.add_argument("--val-seeds", default="",
                        help="validation seeds: the top --finalists training candidates are re-evaluated on them "
                             "and the final pick ranks training + validation together")
        sp.add_argument("--finalists", type=int, default=3)
        sp.add_argument("--dev-seeds", default=None,
                        help="seeds the proposer may evaluate its draft on itself (default: the adaptor's "
                             "declared dev split, else none)")
        sp.add_argument("--dev-evals", type=int, default=2, help="dev evaluations allowed per candidate")
        sp.add_argument("--no-proposer-tools", action="store_true",
                        help="do not tell the proposer about the pre-flight / dev-eval commands")
        sp.add_argument("--no-reflect", action="store_true", help="no lessons.md reflection after each generation")
        sp.add_argument("--heldout", default=None,
                        help="held-out seeds, same game (default: the adaptor's declared split, else 100,101,102)")
        sp.add_argument("--heldout-game", default="",
                        help="held-out GAME for the generality test (registered name or adaptor directory); "
                             "setting it adds the generality requirement to the task")
        sp.add_argument("--heldout-game-kwargs", default="", help="JSON kwargs for the held-out game")
        sp.add_argument("--no-preflight", action="store_true", help="skip the stand-in dry run + repair round")
        sp.add_argument("--no-calibrate", action="store_true",
                        help="do not measure model latency for the task spec when a new archive starts")
        sp.add_argument("--no-seed-check", action="store_true",
                        help="do not reset every seed once before the search")
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

    lp = sub.add_parser("loop", help="run the harness search on registered games and/or adaptors")
    lp.add_argument("--game", default="", help="registered game name(s), comma-separated for several training "
                                               "games (test beds: snake, lanerunner, rulepuzzle, arc3)")
    lp.add_argument("--adaptor", action="append", default=[],
                    help="generated adaptor directory (runs/<game>/adaptor); repeat for several training games")
    lp.add_argument("--study", default="", help="study directory whose documents become prior knowledge")
    search_opts(lp, tier_required=True)
    loop_opts(lp)
    lp.set_defaults(fn=cmd_loop)

    ep = sub.add_parser("eval", help="evaluate one harness file")
    ep.add_argument("--game", default="")
    ep.add_argument("--adaptor", action="append", default=[])
    search_opts(ep, tier_required=True)
    ep.add_argument("--harness", required=True)
    ep.add_argument("--out", default="")
    ep.set_defaults(fn=cmd_eval)

    tp = sub.add_parser("try", help="play a harness for one episode on a training seed of its archive")
    tp.add_argument("harness")
    tp.add_argument("--seed", type=int, default=None, help="a training seed of the archive (default: the first)")
    tp.add_argument("--minutes", type=float, default=5.0, help="episode wall-clock cap (at most the evaluation's)")
    tp.add_argument("--astra", action="store_true", help="System-2 on (default off: tries spend no Astra calls)")
    tp.set_defaults(fn=cmd_try)

    pp = sub.add_parser("preflight", help="(proposer tool) the evaluator's pre-flight of a harness, in an archive")
    pp.add_argument("harness")
    pp.add_argument("--archive", default=".", help="search archive with manifest.json (default: cwd)")
    pp.set_defaults(fn=cmd_preflight)

    dp = sub.add_parser("dev-eval", help="(proposer tool) evaluate a harness on the archive's dev seeds with the real models")
    dp.add_argument("harness")
    dp.add_argument("--archive", default=".")
    dp.add_argument("--seeds", default="", help="a subset of the dev seeds (default: all)")
    dp.add_argument("--out", default="", help="output directory (default: <harness dir>/dev/<timestamp>)")
    dp.set_defaults(fn=cmd_dev_eval)

    vp = sub.add_parser("validate-adaptor", help="run the generic adaptor smoke test")
    vp.add_argument("path")
    vp.add_argument("--game-kwargs", default="")
    vp.set_defaults(fn=cmd_validate_adaptor)

    a = p.parse_args(argv)
    # SIGTERM (kill, cluster preemption) unwinds like Ctrl-C. The children go first: worker threads wait
    # on CLI agents and evaluation processes, and the main thread's pool would otherwise wait for them.
    previous = None
    if threading.current_thread() is threading.main_thread():
        def _terminate(signum, frame):
            from .core.evalworker import kill_children
            from .models._cli import kill_all
            kill_all()
            kill_children()
            raise SystemExit(128 + signum)
        previous = (signal.signal(signal.SIGTERM, _terminate), signal.signal(signal.SIGINT, _terminate))
    try:
        return a.fn(a)
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous[0])
            signal.signal(signal.SIGINT, previous[1])


if __name__ == "__main__":
    sys.exit(main())

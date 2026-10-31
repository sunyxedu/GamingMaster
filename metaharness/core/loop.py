"""The outer loop: propose -> evaluate -> archive -> repeat; then validate and hold out.

A *generation* proposes ``proposals_per_iteration`` candidates at once, each with a directive
(parent and kind of change, ``core/generation.py``), pre-flights them, drops near-duplicates,
evaluates the rest — when the models are named by profile (``eval_spec``) every episode is its own
child process, ``parallel`` at a time (independent-memory evaluations; shared memory keeps a game's
seeds in one process, in order) — and records a reflection (``lessons.md``). After the last generation the
top ``finalists`` are re-evaluated on the validation seeds and the final pick ranks training +
validation together (``champion.json``); only that candidate is tested on the held-out seeds / game.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import hashlib
import json
import math
import os
import sys
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..games.base import close_game
from ..models._cli import STOPPING
from ..models.base import ModelUnavailable, System1Model, System2Model, TraceSink
from . import evalworker, generation
from .archive import Archive, Candidate, combine_evals, ineligibility, write_atomic
from .contract import Budget
from .proposer import propose, repair, tools_text
from .runner import DEFAULT_GRACE_S, PROTOCOL_VERSION, evaluate_harness, preflight_harness


ABLATION = "ablation-no-models"


class SeedRefused(ValueError):
    """A configured seed cannot be played (the game refuses it); choose other seeds."""


@dataclass
class GameSpec:
    """One training game of a search; seeds that are None fall back to the loop's."""
    name: Optional[str]
    factory: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    seeds_train: Optional[list[int]] = None
    seeds_val: Optional[list[int]] = None
    seeds_heldout: Optional[list[int]] = None
    adaptor_dir: Optional[str] = None            # child processes register this directory


@dataclass
class LoopConfig:
    n_iterations: int = 3
    seeds_train: list[int] = field(default_factory=lambda: [0, 1, 2])
    seeds_heldout: list[int] = field(default_factory=lambda: [100, 101, 102])
    clock_mode: str = "wall"
    budget: Budget = field(default_factory=Budget)
    latency_override: Optional[dict[str, float]] = None
    proposer_agentic: bool = True
    proposer_timeout_s: float = 1800.0
    harness_timeout_s: float = 600.0
    stop_on_model_unavailable: bool = True
    heldout_game: Optional[str] = None                       # cross-game generality test
    heldout_game_kwargs: dict[str, Any] = field(default_factory=dict)
    memory_mode: str = "episode"
    preflight: bool = True                                   # cheap dry run + one repair before evaluating
    preflight_timeout_s: float = 60.0
    grace_s: float = DEFAULT_GRACE_S
    qualify_seeds: bool = True                               # reset every seed once before evaluating
    game_source: Optional[str] = None                        # adaptor source hash, pinned by the manifest
    heldout_game_seeds: Optional[list[int]] = None           # a held-out adaptor's own declared seeds
    proposals_per_iteration: int = 1                         # candidates proposed per generation
    parallel: int = 1                                        # concurrent proposers / evaluations
    seeds_val: list[int] = field(default_factory=list)       # validation block for the final pick
    finalists: int = 3                                       # candidates re-evaluated on the validation seeds
    reflect: bool = True                                     # lessons.md after every generation
    eval_spec: Optional[dict[str, Any]] = None               # model profiles + tier: evaluate in child processes
    more_games: list[GameSpec] = field(default_factory=list)  # training games besides the primary one
    heldout_game_adaptor: Optional[str] = None               # its directory, for child processes
    patience: int = 0                                        # stop once the best is this many generations old
    ablate: bool = True                                      # the final pick without runtime models (reported)
    seeds_dev: list[int] = field(default_factory=list)       # the proposer's own dev evaluations (never scored)
    proposer_dev_evals: int = 0                              # ... how many per candidate (0: no dev tool)
    proposer_tools: bool = True                              # pre-flight (and dev-eval) commands in the prompt


def _unavailable(e: Optional[dict[str, Any]]) -> bool:
    return bool(e) and str(e.get("fatal") or "").startswith("model unavailable")


_ENV_ERRORS = ("game reset failed", "game construction failed", "game hung", "invalid trial")


def _env_failed(errors: list[str], n: int) -> bool:
    """Every episode failed because of the game or its infrastructure, not the harness."""
    return n > 0 and len(errors) == n and all(str(e).split(": ", 1)[-1].startswith(_ENV_ERRORS)
                                              or str(e).startswith(_ENV_ERRORS) for e in errors)


def _interrupted(e: Optional[dict[str, Any]]) -> bool:
    """An evaluation the search stopped on (model outage, game down): redone on resume."""
    return _unavailable(e) or (bool(e) and _env_failed(e.get("errors") or [], e.get("n_episodes", 0)))


SIGNATURE_STEPS = 300           # games open with intros that ignore input (one needed ~240 steps to diverge)
SIGNATURE_TIME_S = 30.0         # a signature cut short by this is not used: it could alias distinct seeds


def _episode_signature(game: Any, seed: int) -> Optional[str]:
    """Hash of the observations under a fixed action script from reset(seed) (fields named like a seed
    dropped: they label the episode, they are not its state); None if the game cannot play it, the
    script did not finish within SIGNATURE_TIME_S, or the episode ended before SIGNATURE_STEPS (a
    scripted player that dies early cannot see the seeds diverge later: two seeds sharing a first
    obstacle would look identical)."""
    from ..games.adaptor import episode_state as state
    try:
        names = list(game.action_spec.names)
        step = game.tick if hasattr(game, "tick") else game.step
        h = hashlib.sha256(state(game.observe()).encode())
        t_end = time.monotonic() + SIGNATURE_TIME_S
        complete = True
        for i in range(SIGNATURE_STEPS):
            if game.done:
                complete = False
                break
            if time.monotonic() > t_end:
                complete = False
                break
            step(names[(i // 3) % len(names)])
            h.update(state(game.observe()).encode())
        game.reset(seed)                     # leave it as the seed check found it
        return h.hexdigest()[:16] if complete else None
    except Exception:  # noqa: BLE001 — a signature is a check, never a reason to refuse
        return None


def _jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _slug(name: Optional[str]) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "game")).strip("_") or "game"


def _exposed(d: Path) -> bool:
    """A held-out / validation stage counts as run (test information is in the archive) once any of
    its episodes started — an eval.json with episodes, or any trace written — whatever stopped it."""
    ej = d / "eval.json"
    if ej.exists():
        e = json.loads(ej.read_text() or "{}")
        if e.get("n_episodes") or not _unavailable(e):
            return True
    return any(p.stat().st_size > 0 for p in d.rglob("trace_seed*.jsonl"))


def _exposed_dirs(c: Candidate) -> list[Path]:
    return [d for d in list(c.dir.glob("heldout*")) + [c.dir / "validation"] if d.is_dir()]


@contextlib.contextmanager
def _archive_lock(root) -> Iterator[None]:
    """One search process per archive (a second loop, or an agent orphaned by Ctrl-C, would write into it)."""
    try:
        import fcntl
    except ImportError:            # not POSIX: no lock
        yield
        return
    import errno
    fh = open(root / ".lock", "a+")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise RuntimeError(f"cannot lock the archive {root} ({e}); is it on a filesystem "
                                   "without flock support?") from None
            raise RuntimeError(f"another Meta-Harness process is using the archive {root}") from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def _pool(n: int) -> cf.ThreadPoolExecutor:
    return cf.ThreadPoolExecutor(max_workers=max(1, n))


class MetaHarnessLoop:
    def __init__(
        self,
        *,
        archive: Archive,
        game_factory: Callable[..., Any],
        game_kwargs: dict[str, Any],
        task_md: str,
        contract_doc: str,
        proposer: System2Model,
        system2: Optional[System2Model],
        system1: Optional[System1Model],
        config: LoopConfig,
        proposer_extra: str = "",
        game_name: Optional[str] = None,
        game_adaptor: Optional[str] = None,
    ) -> None:
        self.archive = archive
        self.game_factory = game_factory
        self.game_kwargs = game_kwargs
        self.task_md = task_md
        self.contract_doc = contract_doc
        self.proposer = proposer
        self.system2 = system2
        self.system1 = system1
        self.cfg = config
        self.proposer_extra = proposer_extra
        self.game_name = game_name
        self.games: list[GameSpec] = [GameSpec(game_name, game_factory, game_kwargs, adaptor_dir=game_adaptor)]
        self.games += list(config.more_games)
        self.interrupted: Optional[str] = None
        for g in self.games:
            overlap = set(g.seeds_train or config.seeds_train) & (set(g.seeds_heldout or config.seeds_heldout)
                                                                 | set(g.seeds_val or config.seeds_val))
            if overlap:
                raise ValueError(f"training seeds overlap held-out / validation seeds: {sorted(overlap)}")
        if config.n_iterations > self.proposals_done() // max(1, config.proposals_per_iteration) and any(
                _exposed(d) for c in archive.candidates() for d in _exposed_dirs(c)):
            raise ValueError("This archive contains held-out results (or validation results); start a new search archive.")
        manifest = {
            "protocol_version": PROTOCOL_VERSION, "task": task_md, "seeds_train": config.seeds_train,
            "clock_mode": config.clock_mode, "latency_override": config.latency_override,
            # compared JSON-normalised (a tuple in EVAL_KWARGS is a list once stored); games are
            # still built from the caller's kwargs
            "game_kwargs": _jsonish(game_kwargs), "memory_mode": config.memory_mode,
            "budget": {k: getattr(config.budget, k) for k in
                       ("max_system1_calls", "max_system2_calls", "max_wall_s")},
            "harness_timeout_s": config.harness_timeout_s, "grace_s": config.grace_s,
            "models": {k: None if m is None else {
                "name": m.name, "model": getattr(m, "model", None),
                "player_tools": getattr(m, "player_tools", None),
                "reasoning_effort": getattr(m, "reasoning_effort", None)}
                for k, m in (("system1", system1), ("system2", system2), ("proposer", proposer))},
        }
        if config.game_source:
            manifest["game_source"] = config.game_source
        # Validation and held-out seeds are not pinned: they never touch the training scores, and each
        # candidate's validation result (like champion.json) is written once.
        if config.eval_spec:                 # how to rebuild the models / game outside this process
            manifest["run"] = {**config.eval_spec, "game": game_name, "adaptor": game_adaptor}
        if config.seeds_dev:                 # the proposer's dev-eval tool reads them; never scored
            manifest["seeds_dev"] = list(config.seeds_dev)
            for g in self.games:
                overlap = set(config.seeds_dev) & (set(g.seeds_train or config.seeds_train)
                                                   | set(g.seeds_heldout or config.seeds_heldout)
                                                   | set(g.seeds_val or config.seeds_val))
                if overlap:
                    raise ValueError(f"dev seeds overlap training / validation / held-out seeds: {sorted(overlap)}")
        if len(self.games) > 1:
            manifest["games"] = [{"name": g.name, "kwargs": _jsonish(g.kwargs), "seeds_train": g.seeds_train}
                                 for g in self.games]
        manifest_path = archive.root / "manifest.json"
        scored = any(c.eval() is not None for c in archive.candidates())
        self.task_changed = False
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            old.pop("seeds_val", None)                 # pinned by earlier versions; not part of the protocol
            for g in old.get("games") or []:
                for k in ("seeds_val", "seeds_heldout"):
                    g.pop(k, None)
            changed = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
            # The task text is what proposers were told, not how scores were measured: a resumed search
            # may carry an updated task (every candidate's prompt.md records what its proposer saw).
            # Dev seeds are never scored either.
            self.task_changed = "task" in changed
            changed = [k for k in changed if k not in ("task", "seeds_dev")]
            if changed and scored:
                raise ValueError(f"Search configuration changed ({', '.join(changed)}) for the archive "
                                 f"{archive.root}: its scores were measured differently. Search in a new "
                                 "archive (loop: another --name; run: move that search/ directory away).")
        elif scored:
            raise ValueError("Legacy archive has no protocol manifest; start a new search archive.")
        self._manifest = manifest           # written by run() once the seeds are known to work
        self.log_path = archive.root / "loop.log"

    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    # -- seeds per stage ----------------------------------------------------------------------
    def _tools(self) -> str:
        """The proposer's own tools (pre-flight, dev-eval), when it works agentically and the archive
        carries what the tools need (``manifest["run"]``)."""
        if not self.cfg.proposer_tools or not self.cfg.proposer_agentic or not self.cfg.eval_spec:
            return ""
        from ..config import REPO_ROOT
        return tools_text(sys.executable, str(REPO_ROOT), self.cfg.seeds_dev, self.cfg.proposer_dev_evals)

    def _seeds(self, stage: str, games: Optional[list[GameSpec]] = None) -> dict[str, list[int]]:
        out = {}
        for g in games or self.games:
            own = {"train": g.seeds_train, "val": g.seeds_val, "heldout": g.seeds_heldout}[stage]
            default = {"train": self.cfg.seeds_train, "val": self.cfg.seeds_val, "heldout": self.cfg.seeds_heldout}[stage]
            out[_slug(g.name)] = list(own if own is not None else default)
        return out

    # -- evaluation ---------------------------------------------------------------------------
    def _job(self, cand: Candidate, game: GameSpec, seeds: list[int], out: Path, label: Optional[str],
             no_models: bool = False) -> evalworker.EvalJob:
        b = self.cfg.budget
        models = dict(self.cfg.eval_spec or {})
        if no_models:      # every call fails, as in an outage, with the same budget (tier 1 has no System-1)
            s1, s2 = self._silent_latency()
            models.update(system1=f"silent:{s1:g}" if self.system1 is not None else "none", system2=f"silent:{s2:g}")
        return evalworker.EvalJob(
            harness_path=str(cand.harness_path), out_dir=str(out), game=str(game.name), seeds=list(seeds),
            adaptor_dir=game.adaptor_dir, game_kwargs=_jsonish(game.kwargs), models=models,
            clock_mode=self.cfg.clock_mode,
            budget={"max_system1_calls": b.max_system1_calls, "max_system2_calls": b.max_system2_calls,
                    "max_wall_s": b.max_wall_s},
            harness_timeout_s=self.cfg.harness_timeout_s, memory_mode=self.cfg.memory_mode,
            grace_s=self.cfg.grace_s, latency_override=self.cfg.latency_override, game_name=label)

    def _eval_children(self, units: list[tuple[int, Candidate, GameSpec, list[int], Path, Optional[str]]],
                       no_models: bool = False) -> list[dict[str, Any]]:
        """Every unit's episodes in child processes, ``parallel`` at a time: one process per seed when
        episodes are independent (a game's traces end up in the unit's directory, its eval.json merged),
        else one per unit."""
        jobs: list[evalworker.EvalJob] = []
        owner: list[tuple[int, Optional[int]]] = []
        for k, (_, cand, game, seeds, out, label) in enumerate(units):
            split = self.cfg.memory_mode == "episode" and len(seeds) > 1
            for s in (seeds if split else [None]):
                where = out / "seeds" / f"seed-{s}" if s is not None else out
                jobs.append(self._job(cand, game, [s] if s is not None else seeds, where, label, no_models))
                owner.append((k, s))
        results = evalworker.run_jobs(jobs, self.cfg.parallel)
        evals = []
        for k, (_, cand, game, seeds, out, label) in enumerate(units):
            parts = [(s, r) for (kk, s), r in zip(owner, results) if kk == k]
            if parts[0][0] is None:
                evals.append(parts[0][1])
                continue
            for s, _ in parts:                      # traces where the proposer reads them
                for t in (out / "seeds" / f"seed-{s}").glob("trace_seed*.jsonl"):
                    os.replace(t, out / t.name)
            merged = evalworker.merge_seed_evals([r for _, r in parts], seeds)
            write_atomic(out / "eval.json", json.dumps(merged, indent=2, default=str))
            evals.append(merged)
        return evals

    def _silent_latency(self) -> tuple[float, float]:
        """(System-1, System-2) latency of the ablation's failing calls: the calibrated medians on the wall clock
        (defaults without a calibration), none on a virtual clock, where a failed call is charged as it ran."""
        if self.cfg.clock_mode != "wall":
            return 0.0, 0.0
        try:
            cal = json.loads((self.archive.root / "calibration.json").read_text())
            return float(cal["system1"]["latency_s"]), float(cal["system2_short"]["latency_s"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0.25, 30.0

    def _eval_unit(self, cand: Candidate, game: GameSpec, seeds: list[int], out: Path, label: Optional[str],
                   no_models: bool = False) -> dict[str, Any]:
        """One candidate on one game in this process (models given as objects: tests, scripted runs)."""
        from ..models.mock import SilentModel
        s1, s2 = self._silent_latency()
        res = evaluate_harness(
            cand.harness_path, game.factory, seeds, system2=SilentModel(s2) if no_models else self.system2,
            system1=(SilentModel(s1) if self.system1 is not None else None) if no_models else self.system1, out_dir=out,
            clock_mode=self.cfg.clock_mode, budget_per_episode=self.cfg.budget,
            latency_override=self.cfg.latency_override, harness_timeout_s=self.cfg.harness_timeout_s,
            game_kwargs=game.kwargs, memory_mode=self.cfg.memory_mode, grace_s=self.cfg.grace_s, game_name=label)
        return res.to_json()

    def evaluate_many(self, requests: list[tuple[Candidate, dict[str, list[int]], str, list[GameSpec]]],
                      no_models: bool = False) -> list[dict[str, Any]]:
        """Evaluate several (candidate, seeds per game, tag, games) requests, ``parallel`` game
        evaluations at a time; writes each request's eval.json (combined over its games) and
        returns the eval dicts in request order."""
        units: list[tuple[int, Candidate, GameSpec, list[int], Path, Optional[str]]] = []
        for i, (cand, seeds, tag, games) in enumerate(requests):
            base = cand.dir / tag if tag else cand.dir
            for g in games:
                out = base / f"game-{_slug(g.name)}" if len(games) > 1 else base
                label = g.name if (tag or len(games) > 1 or g is not self.games[0]) else self.game_name
                units.append((i, cand, g, seeds[_slug(g.name)], out, label))
        if self.cfg.eval_spec is not None:
            evals = self._eval_children(units, no_models)
        else:
            with _pool(self.cfg.parallel) as ex:
                evals = list(ex.map(lambda u: self._eval_unit(u[1], u[2], u[3], u[4], u[5], no_models), units))
        out: list[dict[str, Any]] = []
        for i, (cand, seeds, tag, games) in enumerate(requests):
            mine = [(u, e) for u, e in zip(units, evals) if u[0] == i]
            if len(games) > 1:
                e = combine_evals({str(u[2].name): ev for u, ev in mine}, sorted({s for u, _ in mine for s in u[3]}))
                (cand.dir / tag if tag else cand.dir).mkdir(parents=True, exist_ok=True)
                write_atomic((cand.dir / tag if tag else cand.dir) / "eval.json", json.dumps(e, indent=2, default=str))
            else:
                e = mine[0][1]
            out.append(e)
            self._log_eval(cand, tag, e)
        self.archive.write_leaderboard()
        return out

    def _log_eval(self, cand: Candidate, tag: str, e: dict[str, Any]) -> None:
        eps = [ep for g in (e["per_game"].values() if isinstance(e.get("per_game"), dict) else [e])
               for ep in g.get("episodes", [])]
        ended: dict[str, int] = {}
        for ep in eps:
            ended[ep.get("ended_by") or "?"] = ended.get(ep.get("ended_by") or "?", 0) + 1
        self._log(f"eval {cand.dir.name}{'/' + tag if tag else ''}: mean={e.get('mean_score')} "
                  f"stale={e.get('stale_fraction')} fatal={e.get('fatal')} errors={len(e.get('errors') or [])} "
                  f"failed s1/s2={e.get('failed_s1', 0)}/{e.get('failed_s2', 0)} "
                  f"ended: {', '.join(f'{k} {v}' for k, v in ended.items()) or '–'}")

    def evaluate(self, cand: Candidate, seeds: list[int], tag: str = "",
                 game_factory: Optional[Callable[..., Any]] = None,
                 game_kwargs: Optional[dict[str, Any]] = None, game_name: Optional[str] = None,
                 game_adaptor: Optional[str] = None) -> dict[str, Any]:
        """Evaluate on ``seeds`` of the training games, or of the one game given (held-out game)."""
        if game_factory is not None:
            games = [GameSpec(game_name, game_factory, game_kwargs or {}, adaptor_dir=game_adaptor)]
            return self.evaluate_many([(cand, {_slug(game_name): list(seeds)}, tag, games)])[0]
        per_game = {_slug(g.name): list(seeds) for g in self.games}
        return self.evaluate_many([(cand, per_game, tag, self.games)])[0]

    def _stop_for(self, e: dict[str, Any], *, env_retry: bool = False) -> bool:
        """Stop the search (resumable) on a model outage or when the game itself is down. A candidate
        whose evaluation already failed that way once is not a reason to stop again: a hang or an
        invalid trial its own play provokes would otherwise block every resume."""
        fatal = str(e.get("fatal") or "")
        errors = [str(x) for x in (e.get("errors") or [])]
        if fatal.startswith("model unavailable") and self.cfg.stop_on_model_unavailable:
            self.interrupted = fatal
        elif not env_retry and _env_failed(errors, int(e.get("n_episodes") or 0)):
            self.interrupted = "the game failed in every episode: " + errors[0][:300]
        return self.interrupted is not None

    def proposals_done(self) -> int:
        """Proposals the archive holds (a proposer outage or an empty directory left by a killed
        proposer does not count): --iters generations of proposals_per_iteration is their target."""
        n = 0
        for c in self.archive.candidates():
            if "-" in c.dir.name:                                   # a labelled seed baseline
                continue
            e = c.eval()
            if e is not None and str(e.get("fatal") or "").startswith("proposer unavailable"):
                continue
            if c.harness_path.exists() or e is not None:
                n += 1
        return n

    # -- one candidate's life before evaluation ---------------------------------------------------
    def preflight_and_repair(self, cand: Candidate) -> bool:
        """Dry-run the candidate with stand-in models; on failure ask the proposer to repair once.
        A ModelUnavailable from the repair propagates (the search stops)."""
        for attempt in (0, 1):
            if STOPPING.is_set():
                return False
            rep = preflight_harness(cand.harness_path, self.game_factory, game_kwargs=self.game_kwargs,
                                    tier2=self.system1 is not None, timeout_s=self.cfg.preflight_timeout_s,
                                    budget=self.cfg.budget, seed=self._seeds("train")[_slug(self.games[0].name)][0])
            (cand.dir / f"preflight_{attempt}.json").write_text(json.dumps(rep, indent=2, default=str))
            if rep["ok"]:
                note = ("; warnings: " + " | ".join(rep["warnings"])) if rep.get("warnings") else ""
                self._log(f"preflight {cand.dir.name}: ok" + (" after repair" if attempt else "") + note)
                return True
            self._log(f"preflight {cand.dir.name}: FAILED — {rep['error'][:200].replace(chr(10), ' ')}")
            if attempt == 1 or not rep.get("repairable", True):
                return False
            ptrace = TraceSink(cand.dir / "proposer_trace.jsonl")
            try:
                repair(self.proposer, cand, rep["error"], agentic=self.cfg.proposer_agentic, trace=ptrace,
                       timeout_s=min(900.0, self.cfg.proposer_timeout_s), full_access=bool(self._tools()))
            finally:
                ptrace.close()
        return False

    def _preflight_all(self, cands: list[Candidate]) -> None:
        """Pre-flight (and repair) several candidates at once; an outage of the proposer interrupts."""
        if not self.cfg.preflight:
            return
        todo = [c for c in cands if c.harness_path.exists() and "-" not in c.dir.name]

        def one(c: Candidate) -> Optional[str]:
            try:
                self.preflight_and_repair(c)
            except ModelUnavailable as exc:
                return f"model unavailable: {exc}"
            return None

        with _pool(self.cfg.parallel) as ex:
            for problem in ex.map(one, todo):
                if problem and not self.interrupted:
                    self.interrupted = problem

    def _new_candidate_eval(self, cand: Candidate, fatal: str) -> None:
        write_atomic(cand.dir / "eval.json", json.dumps({"fatal": fatal, "mean_score": None, "n_episodes": 0,
                                                         "game": self.game_name}))

    def _propose_one(self, cand: Candidate, directive: dict[str, Any], siblings: tuple[str, ...],
                     iteration: tuple[int, int]) -> Optional[str]:
        """Run one proposer; returns a 'model unavailable' reason when the proposer is gone."""
        ptrace = TraceSink(cand.dir / "proposer_trace.jsonl")
        try:
            tools = self._tools()
            propose(self.proposer, self.archive, contract_doc=self.contract_doc, task_md=self.task_md,
                    extra=self.proposer_extra + tools, agentic=self.cfg.proposer_agentic, trace=ptrace,
                    timeout_s=self.cfg.proposer_timeout_s, candidate=cand, iteration=iteration,
                    directive=directive, siblings=siblings, full_access=bool(tools))
        except ModelUnavailable as e:
            self._log(f"{cand.dir.name}: proposer unavailable: {e}")
            if not cand.harness_path.exists():
                self._new_candidate_eval(cand, f"proposer unavailable: {str(e)[:300]}")
            return f"model unavailable: {e}"
        except Exception as e:  # noqa: BLE001 — a failed proposal is data, not a crash
            if STOPPING.is_set():               # killed by our own stop: a resume proposes it again
                return "stopped"
            self._log(f"{cand.dir.name}: proposer failed: {type(e).__name__}: {str(e)[:300]}")
            if not cand.harness_path.exists():
                self._new_candidate_eval(cand, f"proposer failed: {type(e).__name__}: {str(e)[:300]}")
            return None
        finally:
            ptrace.close()
        if STOPPING.is_set():
            return "stopped"
        if not cand.harness_path.exists():
            self._new_candidate_eval(cand, "proposer produced no harness.py")
            self._log(f"{cand.dir.name}: proposer produced no harness")
        return None

    def _drop_duplicates(self, cands: list[Candidate]) -> list[Candidate]:
        keep = []
        for c in cands:
            if not c.harness_path.exists() or c.eval() is not None:
                continue
            dup = generation.duplicate_of(self.archive, c)
            if dup is None:
                keep.append(c)
                continue
            other, sim = dup
            self._new_candidate_eval(c, f"duplicate of candidates/{other.dir.name} (the same code, comments aside): "
                                        "not evaluated")
            self._log(f"{c.dir.name}: duplicate of {other.dir.name} (the same code); not evaluated")
        return keep

    def reflect(self, gen: int, cands: list[Candidate]) -> None:
        """Record what the generation tried and what happened (``lessons.md``), with the proposer model."""
        cands = [c for c in cands if c.eval() is not None]
        if not cands or STOPPING.is_set():
            return
        prompt = generation.reflection_prompt(self.archive, gen, cands)
        rdir = self.archive.root / "reflections"
        rdir.mkdir(exist_ok=True)
        (rdir / f"gen_{gen:03d}_prompt.md").write_text(prompt)
        lessons = self.archive.root / "lessons.md"
        before = lessons.read_text() if lessons.exists() else ""
        trace = TraceSink(rdir / f"gen_{gen:03d}_trace.jsonl")
        timeout = min(1800.0, self.cfg.proposer_timeout_s)
        try:
            try:
                res = self.proposer.run(prompt, cwd=self.archive.root, agentic=self.cfg.proposer_agentic,
                                        timeout_s=timeout, trace=trace)
            except TimeoutError as e:       # out of time while reading traces: write what it found
                session = getattr(e, "session_id", None)
                if not session or STOPPING.is_set():
                    raise
                res = self.proposer.run(f"Your time is up. Finish the `## Generation {gen}` section of lessons.md "
                                        "now from what you have found (a bullet per candidate of this generation, "
                                        "then the general lessons), then stop.", cwd=self.archive.root,
                                        session_id=session, agentic=self.cfg.proposer_agentic,
                                        timeout_s=timeout / 3, trace=trace)
        except ModelUnavailable as e:
            self.interrupted = f"model unavailable: {e}"
            self._log(f"reflection: proposer unavailable: {e}")
            return
        except Exception as e:  # noqa: BLE001 — the section may be written anyway; else a numeric summary
            section = re.search(rf"^## Generation {gen}\b", lessons.read_text() if lessons.exists() else "", flags=re.M)
            if section is None and not STOPPING.is_set():
                with open(lessons, "a") as fh:
                    fh.write(f"\n## Generation {gen} (automatic summary: the reflection did not finish)\n"
                             + "\n".join(generation.digest(c, limit=400) for c in cands) + "\n\n")
            self._log(f"reflection after generation {gen} " + ("recorded in lessons.md before the call ended"
                      if section else "did not finish; an automatic summary was recorded")
                      + f" ({type(e).__name__}: {str(e)[:120]})")
            return
        finally:
            trace.close()
        (rdir / f"gen_{gen:03d}.md").write_text(res.text or "")
        after = lessons.read_text() if lessons.exists() else ""
        if before and after != before and not after.startswith(before):
            # the agent rewrote the file: keep every earlier section, add only its section for this generation
            m = re.search(rf"^## Generation {gen}\b.*?(?=^## Generation |\Z)", after, flags=re.M | re.S)
            write_atomic(lessons, before.rstrip("\n") + "\n\n" + (m.group(0).rstrip("\n") + "\n\n" if m else ""))
            after = lessons.read_text()
            self._log(f"reflection after generation {gen} rewrote lessons.md; earlier sections restored")
        if after == before and (res.text or "").strip():         # no tools, or it only answered: keep the text
            text = (res.text or "").strip()
            if not text.lower().startswith("## generation"):
                text = f"## Generation {gen}\n{text}"
            with open(lessons, "a") as fh:
                fh.write(("\n" if before and not before.endswith("\n") else "") + text + "\n\n")
        self._log(f"reflection after generation {gen} recorded in lessons.md")

    # -- seeds ---------------------------------------------------------------------------------
    def qualify_seeds(self) -> None:
        """Reset every configured seed once before anything is evaluated. A training seed the game
        refuses makes every candidate ineligible, a refused held-out seed voids the final test —
        found in minutes instead of after the whole search. Cached in seeds.json."""
        from ..games.registry import make_game
        groups: list[tuple[str, Callable[..., Any], dict[str, Any], list[int]]] = []
        for g in self.games:
            tag = "" if g is self.games[0] else f"{_slug(g.name)}:"
            for stage in ("train", "val", "heldout"):
                groups.append((f"{tag}{stage}", g.factory, g.kwargs, self._seeds(stage, [g])[_slug(g.name)]))
        if self.cfg.heldout_game:
            name = self.cfg.heldout_game
            groups.append(("heldout-game", lambda **k: make_game(name, **k), self.cfg.heldout_game_kwargs,
                           self.cfg.heldout_game_seeds or self.cfg.seeds_heldout or self.cfg.seeds_train))
        key = _jsonish({"groups": [[g[0], g[3]] for g in groups], "game_kwargs": self.game_kwargs,
                        "heldout_game": self.cfg.heldout_game, "heldout_game_kwargs": self.cfg.heldout_game_kwargs,
                        "game_source": self.cfg.game_source})
        path = self.archive.root / "seeds.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("key") == key and not cached.get("refused"):
                return
        results: dict[str, Any] = {}
        refused: list[str] = []
        for label, factory, kwargs, seeds in groups:
            if not seeds:
                continue
            game = None
            try:
                game = factory(**kwargs)
                for seed in seeds:
                    try:
                        game.reset(seed)
                        obs = json.dumps(game.observe(), sort_keys=True, default=str)
                        results[f"{label}:{seed}"] = {"ok": True, "obs_sha256": hashlib.sha256(obs.encode()).hexdigest()[:16],
                                                      "episode_sha256": _episode_signature(game, seed)}
                    except ModelUnavailable:
                        raise
                    except Exception as e:  # noqa: BLE001
                        results[f"{label}:{seed}"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
                        refused.append(f"{label} seed {seed} ({type(e).__name__}: {str(e)[:120]})")
            except ModelUnavailable:
                raise
            except Exception as e:  # noqa: BLE001
                refused.append(f"{label}: the game cannot be constructed ({type(e).__name__}: {str(e)[:160]})")
            finally:
                if game is not None:
                    close_game(game)
        # Seeds that replay another seed's episode (a scenario chosen by seed % n, an RNG the seed does not
        # reach): a validation or held-out seed must not repeat a training one, nor validation a held-out one.
        by_sig: dict[str, list[str]] = {}
        for k, v in results.items():
            if v.get("episode_sha256"):
                by_sig.setdefault(v["episode_sha256"], []).append(k)
        warnings = []
        for keys in by_sig.values():
            stages = {k.split(":")[-2] for k in keys}
            if len(keys) > 1 and len(stages) > 1:
                refused.append(f"seeds {', '.join(keys)} play the same episode under the same actions (a validation "
                               "or held-out seed repeating another stage's)")
            elif len(keys) > 1:
                warnings.append(f"seeds {', '.join(keys)} play the same episode under the same actions")
        path.write_text(json.dumps({"key": key, "results": results, "refused": refused, "warnings": warnings},
                                   indent=2) + "\n")
        for w in warnings:
            self._log("seed check: " + w)
        if refused:
            raise SeedRefused("the game refuses these seeds: " + "; ".join(refused)
                              + ". Choose other seeds (--seeds / --val-seeds / --heldout) or fix the game; see seeds.json.")
        self._log(f"seed check: {len(results)} seed(s) reset ok")

    # -- the search --------------------------------------------------------------------------
    def run(self) -> Optional[Candidate]:
        self.interrupted = None
        with _archive_lock(self.archive.root):
            return self._run()

    def _pending(self) -> None:
        """Evaluate unevaluated candidates first: seed baselines, a candidate whose evaluation was
        interrupted by a model outage, or one written by a proposer that was stopped."""
        todo: list[tuple[Candidate, bool]] = []
        for c in self.archive.candidates():
            e = c.eval()
            if not c.harness_path.exists() or (e is not None and not _interrupted(e)):
                continue
            # an evaluation the game itself spoiled is redone once; failing that way again is the candidate's
            env_retry = e is not None and not _unavailable(e) and any(
                _interrupted(json.loads(p.read_text() or "{}")) and not _unavailable(json.loads(p.read_text() or "{}"))
                for p in c.dir.glob("eval.interrupted-*.json"))
            if env_retry:
                continue
            if e is not None:
                c.eval_path.rename(c.dir / f"eval.interrupted-{int(time.time())}.json")
            retried = bool(list(c.dir.glob("eval.interrupted-*.json"))) and e is not None and not _unavailable(e)
            todo.append((c, retried))
        if not todo:
            return
        self._preflight_all([c for c, _ in todo])
        if self.interrupted:
            return
        train = self._seeds("train")
        evals = self.evaluate_many([(c, train, "", self.games) for c, _ in todo])
        for (c, retried), e in zip(todo, evals):
            if self._stop_for(e, env_retry=retried):
                break

    def _plateaued(self, gen: int) -> bool:
        """--patience: the best candidate has not improved during the last ``patience`` generations."""
        p = self.cfg.patience
        best = self.archive.best() if p > 0 and gen > 0 else None
        if best is None:
            return False
        best_gen = int((best.directive() or {}).get("generation", -1))
        if gen - 1 - best_gen < p:
            return False
        self._log(f"no improvement in the last {gen - 1 - best_gen} generation(s) (best {best.dir.name}, "
                  f"patience {p}): the search stops after {gen} of up to {self.cfg.n_iterations}")
        return True

    def _catch_up_reflections(self, done: int) -> None:
        """A generation that has no section in lessons.md (its reflection failed, or the process stopped
        before it): the latest one is reflected on now, before the next proposers read the notes; an older
        gap gets the numeric summary only (a model reflection per old generation cost hours on a shared
        rate limit, for notes the newer sections have superseded)."""
        if not self.cfg.reflect:
            return
        lessons = self.archive.root / "lessons.md"
        text = lessons.read_text(errors="replace") if lessons.exists() else ""
        for g in range(1, done + 1):
            if STOPPING.is_set() or self.interrupted:
                return
            if re.search(rf"^## Generation {g}\b", text, flags=re.M):
                continue
            cands = [c for c in self.archive.candidates() if (c.directive() or {}).get("generation") == g - 1]
            if not cands:
                continue
            if g == done:
                self._log(f"generation {g} has no reflection yet; reflecting now")
                self.reflect(g, cands)
            else:
                with open(lessons, "a") as fh:
                    fh.write(f"\n## Generation {g} (automatic summary: no reflection was recorded)\n"
                             + "\n".join(generation.digest(c, limit=400) for c in cands if c.eval() is not None)
                             + "\n\n")
                self._log(f"generation {g} had no reflection; recorded its numeric summary")

    def _generation(self, gen: int, n: int) -> None:
        k = max(1, self.cfg.proposals_per_iteration)
        self._log(f"generation {gen + 1}/{n}: proposing {k} candidate(s) with {self.proposer.name}")
        directives = generation.plan(self.archive, k, gen)
        cands = [self.archive.new_candidate() for _ in directives]
        names = tuple(c.dir.name for c in cands)
        for c, d in zip(cands, directives):
            self._log(f"{c.dir.name}: {d['mode']} of {', '.join(d.get('parent_dirs') or []) or '–'}")
        with _pool(min(k, max(1, self.cfg.parallel) if self.cfg.parallel > 1 else k)) as ex:
            problems = list(ex.map(lambda cd: self._propose_one(
                cd[0], cd[1], tuple(x for x in names if x != cd[0].dir.name), (gen + 1, n)), zip(cands, directives)))
        if STOPPING.is_set():
            self.interrupted = "stopped"
            return
        for problem in problems:
            if problem and self.cfg.stop_on_model_unavailable:
                self.interrupted = problem
                return
        fresh = self._drop_duplicates(cands)
        self._preflight_all(fresh)
        if self.interrupted:
            return
        fresh = [c for c in fresh if c.harness_path.exists()]
        if fresh:
            evals = self.evaluate_many([(c, self._seeds("train"), "", self.games) for c in fresh])
            for e in evals:
                if self._stop_for(e):
                    return
        if self.cfg.reflect:
            self.reflect(gen + 1, cands)

    def _drop_abandoned(self) -> None:
        """Candidate directories a stopped process left without a harness or an evaluation (proposers
        that were killed) are moved to ``abandoned/``: the generation is proposed again on resume."""
        for c in self.archive.candidates():
            if "-" in c.dir.name or c.harness_path.exists() or c.eval_path.exists():
                continue
            dest = self.archive.root / "abandoned" / f"{c.dir.name}-{int(time.time())}"
            dest.parent.mkdir(exist_ok=True)
            os.replace(c.dir, dest)
            self._log(f"{c.dir.name}: left without a harness by a stopped proposer; moved to {dest.name}")

    def _run(self) -> Optional[Candidate]:
        self._drop_abandoned()
        if self.cfg.qualify_seeds:
            self.qualify_seeds()
        # Pinned only now: a refused seed (or an adaptor fix after it) must not lock the archive
        # into a configuration that never produced a score.
        if self.task_changed:
            self._log("the task text differs from the one this archive was created with; proposals from now on "
                      "see the new text (each candidate's prompt.md records what its proposer saw)")
        write_atomic(self.archive.root / "manifest.json", json.dumps(self._manifest, indent=2) + "\n")
        self.archive.write_task(self.task_md)
        self._pending()
        # --iters is the number of generations this archive should have: a resumed search runs the rest
        n, k = self.cfg.n_iterations, max(1, self.cfg.proposals_per_iteration)
        done = math.ceil(self.proposals_done() / k)
        if done and n:
            self._log(f"the archive already holds {self.proposals_done()} proposal(s) = {done} generation(s) of {k}; "
                      f"running {max(0, n - done)} more to reach {n} (raise --iters for more)")
        if not self.interrupted:
            self._catch_up_reflections(done)
        for gen in range(done, n):
            if self.interrupted:
                break
            if self._plateaued(gen):
                break
            self._generation(gen, n)
        if self.interrupted:
            # no validation / held-out evaluation: it would lock the archive against resuming the search
            self._log(f"search interrupted ({self.interrupted[:200]}); validation and held-out evaluation skipped — "
                      f"rerun the same search on {self.archive.root} to resume")
            self.archive.write_leaderboard()
            return self.archive.best()
        best = self.select_champion() or self.heldout_best()
        if best and self.cfg.seeds_heldout and self._stage_needed(best, "heldout"):
            self._log(f"held-out seeds evaluation of {best.dir.name}")
            self._stop_for(self.evaluate_many([(best, self._seeds("heldout"), "heldout", self.games)])[0])
        if best and self.cfg.heldout_game and not self.interrupted:
            from ..games.registry import make_game
            tag = f"heldout-{self.cfg.heldout_game.replace('/', '_').replace(':', '_')}"
            if self._stage_needed(best, tag):
                self._log(f"held-out GAME evaluation of {best.dir.name} on {self.cfg.heldout_game}")
                name = self.cfg.heldout_game
                self._stop_for(self.evaluate(best, self.cfg.heldout_game_seeds or self.cfg.seeds_heldout
                                             or self.cfg.seeds_train, tag=tag,
                                             game_factory=lambda **k: make_game(name, **k),
                                             game_kwargs=self.cfg.heldout_game_kwargs, game_name=name,
                                             game_adaptor=self.cfg.heldout_game_adaptor))
        if best and self.cfg.ablate and not self.interrupted and self._stage_needed(best, ABLATION):
            # what the runtime models add: the pick again on its training seeds, both models switched off
            self._log(f"ablation of {best.dir.name}: training seeds with every runtime model call failing")
            self._stop_for(self.evaluate_many([(best, self._seeds("train"), ABLATION, self.games)], no_models=True)[0])
        if self.interrupted:
            self._log(f"held-out evaluation interrupted ({self.interrupted[:200]}); rerun the same search on "
                      f"{self.archive.root} to complete it (the search itself stays closed)")
        self.archive.write_leaderboard()
        return best

    # -- final pick ---------------------------------------------------------------------------
    def select_champion(self) -> Optional[Candidate]:
        """Re-evaluate the top ``finalists`` on the validation seeds and pick by training + validation
        together (per game: mean over both blocks, min-max normalised across the finalists, averaged;
        ties keep the training order). Written once to ``champion.json``; None without validation seeds."""
        existing = self.archive.champion()
        if existing is not None:
            return existing
        val = self._seeds("val")
        if not any(val.values()):
            return None
        all_ranked = self.archive.ranked()
        ranked = all_ranked[:max(1, self.cfg.finalists)]
        if not ranked:
            return None
        # candidates tied with the last finalist on the training mean are finalists too: the tie-breaks
        # (fewer calls, fewer steps) must not decide which of several equal scripts the validation judges
        cutoff = (ranked[-1].eval() or {}).get("mean_score")
        ranked += [c for c in all_ranked[len(ranked):] if (c.eval() or {}).get("mean_score") == cutoff]
        todo = [c for c in ranked if self._stage_needed(c, "validation")]
        if todo:
            self._log(f"validation of {len(todo)} finalist(s) on seeds {val}")
            for e in self.evaluate_many([(c, val, "validation", self.games) for c in todo]):
                if self._stop_for(e):
                    return None
        rows = []
        for rank, c in enumerate(ranked):
            v = json.loads((c.dir / "validation" / "eval.json").read_text())
            status = ineligibility(v)
            rows.append({"candidate": c.dir.name, "rank": rank, "status": status or "ok",
                         "train": c.score, "val": v.get("mean_score"),
                         "per_game": self._both_blocks(c, v) if status is None else {}})
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            self._log("validation: no finalist is eligible on the validation seeds; the training best stays")
            pick = rows[0]
        else:
            games = sorted({g for r in ok for g in r["per_game"]})
            for g in games:
                vals = [r["per_game"][g] for r in ok if g in r["per_game"]]
                lo, hi = min(vals), max(vals)
                for r in ok:
                    if g in r["per_game"]:
                        r.setdefault("norm", []).append((r["per_game"][g] - lo) / (hi - lo) if hi > lo else 0.0)
            for r in ok:
                r["combined"] = sum(r["norm"]) / len(r["norm"]) if r.get("norm") else 0.0
            pick = max(ok, key=lambda r: (round(r["combined"], 9), -r["rank"]))
        write_atomic(self.archive.root / "champion.json", json.dumps(
            {"candidate": pick["candidate"], "train": pick["train"], "val": pick["val"],
             "combined": pick.get("combined"), "seeds_val": val, "finalists": rows}, indent=2, default=str) + "\n")
        self._log(f"final pick: {pick['candidate']} (train {pick['train']}, validation {pick['val']})")
        return self.archive.champion()

    @staticmethod
    def _both_blocks(c: Candidate, v: dict[str, Any]) -> dict[str, float]:
        """Per game: mean score over the training and validation episodes together."""
        out: dict[str, float] = {}
        for e in (c.eval() or {}, v):
            games = e["per_game"] if isinstance(e.get("per_game"), dict) else {str(e.get("game", "game")): e}
            for g, ge in games.items():
                scores = [float(ep["score"]) for ep in ge.get("episodes", []) if ep.get("score") is not None]
                out.setdefault(g, []).extend(scores)       # type: ignore[arg-type]
        return {g: sum(s) / len(s) for g, s in out.items() if s}     # type: ignore[misc]

    def heldout_best(self) -> Optional[Candidate]:
        """The candidate the held-out stage tests: the validation pick when there is one; else the
        training best. Once the loop's held-out stage has started on a candidate it stays that one
        (a later best — e.g. an evaluation redone on resume — must not re-roll the test). Held-out
        results on several candidates (evaluated outside the loop, as the audit's test phase does)
        leave the training selection in charge."""
        champion = self.archive.champion()
        if champion is not None:
            return champion
        best = self.archive.best()
        exposed = [c for c in self.archive.candidates()
                   if any(_exposed(d) for d in c.dir.glob("heldout*") if d.is_dir())]
        if len(exposed) == 1 and (best is None or best.dir != exposed[0].dir):
            return exposed[0]
        return best

    def _stage_needed(self, cand: Candidate, tag: str) -> bool:
        """Held-out / validation results are written once: re-running would let a test be re-rolled.
        A stage cut short (model outage, killed process) is completed, never replaced."""
        p = cand.dir / tag / "eval.json"
        if not p.exists():
            return True
        if _interrupted(json.loads(p.read_text() or "{}")):
            return True
        self._log(f"{cand.dir.name}/{tag}: result already recorded; not re-evaluated")
        return False

    _heldout_needed = _stage_needed

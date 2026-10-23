"""The outer loop: propose -> evaluate -> archive -> repeat; then hold-out."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..games.base import close_game
from ..models.base import ModelUnavailable, System1Model, System2Model, TraceSink
from .archive import Archive, Candidate
from .contract import Budget
from .proposer import propose, repair
from .runner import DEFAULT_GRACE_S, PROTOCOL_VERSION, EvalResult, evaluate_harness, preflight_harness


class SeedRefused(ValueError):
    """A configured seed cannot be played (the game refuses it); choose other seeds."""


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


def _unavailable(e: Optional[dict[str, Any]]) -> bool:
    return bool(e) and str(e.get("fatal") or "").startswith("model unavailable")


_ENV_ERRORS = ("game reset failed", "game construction failed", "game hung", "invalid trial")


def _env_failed(errors: list[str], n: int) -> bool:
    """Every episode failed because of the game or its infrastructure, not the harness."""
    return n > 0 and len(errors) == n and all(str(e).startswith(_ENV_ERRORS) for e in errors)


def _interrupted(e: Optional[dict[str, Any]]) -> bool:
    """An evaluation the search stopped on (model outage, game down): redone on resume."""
    return _unavailable(e) or (bool(e) and _env_failed(e.get("errors") or [], e.get("n_episodes", 0)))


def _heldout_exposed(d: Path) -> bool:
    """A held-out stage counts as run (test information is in the archive) once any of its
    episodes started — an eval.json with episodes, or any trace written — whatever stopped it."""
    ej = d / "eval.json"
    if ej.exists():
        e = json.loads(ej.read_text() or "{}")
        if e.get("n_episodes") or not _unavailable(e):
            return True
    return any(p.stat().st_size > 0 for p in d.glob("trace_seed*.jsonl"))


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
        self.interrupted: Optional[str] = None
        overlap = set(config.seeds_train) & set(config.seeds_heldout)
        if overlap:
            raise ValueError(f"train and held-out seeds overlap: {sorted(overlap)}")
        if config.n_iterations and any(_heldout_exposed(d) for d in archive.cand_dir.glob("*/heldout*")
                                       if d.is_dir()):
            raise ValueError("This archive contains held-out results; start a new search archive.")
        # JSON-normalised (a tuple in an adaptor's EVAL_KWARGS would never equal its stored list)
        game_kwargs = json.loads(json.dumps(game_kwargs, default=str))
        self.game_kwargs = game_kwargs
        manifest = {
            "protocol_version": PROTOCOL_VERSION, "task": task_md, "seeds_train": config.seeds_train,
            "clock_mode": config.clock_mode, "latency_override": config.latency_override,
            "game_kwargs": game_kwargs, "memory_mode": config.memory_mode,
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
        manifest_path = archive.root / "manifest.json"
        scored = any(c.eval() is not None for c in archive.candidates())
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old != manifest and scored:
                changed = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
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

    def evaluate(self, cand: Candidate, seeds: list[int], tag: str = "",
                 game_factory: Optional[Callable[..., Any]] = None,
                 game_kwargs: Optional[dict[str, Any]] = None, game_name: Optional[str] = None) -> EvalResult:
        out = cand.dir if not tag else cand.dir / tag
        res = evaluate_harness(
            cand.harness_path, game_factory or self.game_factory, seeds, system2=self.system2,
            system1=self.system1, out_dir=out, clock_mode=self.cfg.clock_mode,
            budget_per_episode=self.cfg.budget, latency_override=self.cfg.latency_override,
            harness_timeout_s=self.cfg.harness_timeout_s,
            game_kwargs=self.game_kwargs if game_kwargs is None else game_kwargs,
            memory_mode=self.cfg.memory_mode, grace_s=self.cfg.grace_s,
            game_name=game_name or (None if tag else self.game_name),
        )
        self.archive.write_leaderboard()
        ended = ", ".join(f"{k} {v}" for k, v in res.to_json()["ended_by"].items())
        self._log(f"eval {cand.dir.name}{'/' + tag if tag else ''}: mean={res.mean_score} "
                  f"stale={res.stale_fraction} fatal={res.fatal} errors={len([e for e in res.episodes if e.error])} "
                  f"failed s1/s2={sum(e.failed_s1 for e in res.episodes)}/{sum(e.failed_s2 for e in res.episodes)} "
                  f"ended: {ended or '–'}")
        return res

    def _stop_for(self, res: EvalResult) -> bool:
        """Stop the search (resumable) on a model outage or when the game itself is down."""
        if res.fatal and res.fatal.startswith("model unavailable") and self.cfg.stop_on_model_unavailable:
            self.interrupted = res.fatal
        elif _env_failed([e.error or "" for e in res.episodes], len(res.episodes)):
            self.interrupted = "the game failed in every episode: " + (res.episodes[0].error or "")[:300]
        return self.interrupted is not None

    def preflight_and_repair(self, cand: Candidate) -> bool:
        """Dry-run the candidate with stand-in models; on failure ask the proposer to repair once.
        A ModelUnavailable from the repair propagates (the search stops)."""
        for attempt in (0, 1):
            rep = preflight_harness(cand.harness_path, self.game_factory, game_kwargs=self.game_kwargs,
                                    tier2=self.system1 is not None, timeout_s=self.cfg.preflight_timeout_s,
                                    budget=self.cfg.budget, seed=self.cfg.seeds_train[0])
            (cand.dir / f"preflight_{attempt}.json").write_text(json.dumps(rep, indent=2, default=str))
            if rep["ok"]:
                note = ("; warnings: " + " | ".join(rep["warnings"])) if rep.get("warnings") else ""
                self._log(f"preflight {cand.dir.name}: ok" + (" after repair" if attempt else "") + note)
                return True
            self._log(f"preflight {cand.dir.name}: FAILED — {rep['error'][:200].replace(chr(10), ' ')}")
            if attempt == 1 or not rep.get("repairable", True):
                return False
            ptrace = TraceSink(self.archive.root / "proposer_trace.jsonl")
            try:
                repair(self.proposer, cand, rep["error"], agentic=self.cfg.proposer_agentic, trace=ptrace,
                       timeout_s=min(900.0, self.cfg.proposer_timeout_s))
            finally:
                ptrace.close()
        return False

    def qualify_seeds(self) -> None:
        """Reset every configured seed once before anything is evaluated. A training seed the game
        refuses makes every candidate ineligible, a refused held-out seed voids the final test —
        found in minutes instead of after the whole search. Cached in seeds.json."""
        from ..games.registry import make_game
        groups: list[tuple[str, Callable[..., Any], dict[str, Any], list[int]]] = [
            ("train", self.game_factory, self.game_kwargs, self.cfg.seeds_train),
            ("heldout", self.game_factory, self.game_kwargs, self.cfg.seeds_heldout)]
        if self.cfg.heldout_game:
            name = self.cfg.heldout_game
            groups.append(("heldout-game", lambda **k: make_game(name, **k), self.cfg.heldout_game_kwargs,
                           self.cfg.heldout_game_seeds or self.cfg.seeds_heldout or self.cfg.seeds_train))
        key = {"groups": [[g[0], g[3]] for g in groups], "game_kwargs": self.game_kwargs,
               "heldout_game": self.cfg.heldout_game, "heldout_game_kwargs": self.cfg.heldout_game_kwargs,
               "game_source": self.cfg.game_source}
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
                        results[f"{label}:{seed}"] = {"ok": True, "obs_sha256": hashlib.sha256(obs.encode()).hexdigest()[:16]}
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
        train = {v.get("obs_sha256") for k, v in results.items() if k.startswith("train:") and v.get("ok")}
        same = [k for k, v in results.items() if k.startswith("heldout:") and v.get("obs_sha256") in train]
        warnings = ([f"held-out seeds start exactly like a training seed: {', '.join(same)}"] if same else [])
        path.write_text(json.dumps({"key": key, "results": results, "refused": refused, "warnings": warnings},
                                   indent=2) + "\n")
        for w in warnings:
            self._log("seed check: " + w)
        if refused:
            raise SeedRefused("the game refuses these seeds: " + "; ".join(refused)
                              + ". Choose other seeds (--seeds / --heldout) or fix the game; see seeds.json.")
        self._log(f"seed check: {len(results)} seed(s) reset ok")

    def _new_candidate_eval(self, cand: Candidate, fatal: str) -> None:
        (cand.dir / "eval.json").write_text(json.dumps({"fatal": fatal, "mean_score": None, "n_episodes": 0,
                                                        "game": self.game_name}))

    def run(self) -> Optional[Candidate]:
        self.interrupted = None
        with _archive_lock(self.archive.root):
            return self._run()

    def _run(self) -> Optional[Candidate]:
        if self.cfg.qualify_seeds:
            self.qualify_seeds()
        # Pinned only now: a refused seed (or an adaptor fix after it) must not lock the archive
        # into a configuration that never produced a score.
        (self.archive.root / "manifest.json").write_text(json.dumps(self._manifest, indent=2) + "\n")
        self.archive.write_task(self.task_md)
        # evaluate unevaluated candidates first: seed baselines, a candidate whose evaluation was
        # interrupted by a model outage, or one written by a proposer that was stopped
        for c in self.archive.candidates():
            e = c.eval()
            if not c.harness_path.exists() or (e is not None and not _interrupted(e)):
                continue
            if e is not None:
                c.eval_path.rename(c.dir / f"eval.interrupted-{int(time.time())}.json")
            if self.cfg.preflight and "-" not in c.dir.name:       # proposed, not a labelled seed baseline
                try:
                    self.preflight_and_repair(c)
                except ModelUnavailable as exc:
                    self.interrupted = f"model unavailable: {exc}"
                    break
            if self._stop_for(self.evaluate(c, self.cfg.seeds_train)):
                break
        # --iters is the number of proposals this archive should have: a resumed search runs the rest
        n = self.cfg.n_iterations
        done = sum(1 for c in self.archive.candidates() if "-" not in c.dir.name
                   and not str((c.eval() or {}).get("fatal") or "").startswith("proposer unavailable"))
        if done and n:
            self._log(f"the archive already holds {done} proposal(s); running {max(0, n - done)} more "
                      f"to reach {n} (raise --iters for more)")
        for it in range(done, n):
            if self.interrupted:
                break
            self._log(f"iteration {it + 1}/{n}: proposing with {self.proposer.name}")
            cand = self.archive.new_candidate()
            ptrace = TraceSink(self.archive.root / "proposer_trace.jsonl")
            try:
                propose(self.proposer, self.archive, contract_doc=self.contract_doc, task_md=self.task_md,
                        extra=self.proposer_extra, agentic=self.cfg.proposer_agentic, trace=ptrace,
                        timeout_s=self.cfg.proposer_timeout_s, candidate=cand, iteration=(it + 1, n))
            except ModelUnavailable as e:
                self._log(f"proposer unavailable: {e}")
                if not cand.harness_path.exists():
                    self._new_candidate_eval(cand, f"proposer unavailable: {str(e)[:300]}")
                if self.cfg.stop_on_model_unavailable:
                    self.interrupted = f"model unavailable: {e}"
                    break
                continue
            except Exception as e:  # noqa: BLE001 — a failed proposal is data, not a crash
                self._log(f"proposer failed: {type(e).__name__}: {str(e)[:300]}")
                if not cand.harness_path.exists():
                    self._new_candidate_eval(cand, f"proposer failed: {type(e).__name__}: {str(e)[:300]}")
                    continue
            finally:
                ptrace.close()
            if not cand.harness_path.exists():
                self._new_candidate_eval(cand, "proposer produced no harness.py")
                self._log(f"{cand.dir.name}: proposer produced no harness")
                continue
            if self.cfg.preflight:
                try:
                    self.preflight_and_repair(cand)
                except ModelUnavailable as e:
                    self._log(f"repair: proposer unavailable: {e}")
                    self.interrupted = f"model unavailable: {e}"
                    break
            if self._stop_for(self.evaluate(cand, self.cfg.seeds_train)):
                break
        if self.interrupted:
            # no held-out evaluation: it would lock the archive against resuming the search
            self._log(f"search interrupted ({self.interrupted[:200]}); held-out evaluation skipped — "
                      f"rerun the same search on {self.archive.root} to resume")
            self.archive.write_leaderboard()
            return self.archive.best()
        best = self.heldout_best()
        if best and self.cfg.seeds_heldout and self._heldout_needed(best, "heldout"):
            self._log(f"held-out seeds evaluation of {best.dir.name}")
            self._stop_for(self.evaluate(best, self.cfg.seeds_heldout, tag="heldout"))
        if best and self.cfg.heldout_game and not self.interrupted:
            from ..games.registry import make_game
            tag = f"heldout-{self.cfg.heldout_game.replace('/', '_').replace(':', '_')}"
            if self._heldout_needed(best, tag):
                self._log(f"held-out GAME evaluation of {best.dir.name} on {self.cfg.heldout_game}")
                name = self.cfg.heldout_game
                self._stop_for(self.evaluate(best, self.cfg.heldout_game_seeds or self.cfg.seeds_heldout
                                             or self.cfg.seeds_train, tag=tag,
                                             game_factory=lambda **k: make_game(name, **k),
                                             game_kwargs=self.cfg.heldout_game_kwargs))
        if self.interrupted:
            self._log(f"held-out evaluation interrupted ({self.interrupted[:200]}); rerun the same search on "
                      f"{self.archive.root} to complete it (the search itself stays closed)")
        self.archive.write_leaderboard()
        return best

    def heldout_best(self) -> Optional[Candidate]:
        """The candidate the held-out stage tests: once a held-out stage has started, always that one
        (a later best — e.g. after an interrupted held-out run — must not re-roll the test)."""
        for c in self.archive.candidates():
            if any(_heldout_exposed(d) for d in c.dir.glob("heldout*") if d.is_dir()):
                return c
        return self.archive.best()

    def _heldout_needed(self, cand: Candidate, tag: str) -> bool:
        """Held-out results are written once: re-running would let a test be re-rolled. A stage cut
        short (model outage, killed process) is completed, never replaced by another candidate's."""
        p = cand.dir / tag / "eval.json"
        if not p.exists():
            return True
        if _interrupted(json.loads(p.read_text() or "{}")):
            return True
        self._log(f"{cand.dir.name}/{tag}: held-out result already recorded; not re-evaluated")
        return False

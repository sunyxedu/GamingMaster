"""The outer loop: propose -> evaluate -> archive -> repeat; then hold-out."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..models.base import ModelUnavailable, System1Model, System2Model, TraceSink
from .archive import Archive, Candidate
from .contract import Budget
from .proposer import propose, repair
from .runner import EvalResult, evaluate_harness, preflight_harness


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
        overlap = set(config.seeds_train) & set(config.seeds_heldout)
        if overlap:
            raise ValueError(f"train and held-out seeds overlap: {sorted(overlap)}")
        if config.n_iterations and any(archive.cand_dir.glob("*/heldout*/eval.json")):
            raise ValueError("This archive contains held-out results; start a new search archive.")
        manifest = {
            "protocol_version": 2, "task": task_md, "seeds_train": config.seeds_train,
            "clock_mode": config.clock_mode, "latency_override": config.latency_override,
            "game_kwargs": game_kwargs, "memory_mode": config.memory_mode,
            "budget": {k: getattr(config.budget, k) for k in
                       ("max_system1_calls", "max_system2_calls", "max_wall_s")},
            "harness_timeout_s": config.harness_timeout_s,
            "models": {k: None if m is None else {
                "name": m.name, "model": getattr(m, "model", None),
                "player_tools": getattr(m, "player_tools", None),
                "reasoning_effort": getattr(m, "reasoning_effort", None)}
                for k, m in (("system1", system1), ("system2", system2), ("proposer", proposer))},
        }
        manifest_path = archive.root / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Search configuration changed; start a new archive to avoid mixing scores.")
        elif any(c.eval() is not None for c in archive.candidates()):
            raise ValueError("Legacy archive has no protocol manifest; start a new search archive.")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        self.archive.write_task(task_md)
        self.log_path = archive.root / "loop.log"

    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def evaluate(self, cand: Candidate, seeds: list[int], tag: str = "",
                 game_factory: Optional[Callable[..., Any]] = None,
                 game_kwargs: Optional[dict[str, Any]] = None) -> EvalResult:
        out = cand.dir if not tag else cand.dir / tag
        res = evaluate_harness(
            cand.harness_path, game_factory or self.game_factory, seeds, system2=self.system2,
            system1=self.system1, out_dir=out, clock_mode=self.cfg.clock_mode,
            budget_per_episode=self.cfg.budget, latency_override=self.cfg.latency_override,
            harness_timeout_s=self.cfg.harness_timeout_s,
            game_kwargs=self.game_kwargs if game_kwargs is None else game_kwargs,
            memory_mode=self.cfg.memory_mode,
        )
        self.archive.write_leaderboard()
        self._log(f"eval {cand.dir.name}{'/' + tag if tag else ''}: mean={res.mean_score} "
                  f"stale={res.stale_fraction} fatal={res.fatal} errors={len([e for e in res.episodes if e.error])}")
        return res

    def preflight_and_repair(self, cand: Candidate) -> bool:
        """Dry-run the candidate with stand-in models; on failure ask the proposer to repair once."""
        for attempt in (0, 1):
            rep = preflight_harness(cand.harness_path, self.game_factory, game_kwargs=self.game_kwargs,
                                    tier2=self.system1 is not None, timeout_s=self.cfg.preflight_timeout_s)
            (cand.dir / f"preflight_{attempt}.json").write_text(json.dumps(rep, indent=2, default=str))
            if rep["ok"]:
                self._log(f"preflight {cand.dir.name}: ok" + (" after repair" if attempt else ""))
                return True
            self._log(f"preflight {cand.dir.name}: FAILED — {rep['error'][:200].replace(chr(10), ' ')}")
            if attempt == 1:
                return False
            ptrace = TraceSink(self.archive.root / "proposer_trace.jsonl")
            try:
                repair(self.proposer, cand, rep["error"], agentic=self.cfg.proposer_agentic, trace=ptrace,
                       timeout_s=min(900.0, self.cfg.proposer_timeout_s))
            finally:
                ptrace.close()
        return False

    def run(self) -> Optional[Candidate]:
        # evaluate any unevaluated candidates (seeds / baselines) first
        for c in self.archive.candidates():
            if c.eval() is None and c.harness_path.exists():
                r = self.evaluate(c, self.cfg.seeds_train)
                if r.fatal and "model unavailable" in r.fatal and self.cfg.stop_on_model_unavailable:
                    self._log("stopping: " + r.fatal)
                    return self.archive.best()
        for it in range(self.cfg.n_iterations):
            self._log(f"iteration {it + 1}/{self.cfg.n_iterations}: proposing with {self.proposer.name}")
            ptrace = TraceSink(self.archive.root / "proposer_trace.jsonl")
            try:
                cand = propose(self.proposer, self.archive, contract_doc=self.contract_doc,
                               task_md=self.task_md, extra=self.proposer_extra,
                               agentic=self.cfg.proposer_agentic, trace=ptrace,
                               timeout_s=self.cfg.proposer_timeout_s)
            except ModelUnavailable as e:
                self._log(f"proposer unavailable: {e}")
                if self.cfg.stop_on_model_unavailable:
                    break
                continue
            except Exception as e:  # noqa: BLE001 — a failed proposal is data, not a crash
                self._log(f"proposer failed: {type(e).__name__}: {str(e)[:300]}")
                continue
            finally:
                ptrace.close()
            if not cand.harness_path.exists():
                (cand.dir / "eval.json").write_text(json.dumps({"fatal": "proposer produced no harness.py",
                                                               "mean_score": None, "n_episodes": 0}))
                self._log(f"{cand.dir.name}: proposer produced no harness")
                continue
            if self.cfg.preflight:
                self.preflight_and_repair(cand)
            r = self.evaluate(cand, self.cfg.seeds_train)
            if r.fatal and "model unavailable" in r.fatal and self.cfg.stop_on_model_unavailable:
                self._log("stopping: " + r.fatal)
                break
        best = self.archive.best()
        if best and self.cfg.seeds_heldout:
            self._log(f"held-out seeds evaluation of {best.dir.name}")
            self.evaluate(best, self.cfg.seeds_heldout, tag="heldout")
        if best and self.cfg.heldout_game:
            from ..games.registry import make_game
            self._log(f"held-out GAME evaluation of {best.dir.name} on {self.cfg.heldout_game}")
            self.evaluate(best, self.cfg.seeds_heldout or self.cfg.seeds_train, tag=f"heldout-{self.cfg.heldout_game}",
                          game_factory=lambda **k: make_game(self.cfg.heldout_game, **k),
                          game_kwargs=self.cfg.heldout_game_kwargs)
        self.archive.write_leaderboard()
        return best

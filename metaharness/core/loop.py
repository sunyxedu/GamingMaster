"""The outer loop: propose -> evaluate -> archive -> repeat; then hold-out."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..models.base import ModelUnavailable, System1Model, System2Model, TraceSink
from .archive import Archive, Candidate
from .contract import Budget
from .proposer import propose
from .runner import EvalResult, evaluate_harness


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
        self.archive.write_task(task_md)
        self.log_path = archive.root / "loop.log"

    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def evaluate(self, cand: Candidate, seeds: list[int], tag: str = "") -> EvalResult:
        out = cand.dir if not tag else cand.dir / tag
        res = evaluate_harness(
            cand.harness_path, self.game_factory, seeds, system2=self.system2, system1=self.system1,
            out_dir=out, clock_mode=self.cfg.clock_mode, budget_per_episode=self.cfg.budget,
            latency_override=self.cfg.latency_override, harness_timeout_s=self.cfg.harness_timeout_s,
            game_kwargs=self.game_kwargs,
        )
        self.archive.write_leaderboard()
        self._log(f"eval {cand.dir.name}{'/' + tag if tag else ''}: mean={res.mean_score} "
                  f"stale={res.stale_fraction} fatal={res.fatal} errors={len([e for e in res.episodes if e.error])}")
        return res

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
            finally:
                ptrace.close()
            if not cand.harness_path.exists():
                (cand.dir / "eval.json").write_text(json.dumps({"fatal": "proposer produced no harness.py",
                                                               "mean_score": None, "n_episodes": 0}))
                self._log(f"{cand.dir.name}: proposer produced no harness")
                continue
            r = self.evaluate(cand, self.cfg.seeds_train)
            if r.fatal and "model unavailable" in r.fatal and self.cfg.stop_on_model_unavailable:
                self._log("stopping: " + r.fatal)
                break
        best = self.archive.best()
        if best and self.cfg.seeds_heldout:
            self._log(f"held-out evaluation of {best.dir.name}")
            self.evaluate(best, self.cfg.seeds_heldout, tag="heldout")
            self.archive.write_leaderboard()
        return best

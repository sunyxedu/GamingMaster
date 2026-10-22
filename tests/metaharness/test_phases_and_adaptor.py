"""Study / adapt phases with a file-writing stand-in agent, the adaptor contract
validator, the pre-flight check and generality-aware selection."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from metaharness.cli import main
from metaharness.core.archive import Archive, combine_evals
from metaharness.core.runner import preflight_harness
from metaharness.games.adaptor import validate_adaptor
from metaharness.models.mock import ScriptedSystem2
from metaharness.phases import run_adapt, run_study
from metaharness.testbeds.snake import SnakeGame

GOOD_ADAPTOR = textwrap.dedent('''
    """A tiny counting game written as a generated adaptor would be."""
    import random
    from metaharness.games.base import ActionSpec

    TIER = 2
    DETERMINISTIC = True

    class Game:
        name = "countgame"
        tick_s = 0.1
        action_spec = ActionSpec(names=["inc", "dec", "noop"], descriptions={"inc": "+1", "dec": "-1"}, noop="noop")

        def __init__(self, target=5):
            self.target = target
            self.reset(0)

        def reset(self, seed):
            self.rng = random.Random(seed)
            self.value = self.rng.randrange(-3, 3)
            self._ticks = 0
            self._done = False
            return self.observe()

        def observe(self):
            return {"value": self.value, "target": self.target, "t": self._ticks}

        def tick(self, action):
            if self._done:
                return
            self._ticks += 1
            if action == "inc":
                self.value += 1
            elif action == "dec":
                self.value -= 1
            if self.value == self.target or self._ticks >= 30:
                self._done = True

        @property
        def done(self):
            return self._done

        @property
        def score(self):
            return float(-abs(self.value - self.target))

        @property
        def ticks(self):
            return self._ticks

        def info(self):
            return {"ticks": self._ticks}

        def describe(self):
            return ("Count game: move an integer to the target with inc/dec, one tick every 0.1 s, "
                    "max 30 ticks. Score is minus the final distance. Observation: value, target, t.")

    def make_game(**kw):
        return Game(**kw)
''')

BAD_ADAPTOR = "def make_game(**kw):\n    return object()\n"

STUDY_FILES = {
    "brief.md": "x" * 300, "interfaces.md": "y" * 300, "strategies.md": "z" * 300,
    "study.json": json.dumps({"game": "countgame", "tier": 2, "tier_rationale": "ticks", "decision_latency_s": 0.1,
                              "interface": {"chosen": "local simulator", "env": []}, "scoring": "distance",
                              "scenarios": "random start", "strategy_summary": "move toward the target"}),
}


def test_validate_adaptor_pass_and_fail(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "game.py").write_text(GOOD_ADAPTOR)
    (good / "ADAPTOR.md").write_text("count game")
    rep = validate_adaptor(good)
    assert rep.ok, rep.to_markdown()
    assert any("DETERMINISTIC" in c["check"] for c in rep.checks)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "game.py").write_text(BAD_ADAPTOR)
    (bad / "ADAPTOR.md").write_text("bad")
    rep = validate_adaptor(bad)
    assert not rep.ok and "action_spec" in rep.to_markdown()
    assert main(["validate-adaptor", str(good)]) == 0
    assert main(["validate-adaptor", str(bad)]) == 1


def test_study_phase_validates_and_repairs(tmp_path):
    # first call writes an incomplete study; the repair call completes it
    incomplete = {k: v for k, v in STUDY_FILES.items() if k != "study.json"}
    agent = ScriptedSystem2(replies=["done"], files=[incomplete, STUDY_FILES])
    study = run_study(agent, tmp_path, "countgame", log=lambda *_: None)
    assert study["tier"] == 2 and len(agent.calls) == 2
    assert "Study the game" in agent.calls[0] and "incomplete" in agent.calls[1]
    # resumable: a valid study is reused without calling the agent
    study2 = run_study(ScriptedSystem2(replies=["never"]), tmp_path, "countgame", log=lambda *_: None)
    assert study2 == study


def test_adapt_phase_feeds_back_validation(tmp_path):
    (tmp_path / "study").mkdir()
    for k, v in STUDY_FILES.items():
        (tmp_path / "study" / k).write_text(v)
    study = json.loads(STUDY_FILES["study.json"])
    agent = ScriptedSystem2(replies=["done"], files=[
        {"game.py": BAD_ADAPTOR, "ADAPTOR.md": "v1"},
        {"game.py": GOOD_ADAPTOR, "ADAPTOR.md": "v2"},
    ])
    rep = run_adapt(agent, tmp_path, "countgame", study, log=lambda *_: None)
    assert rep.ok and len(agent.calls) == 2
    assert "Validation feedback" not in agent.calls[0] and "FAIL" in agent.calls[1]
    assert (tmp_path / "adaptor" / "validation.json").exists()


def test_adapt_phase_stops_on_missing_env(tmp_path, monkeypatch):
    (tmp_path / "study").mkdir()
    for k, v in STUDY_FILES.items():
        (tmp_path / "study" / k).write_text(v)
    monkeypatch.delenv("COUNTGAME_TOKEN", raising=False)
    agent = ScriptedSystem2(files=[{"game.py": GOOD_ADAPTOR, "ADAPTOR.md": "v", "env.example": "COUNTGAME_TOKEN=\n"}])
    rep = run_adapt(agent, tmp_path, "countgame", json.loads(STUDY_FILES["study.json"]), log=lambda *_: None)
    assert not rep.ok and "COUNTGAME_TOKEN" in (tmp_path / "adaptor" / "NEEDS.md").read_text()


def test_full_run_pipeline_offline(tmp_path):
    """study -> adapt -> search with stand-ins; the search runs on the generated adaptor."""
    import metaharness.cli as cli
    agent = ScriptedSystem2(replies=["done"], files=[STUDY_FILES, {"game.py": GOOD_ADAPTOR, "ADAPTOR.md": "ok"}])
    orig = cli._system2

    def fake_system2(name, effort):
        return agent if name == "fake-agent" else orig(name, effort)
    cli._system2 = fake_system2
    try:
        rc = main(["run", "--game", "Count Game", "--agent", "fake-agent", "--system2", "scripted",
                   "--system1", "heuristic", "--proposer", "scripted", "--clock", "virtual", "--iters", "1",
                   "--seeds", "0,1", "--heldout", "100", "--runs-dir", str(tmp_path)])
    finally:
        cli._system2 = orig
    assert rc == 0
    run_dir = tmp_path / "count-game"
    assert (run_dir / "study" / "study.json").exists() and (run_dir / "adaptor" / "validation.json").exists()
    a = Archive(run_dir / "search")
    names = [c.dir.name for c in a.candidates()]
    assert names[:3] == ["0000-tier2_jev_only", "0001-tier2_astra_only", "0002-tier2_idle"] and len(names) == 4
    assert "study/strategies.md" in (a.root / "task.md").read_text()
    for c in a.candidates():
        assert (c.eval() or {}).get("game") == "countgame"
    assert (a.candidates()[3].dir / "preflight_0.json").exists()


def test_preflight_catches_contract_misuse(tmp_path):
    bad = tmp_path / "h.py"
    bad.write_text(textwrap.dedent('''
        class H:
            def __init__(self, ctx): self.ctx = ctx
            def run_episode(self, game):
                self.ctx.think("plan?")
                self.ctx.decide(game.observe(), {"a": {"type": "choice", "instructions": "?", "criteria": {"up": None}}})
                self.ctx.log("event", 1, 2)      # positional args are tolerated
                self.ctx.nonexistent_method()    # this is not
        def build(ctx): return H(ctx)
    '''))
    rep = preflight_harness(bad, lambda **k: SnakeGame(max_ticks=20, **k))
    assert not rep["ok"] and "nonexistent_method" in rep["error"]
    good = tmp_path / "g.py"
    good.write_text(textwrap.dedent('''
        class H:
            def __init__(self, ctx): self.ctx = ctx
            def run_episode(self, game):
                r = self.ctx.think("plan?")
                self.ctx.log("thought", chars=len(r.text))
                while not self.ctx.done:
                    self.ctx.act(game.action_spec.noop)
                    self.ctx.wait_for_tick()
        def build(ctx): return H(ctx)
    '''))
    rep = preflight_harness(good, lambda **k: SnakeGame(max_ticks=20, **k))
    assert rep["ok"] and rep["system2_calls"] == 1


def _write_eval(cand_dir: Path, per_game: dict[str, float], seeds=(0, 1)):
    evals = {}
    for g, m in per_game.items():
        evals[g] = {"game": g, "fatal": None, "mean_score": m, "n_episodes": len(seeds), "errors": [],
                    "system1_calls": 1, "system2_calls": 1, "stale_fraction": 0.0,
                    "episodes": [{"seed": s, "score": m, "stale_ticks": 0, "decided_ticks": 10, "acts": 10,
                                  "acts_after_s1": 5, "acts_after_s2": 1} for s in seeds],
                    "metadata": {"seeds": list(seeds), "game_kind": "realtime"}}
    (cand_dir / "eval.json").write_text(json.dumps(combine_evals(evals, list(seeds))))


def test_generality_aware_selection(tmp_path):
    a = Archive(tmp_path / "arch")
    specialist = a.new_candidate("specialist")        # great on A, dies on B
    generalist = a.new_candidate("generalist")        # decent everywhere
    floor = a.new_candidate("floor")
    for c in (specialist, generalist, floor):
        c.harness_path.write_text("def build(ctx): pass\n")
    _write_eval(specialist.dir, {"A": 30.0, "B": 0.0})
    _write_eval(generalist.dir, {"A": 20.0, "B": 10.0})
    _write_eval(floor.dir, {"A": 0.0, "B": 0.0})
    assert a.best().dir == generalist.dir
    n = a.normalized(generalist)
    assert n == {"A": 20 / 30, "B": 1.0}
    lb = a.write_leaderboard()
    assert "worst" in lb and "Jev→act" in lb and "50%" in lb
    e = generalist.eval()
    assert e["s1_informed_fraction"] == 0.5 and e["metadata"]["expected_episodes"] == 4

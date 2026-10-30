"""The game-name pipeline around the adaptor: run directories (Unicode slugs, run.json ownership),
study reuse (tier override, notes), the adapt loop (resume validates first, NEEDS.md, tier and
kwargs) and the prior documents the search proposer gets (budgeted digest + prior/ copies)."""
from __future__ import annotations

import json
import random
import re
import string
import textwrap
from pathlib import Path

import pytest

import metaharness.cli as cli
import metaharness.phases as phases
from metaharness.cli import _slug, main
from metaharness.core.archive import Archive
from metaharness.games.adaptor import ValidationReport
from metaharness.models.mock import ScriptedSystem2
from metaharness.phases import prior_documents, run_adapt, run_study
from tests.metaharness.test_phases_and_adaptor import BAD_ADAPTOR, GOOD_ADAPTOR, STUDY_FILES
from tests.metaharness.test_pipeline_adaptor import COUNTBASE, TICK_GAME, TURN_GAME, write_adaptor

STUDY = json.loads(STUDY_FILES["study.json"])          # tier 2


def adaptor_files(body: str, kind: str = "TickCount", **extra: str) -> dict[str, str]:
    """What a scripted adapt agent writes: the fake adaptor of test_pipeline_adaptor."""
    game = f"from countbase import {kind}\n\n" + textwrap.dedent(body) + "\n\ndef make_game(**kw):\n    return Game(**kw)\n"
    return {"countbase.py": COUNTBASE, "game.py": game, "ADAPTOR.md": "count adaptor", **extra}


def write_study(run_dir: Path, **over: str) -> None:
    (run_dir / "study").mkdir(parents=True, exist_ok=True)
    for k, v in {**STUDY_FILES, **over}.items():
        (run_dir / "study" / k).write_text(v)


@pytest.fixture(autouse=True)
def clean_registry():
    from metaharness.games import registry as reg
    reg._populate()
    games, adaptors = dict(reg.GAMES), dict(reg._ADAPTORS)
    yield
    reg.GAMES.clear()
    reg.GAMES.update(games)
    reg._ADAPTORS.clear()
    reg._ADAPTORS.update(adaptors)


@pytest.fixture
def fake_agent(monkeypatch):
    """``--agent fake-agent`` resolves to the ScriptedSystem2 the test installs."""
    holder: dict[str, ScriptedSystem2] = {}
    orig = cli._system2
    monkeypatch.setattr(cli, "_system2",
                        lambda name, effort: holder["agent"] if name == "fake-agent" else orig(name, effort))
    return holder


# ----------------------------------------------------------------------------- run directories

def test_slug_is_unicode_aware_and_keeps_ascii_slugs():
    def old(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "game"

    rng = random.Random(0)
    names = ["Count Game", "ARC-AGI-3", "foo_bar", "Minecraft: Java 1.21", "  --  ", "StarCraft II", "C++ & Go!"]
    names += ["".join(rng.choice(string.printable) for _ in range(rng.randint(0, 30))) for _ in range(500)]
    for s in names:
        assert _slug(s) == old(s), s
    assert _slug("原神") == "原神" and _slug("王者荣耀") == "王者荣耀" and _slug("皇室战争") == "皇室战争"
    assert len({_slug("原神"), _slug("王者荣耀"), _slug("皇室战争")}) == 3
    assert _slug("Pokémon") == _slug("Pokémon") == "pokémon"           # composed and decomposed
    assert _slug("हिन्दी खेल") == "हिन्दी-खेल"                                  # combining marks stay
    assert _slug("Ｍｉｎｅｃｒａｆｔ") == "minecraft" and _slug("🎮") == "game"


def run_study_only(game: str, runs: Path, *extra: str) -> int:
    return main(["run", "--game", game, "--agent", "fake-agent", "--stop-after", "study", "--runs-dir", str(runs), *extra])


def test_run_dir_belongs_to_its_game(tmp_path, fake_agent):
    fake_agent["agent"] = agent = ScriptedSystem2(replies=["done"], files=[STUDY_FILES])
    assert run_study_only("🎮", tmp_path) == 0
    meta = json.loads((tmp_path / "game" / "run.json").read_text(encoding="utf-8"))
    assert meta["game"] == "🎮"
    with pytest.raises(SystemExit, match="was created for --game '🎮', not '🕹'"):
        run_study_only("🕹", tmp_path)                          # same slug, another game: refused
    assert run_study_only("🕹", tmp_path, "--name", "joystick") == 0       # ... unless asked for another dir
    n = len(agent.calls)
    assert run_study_only("Count Game", tmp_path) == 0
    assert run_study_only("count  GAME", tmp_path) == 0         # case / spacing: the same game, resumed
    assert len(agent.calls) == n + 1                            # (the second run reused the study)
    with pytest.raises(SystemExit, match="created for --game 'Count Game'"):
        run_study_only("Count_Game", tmp_path)
    with pytest.raises(SystemExit):
        run_study_only("Other", tmp_path, "--name", "count-game")
    # a directory from before run.json: the first pipeline.log line names its game
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "pipeline.log").write_text("[16:12:45] run: game=\"Tom's Game\" dir=runs/legacy agent=codex\n")
    with pytest.raises(SystemExit, match="Tom's Game"):
        run_study_only("Other Game", tmp_path, "--name", "legacy")
    assert run_study_only("tom's game", tmp_path, "--name", "legacy") == 0
    assert json.loads((legacy / "run.json").read_text(encoding="utf-8"))["game"] == "Tom's Game"


# ----------------------------------------------------------------------------- study

def test_study_reuse_applies_tier_override_and_tracks_notes(tmp_path):
    logs: list[str] = []
    agent = ScriptedSystem2(replies=["done"], files=[STUDY_FILES])
    run_study(agent, tmp_path, "countgame", notes="server at localhost:1234", log=logs.append)
    assert (tmp_path / "study" / "notes.md").read_text() == "server at localhost:1234"
    study = run_study(ScriptedSystem2(replies=["never"]), tmp_path, "countgame", notes="server at localhost:1234",
                      tier_hint=1, log=logs.append)
    assert study["tier"] == 1 and json.loads((tmp_path / "study" / "study.json").read_text())["tier"] == 1
    assert not any("WARNING" in m for m in logs)
    study = run_study(ScriptedSystem2(replies=["never"]), tmp_path, "countgame", notes="another server",
                      log=logs.append)
    assert study["tier"] == 1                                   # the override persisted
    assert any("--notes differ" in m for m in logs)
    assert (tmp_path / "study" / "notes.md").read_text() == "server at localhost:1234"   # what it was made with


# ----------------------------------------------------------------------------- adapt

def test_adapt_prompt_carries_notes_contract_and_kwargs(tmp_path):
    write_study(tmp_path)
    agent = ScriptedSystem2(replies=["done"], files=[adaptor_files(textwrap.dedent("""
        TIER = 2

        class Game(TickCount):
            def __init__(self, level, **kw):
                super().__init__(**kw)
    """))])
    rep = run_adapt(agent, tmp_path, "countgame", STUDY, notes="use the EU server", kwargs={"level": 2},
                    log=lambda *_: None)
    assert rep.ok, rep.to_markdown()                            # make_game needs the kwargs: they were used
    prompt = agent.calls[0]
    assert "use the EU server" in prompt and "expected_tier=2, kwargs={'level': 2}" in prompt
    for term in ("close()", "SEEDS", "EVAL_KWARGS", "valid_trial", "invalid_reason", "never overlap",
                 "game error", "MUST", "env.example", "runtime/", "pinned by"):
        assert term in prompt, term
    saved = json.loads((tmp_path / "adaptor" / "validation.json").read_text())
    assert saved["ok"] and "warnings" in saved


def test_adapt_resume_validates_before_calling_the_agent(tmp_path):
    write_study(tmp_path)
    out = tmp_path / "adaptor"
    write_adaptor(out, TICK_GAME)
    (out / "NEEDS.md").write_text("log in to the launcher\n")          # left over from an earlier run
    agent = ScriptedSystem2(replies=["never"])
    rep = run_adapt(agent, tmp_path, "countgame", STUDY, log=lambda *_: None)
    assert rep.ok and agent.calls == []
    assert json.loads((out / "validation.json").read_text())["ok"] and not (out / "NEEDS.md").exists()
    # an existing adaptor that fails: the agent's first prompt already carries the report
    (out / "game.py").write_text(BAD_ADAPTOR)
    agent = ScriptedSystem2(replies=["fixed"], files=[adaptor_files(TICK_GAME)])
    rep = run_adapt(agent, tmp_path, "countgame", STUDY, log=lambda *_: None)
    assert rep.ok and len(agent.calls) == 1
    assert "Validation feedback" in agent.calls[0] and "FAIL" in agent.calls[0]


def test_adapt_feeds_back_a_tier_mismatch(tmp_path):
    study = {**STUDY, "tier": 1}
    write_study(tmp_path, **{"study.json": json.dumps(study)})
    agent = ScriptedSystem2(replies=["done"], files=[adaptor_files(TICK_GAME),
                                                     adaptor_files(TURN_GAME, kind="TurnCount")])
    rep = run_adapt(agent, tmp_path, "countgame", study, log=lambda *_: None)
    assert rep.ok and len(agent.calls) == 2
    assert "tier 1, as the study decided" in agent.calls[1] and "tier 1 needs a TurnGame" in agent.calls[1]


def test_adapt_stale_needs_md_is_feedback_not_a_stop(tmp_path):
    write_study(tmp_path)
    (tmp_path / "adaptor").mkdir()
    (tmp_path / "adaptor" / "NEEDS.md").write_text("start the server (done since)\n")
    agent = ScriptedSystem2(replies=["done"], files=[{"game.py": BAD_ADAPTOR, "ADAPTOR.md": "v1"},
                                                     {"game.py": GOOD_ADAPTOR, "ADAPTOR.md": "v2"}])
    rep = run_adapt(agent, tmp_path, "countgame", STUDY, log=lambda *_: None)
    assert rep.ok and len(agent.calls) == 2 and not (tmp_path / "adaptor" / "NEEDS.md").exists()


@pytest.mark.parametrize("files", [
    {"NEEDS.md": "buy the game and log in\n"},                                  # nothing to validate yet
    {"NEEDS.md": "buy the game and log in\n", "game.py": BAD_ADAPTOR, "ADAPTOR.md": "v1"},
])
def test_adapt_stops_when_the_agent_asks_for_the_user(tmp_path, files):
    write_study(tmp_path)
    agent = ScriptedSystem2(replies=["need the user"], files=[files, {"game.py": GOOD_ADAPTOR, "ADAPTOR.md": "v2"}])
    rep = run_adapt(agent, tmp_path, "countgame", STUDY, log=lambda *_: None)
    assert not rep.ok and len(agent.calls) == 1
    assert "user input required (the model needs user input)" in rep.error and "buy the game" in rep.error
    assert (tmp_path / "adaptor" / "validation.json").exists()


def test_adapt_unset_variable_does_not_block_a_passing_adaptor(tmp_path, monkeypatch):
    write_study(tmp_path)
    monkeypatch.delenv("COUNTGAME_REGION", raising=False)
    logs: list[str] = []
    agent = ScriptedSystem2(files=[{"game.py": GOOD_ADAPTOR, "ADAPTOR.md": "v",
                                    "env.example": "export COUNTGAME_REGION=eu\n"}])
    rep = run_adapt(agent, tmp_path, "countgame", STUDY, log=logs.append)
    assert rep.ok and not (tmp_path / "adaptor" / "NEEDS.md").exists()
    assert any("COUNTGAME_REGION" in m and "validates without them" in m for m in logs)


# ----------------------------------------------------------------------------- prior documents

def _doc(tag: str, n: int) -> str:
    body = "\n".join(f"{tag} line {i:05d} " + "x" * 50 for i in range(n // 62 + 1))
    return f"{tag.upper()}-START\n{body[: max(0, n - 40)]}\n{tag.upper()}-END advice to the search\n"


def test_prior_documents_budget_marks_and_copies(tmp_path):
    study, adaptor, arch = tmp_path / "study", tmp_path / "adaptor", tmp_path / "arch"
    write_study(tmp_path, **{"strategies.md": _doc("strategies", 40000), "brief.md": _doc("brief", 6000),
                             "interfaces.md": _doc("interfaces", 40000),
                             "study.json": json.dumps({**STUDY, "scoring": "distance to the target",
                                                       "interface": {"observation": "json state", "actions": "inc/dec"}})})
    adaptor.mkdir()
    (adaptor / "ADAPTOR.md").write_text(_doc("adaptor", 20000))
    text = prior_documents(study, adaptor, arch)
    for f in ("strategies.md", "brief.md", "interfaces.md", "study.json", "ADAPTOR.md"):
        src = (adaptor if f == "ADAPTOR.md" else study) / f
        assert (arch / "prior" / f).read_text() == src.read_text()
    assert "readable in this archive under `prior/`" in text
    order = ["- strategy_summary: move toward the target", "- scoring: distance to the target",
             "- interface.observation: json state", "- interface.actions: inc/dec", "### study/strategies.md",
             "### study/brief.md", "### adaptor/ADAPTOR.md", "### study/interfaces.md"]
    assert [text.index(s) for s in order] == sorted(text.index(s) for s in order)
    for tag, f in (("strategies", "strategies.md"), ("adaptor", "ADAPTOR.md"), ("interfaces", "interfaces.md")):
        assert f"{tag.upper()}-START" in text and f"{tag.upper()}-END advice to the search" in text   # head + end
        assert f"…(truncated; full text in prior/{f})" in text
    assert "BRIEF-END" in text and "prior/brief.md)" not in text      # its share covers all of it
    sections = dict(re.findall(r"### (\S+)\n(.*?)(?=\n### |\Z)", text, flags=re.S))
    assert 11500 <= len(sections["adaptor/ADAPTOR.md"]) <= 12100
    assert len(sections["study/strategies.md"]) > 2 * len(sections["study/interfaces.md"]) > 0
    assert len(text) < 24000 + 12000 + 1000
    assert prior_documents(study, adaptor, arch) == text           # deterministic (the search manifest)


def test_prior_documents_redistributes_unused_budget(tmp_path):
    write_study(tmp_path, **{"strategies.md": _doc("strategies", 30000)})
    (tmp_path / "adaptor").mkdir()
    (tmp_path / "adaptor" / "ADAPTOR.md").write_text("short adaptor notes " * 10)
    text = prior_documents(tmp_path / "study", tmp_path / "adaptor", tmp_path / "arch")
    assert "full text in prior/strategies.md" not in text and "STRATEGIES-END" in text   # ADAPTOR.md's budget
    text = prior_documents(None, tmp_path / "adaptor", tmp_path / "arch2")     # loop --adaptor alone
    assert "### adaptor/ADAPTOR.md" in text and "study.json" not in text and "### study/" not in text
    assert prior_documents(None, None, tmp_path / "arch3") == ""


# ----------------------------------------------------------------------------- CLI wiring

def test_loop_with_adaptor_and_study_gets_prior_documents(tmp_path):
    adaptor = write_adaptor(tmp_path / "adaptor", TICK_GAME)
    (adaptor / "ADAPTOR.md").write_text("ADAPTOR-MARKER: observation fields value / target / n\n")
    write_study(tmp_path)
    rc = main(["loop", "--tier", "2", "--adaptor", str(adaptor), "--study", str(tmp_path / "study"),
               "--system1", "heuristic", "--system2", "scripted", "--proposer", "scripted", "--clock", "virtual",
               "--iters", "0", "--seeds", "0", "--heldout", "", "--name", "lp", "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0
    arch = Archive(tmp_path / "runs" / "lp")
    task = (arch.root / "task.md").read_text()
    assert "### adaptor/ADAPTOR.md\nADAPTOR-MARKER" in task and "### study/strategies.md" in task
    assert (arch.root / "prior" / "ADAPTOR.md").exists() and (arch.root / "prior" / "strategies.md").exists()


def test_run_searches_the_adaptor_tier_and_passes_notes_and_kwargs(tmp_path, fake_agent, monkeypatch):
    calls: dict[str, object] = {}

    def fake_adapt(model, run_dir, game, study, **kw):           # passes whatever the agent builds
        calls["adapt"] = kw
        write_adaptor(run_dir / "adaptor", TICK_GAME)
        return ValidationReport(ok=True)

    def fake_search(a, games, tier_n, prior, exp_dir):
        (game, adaptor_dir), = games
        assert adaptor_dir == str((exp_dir.parent / "adaptor").resolve())
        calls["search"] = (game, tier_n, prior, exp_dir)

    monkeypatch.setattr(phases, "run_adapt", fake_adapt)
    monkeypatch.setattr(cli, "_run_search", fake_search)
    fake_agent["agent"] = ScriptedSystem2(replies=["done"], files=[STUDY_FILES])
    rc = main(["run", "--game", "Count Game", "--agent", "fake-agent", "--tier", "1", "--notes", "EU server",
               "--game-kwargs", '{"target": 4}', "--runs-dir", str(tmp_path)])
    assert rc == 0
    run_dir = tmp_path / "count-game"
    assert calls["adapt"]["notes"] == "EU server" and calls["adapt"]["kwargs"] == {"target": 4}
    game, tier_n, prior, exp_dir = calls["search"]
    assert tier_n == 2 and exp_dir == run_dir / "search"         # the real-time adaptor decides, not study.json
    assert "the adaptor is tier 2 (real-time) but study.json says tier 1" in (run_dir / "pipeline.log").read_text()
    assert "### adaptor/ADAPTOR.md" in prior and (exp_dir / "prior" / "ADAPTOR.md").exists()
    from metaharness.games.registry import adaptor_spec
    assert adaptor_spec(game).path == (run_dir / "adaptor").resolve()
    with pytest.raises(SystemExit, match="not valid JSON"):
        main(["run", "--game", "Count Game", "--agent", "fake-agent", "--game-kwargs", "{target: 4}",
              "--runs-dir", str(tmp_path)])


def test_a_study_that_times_out_resumes_its_session_and_keeps_its_research(tmp_path):
    from metaharness.models.base import ModelTimeout, System2Result
    from metaharness.phases.study import FINISH_PROMPT, run_study

    class SlowThenDone:
        name = "slow"

        def __init__(self):
            self.calls = []

        def run(self, prompt, *, cwd, session_id=None, timeout_s=0, **kw):
            self.calls.append((prompt, session_id, timeout_s))
            if len(self.calls) == 1:
                raise ModelTimeout("timeout", session_id="s-1")
            for name, text in STUDY_FILES.items():
                (cwd / name).write_text(text)
            return System2Result(text="done", session_id=session_id)

    agent = SlowThenDone()
    data = run_study(agent, tmp_path, "Count Game", timeout_s=3600, log=lambda m: None)
    assert data["game"] and len(agent.calls) == 2
    first, second = agent.calls
    assert "Write a first complete version of all four files" in first[0] and first[1] is None
    assert second == (FINISH_PROMPT, "s-1", 1800.0)                       # the same conversation, finishing
    assert not (tmp_path / "study" / "session.json").exists()            # forgotten once the study validates

    # a process killed after a timeout: the next run resumes the saved session instead of starting over
    other = tmp_path / "again"
    (other / "study").mkdir(parents=True)
    (other / "study" / "session.json").write_text('{"session_id": "s-9"}')
    agent2 = SlowThenDone()
    agent2.calls.append(("(already timed out)", None, 0))
    run_study(agent2, other, "Count Game", timeout_s=3600, log=lambda m: None)
    assert agent2.calls[1][:2] == (FINISH_PROMPT, "s-9")

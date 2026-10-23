"""Generated-adaptor plumbing: fresh loading (sibling helpers, declarations, probe close, source
hash of the code only), the stricter validator (tier, post-action types, tick pace, malformed actions,
fresh-instance determinism, SEEDS, close, code changed by running) and a registry that never lets a
test bed replace an adaptor and keeps each adaptor on its own helper modules."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from metaharness.games import registry as reg
from metaharness.games.adaptor import (MAX_SUBDIR_FILES, DeclarationError, load_adaptor, source_sha256,
                                       validate_adaptor)
from metaharness.testbeds.snake import SnakeGame

# Written next to every fake adaptor's game.py: game.py imports it as a sibling module, the way
# generated adaptors import their helpers (r7: scoring.py).
COUNTBASE = textwrap.dedent('''
    import random
    import time
    from metaharness.games.base import ActionSpec

    SERIAL = [0]                                   # instances made since this module was imported


    class CountGame:
        name = "countgame"
        action_spec = ActionSpec(names=["inc", "dec", "noop"], descriptions={"inc": "+1", "dec": "-1"}, noop="noop")

        def __init__(self, target=5, events=None):
            SERIAL[0] += 1
            self.serial, self.target, self.events = SERIAL[0], target, events
            self.n_close = self.resets = self.n = self.value = 0
            self._done = False
            self.log("make")

        def log(self, event):
            if self.events:
                with open(self.events, "a") as fh:
                    fh.write(event + "\\n")

        def start(self, seed):
            return random.Random(seed).randrange(-3, 3)

        def reset(self, seed):
            self.resets += 1
            self.value, self.n, self._done = self.start(seed), 0, False
            return self.observe()

        def observe(self):
            return {"value": self.value, "target": self.target, "n": self.n}

        def apply(self, action):
            self.n += 1
            self.value += {"inc": 1, "dec": -1}.get(action, 0)
            self._done = self.value == self.target or self.n >= 30

        @property
        def done(self):
            return self._done

        @property
        def score(self):
            return float(-abs(self.value - self.target))

        def info(self):
            return {"n": self.n}

        def describe(self):
            return ("Count game: move an integer to the target with inc / dec, at most 30 moves. Score is minus "
                    "the final distance. Observation: value, target, n.")

        def close(self):
            self.n_close += 1
            if self.n_close == 1:
                self.log("close")


    class TurnCount(CountGame):
        def step(self, action):
            self.apply(action)
            return self.observe()


    class TickCount(CountGame):
        tick_s = 0.1
        tick_sleep = 0.0

        def tick(self, action):
            if self.tick_sleep:
                time.sleep(self.tick_sleep)
            if not self._done:
                self.apply(action)

        @property
        def ticks(self):
            return self.n
''')


def write_adaptor(d: Path, body: str, *, kind: str = "TickCount", **files: str) -> Path:
    """A fake adaptor directory: countbase.py + game.py (``class Game(<kind>)`` + ``body``)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "countbase.py").write_text(COUNTBASE)
    (d / "game.py").write_text(f"from countbase import {kind}\n\n" + textwrap.dedent(body) +
                               "\n\ndef make_game(**kw):\n    return Game(**kw)\n")
    (d / "ADAPTOR.md").write_text("Fake count adaptor for the pipeline tests.\n")
    for name, text in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
    return d


TICK_GAME = "TIER = 2\nDETERMINISTIC = True\n\nclass Game(TickCount):\n    pass\n"
TURN_GAME = "TIER = 1\n\nclass Game(TurnCount):\n    pass\n"


def events(path: Path) -> list[str]:
    return path.read_text().split() if path.exists() else []


@pytest.fixture
def clean_registry():
    reg._populate()
    games, adaptors = dict(reg.GAMES), dict(reg._ADAPTORS)
    yield reg
    reg.GAMES.clear()
    reg.GAMES.update(games)
    reg._ADAPTORS.clear()
    reg._ADAPTORS.update(adaptors)


def checks(rep) -> dict[str, bool]:
    return {c["check"]: c["ok"] for c in rep.checks}


# ----------------------------------------------------------------------------- loading

def test_load_adaptor_sees_fixed_sibling_helpers(tmp_path):
    body = "from mh_tst_helper import VALUE\n\nclass Game(TickCount):\n    name = f'v{VALUE}'\n"
    a = write_adaptor(tmp_path / "a", body, **{"mh_tst_helper.py": "VALUE = 1\n"})
    assert load_adaptor(a).name == "v1"
    (a / "mh_tst_helper.py").write_text("VALUE = 22  # the agent's fix\n")
    assert load_adaptor(a).name == "v22"                    # not the cached first version
    b = write_adaptor(tmp_path / "b", body, **{"mh_tst_helper.py": "VALUE = 333\n"})
    assert load_adaptor(b).name == "v333"                   # not a's same-named helper
    assert load_adaptor(a).name == "v22"                    # and back: a's directory first again
    assert sys.path[0] == str(a.resolve())                  # stays importable for lazy sibling imports
    assert sys.modules["mh_tst_helper"].__file__ == str(a.resolve() / "mh_tst_helper.py")


def test_load_adaptor_closes_probe_and_reads_declarations(tmp_path):
    log = tmp_path / "events.txt"
    body = """
        TIER = 2
        SEEDS = {"train": (10, 11, 12), "heldout": range(20, 22)}
        EVAL_KWARGS = {"level": 2}

        class Game(TickCount):
            def __init__(self, level, **kw):       # needs a kwarg only EVAL_KWARGS provides
                super().__init__(**kw)
                self.name = f"count-l{level}"
    """
    a = write_adaptor(tmp_path / "a", body)
    spec = load_adaptor(a, kwargs={"events": str(log)})
    assert events(log) == ["make", "close"]                 # the probe instance was released
    assert spec.name == "count-l2" and spec.tier == "tier2" and spec.realtime
    assert spec.seeds == {"train": [10, 11, 12], "heldout": [20, 21]} and spec.eval_kwargs == {"level": 2}
    assert load_adaptor(a, kwargs={"level": 3}).name == "count-l3"     # the user's kwargs win
    assert len(spec.source_sha256) == 64


@pytest.mark.parametrize("decl, msg", [
    ('SEEDS = {"train": [], "heldout": [1]}', "non-empty list of ints"),
    ('SEEDS = {"train": [1, 2], "heldout": [2, 3]}', "disjoint"),
    ('SEEDS = {"train": [True], "heldout": [2]}', "list of ints"),
    ('SEEDS = {"train": [1, 1], "heldout": [2]}', "repeats"),
    ('SEEDS = {"training": [1], "heldout": [2]}', "SEEDS must be"),
    ('EVAL_KWARGS = [("level", 1)]', "EVAL_KWARGS must be a dict"),
    ('EVAL_KWARGS = {"level": float("nan")}', "JSON"),
])
def test_malformed_declarations(tmp_path, decl, msg):
    a = write_adaptor(tmp_path / "a", f"{decl}\n\nclass Game(TickCount):\n    pass\n")
    with pytest.raises(DeclarationError, match=msg):
        load_adaptor(a)
    rep = validate_adaptor(a)
    assert not rep.ok and msg in rep.to_markdown() and "SEEDS / EVAL_KWARGS" in rep.checks[0]["check"]


def test_source_sha256_scope(tmp_path):
    a = write_adaptor(tmp_path / "a", TICK_GAME, **{
        "bridge/bridge.js": "console.log(1)\n", "bridge/package.json": "{}", "bridge/node_modules/dep/index.js": "x\n",
        "runtime/state.json": "{}", "logs/run.txt": "x", "validation.json": "{}", "run.json": "{}",
        "prompt_0.md": "p", "NEEDS.md": "n", "notes.md": "doc", "seeds.json": "[1]", "requirements.txt": "numpy\n"})
    h0 = source_sha256(a)
    assert h0 == source_sha256(a) and h0 == load_adaptor(a).source_sha256

    def changed(rel: str, text: str = "changed") -> bool:
        p = a / rel
        old = p.read_bytes() if p.exists() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        h = source_sha256(a)
        if old is None:
            p.unlink()
        else:
            p.write_bytes(old)
        assert source_sha256(a) == h0
        return h != h0

    for rel in ("game.py", "countbase.py", "bridge/bridge.js", "bridge/new.mjs", "bridge/new.cjs", "bridge/new.ts",
                "setup.sh", "run.bash", "stubs.pyi", "requirements.txt", "bridge/package.json", "validation.py"):
        assert changed(rel), rel                                   # code: part of the hash
    for rel in ("ADAPTOR.md", "NEEDS.md", "validation.json", "validation.md", "run.json", "prompt_0.md",
                "agent_output_0.md", "adapt_trace.jsonl", "runtime/state.json", "logs/run.txt",
                "bridge/node_modules/dep/index.js", "__pycache__/game.cpython-312.pyc", ".env", "worlds/w/level.json",
                "recordings/r.json", ".cache/x.json", "server.jar",
                # what adaptors write while they play, next to game.py or in a new data directory
                "seeds.json", "state.json", "verification.json", "eula.txt", "server.port", "config.yaml",
                "settings.toml", "setup.cfg", "bridge/package-lock.json", "bridge/bot.log",
                "observations/3f2a9c.json", "data/frames/0001.json", "data/notes.txt"):
        assert not changed(rel), rel                               # docs, data, runtime, orchestrator files
    vendor = a / "vendor"                                          # e.g. a downloaded game's scripts
    vendor.mkdir()
    (vendor / "lib.py").write_text("x = 1\n")
    with open(vendor / "bundle.js", "wb") as fh:                   # sparse: > 20 MB of code without the disk
        fh.truncate(21 * 2**20)
    assert source_sha256(a) == h0                                  # a heavy directory is not source
    (vendor / "bundle.js").unlink()
    h1 = source_sha256(a)
    assert h1 != h0                                                # small again: vendor/lib.py counts
    with open(a / "bridge" / "world.bin", "wb") as fh:             # only code counts toward that limit:
        fh.truncate(21 * 2**20)                                    # data piling up beside bridge.js...
    for i in range(MAX_SUBDIR_FILES + 1):
        (a / "bridge" / f"frame{i}.json").touch()
    assert source_sha256(a) == h1                                  # ... never drops it from the hash


def test_runtime_outputs_keep_the_source_hash(tmp_path):
    # r6's ARC adaptor wrote observations/<sha>.json frames while it played: with .json in the hash
    # a resumed search was refused ("Search configuration changed (game_source)")
    body = """
        import hashlib
        import json
        from pathlib import Path

        TIER = 2
        HERE = Path(__file__).resolve().parent

        class Game(TickCount):
            def reset(self, seed):
                obs = super().reset(seed)
                frame = json.dumps({"seed": seed, "obs": obs}).encode()
                (HERE / "observations").mkdir(exist_ok=True)
                (HERE / "observations" / f"{hashlib.sha256(frame).hexdigest()}.json").write_bytes(frame)
                (HERE / "state.json").write_text(json.dumps({"seed": seed}))
                (HERE / "eula.txt").write_text("eula=true\\n")
                return obs
    """
    a = write_adaptor(tmp_path / "a", body)
    h0 = source_sha256(a)
    rep = validate_adaptor(a)
    assert rep.ok and not rep.warnings, rep.to_markdown()
    assert len(list((a / "observations").glob("*.json"))) >= 2 and (a / "state.json").exists()
    assert source_sha256(a) == h0 == load_adaptor(a).source_sha256


def test_validator_warns_when_running_changes_code(tmp_path):
    body = """
        from pathlib import Path

        TIER = 2
        HERE = Path(__file__).resolve().parent

        class Game(TickCount):
            def reset(self, seed):
                (HERE / "bot.js").write_text(f"start({seed})\\n")             # a script rewritten per episode
                (HERE / f"plan_{seed}.py").write_text(f"SEED = {seed}\\n")    # generated code
                (HERE / "runtime").mkdir(exist_ok=True)
                (HERE / "runtime" / "plan.py").write_text(f"SEED = {seed}\\n")    # fine: under runtime/
                return super().reset(seed)
    """
    a = write_adaptor(tmp_path / "a", body, **{"bot.js": "start(null)\n"})
    rep = validate_adaptor(a)
    assert rep.ok, rep.to_markdown()                               # a warning, not a failure
    md = rep.to_markdown()
    assert "- [!] running the adaptor changed code files: bot.js (modified), plan_0.py (added), plan_1.py " \
           "(added) — the search manifest pins the code files" in md, md
    assert "runtime/plan.py" not in md and "under runtime/" in md


# ----------------------------------------------------------------------------- validation

def test_validator_warnings_never_fail(tmp_path):
    body = """
        TIER = 1

        class Game(TurnCount):
            def start(self, seed):
                return 0                            # a fixed opening position

            def step(self, action):
                if action not in self.action_spec.names:
                    raise ValueError(f"unknown action {action!r}")
                return super().step(action)
    """
    rep = validate_adaptor(write_adaptor(tmp_path / "a", body, kind="TurnCount"), expected_tier=1)
    assert rep.ok, rep.to_markdown()
    md = rep.to_markdown()
    assert "- [!] step('__not_an_action__') raised ValueError" in md
    assert "- [!] seeds 0 and 1 give identical first observations" in md
    assert rep.to_json()["warnings"] == rep.warnings and len(rep.warnings) == 2


def test_validator_tier_checks(tmp_path):
    tick = write_adaptor(tmp_path / "tick", TICK_GAME)
    turn = write_adaptor(tmp_path / "turn", TURN_GAME, kind="TurnCount")
    assert validate_adaptor(tick, expected_tier=2).ok and validate_adaptor(turn, expected_tier=1).ok
    rep = validate_adaptor(tick, expected_tier=1)
    assert not rep.ok and checks(rep)["tier 1, as the study decided"] is False
    assert "tier 1 needs a TurnGame" in rep.to_markdown()
    rep = validate_adaptor(turn, expected_tier=2)
    assert not rep.ok and "tier 2 needs a RealtimeGame" in rep.to_markdown()
    lying = write_adaptor(tmp_path / "lying", "TIER = 1\n\nclass Game(TickCount):\n    pass\n")
    rep = validate_adaptor(lying)
    assert not rep.ok and checks(rep)["TIER agrees with the game's kind (2 <=> tick())"] is False


@pytest.mark.parametrize("override, failing", [
    ("    @property\n    def score(self):\n        return float('nan') if self.n else 0.0\n",
     "done is bool and score is a finite number after the actions"),
    ("    @property\n    def score(self):\n        return True if self.n else 0.0\n",
     "done is bool and score is a finite number after the actions"),
    ("    def observe(self):\n        return {'value': self.value, 'ratio': float('inf') if self.n else 0.0}\n",
     "observations after the actions are strict JSON"),
    ("    def observe(self):\n        return {'seen': {self.n}} if self.n == 2 else super().observe()\n",
     "observations after the actions are strict JSON"),
])
def test_validator_rechecks_state_after_actions(tmp_path, override, failing):
    rep = validate_adaptor(write_adaptor(tmp_path / "a", f"TIER = 2\n\nclass Game(TickCount):\n{override}"))
    assert not rep.ok and any(c["check"].endswith(failing) and not c["ok"] for c in rep.checks), rep.to_markdown()
    assert all(c["ok"] for c in rep.checks if "after reset" in c["check"])     # fine before the first action


def test_validator_tick_pace(tmp_path):
    slow = write_adaptor(tmp_path / "slow", "TIER = 2\n\nclass Game(TickCount):\n    tick_s = 0.05\n    tick_sleep = 0.08\n")
    rep = validate_adaptor(slow)
    assert not rep.ok and "cannot keep this pace" in rep.to_markdown()
    tight = write_adaptor(tmp_path / "tight",
                          "TIER = 2\n\nclass Game(TickCount):\n    tick_s = 0.2\n    tick_sleep = 0.12\n")
    rep = validate_adaptor(tight)
    assert rep.ok, rep.to_markdown()
    assert any("little slack left" in w for w in rep.warnings)


def test_validator_determinism_uses_a_fresh_instance(tmp_path):
    # the start depends on the instance (not the seed alone): replaying on the same object used to pass
    per_instance = write_adaptor(tmp_path / "inst", textwrap.dedent("""
        TIER = 2
        DETERMINISTIC = True

        class Game(TickCount):
            def start(self, seed):
                return self.serial * 100 + seed
    """))
    rep = validate_adaptor(per_instance)
    assert not rep.ok and checks(rep)[
        "DETERMINISTIC: reset(seed) on a fresh instance reproduces the first observation"] is False
    # the start depends on how often this object was reset: a fresh instance reproduces it
    per_reset = write_adaptor(tmp_path / "reset", textwrap.dedent("""
        TIER = 2
        DETERMINISTIC = True

        class Game(TickCount):
            def start(self, seed):
                return self.resets * 100 + seed
    """))
    rep = validate_adaptor(per_reset)
    assert rep.ok, rep.to_markdown()


def test_validator_uses_declared_seeds(tmp_path):
    body = """
        TIER = 2
        SEEDS = {"train": [10, 11], "heldout": [20, 21]}

        class Game(TickCount):
            def reset(self, seed):
                if seed not in (10, 11, 20, 21):
                    raise ValueError(f"no scenario for seed {seed}")
                return super().reset(seed)
    """
    a = write_adaptor(tmp_path / "a", body)
    rep = validate_adaptor(a)
    assert rep.ok, rep.to_markdown()
    names = list(checks(rep))
    assert "seed 10: observe() JSON-serialisable" in names and "seed 20: observe() JSON-serialisable" in names
    assert not any(n.startswith(("seed 0:", "seed 1:")) for n in names)
    assert "2 train (10, 11), 2 held-out (20, 21)" in rep.to_markdown()
    rep = validate_adaptor(a, seeds=(11,))                 # explicit seeds still cover the first of each list
    assert rep.ok and {f"seed {s}: info() is a dict" for s in (11, 10, 20)} <= set(checks(rep))


def test_validator_closes_every_instance(tmp_path):
    log = tmp_path / "events.txt"
    a = write_adaptor(tmp_path / "a", TICK_GAME)
    rep = validate_adaptor(a, kwargs={"events": str(log)})
    assert rep.ok and not rep.warnings, rep.to_markdown()
    assert events(log) == ["make", "close"] * 3        # probe, played instance, replay: one at a time
    assert checks(rep)["close() releases the instance (called after every episode)"]
    not_idempotent = write_adaptor(tmp_path / "b", textwrap.dedent("""
        TIER = 2

        class Game(TickCount):
            def close(self):
                super().close()
                if self.n_close > 1:
                    raise RuntimeError("bridge already closed")
    """))
    rep = validate_adaptor(not_idempotent)
    assert rep.ok and any("close() must be idempotent" in w for w in rep.warnings)
    broken = write_adaptor(tmp_path / "c", textwrap.dedent("""
        TIER = 2

        class Game(TickCount):
            def close(self):
                raise RuntimeError("server refused shutdown")
    """))
    rep = validate_adaptor(broken)
    assert not rep.ok and "server refused shutdown" in rep.to_markdown()
    crashing = write_adaptor(tmp_path / "d", textwrap.dedent("""
        TIER = 2

        class Game(TickCount):
            def describe(self):
                raise RuntimeError("no client")
    """))
    rep = validate_adaptor(crashing, kwargs={"events": str(tmp_path / "events_d.txt")})
    assert not rep.ok and "no client" in rep.error
    ev = events(tmp_path / "events_d.txt")
    assert ev.count("make") == ev.count("close") == 2                # closed even after the crash


def test_validation_json_and_markdown_carry_warnings(tmp_path):
    a = write_adaptor(tmp_path / "a", "TIER = 2\n\nclass Game(TickCount):\n    def start(self, seed):\n        return 1\n")
    rep = validate_adaptor(a)
    data = json.loads(json.dumps(rep.to_json()))
    assert data["ok"] is True and data["warnings"] and "identical first observations" in data["warnings"][0]


# ----------------------------------------------------------------------------- registry

def test_register_adaptor_is_never_replaced_by_a_test_bed(tmp_path, clean_registry):
    snake = write_adaptor(tmp_path / "snake", "TIER = 2\n\nclass Game(TickCount):\n    name = 'snake'\n")
    key = reg.register_adaptor(snake)
    assert key == "adaptor:snake"
    assert isinstance(reg.make_game("snake"), SnakeGame)            # the test bed keeps its name
    g = reg.make_game("adaptor:snake")
    assert not isinstance(g, SnakeGame) and g.name == "snake"       # ... and the adaptor stays reachable
    assert reg.adaptor_spec(key).path == snake.resolve() and reg.adaptor_spec("snake") is None
    assert reg.register_adaptor(snake) == key                       # the same directory: the same key
    other = write_adaptor(tmp_path / "other", "TIER = 2\n\nclass Game(TickCount):\n    name = 'snake'\n")
    assert reg.register_adaptor(other) == "adaptor:snake:2"
    assert reg.adaptor_spec("adaptor:snake").path == snake.resolve()
    assert reg.game_tier(key) == "tier2"


def test_register_adaptor_in_a_fresh_process(tmp_path):
    # the original failure: `loop --adaptor` registered "snake" before any test bed was imported, and
    # the first make_game() imported the snake test bed over it
    snake = write_adaptor(tmp_path / "snake", "TIER = 2\n\nclass Game(TickCount):\n    name = 'snake'\n")
    code = textwrap.dedent(f"""
        from metaharness.games import registry as reg
        key = reg.register_adaptor({str(snake)!r})
        print(key, type(reg.make_game(key)).__module__.startswith("_adaptor_"), type(reg.make_game("snake")).__module__)
    """)
    proc = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["adaptor:snake", "True", "metaharness.testbeds.snake"]


def test_register_adaptor_keeps_its_own_name(tmp_path, clean_registry):
    name = f"count-{tmp_path.name}"
    body = f"TIER = 1\nEVAL_KWARGS = {{'target': 3}}\n\nclass Game(TurnCount):\n    name = {name!r}\n"
    a = write_adaptor(tmp_path / "a", body, kind="TurnCount")
    key = reg.register_adaptor(a)
    assert key == name and reg.adaptor_spec(key).eval_kwargs == {"target": 3}
    reg.make_game("snake")                                           # _populate() must not touch it
    reg.register(name, "tier2")(SnakeGame)                           # nor a late test-bed registration
    assert reg.GAMES[key][1] is reg.adaptor_spec(key).make_game and reg.game_tier(key) == "tier1"
    assert os.path.isdir(reg.adaptor_spec(key).path)


# game.py of its own: the helper module is imported lazily, in make_game and while playing
LAZY_GAME = textwrap.dedent('''
    from countbase import TurnCount

    TIER = 1


    class Game(TurnCount):
        def __init__(self, helper, **kw):
            super().__init__(**kw)
            self.made_with, self.played_with = helper, None
            self.name = f"lazy-{helper.WHO}"

        def step(self, action):
            import mh_tst_lazy
            self.played_with = mh_tst_lazy
            return super().step(action)


    def make_game(**kw):
        import mh_tst_lazy
        return Game(mh_tst_lazy, **kw)
''')


def test_adaptors_keep_their_own_helpers(tmp_path, clean_registry):
    # loop --adaptor A --heldout-game B, both with a helper of the same name: registering B evicted
    # A's helper and put B first on sys.path, so every later construction / lazy import of A ran B's
    dirs = {who: write_adaptor(tmp_path / who, "", **{"game.py": LAZY_GAME, "mh_tst_lazy.py": f"WHO = {who!r}\n"})
            for who in ("a", "b")}
    keys = {who: clean_registry.register_adaptor(d) for who, d in dirs.items()}
    assert keys == {"a": "lazy-a", "b": "lazy-b"}
    first: dict[str, object] = {}

    def play(who: str) -> str:
        g = clean_registry.make_game(keys[who])
        try:
            g.reset(0)
            g.step("inc")
        finally:
            g.close()
        helper = first.setdefault(who, g.made_with)
        assert g.made_with is g.played_with is helper, who          # the module it loaded, not a re-import
        assert helper.__file__ == str(dirs[who].resolve() / "mh_tst_lazy.py")
        return helper.WHO

    try:
        assert [play(w) for w in "abab"] == list("abab")
        (dirs["a"] / "mh_tst_lazy.py").write_text("WHO = 'a2'  # the agent's fix\n")
        keys["a"] = clean_registry.register_adaptor(dirs["a"])      # reloaded while b's helper is active
        del first["a"]
        assert [play(w) for w in "ba"] == ["b", "a2"]                # b's own module back; a's fresh one
    finally:
        sys.modules.pop("mh_tst_lazy", None)


NS_GAME = LAZY_GAME.replace("import mh_tst_lazy\n", "import mh_tst_ns.helper as mh_tst_lazy\n")


def test_adaptors_keep_their_own_namespace_package_helpers(tmp_path, clean_registry):
    # helper directories without __init__.py (namespace packages) are shadowed the same way
    dirs = {who: write_adaptor(tmp_path / who, "", **{"game.py": NS_GAME,
                                                      "mh_tst_ns/helper.py": f"WHO = {who!r}\n"})
            for who in ("a", "b")}
    keys = {who: clean_registry.register_adaptor(d) for who, d in dirs.items()}

    def play(who: str) -> str:
        g = clean_registry.make_game(keys[who])
        try:
            g.reset(0)
            g.step("inc")
        finally:
            g.close()
        assert g.made_with.WHO == g.played_with.WHO, who
        return g.played_with.WHO

    try:
        assert [play(w) for w in "abab"] == list("abab")
    finally:
        for name in [n for n in sys.modules if n == "mh_tst_ns" or n.startswith("mh_tst_ns.")]:
            sys.modules.pop(name, None)

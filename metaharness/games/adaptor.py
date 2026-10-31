"""Generated game adaptors — the contract the ``adapt`` phase must satisfy.

An adaptor is a directory (written by Astra, never by hand in this repo):

    <adaptor>/
      game.py            defines make_game(**kwargs) -> TurnGame | RealtimeGame
      ADAPTOR.md         how it connects to the real game, what it observes, how
                         seeds map to scenarios, what score means, its limits
      requirements.txt   (optional) extra pip packages the adaptor needs
      env.example        (optional) environment variables it reads (credentials,
                         server address...). Real values live in the user's .env.
                         An uncommented ``NAME=`` line is a variable the user MUST
                         set (the orchestrator stops and asks for it); keep optional
                         ones commented (``# NAME=default``).
      runtime/           (optional) everything the adaptor writes while it runs: screenshots,
                         caches, logs, server / world directories, generated configs

``game.py`` may import anything installed, ``metaharness.games.base`` and sibling modules of
the adaptor directory. The object returned by ``make_game`` must implement ``TurnGame``
(``step``) or ``RealtimeGame`` (``tick`` + ``tick_s`` + ``ticks``) from ``games/base.py`` and
additionally ``name`` (str). Optional module-level declarations:

    TIER = 1 | 2           must agree with the kind: 2 <=> the game has tick()
    DETERMINISTIC = True   reset(seed) on a fresh instance reproduces the first observation
    SEEDS = {"train": [...], "heldout": [...]}   non-empty, disjoint lists of ints: the
                           scenarios the search trains on and is judged on (declare them when
                           only some seeds are valid, e.g. a fixed set of levels or maps); optional
                           "dev" (the proposer's own trials) and "validation" (top-k re-evaluation)
    EVAL_KWARGS = {...}    JSON kwargs the evaluator passes to make_game

Optional methods: ``score_breakdown() -> dict[str, float]`` (numeric components of the current
score, e.g. progress vs survival; recorded per episode so the harness designer sees where points
came from) and ``close()``: called on every instance after its episode (and by validation); it
must release every process, server and connection the instance started, and be idempotent.

How the evaluator drives a game: a fresh instance per episode — make_game, reset(seed), play,
read the score, close(). It serialises its own calls into the object (tick / step / observe /
score / done from different threads never overlap); only the adaptor's own background threads
need locking. An exception from tick() / step() ends the episode as a game error: ignore a
malformed or illegal action, or report it in the next observation. ``info()`` is evaluator-only
(never shown to the harness). For an episode spoiled by the infrastructure (server crash, lost
connection — not bad play) report ``info()["valid_trial"] = False`` with
``info()["invalid_reason"]``: the evaluator retries that seed once on a fresh game and otherwise
records an error. Observations must be strict JSON (dicts/lists/str/finite numbers/bools,
< 200 kB); binary perception (screenshots) is exposed as file paths so a harness can hand them
to ``ctx.think(images=[...])``.

Runtime outputs (screenshots, caches, logs, recordings, server / world directories, anything
generated while playing) belong under ``runtime/`` or another directory the source hash skips
(node_modules, .cache, logs, recordings, worlds). The code files elsewhere in the directory
(``*.py *.pyi *.js *.mjs *.cjs *.ts *.sh *.bash``, requirements.txt, package.json) are pinned by
the search manifest (``source_sha256``): one written or changed while the adaptor runs makes a
resumed search refuse to continue.

``validate_adaptor`` is the generic smoke test the orchestrator runs (and shows the model when it
fails): load, construct, the tier the study chose, reset two seeds (the first train and held-out
seed when SEEDS is declared), observe, describe, action spec, apply the noop and every action
once, done/score/observation types after them, tick() time against tick_s, one malformed action,
determinism of reset(seed) on a fresh instance (``DETERMINISTIC``), close(), and that running
changed no code file. Warnings (``[!]``) do not fail it.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import json
import math
import numbers
import os
import re
import statistics
import sys
import sysconfig
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .base import ActionSpec, close_game

REQUIRED_FILES = ("game.py", "ADAPTOR.md")
MALFORMED_ACTION = "__not_an_action__"
MAX_OBS_CHARS = 200_000
VARIETY_ACTIONS = 300             # fixed script played on two seeds: identical trajectories fail validation
                                  # (intros that ignore input took one game ~240 steps to diverge)

# source_sha256 scope (see its docstring): code only — data files are what adaptors write while they play
# (r6's observations/<sha>.json frames, state.json, eula.txt, package-lock.json), and pinning them refused resumes
SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".sh", ".bash"})
SOURCE_NAMES = frozenset({"requirements.txt", "package.json"})
SKIP_DIRS = frozenset({"node_modules", "runtime", "__pycache__", ".cache", "logs", "recordings", "worlds", ".venv"})
ORCHESTRATOR_FILES = re.compile(r"(validation\.(md|json)|prompt_\d+\.md|agent_output_\d+\.md|adapt_trace\.jsonl"
                                r"|NEEDS\.md|run\.json)\Z")
MAX_SUBDIR_BYTES = 20 * 2**20
MAX_SUBDIR_FILES = 5000
MAX_HASHED_BYTES = 2 * 2**20


_SEED_GROUPS = ("train", "heldout", "dev", "validation")


class DeclarationError(ValueError):
    """A module-level declaration of game.py (SEEDS / EVAL_KWARGS) is malformed."""


@dataclass
class AdaptorSpec:
    path: Path
    name: str
    tier: str                         # "tier1" | "tier2"
    make_game: Callable[..., Any]
    realtime: bool
    deterministic: bool = False
    module: Any = None
    seeds: dict[str, list[int]] | None = None                  # SEEDS {"train": [...], "heldout": [...]}
    eval_kwargs: dict[str, Any] = field(default_factory=dict)  # EVAL_KWARGS for make_game
    source_sha256: str = ""                                     # source_sha256(path) at load time


def _lib_roots() -> tuple[str, ...]:
    roots = {sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    try:
        import site
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except Exception:  # noqa: BLE001 — some embedded interpreters lack these
        pass
    out = set()
    for r in filter(None, roots):
        out.update({os.path.abspath(r) + os.sep, os.path.realpath(r) + os.sep})
    return tuple(out)


def _module_paths(mod: Any) -> list[str]:
    paths = [f for f in (getattr(mod, "__file__", None),) if isinstance(f, str)]
    try:
        paths += [p for p in (getattr(mod, "__path__", None) or ()) if isinstance(p, str)]
    except Exception:  # noqa: BLE001 — exotic loaders
        pass
    return paths


_PARKED: dict[str, dict[str, Any]] = {}       # loaded adaptor root -> its modules another adaptor evicted


def _forget_modules(path: Path, *, own: bool = True) -> None:
    """Drop cached imports that would shadow this adaptor directory's modules on (re)load.

    A long-lived process validates after every adapt attempt: r7's game.py imports scoring.py,
    and the agent's fix to scoring.py stayed invisible because ``scoring`` was still cached; a
    same-named module of another adaptor directory would shadow it the same way. Evicts modules
    loaded from this directory and modules named after its ``*.py`` files / packages that come
    from elsewhere. Built-in, standard-library, installed and metaharness modules are never touched.
    ``own=False`` (``_activate``, before every construction) keeps this directory's modules. Evicted
    modules of another loaded adaptor are parked and put back by its next activation: its lazy
    imports then get the modules its game.py imported, not a re-executed second copy."""
    root = str(path) + os.sep
    try:
        local = {p.stem for p in path.glob("*.py")}
        local |= {p.name for p in path.iterdir() if (p / "__init__.py").is_file()}
        # helper directories without __init__.py are namespace packages: same shadowing problem
        local |= {p.name for p in path.iterdir() if p.is_dir() and p.name.isidentifier()
                  and p.name not in SKIP_DIRS and any(p.glob("*.py"))}
    except OSError:                                    # the directory is gone: nothing of it to shadow
        local = set()
    protected = set(sys.builtin_module_names) | set(getattr(sys, "stdlib_module_names", ()))
    libs = _lib_roots()
    for name, mod in list(sys.modules.items()):
        top = name.partition(".")[0]
        if not (own or top in local):
            continue
        paths = _module_paths(mod)
        if not paths or top in protected or top == "metaharness" or name == "__main__":
            continue
        where = [os.path.abspath(p) for p in paths]
        if top in local:                               # few: worth resolving symlinked spellings
            where += [os.path.realpath(p) for p in paths]
        if any(p.startswith(libs) for p in where):
            continue
        # a namespace package's __path__ spans every directory on sys.path: never anyone's own
        ns = getattr(mod, "__file__", None) is None and hasattr(mod, "__path__")
        mine = not ns and any(p.startswith(root) for p in where)
        if mine and own:
            del sys.modules[name]
        elif top in local and not mine:
            del sys.modules[name]
            owner = max((r for r in _PARKED if any(p.startswith(r) for p in where)), key=len, default="")
            if owner and not ns:
                _PARKED[owner][name] = mod
    parked = _PARKED.pop(root, {})
    _PARKED[root] = {}                                 # a loaded adaptor: its evicted modules get parked
    if not own:                                        # (a reload drops them instead: stale)
        for name, mod in parked.items():
            sys.modules.setdefault(name, mod)


def _activate(path: Path, *, fresh: bool = False) -> None:
    """Point sibling imports at this adaptor directory: evict what would shadow its modules
    (``_forget_modules``; ``fresh``: its own cached ones too), put it first on sys.path and drop
    the finders' directory caches (helper files added since the last import)."""
    _forget_modules(path, own=fresh)
    if str(path) in sys.path:                          # sibling helper modules, also imported lazily:
        sys.path.remove(str(path))                     # this directory ahead of other adaptors'
    sys.path.insert(0, str(path))
    importlib.invalidate_caches()


def _activating(path: Path, make_game: Callable[..., Any]) -> Callable[..., Any]:
    """``make_game`` that activates its adaptor before every construction. Loading a second adaptor
    (``loop --adaptor A --heldout-game B``, both with a bridge.py) evicts A's same-named helpers
    and puts B first on sys.path; without this, A's lazy imports — in make_game or while playing —
    ran B's helper. The evaluator constructs one game at a time, so imports while playing resolve
    to the adaptor of the game being played."""
    @functools.wraps(make_game)
    def activated(**kwargs: Any) -> Any:
        _activate(path)
        return make_game(**kwargs)
    return activated


def _seed_list(value: Any, key: str) -> list[int]:
    if not isinstance(value, (list, tuple, range)) or not value or \
            not all(isinstance(s, int) and not isinstance(s, bool) for s in value):
        raise DeclarationError(f"SEEDS[{key!r}] must be a non-empty list of ints, got {value!r:.200}")
    if len(set(value)) != len(value):
        raise DeclarationError(f"SEEDS[{key!r}] repeats a seed")
    return [int(s) for s in value]


def _declared_seeds(module: Any) -> dict[str, list[int]] | None:
    raw = getattr(module, "SEEDS", None)
    if raw is None:
        return None
    if not isinstance(raw, dict) or not {"train", "heldout"} <= set(raw) or not set(raw) <= set(_SEED_GROUPS):
        raise DeclarationError(f'SEEDS must be {{"train": [int, ...], "heldout": [int, ...]}} (optional "dev", '
                               f'"validation"), got {raw!r:.200}')
    seeds = {k: _seed_list(raw[k], k) for k in _SEED_GROUPS if k in raw}
    names = list(seeds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sorted(set(seeds[a]) & set(seeds[b]))
            if overlap:
                raise DeclarationError(f"SEEDS {a} and {b} must be disjoint; both contain {overlap[:10]}")
    return seeds


def _declared_eval_kwargs(module: Any) -> dict[str, Any]:
    raw = getattr(module, "EVAL_KWARGS", None)
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(isinstance(k, str) for k in raw):
        raise DeclarationError(f"EVAL_KWARGS must be a dict of make_game keyword arguments, got {raw!r:.200}")
    try:
        json.dumps(raw, allow_nan=False)          # recorded in the search manifest
    except (TypeError, ValueError) as e:
        raise DeclarationError(f"EVAL_KWARGS must be JSON-serialisable: {e}") from None
    return dict(raw)


def _source_files(d: Path, rel: str, top: bool, out: list[tuple[str, str, int]]) -> tuple[int, int] | None:
    """Collect (relative path, path, size) of the code files under ``d`` into ``out``; returns the
    subtree's code (bytes, files), or None for a subdirectory too heavy to be source (nothing
    collected). Only code counts: runtime outputs piling up beside it never drop it from the hash."""
    found: list[tuple[str, str, int]] = []
    size = count = 0
    try:
        entries = sorted(os.scandir(d), key=lambda e: e.name)
    except OSError:
        return 0, 0
    for e in entries:
        try:
            if e.name.startswith(".") or e.is_symlink():
                continue
            if e.is_dir():
                if e.name in SKIP_DIRS:
                    continue
                sub = _source_files(Path(e.path), f"{rel}{e.name}/", False, found)
                if sub:
                    size, count = size + sub[0], count + sub[1]
            elif (os.path.splitext(e.name)[1] in SOURCE_SUFFIXES or e.name in SOURCE_NAMES) and e.is_file() \
                    and not (top and ORCHESTRATOR_FILES.match(e.name)):
                n = e.stat().st_size
                size, count = size + n, count + 1
                found.append((rel + e.name, e.path, n))
        except OSError:
            continue
        if not top and (size > MAX_SUBDIR_BYTES or count > MAX_SUBDIR_FILES):
            return None
    out += found
    return size, count


def _source_digests(path: Path | str) -> tuple[str, dict[str, str]]:
    """(``source_sha256``, {relative path: sha256 of that file's part of it}): the parts name what changed."""
    files: list[tuple[str, str, int]] = []
    _source_files(Path(path), "", True, files)
    h = hashlib.sha256()
    parts: dict[str, str] = {}
    for rel, p, size in sorted(files):
        try:
            with open(p, "rb") as fh:
                data = fh.read(MAX_HASHED_BYTES)
        except OSError:
            data = b"<unreadable>"
        part = f"{rel}\0{size}\0{len(data)}\0".encode("utf-8", "surrogateescape") + data   # undecodable names too
        h.update(part)
        parts[rel] = hashlib.sha256(part).hexdigest()
    return h.hexdigest(), parts


def source_sha256(path: Path | str) -> str:
    """sha256 of an adaptor's code, so an archive can tell which adaptor version produced its scores
    (the search manifest pins it as ``game_source``; a resumed search with another hash is refused).

    Covered: code files — ``SOURCE_SUFFIXES`` (.py .pyi .js .mjs .cjs .ts .sh .bash) and
    ``SOURCE_NAMES`` (requirements.txt, package.json) — at the top level and in subdirectories (e.g.
    ``bridge/*.js``), in sorted path order, each as its relative path, size and first
    ``MAX_HASHED_BYTES`` bytes. Not covered: every other file (.json .txt .yaml .toml .md, images...:
    adaptors write such files while they play — observation frames, state.json, eula.txt,
    package-lock.json — and that must not change the hash), hidden files and directories, symlinks,
    runtime / dependency directories (``SKIP_DIRS``: node_modules, runtime, __pycache__, .cache, logs,
    recordings, worlds, .venv), any subdirectory with over 20 MB or 5000 files of code, and what the
    orchestrator writes into the directory (validation.md / .json, prompt_N.md, agent_output_N.md,
    adapt_trace.jsonl, NEEDS.md, run.json)."""
    return _source_digests(path)[0]


def load_adaptor(path: Path | str, *, kwargs: dict[str, Any] | None = None) -> AdaptorSpec:
    """Import ``<path>/game.py`` afresh and read its declarations; one probe instance (EVAL_KWARGS
    updated by ``kwargs``) tells the kind and name and is closed again. The spec's ``make_game``
    activates the adaptor before every construction (``_activating``). Raises on any failure
    (``DeclarationError`` for malformed SEEDS / EVAL_KWARGS)."""
    path = Path(path).resolve()
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"adaptor at {path} is missing {missing}")
    _activate(path, fresh=True)
    mod_name = f"_adaptor_{abs(hash(str(path)))}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(mod_name, path / "game.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    if not hasattr(module, "make_game"):
        raise ImportError("game.py must define make_game(**kwargs)")
    seeds = _declared_seeds(module)
    eval_kwargs = _declared_eval_kwargs(module)
    make = _activating(path, module.make_game)
    probe = make(**{**eval_kwargs, **(kwargs or {})})
    try:
        realtime = hasattr(probe, "tick")
        name = str(getattr(probe, "name", path.name))
    finally:
        close_game(probe)                              # it may own a server / emulator / connection
    tier = getattr(module, "TIER", 2 if realtime else 1)
    return AdaptorSpec(path=path, name=name, tier=f"tier{int(tier)}", make_game=make,
                       realtime=realtime, deterministic=bool(getattr(module, "DETERMINISTIC", False)),
                       module=module, seeds=seeds, eval_kwargs=eval_kwargs, source_sha256=source_sha256(path))


@dataclass
class ValidationReport:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)    # shown as [!]; never flip ``ok``

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "ok": ok, "detail": detail})
        if not ok:
            self.ok = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "warnings": self.warnings, "error": self.error}

    def to_markdown(self) -> str:
        lines = [f"# Adaptor validation: {'PASS' if self.ok else 'FAIL'}", ""]
        for c in self.checks:
            lines.append(f"- [{'x' if c['ok'] else ' '}] {c['check']}" + (f" — {c['detail']}" if c["detail"] else ""))
        lines += [f"- [!] {w}" for w in self.warnings]
        if self.error:
            lines += ["", "```", self.error[-4000:], "```"]
        return "\n".join(lines) + "\n"


def _json_ok(obj: Any) -> tuple[bool, str]:
    try:
        s = json.dumps(obj, allow_nan=False)
    except (TypeError, ValueError) as e:
        return False, f"observation is not strict JSON: {e}"
    if len(s) > MAX_OBS_CHARS:
        return False, f"observation is {len(s)} bytes; keep it under 200 kB (summarise / crop)"
    return True, f"{len(s)} bytes"


def _finite_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def _state_ok(g: Any) -> tuple[bool, str]:
    done, score = g.done, g.score
    return isinstance(done, bool) and _finite_number(score), f"done={done!r} score={score!r}"


def _seeds_text(seeds: list[int]) -> str:
    return ", ".join(map(str, seeds[:5])) + (", …" if len(seeds) > 5 else "")


def episode_state(obs: Any) -> str:
    """An observation as episode state: top-level fields named after the seed label the episode, they are
    not its state (two seeds that differ only there play the same game)."""
    if isinstance(obs, dict):
        obs = {k: v for k, v in obs.items() if "seed" not in str(k).lower()}
    return json.dumps(obs, sort_keys=True, default=str)


def _trajectory(new_game: Callable[[], Any], release: Callable[..., None], seed: int, actions: list[str]
                ) -> tuple[str, int]:
    """(hash of the episode states, actions applied) of a fixed action script from reset(seed) on a fresh instance."""
    g = new_game()
    try:
        g.reset(seed)
        h = hashlib.sha256(episode_state(g.observe()).encode())
        step = g.tick if hasattr(g, "tick") else g.step
        n = 0
        for i in range(VARIETY_ACTIONS):
            if g.done:
                break
            step(actions[(i // 3) % len(actions)])
            n += 1
            h.update(episode_state(g.observe()).encode())
        return h.hexdigest(), n
    finally:
        release(g, check=False)


def validate_adaptor(path: Path | str, *, kwargs: dict[str, Any] | None = None,
                     seeds: tuple[int, ...] | None = None, max_actions: int = 12,
                     time_budget_s: float = 120.0, expected_tier: int | None = None) -> ValidationReport:
    """The generic smoke test (module docstring). ``kwargs`` update the adaptor's EVAL_KWARGS for
    every instance; ``seeds`` default to the first train and held-out seed of SEEDS, else (0, 1);
    ``expected_tier`` is the tier the study chose. Every instance it creates is closed. The code
    files are hashed before and after: one the run wrote or changed is a warning, since the search
    manifest pins them (``source_sha256``) and code written while playing would break a resume."""
    path = Path(path).resolve()                        # an adaptor that chdirs: still this directory
    before = _source_digests(path)[1]
    rep = _smoke_test(path, kwargs=kwargs, seeds=seeds, max_actions=max_actions, time_budget_s=time_budget_s,
                      expected_tier=expected_tier)
    after = _source_digests(path)[1]
    changed = [f"{rel} ({'added' if rel not in before else 'removed' if rel not in after else 'modified'})"
               for rel in sorted(before.keys() | after.keys()) if before.get(rel) != after.get(rel)]
    if changed:
        rep.warn(f"running the adaptor changed code files: {', '.join(changed[:10])}"
                 + (f" and {len(changed) - 10} more" if len(changed) > 10 else "")
                 + " — the search manifest pins the code files (source_sha256), so a resumed search would "
                   "refuse to continue; write what the adaptor generates while it runs under runtime/")
    return rep


def _smoke_test(path: Path, *, kwargs: dict[str, Any] | None, seeds: tuple[int, ...] | None, max_actions: int,
                time_budget_s: float, expected_tier: int | None) -> ValidationReport:
    rep = ValidationReport(ok=True)
    t0 = time.time()
    try:
        spec = load_adaptor(path, kwargs=kwargs)
        rep.add("load game.py and make_game()", True, f"name={spec.name} tier={spec.tier} realtime={spec.realtime}")
    except DeclarationError as e:
        rep.add("SEEDS / EVAL_KWARGS declarations are well-formed", False, str(e))
        return rep
    except Exception:  # noqa: BLE001
        rep.add("load game.py and make_game()", False)
        rep.error = traceback.format_exc()
        return rep
    kind = "real-time (has tick)" if spec.realtime else "turn-based (no tick)"
    declared = getattr(spec.module, "TIER", None)
    if declared is not None:
        rep.add("TIER agrees with the game's kind (2 <=> tick())",
                declared in (1, 2) and not isinstance(declared, bool) and (declared == 2) == spec.realtime,
                f"TIER={declared!r}; the game is {kind}")
    if expected_tier is not None:
        want = int(expected_tier)
        ok = spec.realtime == (want == 2) and spec.tier == f"tier{want}"
        rep.add(f"tier {want}, as the study decided", ok, f"the game is {kind}, {spec.tier}" + ("" if ok else (
            "; tier 2 needs a RealtimeGame (tick / tick_s / ticks)" if want == 2 else
            "; tier 1 needs a TurnGame (step, no tick)")))
    if spec.seeds:
        tr, ho = spec.seeds["train"], spec.seeds["heldout"]
        rep.add("SEEDS: non-empty, disjoint train / held-out lists of ints", True,
                f"{len(tr)} train ({_seeds_text(tr)}), {len(ho)} held-out ({_seeds_text(ho)})")
        first = (tr[0], ho[0])                         # the evaluator only plays declared seeds
        seeds = first if seeds is None else tuple(seeds) + tuple(s for s in first if s not in seeds)
    elif seeds is None:
        seeds = (0, 1)
    kw = {**spec.eval_kwargs, **(kwargs or {})}
    live: list[Any] = []

    def new_game() -> Any:
        inst = spec.make_game(**kw)
        live.append(inst)
        return inst

    def release(inst: Any, *, check: bool) -> None:
        live[:] = [x for x in live if x is not inst]
        err = close_game(inst)
        if not check:
            if err:
                rep.warn(f"close() of a validation instance failed: {err}")
            return
        has_close = callable(getattr(inst, "close", None))
        rep.add("close() releases the instance (called after every episode)", err is None,
                err or ("" if has_close else "no close(); fine only if the adaptor starts no process, server or connection"))
        if has_close and err is None:
            err2 = close_game(inst)
            if err2:
                rep.warn(f"a second close() failed ({err2}); close() must be idempotent")

    try:
        g = new_game()
        spec_ok = isinstance(getattr(g, "action_spec", None), ActionSpec) and len(g.action_spec.names) > 0
        rep.add("action_spec is an ActionSpec with >=1 action", spec_ok,
                "" if spec_ok else "expose action_spec = ActionSpec(names=[...], descriptions={...}, noop=...)")
        tick_s = 0.0
        if spec.realtime:
            tick_s = float(getattr(g, "tick_s", 0))
            rep.add("realtime: tick_s > 0 and noop action set",
                    tick_s > 0 and g.action_spec.noop in g.action_spec.names,
                    f"tick_s={getattr(g, 'tick_s', None)} noop={g.action_spec.noop}")
        d = g.describe()
        rep.add("describe() returns a non-trivial string", isinstance(d, str) and len(d) > 80,
                f"{len(d) if isinstance(d, str) else type(d).__name__} chars")
        first_obs: dict[int, str] = {}
        for seed in seeds:
            if time.time() - t0 > time_budget_s:
                rep.add(f"seed {seed}", False, "validation time budget exhausted")
                break
            g.reset(seed)
            obs = g.observe()
            ok, detail = _json_ok(obs)
            rep.add(f"seed {seed}: observe() JSON-serialisable", ok, detail)
            first_obs[seed] = episode_state(obs)
            rep.add(f"seed {seed}: done is bool and score is a number after reset", *_state_ok(g))
            names = list(g.action_spec.names)
            actions = ([g.action_spec.noop] if g.action_spec.noop else []) + names
            applied = 0
            err = bad_obs = None
            tick_times: list[float] = []
            for a in actions[:max_actions]:
                if g.done:
                    break
                try:
                    if spec.realtime:
                        t = time.perf_counter()
                        g.tick(a)
                        tick_times.append(time.perf_counter() - t)
                    else:
                        g.step(a)
                    applied += 1
                    obs = g.observe()
                except Exception as e:  # noqa: BLE001
                    err = f"action {a!r}: {type(e).__name__}: {e}"
                    break
                ok, detail = _json_ok(obs)
                if not ok and bad_obs is None:
                    bad_obs = f"after {a!r}: {detail}"
            rep.add(f"seed {seed}: applied {applied} actions without exception", err is None, err or "")
            if err is None:                            # the state the harness sees after play (and at done)
                rep.add(f"seed {seed}: done is bool and score is a finite number after the actions", *_state_ok(g))
                ok, detail = _json_ok(g.observe())
                rep.add(f"seed {seed}: observations after the actions are strict JSON", ok and bad_obs is None,
                        bad_obs or ("" if ok else f"final observation: {detail}"))
            if spec.realtime:
                rep.add(f"seed {seed}: ticks counter advanced", int(getattr(g, "ticks", 0)) == applied,
                        f"ticks={getattr(g, 'ticks', None)} applied={applied}")
                if tick_times and tick_s > 0:
                    med = statistics.median(tick_times)
                    pace = f"median tick() {med * 1000:.1f} ms, tick_s {tick_s * 1000:.0f} ms"
                    rep.add(f"seed {seed}: tick() keeps the declared pace", med <= tick_s, pace if med <= tick_s else
                            pace + ": the ticker cannot keep this pace — return from tick() quickly (send the "
                            "action, update state asynchronously) or declare a longer tick_s")
                    if tick_s * 0.5 < med <= tick_s:
                        rep.warn(f"seed {seed}: {pace} — little slack left for the ticker")
            info = g.info()
            rep.add(f"seed {seed}: info() is a dict", isinstance(info, dict), "")
            if isinstance(info, dict) and info.get("valid_trial") is False:
                rep.warn(f"seed {seed}: info() reports an invalid trial ({info.get('invalid_reason')!r:.300}); "
                         "the evaluator would retry this seed once and then record an error")
        if first_obs:
            if g.done:
                g.reset(next(iter(first_obs)))
            try:
                (g.tick if spec.realtime else g.step)(MALFORMED_ACTION)
                rep.add(f"malformed action {MALFORMED_ACTION!r} does not raise", True, "")
            except Exception as e:  # noqa: BLE001 — a warning: the evaluator survives it, the episode does not
                rep.warn(f"{'tick' if spec.realtime else 'step'}({MALFORMED_ACTION!r}) raised {type(e).__name__}: "
                         f"{str(e)[:200]} — an exception from tick()/step() ends the episode as a game error; "
                         "ignore malformed actions or report them in the next observation")
        release(g, check=True)                         # before a second instance: one server / port at a time
        if spec.deterministic and first_obs:
            s0 = next(iter(first_obs))
            g2 = new_game()
            g2.reset(s0)
            same = episode_state(g2.observe()) == first_obs[s0]
            rep.add("DETERMINISTIC: reset(seed) on a fresh instance reproduces the first observation", same,
                    "" if same else f"seed {s0}: a fresh make_game() + reset({s0}) observed something else")
            release(g2, check=False)
        if len(first_obs) >= 2:
            (s_a, v_a), (s_b, v_b) = list(first_obs.items())[:2]
            if v_a != v_b:
                rep.add("different seeds give different starts (scenario variety)", True, "")
            else:
                rep.warn(f"seeds {s_a} and {s_b} give identical first observations — fine for a fixed opening "
                         "position if the seed still changes the episode (checked next)")
                # A fixed opening is fine, but the seed must still change the episode: with identical
                # play two seeds must diverge, else held-out seeds replay the training seeds.
                traj = {s: _trajectory(new_game, release, s, actions) for s in (s_a, s_b)}
                same = traj[s_a] == traj[s_b]
                rep.add("different seeds give different episodes (the same actions on two seeds diverge)", not same,
                        f"seeds {s_a} and {s_b}: identical observations through {traj[s_a][1]} identical actions — "
                        "held-out seeds would replay the training seeds and a fixed action sequence could be "
                        "memorised; let the seed drive the game's own randomness (the platform's standard "
                        "stochasticity, e.g. randomised starts or action noise, or a seeded scenario)"
                        if same else f"seeds {s_a} and {s_b} start alike and diverge under identical play")
    except Exception:  # noqa: BLE001
        rep.ok = False
        rep.error = traceback.format_exc()
    finally:
        for inst in list(live):
            release(inst, check=False)
    return rep

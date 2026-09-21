"""Generated game adaptors — the contract the ``adapt`` phase must satisfy.

An adaptor is a directory (written by Astra, never by hand in this repo):

    <adaptor>/
      game.py            defines make_game(**kwargs) -> TurnGame | RealtimeGame
      ADAPTOR.md         how it connects to the real game, what it observes, how
                         seeds map to scenarios, what score means, its limits
      requirements.txt   (optional) extra pip packages the adaptor needs
      env.example        (optional) environment variables it reads (credentials,
                         server address...). Real values live in the user's .env.

``game.py`` may import anything installed plus ``metaharness.games.base``. The
object returned by ``make_game`` must implement ``TurnGame`` (``step``) or
``RealtimeGame`` (``tick`` + ``tick_s`` + ``ticks``) from ``games/base.py`` and
additionally ``name`` (str) and ``TIER`` may be declared at module level as
``TIER = 1 | 2``. Observations must be JSON-serialisable (dicts/lists/str/
numbers/bools); binary perception (screenshots) is exposed as file paths so a
harness can hand them to ``ctx.think(images=[...])``.

``validate_adaptor`` is the generic smoke test the orchestrator runs (and shows
the model when it fails): load, construct, reset two seeds, observe, describe,
action spec, apply the noop and every action once, done/score types, and
determinism of ``reset(seed)`` when the adaptor claims it (``DETERMINISTIC``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .base import ActionSpec

REQUIRED_FILES = ("game.py", "ADAPTOR.md")


@dataclass
class AdaptorSpec:
    path: Path
    name: str
    tier: str                         # "tier1" | "tier2"
    make_game: Callable[..., Any]
    realtime: bool
    deterministic: bool = False
    module: Any = None


def load_adaptor(path: Path | str) -> AdaptorSpec:
    path = Path(path).resolve()
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"adaptor at {path} is missing {missing}")
    mod_name = f"_adaptor_{abs(hash(str(path)))}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(mod_name, path / "game.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    if str(path) not in sys.path:                      # allow sibling helper modules
        sys.path.insert(0, str(path))
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "make_game"):
        raise ImportError("game.py must define make_game(**kwargs)")
    probe = module.make_game()
    realtime = hasattr(probe, "tick")
    tier = getattr(module, "TIER", 2 if realtime else 1)
    return AdaptorSpec(path=path, name=str(getattr(probe, "name", path.name)), tier=f"tier{int(tier)}",
                       make_game=module.make_game, realtime=realtime,
                       deterministic=bool(getattr(module, "DETERMINISTIC", False)), module=module)


@dataclass
class ValidationReport:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "ok": ok, "detail": detail})
        if not ok:
            self.ok = False

    def to_markdown(self) -> str:
        lines = [f"# Adaptor validation: {'PASS' if self.ok else 'FAIL'}", ""]
        for c in self.checks:
            lines.append(f"- [{'x' if c['ok'] else ' '}] {c['check']}" + (f" — {c['detail']}" if c["detail"] else ""))
        if self.error:
            lines += ["", "```", self.error[-4000:], "```"]
        return "\n".join(lines) + "\n"


def _json_ok(obj: Any) -> tuple[bool, str]:
    try:
        s = json.dumps(obj)
    except (TypeError, ValueError) as e:
        return False, f"observation is not JSON-serialisable: {e}"
    if len(s) > 200_000:
        return False, f"observation is {len(s)} bytes; keep it under 200 kB (summarise / crop)"
    return True, f"{len(s)} bytes"


def validate_adaptor(path: Path | str, *, kwargs: dict[str, Any] | None = None,
                     seeds: tuple[int, ...] = (0, 1), max_actions: int = 12,
                     time_budget_s: float = 120.0) -> ValidationReport:
    rep = ValidationReport(ok=True)
    t0 = time.time()
    try:
        spec = load_adaptor(path)
        rep.add("load game.py and make_game()", True, f"name={spec.name} tier={spec.tier} realtime={spec.realtime}")
    except Exception:  # noqa: BLE001
        rep.add("load game.py and make_game()", False)
        rep.error = traceback.format_exc()
        return rep
    kwargs = kwargs or {}
    try:
        g = spec.make_game(**kwargs)
        spec_ok = isinstance(getattr(g, "action_spec", None), ActionSpec) and len(g.action_spec.names) > 0
        rep.add("action_spec is an ActionSpec with >=1 action", spec_ok,
                "" if spec_ok else "expose action_spec = ActionSpec(names=[...], descriptions={...}, noop=...)")
        if spec.realtime:
            rep.add("realtime: tick_s > 0 and noop action set",
                    float(getattr(g, "tick_s", 0)) > 0 and g.action_spec.noop in g.action_spec.names,
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
            first_obs[seed] = json.dumps(obs, sort_keys=True, default=str)
            rep.add(f"seed {seed}: done is bool and score is a number after reset",
                    isinstance(g.done, bool) and isinstance(g.score, (int, float)),
                    f"done={g.done!r} score={g.score!r}")
            names = list(g.action_spec.names)
            actions = ([g.action_spec.noop] if g.action_spec.noop else []) + names
            applied = 0
            err = None
            for a in actions[:max_actions]:
                if g.done:
                    break
                try:
                    if spec.realtime:
                        g.tick(a)
                    else:
                        g.step(a)
                    applied += 1
                    json.dumps(g.observe(), default=str)
                except Exception as e:  # noqa: BLE001
                    err = f"action {a!r}: {type(e).__name__}: {e}"
                    break
            rep.add(f"seed {seed}: applied {applied} actions without exception", err is None, err or "")
            if spec.realtime:
                rep.add(f"seed {seed}: ticks counter advanced", int(getattr(g, "ticks", 0)) == applied,
                        f"ticks={getattr(g, 'ticks', None)} applied={applied}")
            info = g.info()
            rep.add(f"seed {seed}: info() is a dict", isinstance(info, dict), "")
        if spec.deterministic and len(first_obs) >= 1:
            s0 = list(first_obs)[0]
            g.reset(s0)
            rep.add("DETERMINISTIC: reset(seed) reproduces the first observation",
                    json.dumps(g.observe(), sort_keys=True, default=str) == first_obs[s0], "")
        if len(first_obs) >= 2:
            vals = list(first_obs.values())
            rep.add("different seeds give different starts (scenario variety)", vals[0] != vals[1],
                    "identical first observations for two seeds" if vals[0] == vals[1] else "")
    except Exception:  # noqa: BLE001
        rep.ok = False
        rep.error = traceback.format_exc()
    return rep

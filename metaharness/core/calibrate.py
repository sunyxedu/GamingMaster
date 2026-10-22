"""Measure the models' latency once per search archive, before the first proposal.

Without it the first proposer only sees the Astra-only baseline's one-word answers (~5 s) and sizes
its timeouts from them, while longer answers take tens of seconds (archived calls fit ~4 s + 31 ms
per output token): in the archived general-pipeline runs every Astra call of two of the six first
proposals ran into the harness's own timeout. The result is stored in ``calibration.json`` and reused
on resume, because it becomes part of task.md, which the search manifest pins."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..games.base import close_game
from ..models.base import ModelUnavailable, System1Model, System2Model

SHORT_PROMPT = "Reply with the single word: ready"
LONG_PROMPT = ("Write a plain-text checklist of 40 numbered, one-sentence rules that a careful player of an "
               "unknown video game could follow (about 600 words). No preamble, no closing remarks.")
OFFLINE = ("scripted", "heuristic")         # stand-ins report declared latencies; nothing to measure


def _measurable(model: Any) -> bool:
    return model is not None and getattr(model, "name", "") not in OFFLINE


def calibrate(root: Path, *, system2: Optional[System2Model], system1: Optional[System1Model],
              game_factory: Callable[..., Any], game_kwargs: dict[str, Any], seed: int,
              log: Callable[[str], None] = print) -> Optional[dict[str, Any]]:
    path = Path(root) / "calibration.json"
    if path.exists():
        return json.loads(path.read_text())
    if not (_measurable(system2) or _measurable(system1)):
        return None
    out: dict[str, Any] = {"measured": time.strftime("%Y-%m-%d %H:%M")}
    if _measurable(system2):
        for label, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT)):
            t0 = time.monotonic()
            try:
                res = system2.run(prompt, agentic=False, timeout_s=900.0)
                out[f"system2_{label}"] = {"latency_s": round(res.latency_s or time.monotonic() - t0, 2),
                                           "output_tokens": (res.usage or {}).get("output_tokens")}
            except ModelUnavailable:
                raise
            except Exception as e:  # noqa: BLE001 — a failed measurement is reported, not fatal
                out[f"system2_{label}"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    if _measurable(system1):
        from ..models.jev import Choice
        game = None
        try:
            game = game_factory(**game_kwargs)
            game.reset(seed)
            obs = game.observe()
            spec = game.action_spec
            names = list(spec.names)[:255]
            q = {"action": Choice("Which action should the player take now?",
                                  {n: spec.descriptions.get(n) or None for n in names})}
            lat = []
            for _ in range(2):          # the first request also pays for the connection
                t0 = time.monotonic()
                system1.decide(obs, q, timeout_s=30.0)
                lat.append(time.monotonic() - t0)
            out["system1"] = {"latency_s": round(lat[1], 3), "first_latency_s": round(lat[0], 3)}
        except ModelUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            out["system1"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        finally:
            if game is not None:
                close_game(game)
    path.write_text(json.dumps(out, indent=2) + "\n")
    log("calibration: " + json.dumps(out))
    return out

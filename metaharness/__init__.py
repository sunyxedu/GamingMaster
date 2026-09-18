"""Meta-Harness: an outer loop that *designs* the harness gluing foundation
models to a game, instead of hand-writing a per-game adapter.

Three game tiers (see docs/meta-harness.md):

* Tier 1 — no latency pressure (ARC-AGI-3 style). Astra (System-2) plays
  directly; the harness decides what to store / retrieve / present.
* Tier 2 — some latency pressure (Minecraft, Genshin, snake). Astra thinks
  at the slow timescale, Jev (System-1) decides at the fast timescale; the
  harness designs the bridge between them and the game's action interface.
* Tier 3 — latency-critical (StarCraft, Clash Royale): the trained network in
  ``model/`` (not handled here).

The harness for a game is an ordinary Python module obeying a small contract
(``metaharness.core.contract``). The meta-harness (``metaharness.core.loop``)
asks a coding model to write that module, evaluates it, archives code +
traces + scores on the filesystem, and iterates — following
"Meta-Harness: End-to-End Optimization of Model Harnesses" (arXiv 2603.28052).
"""

from .config import load_dotenv

__all__ = ["load_dotenv"]

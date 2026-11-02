"""Meta-Harness: an outer loop that *designs* the harness gluing foundation
models to a game, instead of hand-writing a per-game adapter.

Two ways to play a game (see docs/meta-harness.md): this package — one Meta-Harness for every
game that is not latency-critical, with the same models for every game, Astra (System-2, slow
reasoning) and Jev (System-1, fast typed judgments), and a task spec that states the game's pace
(turn-based: the world waits; real-time: it ticks on its own clock) — or the trained network in
``model/`` for latency-critical games (StarCraft, Clash Royale; not handled here).

The harness for a game is an ordinary Python module obeying a small contract
(``metaharness.core.contract``). The meta-harness (``metaharness.core.loop``)
asks a coding model to write that module, evaluates it, archives code +
traces + scores on the filesystem, and iterates — following
"Meta-Harness: End-to-End Optimization of Model Harnesses" (arXiv 2603.28052).
"""

from .config import load_dotenv

__all__ = ["load_dotenv"]

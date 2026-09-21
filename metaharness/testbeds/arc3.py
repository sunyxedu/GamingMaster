"""ARC-AGI-3 as a tier-1 TurnGame (optional dependency: ``pip install arc-agi``).

Uses ``arc_agi.Arcade`` in OFFLINE mode with locally cached game files
(``ENVIRONMENTS_DIR``, default ``environment_files``; the arc3-wm project on
this machine has a cached set). Set ``OPERATION_MODE=offline`` and
``ENVIRONMENTS_DIR=/path/to/environment_files`` in ``.env``.

Observation: the last frame layer as a 64x64 list of palette indices plus
``available_actions``. Actions: ``ACTION1``..``ACTION5``, ``ACTION7`` and
``ACTION6 x y`` (click). Score = levels completed. This mirrors the official
harness surface (no rules given), so tier-1 harnesses can transfer directly.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from ..config import load_dotenv
from ..games.base import ActionSpec
from ..games.registry import register

ARC_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]


@register("arc3", "tier1")
class ARC3Game:
    name = "arc3"
    action_spec = ActionSpec(
        names=ARC_ACTIONS,
        descriptions={
            "ACTION1": "unknown (parameter-less)", "ACTION2": "unknown (parameter-less)",
            "ACTION3": "unknown (parameter-less)", "ACTION4": "unknown (parameter-less)",
            "ACTION5": "unknown (parameter-less)",
            "ACTION6": "click at a grid cell: pass as 'ACTION6 x y' with 0<=x,y<64",
            "ACTION7": "unknown (parameter-less)",
        },
        notes="Only actions listed in observation['available_actions'] do anything. "
              "Effects must be discovered by experiment; goals are not stated.",
    )

    def __init__(self, game_id: str = "vc33", max_actions: int = 400, arcade: Any = None) -> None:
        load_dotenv()
        os.environ.setdefault("OPERATION_MODE", "offline")
        import arc_agi  # noqa: WPS433  (optional dep)
        from arcengine import GameAction, GameState
        self._GameAction, self._GameState = GameAction, GameState
        self._arcade = arcade or arc_agi.Arcade()
        self.game_id = game_id
        self.max_actions = max_actions
        self._wrapper = None
        self.reset(0)

    def reset(self, seed: int) -> dict[str, Any]:
        self._wrapper = self._arcade.make(self.game_id, seed=int(seed))
        if self._wrapper is None:
            raise RuntimeError(f"Arcade.make({self.game_id!r}) returned None — is it cached under ENVIRONMENTS_DIR?")
        self._fd = self._wrapper.reset()
        self._actions = 0
        self._done = False
        return self.observe()

    def step(self, action: Any) -> dict[str, Any]:
        if self._done:
            return self.observe()
        parts = str(action).split()
        name = parts[0].upper()
        if name not in ARC_ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        ga = getattr(self._GameAction, name)
        data: Optional[dict] = None
        if name == "ACTION6":
            if len(parts) != 3:
                raise ValueError("ACTION6 needs 'ACTION6 x y'")
            data = {"x": int(parts[1]), "y": int(parts[2])}
        fd = self._wrapper.step(ga, data=data)
        if fd is not None:
            self._fd = fd
        self._actions += 1
        if self._fd.state in (self._GameState.WIN, self._GameState.GAME_OVER) or self._actions >= self.max_actions:
            self._done = True
        return self.observe()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        return float(self._fd.levels_completed)

    def observe(self) -> dict[str, Any]:
        frame = self._fd.frame[-1] if self._fd.frame else []
        grid = [[int(v) for v in row] for row in frame]
        return {
            "game_id": self.game_id, "state": str(self._fd.state.name),
            "levels_completed": int(self._fd.levels_completed), "win_levels": int(self._fd.win_levels),
            "actions_used": self._actions, "actions_left": self.max_actions - self._actions,
            "available_actions": [f"ACTION{i}" for i in self._fd.available_actions],
            "grid": grid, "grid_size": [len(grid[0]) if grid else 0, len(grid)],
        }

    def info(self) -> dict[str, Any]:
        return {"levels_done": int(self._fd.levels_completed), "actions_used": self._actions,
                "state": str(self._fd.state.name)}

    def describe(self) -> str:
        return (
            f"ARC-AGI-3 game '{self.game_id}': a 64x64 grid of colour indices (0-15). Nothing about the rules "
            f"or goal is given; discover them by acting. Budget {self.max_actions} actions. Score = levels "
            "completed (observation['levels_completed'], out of 'win_levels')."
        )

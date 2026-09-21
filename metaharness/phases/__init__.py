"""The Meta-Harness phases for an arbitrary game named only by its title.

    study   -> runs/<game>/study/      what the game is, how to reach it, how to play it
    adapt   -> runs/<game>/adaptor/    generated game adaptor (validated by a generic smoke test)
    search  -> runs/<game>/search/     harness search archive (metaharness.core)

Nothing here knows any particular game; every game-specific artefact is
produced by the model into the run directory.
"""
from .adapt import run_adapt
from .study import run_study

__all__ = ["run_study", "run_adapt"]

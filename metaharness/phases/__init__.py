"""The Meta-Harness phases for an arbitrary game named only by its title.

    study   -> runs/<game>/study/      what the game is, how to reach it, how to play it
    adapt   -> runs/<game>/adaptor/    generated game adaptor (validated by a generic smoke test)
    search  -> runs/<game>/search/     harness search archive (metaharness.core); search/prior/
                                       holds the study documents + ADAPTOR.md the proposer reads

``runs/<game>/run.json`` records the game the directory was created for, so another name
with the same slug cannot silently reuse its study and adaptor.

Nothing here knows any particular game; every game-specific artefact is
produced by the model into the run directory.
"""
from .adapt import run_adapt
from .study import prior_documents, run_study

__all__ = ["run_study", "run_adapt", "prior_documents"]

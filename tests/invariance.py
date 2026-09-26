"""Causality / privacy invariance checks for imperfect-information games — shared by
every game's tests (the Rust side: gmrules ``invariance::check_game``).

Two levels:

* simulator / rules (gmsim): :func:`check_sim_causality` truncates a trajectory
  after every decision and re-replays it; :func:`check_sim_privacy` re-plays the
  game's own perturbations of hidden information (``gmsim.perturb_hidden``);
  :func:`check_sim_battery` runs both over random playouts.
* converter (the real binary on real record files): :func:`check_converter_causality`
  (convert a record and its truncations: every stream of a truncated record must be
  a frame-prefix of the same stream of the full record) and
  :func:`check_converter_privacy` (convert a record and versions with the other
  seats' private information changed: the seat's samples must be identical).
  ``convert(records) -> samples`` is the game's own glue (write the records in the
  source format, run gmdata with ``--allow-truncated``, ``load_gm_samples``); samples
  are ``{"obs", "tgt", "meta"}`` dicts matched by ``meta["seat"]`` (override
  ``key``).

Frames compare bit-exactly (``torch.equal`` / ``np.array_equal``, dtype and shape
included) over every ``obs`` tensor plus the ``tgt`` keys in ``TGT_KEYS`` (the
label of a frame may not change under truncation / perturbation either; the value
target of course does).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import gmsim
import numpy as np
import torch

#: tgt tensors that are part of a frame (compared); tgt.value is an outcome
TGT_KEYS = ("action_idx",)

Sample = Dict[str, Any]


def _eq(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        a, b = torch.as_tensor(a), torch.as_tensor(b)
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def assert_frames_prefix(full: Dict[str, Any], part: Dict[str, Any], n: int, what: str) -> None:
    """``part[k][:n] == full[k][:n]`` bit-exactly for every key of ``part``."""
    for k, v in part.items():
        assert k in full, f"{what}: key {k!r} missing"
        a, b = full[k][:n], v[:n]
        if not _eq(a, b):
            bad = next((t for t in range(n) if not _eq(a[t:t + 1], b[t:t + 1])), None)
            raise AssertionError(f"{what}: {k} differs at frame {bad}")


# ---------------------------------------------------------------------------
# simulator / rules level
# ---------------------------------------------------------------------------

def _decision_frames(r: Dict[str, Any], idx: Sequence[int]) -> Dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    out = {k: r[k][idx] for k in gmsim.FRAME_KEYS}
    out["action_idx"] = r["action_idx"][idx]
    out["seat"] = r["seat"][idx]
    out["legal"] = [np.asarray(r["legal"][i], dtype=np.int64) for i in idx]
    return out


def _legal_equal(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def check_sim_causality(game: str, config, events) -> int:
    """Truncating ``events`` right after each decision leaves every earlier
    decision (seat, candidates, frame, action index) bit-identical.  Returns the
    number of truncations checked."""
    events = gmsim.events_array(events)
    full = gmsim.replay(game, config, events)
    for t in range(full["n"]):
        cut = int(full["event"][t]) + 1
        part = gmsim.replay(game, config, events[:cut])
        assert part["n"] == t + 1, f"{game}: truncated after decision {t}: {part['n']} decisions"
        a, b = _decision_frames(full, range(t + 1)), _decision_frames(part, range(t + 1))
        legal_a, legal_b = a.pop("legal"), b.pop("legal")
        assert _legal_equal(legal_a, legal_b), f"{game}: candidates change when truncated after {t}"
        assert_frames_prefix(a, b, t + 1, f"{game} causality (cut after decision {t})")
    return full["n"]


def check_sim_privacy(game: str, config, events, seats: Optional[Iterable[int]] = None,
                      seeds: Iterable[int] = range(4)) -> int:
    """Every frame of seat s is bit-identical under the game's perturbations of
    information hidden from s (``gmsim.perturb_hidden``).  Returns the number of
    perturbations checked (0 = the game never perturbed this record)."""
    events = gmsim.events_array(events)
    full = gmsim.replay(game, config, events)
    n_seats = full["spec"]["n_seats"]
    n = 0
    for seat in (range(n_seats) if seats is None else seats):
        mine = np.flatnonzero(full["seat"] == seat)
        for seed in seeds:
            p = gmsim.perturb_hidden(game, config, events, seat, seed)
            if p is None:
                continue
            assert not np.array_equal(p, events), f"{game}: perturbation of seat {seat} is a no-op"
            other = gmsim.replay(game, config, p)
            theirs = np.flatnonzero(other["seat"] == seat)
            assert len(theirs) == len(mine), (
                f"{game}: seat {seat} has {len(mine)} decisions, {len(theirs)} after perturbing")
            a, b = _decision_frames(full, mine), _decision_frames(other, theirs)
            assert _legal_equal(a.pop("legal"), b.pop("legal")), f"{game}: seat {seat} candidates"
            assert_frames_prefix(a, b, len(mine),
                                 f"{game} privacy (seat {seat}'s frames under a perturbation "
                                 f"of hidden information, seed {seed})")
            n += 1
    return n


def check_sim_battery(game: str, config=None, n_games: int = 50, seed: int = 0) -> Dict[str, int]:
    """Causality on ``n_games`` random playouts, privacy on each (hidden-information
    games must produce perturbations)."""
    info = gmsim.game_info(game, config)["spec"]
    stats = {"games": 0, "decisions": 0, "privacy": 0}
    for i in range(n_games):
        ev = gmsim.random_trajectory(game, config, seed + i)
        stats["decisions"] += check_sim_causality(game, config, ev)
        stats["privacy"] += check_sim_privacy(game, config, ev, seeds=(seed + i, seed + i + 1))
        stats["games"] += 1
    if info["hidden_info"] and n_games >= 10:
        assert stats["privacy"] > 0, f"{game}: hidden information, but no perturbation"
    return stats


# ---------------------------------------------------------------------------
# converter level
# ---------------------------------------------------------------------------

def _frames(s: Sample) -> Dict[str, Any]:
    out = dict(s["obs"])
    out.update({k: s["tgt"][k] for k in TGT_KEYS if k in s["tgt"]})
    return out


def _by_key(samples: List[Sample], key: Callable[[Sample], Any]) -> Dict[Any, Sample]:
    out: Dict[Any, Sample] = {}
    for s in samples:
        k = key(s)
        assert k not in out, f"two samples with key {k!r}"
        out[k] = s
    return out


def seat_key(s: Sample):
    return s["meta"].get("seat")


def check_converter_causality(convert: Callable[[List[Any]], List[Sample]], record: Any,
                              truncations: Sequence[Any],
                              key: Callable[[Sample], Any] = seat_key) -> int:
    """``convert([record])`` vs ``convert([t])`` for every truncation ``t`` of the
    record: each stream of a truncated record is a frame-prefix of the full
    record's stream with the same key (every stream present in the truncation
    must exist in the full record).  Returns the number of streams compared."""
    full = _by_key(convert([record]), key)
    assert full, "the full record converted to no sample"
    n = 0
    for i, t in enumerate(truncations):
        for k, s in _by_key(convert([t]), key).items():
            assert k in full, f"truncation {i}: stream {k!r} not in the full record"
            T = next(iter(s["obs"].values())).shape[0]
            F = next(iter(full[k]["obs"].values())).shape[0]
            assert T <= F, f"truncation {i}: stream {k!r} has {T} frames > {F}"
            assert_frames_prefix(_frames(full[k]), _frames(s), T,
                                 f"converter causality (truncation {i}, stream {k!r})")
            n += 1
    return n


def check_converter_privacy(convert: Callable[[List[Any]], List[Sample]], record: Any,
                            perturbed: Sequence[Any], seat: Any,
                            key: Callable[[Sample], Any] = seat_key) -> int:
    """``seat``'s samples of ``record`` and of every ``perturbed`` version (the
    other seats' private information changed, every recorded action still legal)
    are identical frame for frame.  Returns the number of versions compared."""
    mine = _by_key(convert([record]), key)[seat]
    F = next(iter(mine["obs"].values())).shape[0]
    for i, p in enumerate(perturbed):
        other = _by_key(convert([p]), key)
        assert seat in other, f"perturbation {i}: no sample for seat {seat!r}"
        T = next(iter(other[seat]["obs"].values())).shape[0]
        assert T == F, f"perturbation {i}: seat {seat!r} has {T} frames, {F} before"
        assert_frames_prefix(_frames(mine), _frames(other[seat]), F,
                             f"converter privacy (perturbation {i}, seat {seat!r})")
    return len(perturbed)

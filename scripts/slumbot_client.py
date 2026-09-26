"""Play heads-up no-limit hold'em against Slumbot (https://slumbot.com: a strong ACPC
heads-up no-limit bot with a free public API; 50 / 100 blinds, 20 000-chip stacks =
200 big blinds, reset every hand — exactly the gmrules ``hunl`` default config), an
optional evaluation of a checkpoint's hunl policy or of a baseline:

    uv run python scripts/slumbot_client.py --checkpoint ck/multi_final.pt --hands 300 \\
        --out runs/slumbot/ck1              # runs/slumbot/ck1/hands.jsonl + summary.json
    uv run python scripts/slumbot_client.py --policy call --hands 20      # check / call

Be polite, it is a free service: one request at a time, at least ``--delay`` seconds
apart (default 1 s, never below 0.5), a few hundred hands per run (``--hands`` above
2 000 needs ``--force``); retries back off.  Login is optional: SLUMBOT_USERNAME /
SLUMBOT_PASSWORD from the environment (never the command line or the repo; register on
the site first) put the hands on the site's leaderboard.

A hand: the API reports its betting as an action string (``b200c/kb400c/kk/kk``: k
check, c call, f fold, bN = bet / raise to N chips put in ON THIS STREET — preflop
counting the blinds —, "/" ends a street), our ``client_pos`` (0 = big blind, 1 =
small blind / button: the ACPC and gmrules seats), our hole cards and the board.  The
client rebuilds the gmrules hunl trajectory — chance: seat 0's hole cards, seat 1's,
then the board as dealt; bets as total commitments — and takes our seat's frames from
``gmsim`` (the encoder of the IL shards and of the RL env).  Slumbot's hole cards are
unknown until a showdown: two placeholder cards outside every visible card stand in,
which the frames never read (the privacy batteries of tests/test_hunl_pipeline.py;
checked again here).  The policy sees the hand's frames of our seat so far (the RL
env's per-seat stream) and picks a candidate (fold, check / call, all-in, min-raise,
pot fractions); its exact amount (``gmsim.State.decode``) goes back as k / c / f / bN.
Every finished hand is verified: the replayed payoff (with Slumbot's cards when shown)
must equal the API's ``winnings``.  Result: big blinds per 100 hands ± standard error
(poker is noisy — about 10 bb of standard deviation per hand, so ± 5 bb/100 takes some
40 000 hands; at ~4 s per hand a 2 000-hand run takes ~2 h and resolves ± ~20 bb/100).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gmsim
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = "https://slumbot.com/slumbot/api"
GAME = "hunl"
SMALL_BLIND, BIG_BLIND, STACK = 50, 100, 20000
CONFIG = {"small_blind": SMALL_BLIND, "big_blind": BIG_BLIND, "stack": STACK}
RANKS, SUITS = "23456789TJQKA", "cdhs"
FOLD, CALL = 0, 1
SLOT_CALL = 1
MIN_DELAY = 0.5
MAX_HANDS_UNFORCED = 2000


def card_id(s: str) -> int:
    """``"Ac"`` → the ACPC / gmrules card id (rank * 4 + suit)."""
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def parse_action(action: str) -> List[List[str]]:
    """A Slumbot action string → per-street tokens (``k``, ``c``, ``f``, ``b<N>``)."""
    streets: List[List[str]] = [[]]
    i = 0
    while i < len(action):
        ch = action[i]
        if ch == "/":
            streets.append([])
            i += 1
        elif ch in "kcf":
            streets[-1].append(ch)
            i += 1
        elif ch == "b":
            j = i + 1
            while j < len(action) and action[j].isdigit():
                j += 1
            if j == i + 1:
                raise ValueError(f"bet without a size in {action!r}")
            streets[-1].append(action[i:j])
            i = j
        else:
            raise ValueError(f"unexpected {ch!r} in action {action!r}")
    if len(streets) > 4:
        raise ValueError(f"more than four streets in {action!r}")
    return streets


@dataclass
class Rebuilt:
    """A Slumbot hand as a gmrules trajectory (see :func:`rebuild`)."""
    events: List[Tuple[int, int]]
    state: Any                     # gmsim.State after the events
    spent: List[int]               # total commitments of seat 0 / seat 1
    base: int                      # commitment when the current street began
    placeholder: bool              # the opponent's hole cards are stand-ins


def rebuild(seat: int, hole: Sequence[str], action: str, board: Sequence[str],
            bot_hole: Optional[Sequence[str]] = None) -> Rebuilt:
    """The gmrules trajectory of a hand seen from ``seat``: our hole cards, the
    opponent's (``bot_hole`` or two placeholders outside every visible card), the
    betting of ``action`` with the ``board`` cards dealt as the streets begin (an all-in
    deals the rest).  Raises ValueError when the action string is not a legal hunl line
    (streets must match the rules' rounds)."""
    mine = [card_id(c) for c in hole]
    board_ids = [card_id(c) for c in board]
    if bot_hole:
        opp = [card_id(c) for c in bot_hole]
    else:
        seen = set(mine) | set(board_ids)
        opp = [c for c in range(52) if c not in seen][:2]
    holes = (mine, opp) if seat == 0 else (opp, mine)
    st = gmsim.State(GAME, CONFIG)
    events: List[Tuple[int, int]] = [(-1, c) for c in holes[0] + holes[1]]
    for e in events:
        st.apply(e)
    spent, base, nb = [BIG_BLIND, SMALL_BLIND], 0, 0

    def deal_pending(upto: int) -> None:
        nonlocal base, nb
        while st.turn()[0] == "chance" and nb < upto:
            base = max(spent)
            e = (-1, board_ids[nb])
            st.apply(e)
            events.append(e)
            nb += 1

    for street, toks in enumerate(parse_action(action)):
        for t in toks:
            deal_pending(len(board_ids))
            kind, seats = st.turn()
            if kind != "act":
                raise ValueError(f"{action!r}: {t!r} but the hand is at {kind} ({st})")
            rnd = {0: 0, 3: 1, 4: 2, 5: 3}[nb]
            if rnd != street:
                raise ValueError(f"{action!r}: {t!r} on street {street}, the rules are on {rnd}")
            s = seats[0]
            if t == "f":
                a = FOLD
            elif t in ("k", "c"):
                owe = min(max(spent), STACK) - spent[s]
                if (t == "c") != (owe > 0):
                    raise ValueError(f"{action!r}: {t!r} with {owe} chips to call")
                a = CALL
                spent[s] += owe
            else:
                a = base + int(t[1:])
                spent[s] = a
            st.apply((s, a))
            events.append((s, a))
    deal_pending(len(board_ids))
    return Rebuilt(events, st, spent, base, placeholder=not bot_hole)


def to_incr(r: Rebuilt, seat: int, action: int) -> str:
    """Our canonical action as the API's incremental action."""
    if action == FOLD:
        return "f"
    if action == CALL:
        return "k" if r.spent[seat] >= min(max(r.spent), STACK) else "c"
    return f"b{action - r.base}"


def seat_frames(r: Rebuilt, seat: int) -> Dict[str, np.ndarray]:
    """``seat``'s frames of the hand so far + its current decision frame, ``[T, ...]``
    (the RL env's per-seat stream = the IL sample)."""
    past = gmsim.replay(GAME, CONFIG, r.events)
    mine = np.flatnonzero(past["seat"] == seat)
    now = r.state.frame(seat)["obs"]
    return {k: np.concatenate([past["obs"][k][mine], v[None]]) for k, v in now.items()}


# ---------------------------------------------------------------------------
# policies: frames [T, ...] of our stream → candidate index of the last frame
# ---------------------------------------------------------------------------

class CallPolicy:
    """Always check / call."""

    def __call__(self, frames: Dict[str, np.ndarray]) -> int:
        return SLOT_CALL


class RandomPolicy:
    """A uniformly random available candidate."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, frames: Dict[str, np.ndarray]) -> int:
        mask = frames["action_mask"][-1]
        return self.rng.choice([k for k in range(len(mask)) if mask[k]])


class ModelPolicy:
    """The hunl network of a train_multi_game / train_rl_joint checkpoint (EMA weights):
    the hand's frames so far as one sequence, the last frame's masked candidate logits
    sampled at ``temperature`` (0 = argmax)."""

    def __init__(self, checkpoint: str, device: str = "cpu", temperature: float = 1.0,
                 seed: int = 0):
        import torch

        from training.eval_multi_game import load_multi

        self.torch = torch
        self.device = torch.device(device)
        model = load_multi(checkpoint, self.device)
        if GAME not in model.games:
            raise SystemExit(f"{checkpoint}: no {GAME} network (games {model.games})")
        self.net = model[GAME].eval()
        self.temperature = temperature
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, frames: Dict[str, np.ndarray]) -> int:
        torch = self.torch
        # the frames' dtypes are the shards' / the env's (f32, bool, i32)
        obs = {k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).to(self.device)
               for k, v in frames.items()}
        with torch.no_grad():
            pred = self.net(obs, act_steps=self.net.default_depth)
        logits = pred["action_logits"][0, -1].float().cpu()
        mask = pred["action_mask"][0, -1].bool().cpu()
        logits = logits.masked_fill(~mask, float("-inf"))
        if self.temperature <= 0:
            return int(logits.argmax())
        p = torch.softmax(logits / self.temperature, dim=-1)
        return int(torch.multinomial(p, 1, generator=self.gen))


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------

class SlumbotApi:
    """JSON POSTs to the Slumbot API, one at a time, ``delay`` seconds apart, with
    retries (exponential back-off) on network errors and 5xx answers."""

    def __init__(self, base: str = API, delay: float = 1.0, timeout: float = 30.0,
                 retries: int = 4):
        self.base, self.timeout, self.retries = base, timeout, retries
        self.delay = max(delay, MIN_DELAY)
        self._last = 0.0

    def post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            wait = self._last + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(f"{self.base}/{endpoint}", data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    r = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.retries:
                    raise RuntimeError(f"{endpoint}: HTTP {e.code} {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"{endpoint}: {e}") from e
            else:
                if "error_msg" in r:
                    raise RuntimeError(f"{endpoint}: {r['error_msg']}")
                return r
            time.sleep(min(60.0, 2.0 * 2 ** attempt))
        raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# playing
# ---------------------------------------------------------------------------

@dataclass
class HandResult:
    client_pos: int
    hole: List[str]
    board: List[str]
    bot_hole: Optional[List[str]]
    action: str
    winnings: int
    replayed: Optional[float]        # our replayed payoff (None: showdown, cards unshown)
    decisions: List[Dict[str, Any]] = field(default_factory=list)


def finish(r: Dict[str, Any], seat: int, decisions: List[Dict[str, Any]]) -> HandResult:
    """Check a finished hand's ``winnings`` against the rules' payoff."""
    bot = r.get("bot_hole_cards") or None
    rb = rebuild(seat, r["hole_cards"], r["action"], r.get("board") or [], bot)
    kind, _ = rb.state.turn()
    if kind != "terminal":
        raise ValueError(f"hand over per the API, not per the rules: {r['action']!r} ({rb.state})")
    folded = parse_action(r["action"])[-1][-1:] == ["f"]
    replayed = None
    if bot or folded:
        replayed = rb.state.returns()[seat]
        if abs(replayed - r["winnings"]) > 1e-6:
            raise ValueError(f"winnings {r['winnings']} but the rules pay {replayed}: {r}")
    elif abs(r["winnings"]) not in (0, min(rb.spent)):
        raise ValueError(f"showdown winnings {r['winnings']} but the matched pot is "
                         f"{min(rb.spent)}: {r}")
    return HandResult(seat, list(r["hole_cards"]), list(r.get("board") or []),
                      list(bot) if bot else None, r["action"], int(r["winnings"]), replayed,
                      decisions)


def play_hand(post: Callable[[str, Dict[str, Any]], Dict[str, Any]], token: Optional[str],
              policy: Callable[[Dict[str, np.ndarray]], int]) -> Tuple[Optional[str], HandResult]:
    """One hand through ``post(endpoint, body)``; returns the (possibly new) token."""
    r = post("new_hand", {"token": token} if token else {})
    token = r.get("token") or token
    seat = int(r["client_pos"])
    decisions: List[Dict[str, Any]] = []
    while r.get("winnings") is None:
        rb = rebuild(seat, r["hole_cards"], r["action"], r.get("board") or [])
        kind, seats = rb.state.turn()
        if (kind, seats) != ("act", [seat]):
            raise ValueError(f"not our turn per the rules ({kind} {seats}): {r}")
        frames = seat_frames(rb, seat)
        k = int(policy(frames))
        action = rb.state.decode(seat, [k])
        incr = to_incr(rb, seat, action)
        decisions.append({"candidate": k, "action": action, "incr": incr})
        r = post("act", {"token": token, "incr": incr})
        token = r.get("token") or token
    return token, finish(r, seat, decisions)


def summarize(results: List[HandResult]) -> Dict[str, Any]:
    w = np.array([h.winnings for h in results], dtype=np.float64)
    n = len(w)
    mean = float(w.mean()) if n else 0.0
    se = float(w.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    per100 = 100.0 / BIG_BLIND
    return {"hands": n, "winnings": float(w.sum()), "bb100": mean * per100,
            "bb100_se": se * per100,
            "showdowns_unverified": sum(h.replayed is None for h in results)}


def login(api: SlumbotApi) -> Optional[str]:
    user, pw = os.environ.get("SLUMBOT_USERNAME"), os.environ.get("SLUMBOT_PASSWORD")
    if not (user and pw):
        return None
    tok = api.post("login", {"username": user, "password": pw}).get("token")
    if not tok:
        raise RuntimeError("login: no token in the answer")
    return tok


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--policy", choices=("model", "call", "random"), default=None,
                   help="default: model with --checkpoint, else call")
    p.add_argument("--checkpoint", default=None, help="multi_final.pt / rl_final.pt")
    p.add_argument("--temperature", type=float, default=1.0, help="0 = argmax")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=1.0, help=f"seconds between requests "
                   f"(>= {MIN_DELAY})")
    p.add_argument("--out", default=None, help="directory: hands.jsonl + summary.json")
    p.add_argument("--api", default=API)
    p.add_argument("--force", action="store_true", help=f"allow --hands > {MAX_HANDS_UNFORCED}")
    a = p.parse_args(argv)
    if a.hands > MAX_HANDS_UNFORCED and not a.force:
        raise SystemExit(f"--hands {a.hands} > {MAX_HANDS_UNFORCED}: pass --force (be gentle "
                         "with a free service)")
    kind = a.policy or ("model" if a.checkpoint else "call")
    if kind == "model":
        if not a.checkpoint:
            raise SystemExit("--policy model needs --checkpoint")
        policy: Callable = ModelPolicy(a.checkpoint, a.device, a.temperature, a.seed)
    else:
        policy = CallPolicy() if kind == "call" else RandomPolicy(a.seed)
    api = SlumbotApi(a.api, a.delay)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    token = login(api)
    results: List[HandResult] = []
    log = open(out / "hands.jsonl", "a") if out else None
    try:
        for i in range(a.hands):
            token, h = play_hand(api.post, token, policy)
            results.append(h)
            if log:
                log.write(json.dumps(h.__dict__) + "\n")
                log.flush()
            if (i + 1) % 10 == 0 or i + 1 == a.hands:
                s = summarize(results)
                print(f"{i + 1} hands: {s['bb100']:+.1f} ± {s['bb100_se']:.1f} bb/100", flush=True)
    finally:
        if log:
            log.close()
    s = {**summarize(results), "policy": kind, "checkpoint": a.checkpoint,
         "temperature": a.temperature if kind == "model" else None}
    if out:
        (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s))
    return s


if __name__ == "__main__":
    main()

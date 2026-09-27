"""scripts/slumbot_client.py without the network: a local fake of the Slumbot API
(:class:`FakeSlumbot`) deals from a seeded deck, plays random candidates of the hunl
rules for the bot, writes the API's action strings (street amounts — preflop counting
the blinds —, "/" between streets, ``b20000c///`` all-in run-outs) and answers with
``winnings`` / ``bot_hole_cards`` like the real server, refusing illegal client
actions.  The client must rebuild every hand, act legally and reproduce every payoff;
its frames with placeholder opponent cards must equal the frames with the real ones
(nothing hidden reaches the policy); the CLI runs end to end over HTTP (a local
server) with its JSON log and summary."""

from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import gmsim
import numpy as np
import pytest

from scripts import slumbot_client as sc


class FakeSlumbot:
    """The Slumbot API's protocol over the gmrules hunl rules (see the module doc).
    ``show_cards``: report ``bot_hole_cards`` after a showdown; ``bot`` picks the
    bot's candidate from its legal mask (default: uniform)."""

    def __init__(self, seed: int = 0, show_cards: bool = True, bot=None):
        self.rng = random.Random(seed)
        self.show_cards = show_cards
        self.bot = bot or (lambda mask, rng: rng.choice([k for k in range(len(mask)) if mask[k]]))
        self.hands = 0
        self.requests: List[str] = []
        self.client_actions: List[str] = []

    # -- the API --------------------------------------------------------------------
    def post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        self.requests.append(endpoint)
        if endpoint == "new_hand":
            return self._new_hand()
        if endpoint == "act":
            return self._act(body["incr"])
        if endpoint == "login":
            return {"token": "tok-login"}
        return {"error_msg": f"unknown endpoint {endpoint}"}

    # -- dealing / betting ------------------------------------------------------------
    def _new_hand(self) -> Dict[str, Any]:
        self.client = self.hands % 2
        self.hands += 1
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.st = gmsim.State("hunl", sc.CONFIG)
        for c in deck[:4]:
            self.st.apply_chance(c)
        self.hole = [deck[0:2], deck[2:4]]
        self.board_all, self.board = deck[4:9], []
        self.streets: List[List[str]] = [[]]
        self.spent, self.base = [sc.BIG_BLIND, sc.SMALL_BLIND], 0
        self.old = ""
        self._bot_moves()
        return self._answer()

    def _deal(self) -> None:
        while self.st.turn()[0] == "chance":
            if len(self.board) in (0, 3, 4):
                self.streets.append([])
                self.base = max(self.spent)
            c = self.board_all[len(self.board)]
            self.st.apply_chance(c)
            self.board.append(c)

    def _apply(self, seat: int, a: int) -> None:
        if a == sc.FOLD:
            tok = "f"
        elif a == sc.CALL:
            owe = min(max(self.spent), sc.STACK) - self.spent[seat]
            tok = "c" if owe > 0 else "k"
            self.spent[seat] += owe
        else:
            tok = f"b{a - self.base}"
            self.spent[seat] = a
        self.st.apply_action(seat, a)
        self.streets[-1].append(tok)
        self._deal()

    def _bot_moves(self) -> None:
        bot = 1 - self.client
        while self.st.turn() == ("act", [bot]):
            mask = self.st.frame(bot)["obs"]["action_mask"]
            self._apply(bot, self.st.decode(bot, [self.bot(mask, self.rng)]))

    def _act(self, incr: str) -> Dict[str, Any]:
        s = self.client
        if self.st.turn() != ("act", [s]):
            return {"error_msg": "not your turn"}
        owe = min(max(self.spent), sc.STACK) - self.spent[s]
        if incr == "f":
            a = sc.FOLD
        elif incr in ("k", "c"):
            if (incr == "c") != (owe > 0):
                return {"error_msg": f"illegal {incr!r} (owing {owe})"}
            a = sc.CALL
        elif incr.startswith("b") and incr[1:].isdigit():
            a = self.base + int(incr[1:])
        else:
            return {"error_msg": f"bad incr {incr!r}"}
        if a not in self.st.legal_actions(s):
            return {"error_msg": f"illegal action {incr!r}"}
        self.client_actions.append(incr)
        self._apply(s, a)
        self._bot_moves()
        return self._answer()

    def action(self) -> str:
        return "/".join("".join(t) for t in self.streets)

    def _answer(self) -> Dict[str, Any]:
        r = {"old_action": self.old, "action": self.action(), "client_pos": self.client,
             "hole_cards": [sc.card_str(c) for c in self.hole[self.client]],
             "board": [sc.card_str(c) for c in self.board], "token": f"tok{self.hands}"}
        self.old = r["action"]
        if self.st.turn()[0] == "terminal":
            r["winnings"] = int(self.st.returns()[self.client])
            if self.show_cards and not self.action().endswith("f"):
                r["bot_hole_cards"] = [sc.card_str(c) for c in self.hole[1 - self.client]]
        return r


def test_parse_action():
    assert sc.parse_action("") == [[]]
    assert sc.parse_action("b200c/kb400c/kk/kk") == [["b200", "c"], ["k", "b400", "c"],
                                                     ["k", "k"], ["k", "k"]]
    assert sc.parse_action("b20000c///") == [["b20000", "c"], [], [], []]
    assert sc.parse_action("b200c/") == [["b200", "c"], []]
    for bad in ("b", "x", "b200c////k", "bc"):
        with pytest.raises(ValueError):
            sc.parse_action(bad)
    assert [sc.card_id(c) for c in ("2c", "Ac", "As", "td")] == [0, 48, 51, 33]
    assert sc.card_str(51) == "As"


def test_rebuild_known_lines():
    # SB (seat 1) raises to 300, BB calls; flop bet 400 (= 700 total), call; checks down
    r = sc.rebuild(0, ["Ac", "Kd"], "b300c/kb400c/kk/kk", ["2c", "7d", "9h", "Js", "3c"],
                   ["Qh", "Qs"])
    assert r.events[4:6] == [(1, 300), (0, 1)]
    assert (0, 1) in r.events and (1, 700) in r.events
    assert r.state.turn()[0] == "terminal" and r.spent == [700, 700]
    assert r.state.returns() == [-700.0, 700.0]             # queens beat ace-king high
    # the all-in run-out: every street without action, the board dealt at the end
    r = sc.rebuild(1, ["Ac", "Kd"], "b20000c///", ["2c", "7d", "9h", "Js", "3c"])
    assert r.placeholder and r.state.turn()[0] == "terminal" and r.spent == [20000, 20000]
    # our turn on the flop as the big blind: the board is dealt, the street base moved
    r = sc.rebuild(0, ["Ac", "Kd"], "b250c/", ["2c", "7d", "9h"])
    assert r.state.turn() == ("act", [0]) and r.base == 250
    assert sc.to_incr(r, 0, 1) == "k" and sc.to_incr(r, 0, 500) == "b250"
    # a line the rules refuse (a check facing a bet; a second-street action too early)
    with pytest.raises(ValueError):
        sc.rebuild(0, ["Ac", "Kd"], "b300k", [])
    with pytest.raises(ValueError):
        sc.rebuild(0, ["Ac", "Kd"], "b300c/k", [])      # flop action without flop cards


def _play(fake: FakeSlumbot, policy, n: int) -> List[sc.HandResult]:
    token, out = None, []
    for _ in range(n):
        token, h = sc.play_hand(fake.post, token, policy)
        out.append(h)
    return out


def test_client_plays_legal_hands_and_reproduces_every_payoff():
    fake = FakeSlumbot(seed=1)
    res = _play(fake, sc.RandomPolicy(2), 300)
    assert {h.client_pos for h in res} == {0, 1}
    assert all(h.replayed == h.winnings for h in res)       # every payoff verified
    kinds = {"fold": 0, "showdown": 0, "all_in": 0, "run_out": 0, "raise": 0}
    for h in res:
        rb = sc.rebuild(h.client_pos, h.hole, h.action, h.board, h.bot_hole)
        kinds["fold"] += h.action.endswith("f")
        kinds["showdown"] += h.bot_hole is not None
        kinds["all_in"] += max(rb.spent) == sc.STACK
        kinds["run_out"] += h.action.endswith("/")             # e.g. b20000c///
        kinds["raise"] += any(d["incr"].startswith("b") for d in h.decisions)
    assert all(v > 10 for v in kinds.values()), kinds
    n_dec = sum(len(h.decisions) for h in res)
    assert n_dec == len(fake.client_actions) > 300
    s = sc.summarize(res)
    assert s["hands"] == 300 and s["showdowns_unverified"] == 0
    assert s["winnings"] == sum(h.winnings for h in res)
    assert s["bb100"] == pytest.approx(s["winnings"] / 300)


def test_unshown_showdowns_and_the_call_baseline():
    fake = FakeSlumbot(seed=3, show_cards=False)
    res = _play(fake, sc.CallPolicy(), 120)
    assert all(d["action"] == 1 and d["incr"] in ("k", "c") for h in res for d in h.decisions)
    # showdowns without the bot's cards: only the pot size is checkable (finish did)
    unverified = [h for h in res if h.replayed is None]
    assert len(unverified) > 10 and all(h.bot_hole is None for h in unverified)
    assert sc.summarize(res)["showdowns_unverified"] == len(unverified)


def test_placeholder_cards_never_reach_the_frames():
    """At every client decision: the frames rebuilt with placeholder opponent cards ==
    the frames with the real ones (the fake knows them)."""
    fake = FakeSlumbot(seed=5)
    checked = 0

    def post(endpoint, body):
        nonlocal checked
        r = fake.post(endpoint, body)
        if r.get("winnings") is None:
            seat = r["client_pos"]
            blind = sc.rebuild(seat, r["hole_cards"], r["action"], r["board"])
            true = sc.rebuild(seat, r["hole_cards"], r["action"], r["board"],
                              [sc.card_str(c) for c in fake.hole[1 - seat]])
            a, b = sc.seat_frames(blind, seat), sc.seat_frames(true, seat)
            for k in a:
                assert np.array_equal(a[k], b[k]), k
            assert blind.placeholder and not true.placeholder
            checked += 1
        return r

    token = None
    for _ in range(80):
        token, _ = sc.play_hand(post, token, sc.RandomPolicy(6))
    assert checked > 80


class _Handler(BaseHTTPRequestHandler):
    fake: Optional[FakeSlumbot] = None
    fail_next = 0

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if type(self).fail_next > 0:
            type(self).fail_next -= 1
            self.send_response(503)
            self.end_headers()
            return
        endpoint = self.path.rsplit("/", 1)[-1]
        data = json.dumps(self.fake.post(endpoint, body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_cli_over_http(tmp_path, monkeypatch):
    fake = FakeSlumbot(seed=7)
    _Handler.fake, _Handler.fail_next = fake, 1              # one 503: retried
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    monkeypatch.setattr(sc, "MIN_DELAY", 0.0)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)     # no back-off waits in the test
    monkeypatch.setenv("SLUMBOT_USERNAME", "u")
    monkeypatch.setenv("SLUMBOT_PASSWORD", "p")
    try:
        api = f"http://127.0.0.1:{srv.server_port}/slumbot/api"
        s = sc.main(["--hands", "12", "--policy", "random", "--delay", "0", "--api", api,
                     "--out", str(tmp_path / "run")])
    finally:
        srv.shutdown()
    assert fake.requests[0] == "login" and fake.requests.count("new_hand") == 12
    assert s["hands"] == 12 and s["policy"] == "random"
    lines = (tmp_path / "run" / "hands.jsonl").read_text().splitlines()
    assert len(lines) == 12 and json.loads(lines[0])["client_pos"] in (0, 1)
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["hands"] == 12
    with pytest.raises(SystemExit):
        sc.main(["--hands", "5000"])                          # be gentle: needs --force

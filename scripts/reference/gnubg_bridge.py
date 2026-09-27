"""gnubg as a line-protocol engine process — the backgammon reference and teacher.

Runs in a THROW-AWAY environment with GNU Backgammon's Python bindings (the PyPI
``gnubg`` wheel: gnubg's engine, neural nets and bearoff databases; GPL — used as an
external program, never linked into or vendored by this repository)::

    uv run --no-project --python 3.12 --with gnubg==1.1.0a16 \\
        python scripts/reference/gnubg_bridge.py --plies 2

One JSON request per stdin line, one JSON reply per stdout line (gnubg's own output
is sent to /dev/null).  Moves are read from gnubg's `hint` listing (gnubg's move
notation, parsed here): in gnubg 1.1.0a16 the bindings' findbestmoves tuples are cut
at the first bear-off and ``gnubg.hint()`` crashes.  Positions are from the point of
view of the player on roll, in gnubg's convention: ``board = [opponent[25],
mover[25]]``, each side in its own numbering (index i = point i+1, index 24 = the bar)
— exactly gmrules' ``Side``.

  {"op": "hello"}
      -> {"ok": true, "engine": "gnubg <version>", "plies": P}
  {"op": "legal", "board": B, "dice": [a, b]}
      -> {"moves": [[from, to, from, to, ...], ...]}   every legal move (0-ply list)
  {"op": "move", "board": B, "dice": [a, b], "cube": C}
      -> {"moves": [[[from, to, ...], equity, ply], ...], "mwc": m}   every legal move,
         best first (P-ply cubeful with gnubg's move filters: the moves kept by the
         filters at ply P, then the ones cut at lower plies — compare equities of one ply
         only); m = the best move's match-winning chance (match play; else null)
  {"op": "cube", "board": B, "cube": C}
      -> {"proper": "No double, take", "double": false, "take": true,
          "cannot_double": false, "nd": x, "dt": y, "dp": z}   the player on roll's
         cube decision before the roll (P-ply cubeful; equities in EMG, doubler's
         view; "take" = the proper answer to a double; "mwc": {"nd", "dt", "dp"} the
         same as match-winning chances, match play); "cannot_double": true (no
         equities) when gnubg considers the cube dead for the player on roll

``C = {"value": v, "owner": o, "match": N, "score": [mover, opponent],
"crawford": bool, "jacoby": bool}`` with ``o`` = -1 centred, 0 = the mover owns
the cube, 1 = the opponent does (``match`` 0 = money play).  Moves are flat
(from, to) lists, 1-based from the mover's side, 25 = bar, 0 = off; a pair may span
several dice (one checker's path: 13/7/2 is [13, 2]) — the resulting position is what
counts (gnubg names every point where a checker hits on its way).

Errors: {"error": "..."} (the process keeps running).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

import gnubg

# gnubg prints to the C-level stdout: keep a private handle for the protocol and send
# everything else to /dev/null (captured into a file where a command's text is needed)
_PROTO = os.fdopen(os.dup(1), "w", buffering=1)
_NULL = os.open(os.devnull, os.O_WRONLY)
os.dup2(_NULL, 1)
sys.stdout = sys.stderr


def command(cmd: str) -> str:
    """Run a gnubg CLI command; return what it printed (stdout and stderr: gnubg
    reports refusals such as "You cannot double." on stderr)."""
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as tf:
        saved_err = os.dup(2)
        os.dup2(tf.fileno(), 1)
        os.dup2(tf.fileno(), 2)
        try:
            gnubg.command(cmd)
        finally:
            os.dup2(_NULL, 1)
            os.dup2(saved_err, 2)
            os.close(saved_err)
        tf.seek(0)
        return tf.read().decode("utf-8", "replace")


def setup(plies: int, threads: int = 1) -> None:
    # gnubg's evaluation thread pool is empty by default in the bindings: every
    # threaded path (hint, the game loop) then waits forever
    for c in (f"set threads {threads}", "set sound enable off", "set display off", "set beavers 0",
              "set automatic roll off", "set output mwc off", "set output digits 4",
              f"set evaluation chequerplay evaluation plies {plies}",
              f"set evaluation cubedecision evaluation plies {plies}",
              "set evaluation chequerplay evaluation cubeful on",
              "set evaluation cubedecision evaluation cubeful on",
              "set evaluation chequerplay evaluation prune on",
              "set evaluation cubedecision evaluation prune on",
              "set evaluation chequerplay evaluation noise 0",
              "set evaluation cubedecision evaluation noise 0"):
        command(c)


def board_of(req) -> tuple:
    b = req["board"]
    if len(b) != 2 or any(len(s) != 25 for s in b):
        raise ValueError("board must be [opponent[25], mover[25]]")
    opp, me = (tuple(int(x) for x in b[0]), tuple(int(x) for x in b[1]))
    if min(opp + me) < 0 or sum(opp) > 15 or sum(me) > 15:
        raise ValueError("a side has more than 15 (or negative) checkers")
    for q in range(1, 25):  # my point q is the opponent's point 25 - q
        if me[q - 1] and opp[24 - q]:
            raise ValueError(f"point {q} is held by both sides")
    return (opp, me)


def cubeinfo_of(c):
    """gnubg cubeinfo with the mover as player 0."""
    owner = int(c.get("owner", -1))
    command(f"set jacoby {'on' if c.get('jacoby', False) else 'off'}")
    ci = gnubg.cubeinfo(int(c.get("value", 1)), owner, 0, int(c.get("match", 0)),
                        tuple(int(x) for x in c.get("score", (0, 0))),
                        1 if c.get("crawford", False) else 0, 0)
    ci["jacoby"] = 1 if c.get("jacoby", False) else 0
    ci["beavers"] = 0
    return ci


def set_position(board, c, dice=(0, 0)):
    """Make (board, cube context, dice) gnubg's current position (the mover = player 0,
    on roll); refuse silently-kept boards.  Returns the cubeinfo."""
    pi = gnubg.posinfo(0, 0, 0, 1, tuple(dice))
    ci = cubeinfo_of(c)
    gid = gnubg.gnubgid(board, ci, pi)
    out = command(f"set gnubgid {gid}")
    # gnubg keeps its previous board when it refuses a position ("Illegal position.")
    if "Illegal position" in out or tuple(map(tuple, gnubg.board())) != board:
        raise ValueError(f"gnubg refused position {gid}: {out.strip()[:200]!r}")
    return ci


def mwc(eq, c, ci):
    """Match play: gnubg's match-winning chance of the player on roll for a cubeful
    equity (EMG, normalised to the current cube); money play / no equity: None."""
    if eq is None or not int(c.get("match", 0)):
        return None
    return round(float(gnubg.eq2mwc(float(eq), ci)), 6)


# a move line of `hint`:  "   12. Cubeful 2-ply    13/7* 8/7(2)          Eq.: +0.1234 (-0.0101)"
_MOVE_LINE = re.compile(r"^\s*\d+\.\s+\S+\s+(\d+)-ply\s+(.+?)\s+Eq\.:\s+([+-]?\d+\.\d+)")
_TOKEN = re.compile(r"^(.+?)(?:\((\d)\))?$")


def parse_move(text: str) -> list:
    """gnubg's move notation → flat (from, to) steps (25 = bar, 0 = off). A path
    a/b/c is one checker (a→b, b→c); gnubg names intermediate points only where it hits,
    so a/c without them is the same resulting position; "(n)" repeats the path."""
    steps = []
    for tok in text.split():
        m = _TOKEN.match(tok)
        if m is None:
            raise ValueError(f"move token {tok!r}")
        pts = []
        for p in m.group(1).split("/"):
            p = p.rstrip("*")
            pts.append(25 if p == "bar" else 0 if p == "off" else int(p))
        for _ in range(int(m.group(2) or 1)):
            for a, b in zip(pts, pts[1:]):
                steps += [a, b]
    return steps


def hint_moves(plies: int) -> list:
    """Every legal move of the current position as [steps, equity, ply], from gnubg's
    `hint` listing at `plies`: best first — the moves its filters kept, evaluated at
    full depth, then the ones cut at lower plies (whose equities are NOT comparable)."""
    command(f"set evaluation chequerplay evaluation plies {plies}")
    out = command("hint 100000")
    moves = []
    for line in out.splitlines():
        m = _MOVE_LINE.match(line)
        if m:
            moves.append([parse_move(m.group(2)), float(m.group(3)), int(m.group(1))])
    if not moves and "Eq.:" in out:
        raise ValueError(f"unparsed hint output: {out[-400:]!r}")
    return moves


_EQ = re.compile(r"^\s*\d\.\s+(.*?)\s+([+-]\d+\.\d+)")


def parse_cube_hint(text: str):
    if "cannot double" in text.lower():
        # gnubg's GetDPEq: the cube is dead for the player on roll (a post-Crawford
        # leader, or score + cube already reaching the match length) — gnubg never
        # doubles there (the rules still allow it; gmrules checks the two agree)
        return {"proper": "cannot double", "double": False, "take": True, "cannot_double": True,
                "nd": None, "dt": None, "dp": None}
    nd = dt = dp = None
    proper = None
    for line in text.splitlines():
        if line.startswith("Proper cube action:"):
            proper = line.split(":", 1)[1].strip()
            proper = re.sub(r"\s*\(.*\)\s*$", "", proper)
            continue
        m = _EQ.match(line)
        if not m:
            continue
        label, val = m.group(1).lower(), float(m.group(2))
        if label.startswith("no ") or label.startswith("never"):
            nd = val
        elif "pass" in label:
            dp = val
        elif "take" in label or "beaver" in label:
            dt = val
    if proper is None:
        raise ValueError(f"no cube analysis in gnubg's hint output: {text[-400:]!r}")
    p = proper.lower()
    double = p.startswith("double") or p.startswith("redouble") or p.startswith("optional")
    answer = p.split(",", 1)[1].strip() if "," in p else "take"
    take = not answer.startswith("pass")
    return {"proper": proper, "double": double, "take": take, "cannot_double": False,
            "nd": nd, "dt": dt, "dp": dp}


def handle(req, plies: int):
    op = req.get("op")
    if op == "hello":
        return {"ok": True, "engine": f"gnubg {gnubg.version}", "plies": plies}
    # moves come from gnubg's `hint` listing: the bindings' move tuples (findbestmoves)
    # are cut at the first bear-off in gnubg 1.1.0a16, and gnubg.hint() crashes
    if op == "legal":
        set_position(board_of(req), req.get("cube", {}), req["dice"])
        return {"moves": [m[0] for m in hint_moves(0)]}
    if op == "move":
        c = req.get("cube", {})
        ci = set_position(board_of(req), c, req["dice"])
        moves = hint_moves(int(req.get("plies", plies)))
        return {"moves": moves, "mwc": mwc(moves[0][1] if moves else None, c, ci)}
    if op == "cube":
        c = req.get("cube", {})
        ci = set_position(board_of(req), c)
        r = parse_cube_hint(command("hint"))
        r["mwc"] = {k: mwc(r[k], c, ci) for k in ("nd", "dt", "dp")}
        return r
    raise ValueError(f"unknown op {op!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plies", type=int, default=2, help="evaluation plies (0 = gnubg's 0-ply net)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    setup(args.plies, args.threads)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = handle(json.loads(line), args.plies)
        except Exception as e:  # keep serving: the caller decides what an error means
            out = {"error": f"{type(e).__name__}: {e}"}
        _PROTO.write(json.dumps(out) + "\n")
        _PROTO.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

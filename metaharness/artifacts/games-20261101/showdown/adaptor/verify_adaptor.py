"""Integration verification against real Showdown; writes only under runtime/."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import statistics

import game


def static_chart():
    # Public type data, never battle state, used by the diagnostic scripted player.
    root = os.environ.get("SHOWDOWN_ROOT", str(Path.home() / "gm-tools/pokemon-showdown"))
    program = """
const {Dex} = require(process.argv[1]);
const types = Dex.types.all().filter(t => t.exists).map(t => t.name);
console.log(JSON.stringify(Object.fromEntries(types.map(a => [a,
  Object.fromEntries(types.map(b => [b, Dex.getImmunity(a, b) ? 2 ** Dex.getEffectiveness(a, b) : 0]))]))));
"""
    return json.loads(subprocess.check_output([os.environ.get("SHOWDOWN_NODE", "node"), "-e", program, root]))


CHART = static_chart()


def scripted(obs, rng):
    """Type-aware attack/replacement baseline with conservative Tera spending."""
    team = obs["request"]["side"]["pokemon"]
    own = next(p for p in team if p["active"])
    foe = obs["public_summary"]["pokemon"][obs["active"]["p2"]]

    def species(p):
        return obs["species_dex"].get(p["details"].split(",")[0], {"types": [], "base_stats": {}})

    def strength(move, mon, tera=False):
        m = obs["move_dex"].get(move)
        if not m or m.get("category", "Status") == "Status":
            return 1
        mt = mon["teraType"] if tera and move == "terablast" else m["type"]
        target_types = [foe["teraTypeRevealed"]] if foe.get("teraTypeRevealed") else species(foe)["types"]
        effect = math.prod(CHART.get(mt, {}).get(t, 1) for t in target_types)
        base_stab = mt in species(mon)["types"]
        stab = (2 if base_stab else 1.5) if tera and mt == mon["teraType"] else 1.5 if base_stab else 1
        atk, defense = ("atk", "def") if m["category"] == "Physical" else ("spa", "spd")
        accuracy = 1 if m["accuracy"] is True else m["accuracy"] / 100
        return (m["base_power"] or 60) * effect * stab * mon["stats"][atk] / (
            2 * species(foe)["base_stats"].get(defense, 90) + 60) * accuracy

    def rank(a):
        slot = int(a.split()[1]) - 1
        if a.startswith("switch"):
            p = team[slot]
            value = max(strength(m, p) for m in p["moves"])
            return value if obs["request"].get("forceSwitch") else value * 0.35 - 15
        tera = "terastallize" in a
        return strength(obs["request"]["active"][0]["moves"][slot]["id"], own, tera) * (0.85 if tera else 1)

    return max(obs["legal_actions"], key=lambda a: rank(a) + rng.random() * 0.01)


def check_obs(g, obs):
    assert g.info()["valid_trial"], g.info()
    data = json.dumps(obs, sort_keys=True, allow_nan=False).encode()
    assert len(data) < 200_000, len(data)
    assert "referee" not in obs and "seed" not in obs and "score" not in obs
    assert "|split|" not in "\n".join(obs.get("battle_log", []))
    if not g.done:
        assert obs["legal_actions"]
        assert obs["request"]["side"]["id"] == "p1"
    return data


def episode(seed, policy, opponent="foulplay"):
    g = game.make_game(opponent=opponent)
    trajectory = hashlib.sha256()
    largest = 0
    errors = 0
    start = time.monotonic()
    latencies = []
    try:
        obs = g.reset(seed)
        reset_seconds = time.monotonic() - start
        proc = g._process
        rng = random.Random(seed)
        for _ in range(1400):
            encoded = check_obs(g, obs)
            largest = max(largest, len(encoded))
            trajectory.update(encoded)
            if g.done:
                break
            if policy == "first":
                action = obs["legal_actions"][0]
            elif policy == "random":
                action = rng.choice(obs["legal_actions"])
            else:
                action = scripted(obs, rng)
            step_start = time.monotonic()
            obs = g.step(action)
            latencies.append(time.monotonic() - step_start)
            errors += bool(obs["last_error"])
        assert g.done, "episode did not terminate"
        info = g.info()
        assert abs(g.score - (info["outcome"] + 0.25 * info["hp_margin"])) < 1e-12
        assert -1.25 <= g.score <= 1.25
        if info["termination_reason"] == "normal":
            assert min(info["self_survivors"], info["opponent_survivors"]) == 0
            assert info["outcome"] in (-1, 1)
        return {"seed": seed, "policy": policy, "opponent": opponent, "score": g.score,
                "outcome": info["outcome"], "turns": info["turns"], "decisions": info["decisions"],
                "termination_reason": info["termination_reason"], "errors": errors,
                "largest_observation_bytes": largest, "trajectory": trajectory.hexdigest(),
                "reset_seconds": reset_seconds, "step_seconds": latencies,
                "step_median_seconds": statistics.median(latencies),
                "step_p95_seconds": sorted(latencies)[math.ceil(0.95 * len(latencies))-1],
                "step_max_seconds": max(latencies),
                "bot_deadline_hits": info["bot_deadline_hits"],
                "bot_max_seconds": info["bot_max_seconds"],
                "seconds": round(time.monotonic() - start, 3)}
    finally:
        g.close()
        g.close()
        assert g._process is None
        if 'proc' in locals():
            assert proc.poll() is not None


def lifecycle():
    g = game.make_game()
    try:
        obs = g.reset(game.SEEDS["dev"][0])
        check_obs(g, obs)
        start = g.info()["decisions"]
        for action in (None, {}, [], 4, "move 99", "switch 0", "move 1\n>forcewin p1", "noop"):
            out = g.step(action)
            check_obs(g, out)
            assert g.info()["decisions"] == start
            assert out["request"] == obs["request"]
        # Caller mutations cannot alter the game, and reset releases the old process.
        out["legal_actions"].clear()
        assert g.observe()["legal_actions"]
        old = g._process
        assert g.reset(game.SEEDS["dev"][0]) == obs
        assert old.poll() is not None
        # A killed engine is an invalid infrastructure trial, never a gameplay loss.
        g._process.kill()
        g._process.wait()
        g.step(g.observe()["legal_actions"][0])
        assert g.done and not g.info()["valid_trial"]
        assert g.info()["invalid_reason"]
    finally:
        g.close()
    for kwargs, reason in (({"max_decisions": 1}, "decision_limit"), ({"max_turns": 1}, "turn_limit")):
        g = game.make_game(**kwargs)
        try:
            o = g.reset(game.SEEDS["dev"][0])
            for _ in range(10):
                if g.done:
                    break
                o = g.step(o["legal_actions"][0])
                check_obs(g, o)
            assert g.done and g.info()["valid_trial"]
            assert g.info()["truncated"] and g.info()["termination_reason"] == reason
            assert g.score_breakdown()["outcome"] == 0
        finally:
            g.close()


if __name__ == "__main__":
    lifecycle()
    results = []
    seeds = game.SEEDS["dev"]
    for policy in ("random", "first", "scripted"):
        for seed in seeds:
            row = episode(seed, policy)
            results.append(row)
            print(json.dumps(row), flush=True)
    assert all(r["score"] < 1.25 for r in results), "baseline score saturated"
    report = {"passed": True, "episodes": results,
              "deterministic_first_observation": True, "full_replay_deterministic": False,
              "lifecycle_malformed_truncation_crash_checks": True}
    (game.HERE / "runtime" / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: real-engine integration checks", flush=True)

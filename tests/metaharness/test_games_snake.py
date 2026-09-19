from metaharness.games.snake import SnakeGame


def test_snake_rules():
    g = SnakeGame(width=6, height=6, tick_s=0.1, max_ticks=100)
    g.reset(0)
    assert g.observe()["heading"] == "right"
    g.tick("left")                       # reverse ignored
    assert g.heading == "right"
    for _ in range(10):
        g.tick("keep")
    assert g.done and g.info()["death"] == "wall"
    g.reset(1)
    g.food = (g.body[0][0] + 1, g.body[0][1])
    g.tick("keep")
    assert g.score == 1 and len(g.body) == 4


def test_snake_observation_is_json_serialisable_and_danger_consistent():
    import json
    g = SnakeGame()
    o = g.reset(3)
    json.dumps(o)
    assert o["danger"]["left"] is True   # body is to the left of the head at start
    assert o["danger"]["right"] is False




def test_lanerunner_rules_and_heldout_loop(tmp_path):
    from metaharness.games.lanerunner import LaneRunnerGame
    g = LaneRunnerGame(tick_s=0.1, max_ticks=50)
    o = g.reset(0)
    assert o["lane"] == 1 and len(o["upcoming"]) == 6 and "snake" not in g.describe()
    g.rows[0] = ["high", "high", "."]
    g.tick("right")
    assert not g.done and g.lane == 2 and g.score > 0
    g.rows[0] = [".", ".", "low"]
    g.tick("stay")
    assert g.done and g.info()["death"] == "hit_low"


def test_loop_heldout_game_is_reported(tmp_path):
    from metaharness.cli import main
    from metaharness.core.archive import Archive
    runs = tmp_path / "runs"
    rc = main(["loop", "--tier", "2", "--game", "snake", "--system1", "heuristic", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "0", "--seeds", "0", "--heldout", "100",
               "--heldout-game", "lanerunner", "--heldout-game-kwargs", '{"max_ticks": 30}',
               "--name", "hg", "--runs-dir", str(runs)])
    assert rc == 0
    a = Archive(runs / "hg")
    best = a.best()
    e = (best.dir / "heldout-lanerunner" / "eval.json")
    assert e.exists()
    lb = (a.root / "leaderboard.md").read_text()
    assert "heldout-lanerunner (lanerunner)" in lb and "not used for selection" in lb

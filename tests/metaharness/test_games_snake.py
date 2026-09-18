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



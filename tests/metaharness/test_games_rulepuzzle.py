from collections import deque

from metaharness.testbeds.rulepuzzle import ACTIONS, RulePuzzleGame


def _solvable(game: RulePuzzleGame, seed: int) -> bool:
    """BFS over (pos, have_key, grid-hash) using the true dynamics: every level reachable?"""
    import copy
    game.reset(seed)
    for _level in range(game.n_levels):
        start = copy.deepcopy(game)
        key = lambda g: (g.pos, g.have_key, tuple(map(tuple, g.grid)))
        seen = {key(start)}
        q = deque([(start, 0)])
        found = None
        while q:
            g, d = q.popleft()
            if d >= game.budget_per_level:
                continue
            for a in ACTIONS:
                g2 = copy.deepcopy(g)
                lvl = g2.level
                g2.step(a)
                if g2.level != lvl or g2.done and g2.levels_done > g.levels_done:
                    found = g2
                    break
                k = key(g2)
                if k not in seen and not g2.done:
                    seen.add(k)
                    q.append((g2, d + 1))
            if found:
                break
        if found is None:
            return False
        game = found
        if game.done:
            return game.levels_done == game.n_levels
    return True


def test_rulepuzzle_levels_are_solvable_for_many_seeds():
    families = set()
    for seed in range(12):
        g = RulePuzzleGame(size=8, n_levels=2, budget_per_level=40)
        assert _solvable(g, seed), f"seed {seed} unsolvable ({g.family})"
        families.add(g.family)
    assert len(families) >= 3


def test_rulepuzzle_hides_rules_in_observation():
    g = RulePuzzleGame()
    o = g.reset(0)
    text = str(o) + g.describe() + g.action_spec.describe()
    assert "move_map" not in str(o) and "family" not in str(o)
    assert str(g.move_map) not in text and g.move_map["ACTION1"] not in g.action_spec.describe()
    assert o["grid"][1][1] == 9

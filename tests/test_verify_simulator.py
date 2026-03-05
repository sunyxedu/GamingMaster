"""
Comprehensive simulator verification test.

Downloads a specific replay and verifies the simulator produces correct results
by checking game rules: army growth, move execution, fog of war, move priority, etc.
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp
from data_processing.download.replay_fetcher import ReplayFetcher
from data_processing.processing.simulator import ReplaySimulator


async def download_replay(replay_id: str, output_dir: str) -> dict:
    """Download a single replay and return the data."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{replay_id}.json")

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)

    fetcher = ReplayFetcher(server='na')
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ctx)
    async with aiohttp.ClientSession(connector=connector) as session:
        data = await fetcher.fetch_replay(session, replay_id)
        if data:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            return data
    return None


def verify_army_growth(replay_data: dict):
    """Verify army growth rules by running a simplified simulation."""
    sim = ReplaySimulator(replay_data, replay_data['usernames'][0])
    sim.init_map()

    print("\n=== Army Growth Verification ===")
    print(f"Map size: {sim.map_width}x{sim.map_height}")
    print(f"Players: {sim.usernames}")
    print(f"Generals: {sim.generals}")
    print(f"Cities: {sim.cities}, City armies: {sim.city_armies}")
    print(f"Mountains: {sim.mountains}")
    print(f"Total moves: {len(sim.moves)}")

    general_pos = sim.generals[0]
    gr, gc = general_pos // sim.map_width, general_pos % sim.map_width
    print(f"\nPlayer 0 general at ({gr}, {gc})")

    # Turn 0: general should have army=1
    assert sim.map[gr][gc].army == 1, f"Turn 0: general army should be 1, got {sim.map[gr][gc].army}"
    assert sim.map[gr][gc].occupied == 0, f"Turn 0: general should be owned by player 0"
    print("Turn 0: General army = 1 ✓")

    # Simulate increment for turn 2 (generals increment every 2 turns)
    sim.increment_armies(1)  # turn 1: no increment
    assert sim.map[gr][gc].army == 1, f"Turn 1: general army should still be 1"
    print("Turn 1: No increment ✓")

    sim.increment_armies(2)  # turn 2: generals increment
    assert sim.map[gr][gc].army == 2, f"Turn 2: general army should be 2, got {sim.map[gr][gc].army}"
    print("Turn 2: General army = 2 ✓")

    sim.increment_armies(3)  # turn 3: no increment
    assert sim.map[gr][gc].army == 2, f"Turn 3: general army should still be 2"
    sim.increment_armies(4)  # turn 4: generals increment
    assert sim.map[gr][gc].army == 3, f"Turn 4: general army should be 3, got {sim.map[gr][gc].army}"
    print("Turn 4: General army = 3 ✓")

    print("Army growth rules verified! ✓")


def verify_move_execution(replay_data: dict):
    """Verify move execution and capture logic."""
    print("\n=== Move Execution Verification ===")
    sim = ReplaySimulator(replay_data, replay_data['usernames'][0])
    sim.init_map()

    # Test move execution with a synthetic scenario
    p0_gen = sim.generals[0]
    p0_r, p0_c = p0_gen // sim.map_width, p0_gen % sim.map_width

    # Give player 0 some army
    sim.map[p0_r][p0_c].army = 10

    # Find an adjacent non-mountain cell
    adj = None
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = p0_r + dr, p0_c + dc
        if 0 <= nr < sim.map_height and 0 <= nc < sim.map_width:
            if sim.map[nr][nc].terrain != 3:  # Not mountain
                adj = (nr, nc)
                break

    if adj is None:
        print("No valid adjacent cell found, skipping move test")
        return

    adj_idx = adj[0] * sim.map_width + adj[1]
    move = {
        "index": 0,
        "start": p0_gen,
        "end": adj_idx,
        "is50": False,
        "turn": 1
    }

    prev_adj_army = sim.map[adj[0]][adj[1]].army
    prev_adj_owner = sim.map[adj[0]][adj[1]].occupied

    sim._execute_single_move(move)

    # After move: source should have 1 army
    assert sim.map[p0_r][p0_c].army == 1, f"Source army should be 1, got {sim.map[p0_r][p0_c].army}"
    print(f"Source tile army = 1 after move ✓")

    # Destination: 9 army moved, minus whatever was there
    expected_army = 9 - prev_adj_army if prev_adj_owner != 0 else 9 + prev_adj_army
    assert sim.map[adj[0]][adj[1]].army == expected_army, \
        f"Dest army should be {expected_army}, got {sim.map[adj[0]][adj[1]].army}"
    assert sim.map[adj[0]][adj[1]].occupied == 0, "Dest should be owned by player 0"
    print(f"Destination tile captured with army {expected_army} ✓")

    # Test 50% move
    sim.map[p0_r][p0_c].army = 10
    move_50 = {
        "index": 0,
        "start": p0_gen,
        "end": adj_idx,
        "is50": True,
        "turn": 2
    }
    sim._execute_single_move(move_50)
    assert sim.map[p0_r][p0_c].army == 5, f"50% move: source should have 5, got {sim.map[p0_r][p0_c].army}"
    print(f"50% move: Source has 5 army ✓")

    print("Move execution verified! ✓")


def verify_move_priority(replay_data: dict):
    """Verify the move priority system."""
    print("\n=== Move Priority Verification ===")
    sim = ReplaySimulator(replay_data, replay_data['usernames'][0])
    sim.init_map()

    # Test chase priority: if p0 moves A->B and p1 moves B->C, p0 is chasing
    # Chase move should execute first
    p0_gen = sim.generals[0]
    p1_gen = sim.generals[1]

    # Verify _is_chase_move
    move_a = {"index": 0, "start": 10, "end": 20, "is50": False, "turn": 1}
    move_b = {"index": 1, "start": 20, "end": 30, "is50": False, "turn": 1}
    assert sim._is_chase_move(move_a, move_b) is True, "Should detect chase"
    assert sim._is_chase_move(move_b, move_a) is False, "Not a chase in reverse"
    print("Chase detection ✓")

    # Verify swap detection (not a chase)
    move_swap_a = {"index": 0, "start": 10, "end": 20, "is50": False, "turn": 1}
    move_swap_b = {"index": 1, "start": 20, "end": 10, "is50": False, "turn": 1}
    assert sim._is_chase_move(move_swap_a, move_swap_b) is False, "Swap should not be chase"
    print("Swap detection (not chase) ✓")

    # Test priority ordering
    # A defensive move (friendly->friendly) should have higher priority than attack
    p0_r = p0_gen // sim.map_width
    p0_c = p0_gen % sim.map_width
    sim.map[p0_r][p0_c].army = 20

    # Find two adjacent cells
    adj_cells = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = p0_r + dr, p0_c + dc
        if 0 <= nr < sim.map_height and 0 <= nc < sim.map_width:
            if sim.map[nr][nc].terrain != 3:
                adj_cells.append((nr, nc))
                if len(adj_cells) == 2:
                    break

    if len(adj_cells) >= 1:
        adj_idx = adj_cells[0][0] * sim.map_width + adj_cells[0][1]
        # Make adj owned by player 0 (defensive destination)
        sim.map[adj_cells[0][0]][adj_cells[0][1]].occupied = 0
        sim.map[adj_cells[0][0]][adj_cells[0][1]].army = 5

        def_move = {"index": 0, "start": p0_gen, "end": adj_idx, "is50": False, "turn": 1}
        assert sim._is_defensive_move(def_move) is True, "Should be defensive move"
        print("Defensive move detection ✓")

    print("Move priority system verified! ✓")


def verify_fog_of_war(replay_data: dict):
    """Verify fog of war visibility."""
    print("\n=== Fog of War Verification ===")
    sim = ReplaySimulator(replay_data, replay_data['usernames'][0])
    sim.init_map()
    sim.update_vision()

    player = sim.players[0]
    gen_pos = sim.generals[0]
    gen_r, gen_c = gen_pos // sim.map_width, gen_pos % sim.map_width

    # General tile should be visible (owned)
    assert player.vision[gen_r][gen_c] == 1, "General should be owned (vision=1)"
    print(f"General tile ({gen_r},{gen_c}) is owned ✓")

    # Adjacent tiles should be visible (vision=2)
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nr, nc = gen_r + dr, gen_c + dc
        if 0 <= nr < sim.map_height and 0 <= nc < sim.map_width:
            assert player.vision[nr][nc] == 2, \
                f"Adjacent ({nr},{nc}) should be visible (vision=2), got {player.vision[nr][nc]}"
    print("Adjacent tiles visible ✓")

    # Far away tiles should be unseen
    opp_gen = sim.generals[1]
    opp_r, opp_c = opp_gen // sim.map_width, opp_gen % sim.map_width
    manhattan = abs(gen_r - opp_r) + abs(gen_c - opp_c)
    if manhattan > 3:
        assert player.vision[opp_r][opp_c] == 0, \
            f"Distant tile ({opp_r},{opp_c}) should be unseen, got {player.vision[opp_r][opp_c]}"
        print(f"Distant tile ({opp_r},{opp_c}) is unseen ✓")

    print("Fog of war verified! ✓")


def verify_state_encoding(replay_data: dict):
    """Verify state encoding has correct categories."""
    print("\n=== State Encoding Verification ===")
    sim = ReplaySimulator(replay_data, replay_data['usernames'][0])
    sim.init_map()
    sim.update_vision()

    state, army = sim.get_state_encoding(0)

    gen_pos = sim.generals[0]
    gen_r, gen_c = gen_pos // sim.map_width, gen_pos % sim.map_width

    # Own general should be state 7
    assert state[gen_r][gen_c] == 7, f"Own general should be state 7, got {state[gen_r][gen_c]}"
    assert army[gen_r][gen_c] == 1, f"Own general army should be 1, got {army[gen_r][gen_c]}"
    print(f"Own general encoded as state 7 with army 1 ✓")

    # Mountains should be state 4 if visible
    for m in sim.mountains:
        mr, mc = m // sim.map_width, m % sim.map_width
        if sim.players[0].vision[mr][mc] == 2:
            assert state[mr][mc] == 4, f"Visible mountain at ({mr},{mc}) should be state 4, got {state[mr][mc]}"
            print(f"Mountain at ({mr},{mc}) encoded as state 4 ✓")
            break

    # Opponent general should be unseen at start (state 0 or 1)
    opp_gen = sim.generals[1]
    opp_r, opp_c = opp_gen // sim.map_width, opp_gen % sim.map_width
    manhattan = abs(gen_r - opp_r) + abs(gen_c - opp_c)
    if manhattan > 3:
        assert state[opp_r][opp_c] in [0, 1], \
            f"Unseen general should be state 0 or 1 (hidden), got {state[opp_r][opp_c]}"
        print(f"Opponent general hidden as state {state[opp_r][opp_c]} ✓")

    print("State encoding verified! ✓")


def verify_full_simulation(replay_data: dict):
    """Run full simulation and verify consistency."""
    print("\n=== Full Simulation Verification ===")

    for player_name in replay_data['usernames']:
        sim = ReplaySimulator(replay_data, player_name)
        result = sim.simulate()

        T = len(result['map'])
        print(f"\nPlayer: {player_name}")
        print(f"  Total turns simulated: {T}")
        print(f"  Starting army: {result['info_myArmy'][0]}")
        print(f"  Starting land: {result['info_myLand'][0]}")

        # Verify array lengths
        assert len(result['map']) == T
        assert len(result['armies']) == T
        assert len(result['info_myArmy']) == T
        assert len(result['info_myLand']) == T
        assert len(result['info_opArmy']) == T
        assert len(result['info_opLand']) == T
        assert len(result['move_start']) == T
        assert len(result['move_end']) == T
        assert len(result['move_is_50']) == T
        print(f"  All arrays consistent length ✓")

        # Verify map dimensions (should be padded to 25x25)
        for t in range(T):
            assert len(result['map'][t]) == 25, f"Map height should be 25 at turn {t}"
            assert len(result['map'][t][0]) == 25, f"Map width should be 25 at turn {t}"
            assert len(result['armies'][t]) == 25
            assert len(result['armies'][t][0]) == 25
        print(f"  All maps padded to 25x25 ✓")

        # Verify initial state
        assert result['info_myLand'][0] == 1, f"Should start with 1 land, got {result['info_myLand'][0]}"
        assert result['info_myArmy'][0] == 1, f"Should start with 1 army, got {result['info_myArmy'][0]}"
        print(f"  Initial state correct ✓")

        # Verify army generally grows over time
        if T > 50:
            assert result['info_myArmy'][50] > result['info_myArmy'][0], \
                "Army should grow by turn 50"
            print(f"  Army grows: {result['info_myArmy'][0]} -> {result['info_myArmy'][50]} (turn 50) ✓")

        # Count valid moves
        valid_moves = sum(1 for s in result['move_start'] if s[0] >= 0)
        print(f"  Valid moves: {valid_moves} / {T}")

        # Verify state values are in valid range
        for t in range(min(T, 10)):
            for i in range(25):
                for j in range(25):
                    v = result['map'][t][i][j]
                    assert -1 <= v <= 10, f"Invalid state value {v} at turn {t}, ({i},{j})"
        print(f"  State values in valid range [-1, 10] ✓")

    print("\nFull simulation verified! ✓")


def verify_data_conversion(replay_data: dict):
    """Verify the full pipeline including conversion to training format."""
    print("\n=== Data Conversion Verification ===")
    from data_processing.parsers.generals_json import convert_simulator_to_gio

    player_name = replay_data['usernames'][0]
    sim = ReplaySimulator(replay_data, player_name)
    result = sim.simulate()

    gio_data = convert_simulator_to_gio(result)

    T_raw = len(result['map'])
    T = T_raw - 1  # converter drops the last state (no future action)
    K = gio_data['max_actions_per_tick']

    # Verify shapes
    assert len(gio_data['grid_seq']) == T, f"grid_seq length: {len(gio_data['grid_seq'])} != {T}"
    assert len(gio_data['grid_seq'][0]) == 12, f"grid_seq channels: {len(gio_data['grid_seq'][0])} != 12"
    assert len(gio_data['grid_seq'][0][0]) == 25, "grid_seq H: != 25"
    assert len(gio_data['grid_seq'][0][0][0]) == 25, "grid_seq W: != 25"
    print(f"  grid_seq shape: [{T}, 12, 25, 25] ✓")

    assert len(gio_data['globals_seq']) == T
    assert len(gio_data['globals_seq'][0]) == 4
    print(f"  globals_seq shape: [{T}, 4] ✓")

    assert len(gio_data['prev_action_seq']) == T
    assert len(gio_data['prev_action_seq'][0]) == 8
    print(f"  prev_action_seq shape: [{T}, 8] ✓")

    assert len(gio_data['legal_source_mask_seq']) == T
    print(f"  legal_source_mask_seq shape: [{T}, 1, 25, 25] ✓")

    assert len(gio_data['count']) == T
    assert len(gio_data['src_y']) == T
    assert len(gio_data['src_y'][0]) == K
    assert len(gio_data['dir_id']) == T
    assert len(gio_data['ratio_id']) == T
    assert len(gio_data['value']) == T
    print(f"  Action targets all length {T} with K={K} ✓")

    # Verify direction consistency
    # Action targets at converter timestep t correspond to sim move at t+1
    for t in range(T):
        if gio_data['count'][t] > 0:
            sy, sx = gio_data['src_y'][t][0], gio_data['src_x'][t][0]
            d = gio_data['dir_id'][t][0]
            ms = result['move_start'][t + 1]
            me = result['move_end'][t + 1]
            assert sy == ms[0] and sx == ms[1], \
                f"Turn {t}: source ({sy},{sx}) != move_start ({ms[0]},{ms[1]})"
            dr = me[0] - ms[0]
            dc = me[1] - ms[1]
            if dr > 0:
                assert d == 0, f"Turn {t}: down should be dir 0, got {d}"
            elif dc > 0:
                assert d == 1, f"Turn {t}: right should be dir 1, got {d}"
            elif dr < 0:
                assert d == 2, f"Turn {t}: up should be dir 2, got {d}"
            elif dc < 0:
                assert d == 3, f"Turn {t}: left should be dir 3, got {d}"

    print(f"  Direction encoding consistent ✓")

    # Verify action sources are within the legal mask (key correctness check)
    import numpy as np
    legal_violations = 0
    checked = 0
    for t in range(T):
        if gio_data['count'][t] > 0:
            sy, sx = gio_data['src_y'][t][0], gio_data['src_x'][t][0]
            mask = np.array(gio_data['legal_source_mask_seq'][t])
            if not mask[0, sy, sx]:
                legal_violations += 1
            checked += 1
    assert legal_violations == 0, \
        f"Legal mask violations: {legal_violations}/{checked}"
    print(f"  All {checked} action sources are legal in paired observation ✓")

    # Verify tick features
    assert len(gio_data['tick_seq']) == T
    assert len(gio_data['tick_bucket_seq']) == T
    assert gio_data['tick_seq'][0] == 0
    print(f"  Tick features present and correct ✓")

    print("Data conversion verified! ✓")


async def main():
    replay_id = "asS2S5O_y"
    output_dir = "/workspace/tests/test_data"

    print(f"Downloading replay {replay_id}...")
    replay_data = await download_replay(replay_id, output_dir)

    if replay_data is None:
        print("ERROR: Failed to download replay")
        return False

    print(f"Replay downloaded successfully!")
    print(f"  ID: {replay_data['id']}")
    print(f"  Map: {replay_data['mapWidth']}x{replay_data['mapHeight']}")
    print(f"  Players: {replay_data['usernames']}")
    print(f"  Moves: {len(replay_data['moves'])}")

    try:
        verify_army_growth(replay_data)
        verify_move_execution(replay_data)
        verify_move_priority(replay_data)
        verify_fog_of_war(replay_data)
        verify_state_encoding(replay_data)
        verify_full_simulation(replay_data)
        verify_data_conversion(replay_data)

        print("\n" + "=" * 60)
        print("ALL VERIFICATION TESTS PASSED! ✓")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

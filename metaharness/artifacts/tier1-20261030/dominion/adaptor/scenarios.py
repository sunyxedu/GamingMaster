"""Fixed scenario manifest v2. Seeds are evaluator metadata, never observations."""
ENGINE_COMMIT = 'd51e94604894fd76eecfbbd4fe74d1fe875c0be1'
ADAPTOR_VERSION = '2.0.0'
KINGDOMS = {
    'base_intro': 'Cellar Market Merchant Militia Mine Moat Remodel Smithy Village Workshop'.split(),
    'chapel_engine': 'Chapel Village Smithy Market Festival Laboratory Militia Moat Remodel Merchant'.split(),
    'curse_race': 'Witch Chapel Sentry Village Smithy Market Moat Militia Festival Laboratory'.split(),
    'gardens_rush': 'Gardens Workshop Cellar Village Smithy Market Militia Moat Remodel Merchant'.split(),
    'heldout_thin_gardens': 'Gardens Workshop Chapel Sentry Festival Laboratory Merchant Mine Moat Smithy'.split(),
    'heldout_witch_gain': 'Witch Workshop Cellar Mine Remodel Merchant Market Village Moat Smithy'.split(),
    'heldout_base_draw': 'Cellar Market Merchant Militia Laboratory Moat Remodel Smithy Village Workshop'.split(),
    'heldout_chapel_sentry': 'Chapel Village Smithy Market Festival Laboratory Sentry Moat Remodel Merchant'.split(),
}
SEEDS = {
    'train': [1100, 1101, 1200, 1201, 1300, 1301, 1400, 1401],
    'heldout': [2100, 2101, 2200, 2201, 2300, 2301, 2400, 2401],
    'dev': [3100, 3101],
    'validation': [4100, 4101, 4200, 4201, 4300, 4301, 4400, 4401],
}
SCENARIOS = {}
for pair, kingdom, opponent in [
    (1100, 'base_intro', 'BMU-1'), (1200, 'chapel_engine', 'BMU-1'),
    (1300, 'curse_race', 'BMU-1'), (1400, 'gardens_rush', 'BMU-1'),
    (2100, 'heldout_base_draw', 'BMU-1'), (2200, 'heldout_chapel_sentry', 'BMU-1'),
    (2300, 'heldout_witch_gain', 'BMU-1'), (2400, 'heldout_thin_gardens', 'BMU-1'),
    (3100, 'gardens_rush', 'BMU-1'),
    (4100, 'base_intro', 'BMU-1'), (4200, 'chapel_engine', 'BMU-1'),
    (4300, 'curse_race', 'BMU-1'), (4400, 'gardens_rush', 'BMU-1'),
]:
    for seat in (0, 1):
        SCENARIOS[pair + seat] = dict(kingdom_id=kingdom, opponent_id=opponent,
                                    agent_seat=seat, shuffle_seed=pair)

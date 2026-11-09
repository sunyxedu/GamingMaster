"""Evidence-calibrated matchup policies: Astra supplies missing public mechanics.

Only public observation data is used. Damage estimates are explicitly rough;
observed damage and action ordering override estimates when comparable.
"""
from __future__ import annotations
import json
import copy
import math
import re
import time
from collections import deque

_CHART = {
    "Normal": {"Rock": .5, "Steel": .5, "Ghost": 0},
    "Fire": {"Fire": .5, "Water": .5, "Rock": .5, "Dragon": .5, "Grass": 2, "Ice": 2, "Bug": 2, "Steel": 2},
    "Water": {"Fire": 2, "Water": .5, "Grass": .5, "Dragon": .5, "Ground": 2, "Rock": 2},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Dragon": .5, "Ground": 0, "Flying": 2},
    "Grass": {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5, "Flying": .5, "Bug": .5, "Dragon": .5, "Steel": .5, "Ground": 2, "Rock": 2},
    "Ice": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Rock": 2, "Dark": 2, "Steel": 2, "Poison": .5, "Flying": .5, "Psychic": .5, "Bug": .5, "Fairy": .5, "Ghost": 0},
    "Poison": {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5, "Ghost": .5, "Steel": 0, "Fairy": 2},
    "Ground": {"Fire": 2, "Electric": 2, "Poison": 2, "Rock": 2, "Steel": 2, "Grass": .5, "Bug": .5, "Flying": 0},
    "Flying": {"Grass": 2, "Fighting": 2, "Bug": 2, "Electric": .5, "Rock": .5, "Steel": .5},
    "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": .5, "Steel": .5, "Dark": 0},
    "Bug": {"Grass": 2, "Psychic": 2, "Dark": 2, "Fire": .5, "Fighting": .5, "Poison": .5, "Flying": .5, "Ghost": .5, "Steel": .5, "Fairy": .5},
    "Rock": {"Fire": 2, "Ice": 2, "Flying": 2, "Bug": 2, "Fighting": .5, "Ground": .5, "Steel": .5},
    "Ghost": {"Psychic": 2, "Ghost": 2, "Dark": .5, "Normal": 0},
    "Dragon": {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark": {"Psychic": 2, "Ghost": 2, "Fighting": .5, "Dark": .5, "Fairy": .5},
    "Steel": {"Ice": 2, "Rock": 2, "Fairy": 2, "Fire": .5, "Water": .5, "Electric": .5, "Steel": .5},
    "Fairy": {"Fighting": 2, "Dragon": 2, "Dark": 2, "Fire": .5, "Poison": .5, "Steel": .5},
}


# Generic public move reference data from observed move_dex, no species sets.
_PUBLIC_MOVES = {
    'acrobatics': {'name': 'Acrobatics', 'type': 'Flying', 'category': 'Physical', 'base_power': 55, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if the user has no held item.'},
    'alluringvoice': {'name': 'Alluring Voice', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% confuse target that had a stat rise this turn.'},
    'bellydrum': {'name': 'Belly Drum', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User loses 50% max HP. Maximizes Attack.'},
    'bodypress': {'name': 'Body Press', 'type': 'Fighting', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "Uses user's Def stat as Atk in damage calculation."},
    'bodyslam': {'name': 'Body Slam', 'type': 'Normal', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': '30% chance to paralyze the target.'},
    'bravebird': {'name': 'Brave Bird', 'type': 'Flying', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'bugbite': {'name': 'Bug Bite', 'type': 'Bug', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': "User steals and eats the target's Berry."},
    'bugbuzz': {'name': 'Bug Buzz', 'type': 'Bug', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'bulkup': {'name': 'Bulk Up', 'type': 'Fighting', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Defense by 1."},
    'bulletseed': {'name': 'Bullet Seed', 'type': 'Grass', 'category': 'Physical', 'base_power': 25, 'accuracy': 100, 'priority': 0, 'description': 'Hits 2-5 times in one turn.'},
    'calmmind': {'name': 'Calm Mind', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk and Sp. Def by 1."},
    'closecombat': {'name': 'Close Combat', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Defense and Sp. Def by 1."},
    'curse': {'name': 'Curse', 'type': 'Ghost', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Curses if Ghost, else -1 Spe, +1 Atk, +1 Def.'},
    'darkpulse': {'name': 'Dark Pulse', 'type': 'Dark', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '20% chance to make the target flinch.'},
    'dazzlinggleam': {'name': 'Dazzling Gleam', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'},
    'defog': {'name': 'Defog', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': '-1 evasion; ends user and target hazards/terrain.'},
    'doubleedge': {'name': 'Double-Edge', 'type': 'Normal', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'dragondance': {'name': 'Dragon Dance', 'type': 'Dragon', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Speed by 1."},
    'drainingkiss': {'name': 'Draining Kiss', 'type': 'Fairy', 'category': 'Special', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 75% of the damage dealt.'},
    'drainpunch': {'name': 'Drain Punch', 'type': 'Fighting', 'category': 'Physical', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'},
    'dualwingbeat': {'name': 'Dual Wingbeat', 'type': 'Flying', 'category': 'Physical', 'base_power': 40, 'accuracy': 90, 'priority': 0, 'description': 'Hits 2 times in one turn.'},
    'earthpower': {'name': 'Earth Power', 'type': 'Ground', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'earthquake': {'name': 'Earthquake', 'type': 'Ground', 'category': 'Physical', 'base_power': 100, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dig.'},
    'encore': {'name': 'Encore', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'Target repeats its last move for its next 3 turns.'},
    'energyball': {'name': 'Energy Ball', 'type': 'Grass', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'esperwing': {'name': 'Esper Wing', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% chance to raise user Speed by 1. High crit.'},
    'facade': {'name': 'Facade', 'type': 'Normal', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if user is burn/poison/paralyzed.'},
    'fireblast': {'name': 'Fire Blast', 'type': 'Fire', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': '10% chance to burn the target.'},
    'flamethrower': {'name': 'Flamethrower', 'type': 'Fire', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to burn the target.'},
    'flareblitz': {'name': 'Flare Blitz', 'type': 'Fire', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil. 10% chance to burn. Thaws user.'},
    'flipturn': {'name': 'Flip Turn', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'},
    'focusblast': {'name': 'Focus Blast', 'type': 'Fighting', 'category': 'Special', 'base_power': 120, 'accuracy': 70, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'freezedry': {'name': 'Freeze-Dry', 'type': 'Ice', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze. Super effective on Water.'},
    'grassknot': {'name': 'Grass Knot', 'type': 'Grass', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the target.'},
    'haze': {'name': 'Haze', 'type': 'Ice', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Eliminates all stat changes.'},
    'healbell': {'name': 'Heal Bell', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Cures the user's party of all status conditions."},
    'heatcrash': {'name': 'Heat Crash', 'type': 'Fire', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the user than the target.'},
    'highhorsepower': {'name': 'High Horsepower', 'type': 'Ground', 'category': 'Physical', 'base_power': 95, 'accuracy': 95, 'priority': 0, 'description': 'No additional effect.'},
    'hurricane': {'name': 'Hurricane', 'type': 'Flying', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to confuse target. Can't miss in rain."},
    'hydropump': {'name': 'Hydro Pump', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 80, 'priority': 0, 'description': 'No additional effect.'},
    'hypervoice': {'name': 'Hyper Voice', 'type': 'Normal', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'},
    'icebeam': {'name': 'Ice Beam', 'type': 'Ice', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze the target.'},
    'iceshard': {'name': 'Ice Shard', 'type': 'Ice', 'category': 'Physical', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'},
    'icespinner': {'name': 'Ice Spinner', 'type': 'Ice', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Ends the effects of terrain.'},
    'iciclecrash': {'name': 'Icicle Crash', 'type': 'Ice', 'category': 'Physical', 'base_power': 85, 'accuracy': 90, 'priority': 0, 'description': '30% chance to make the target flinch.'},
    'ironhead': {'name': 'Iron Head', 'type': 'Steel', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to make the target flinch.'},
    'jetpunch': {'name': 'Jet Punch', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'},
    'knockoff': {'name': 'Knock Off', 'type': 'Dark', 'category': 'Physical', 'base_power': 65, 'accuracy': 100, 'priority': 0, 'description': '1.5× damage if foe holds an item. Removes item.'},
    'leafblade': {'name': 'Leaf Blade', 'type': 'Grass', 'category': 'Physical', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'High critical hit ratio.'},
    'leafstorm': {'name': 'Leaf Storm', 'type': 'Grass', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."},
    'leechlife': {'name': 'Leech Life', 'type': 'Bug', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'},
    'liquidation': {'name': 'Liquidation', 'type': 'Water', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Defense by 1."},
    'megahorn': {'name': 'Megahorn', 'type': 'Bug', 'category': 'Physical', 'base_power': 120, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect.'},
    'meteorbeam': {'name': 'Meteor Beam', 'type': 'Rock', 'category': 'Special', 'base_power': 120, 'accuracy': 90, 'priority': 0, 'description': "Raises user's Sp. Atk by 1 on turn 1. Hits turn 2."},
    'mirrorcoat': {'name': 'Mirror Coat', 'type': 'Psychic', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': -5, 'description': 'If hit by special attack, returns double damage.'},
    'moonblast': {'name': 'Moonblast', 'type': 'Fairy', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': "30% chance to lower the target's Sp. Atk by 1."},
    'moonlight': {'name': 'Moonlight', 'type': 'Fairy', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'},
    'nastyplot': {'name': 'Nasty Plot', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk by 2."},
    'nuzzle': {'name': 'Nuzzle', 'type': 'Electric', 'category': 'Physical', 'base_power': 20, 'accuracy': 100, 'priority': 0, 'description': '100% chance to paralyze the target.'},
    'originpulse': {'name': 'Origin Pulse', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'},
    'outrage': {'name': 'Outrage', 'type': 'Dragon', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Lasts 2-3 turns. Confuses the user afterwards.'},
    'overheat': {'name': 'Overheat', 'type': 'Fire', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."},
    'poisonjab': {'name': 'Poison Jab', 'type': 'Poison', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'},
    'protect': {'name': 'Protect', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 4, 'description': 'Prevents moves from affecting the user this turn.'},
    'psychic': {'name': 'Psychic', 'type': 'Psychic', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'psychicnoise': {'name': 'Psychic Noise', 'type': 'Psychic', 'category': 'Special', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'For 2 turns, the target is prevented from healing.'},
    'psychoboost': {'name': 'Psycho Boost', 'type': 'Psychic', 'category': 'Special', 'base_power': 140, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."},
    'psyshock': {'name': 'Psyshock', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Damages target based on Defense, not Sp. Def.'},
    'rapidspin': {'name': 'Rapid Spin', 'type': 'Normal', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'Free user from hazards/bind/Leech Seed; +1 Spe.'},
    'recover': {'name': 'Recover', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'},
    'rest': {'name': 'Rest', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User sleeps 2 turns and restores HP and status.'},
    'revivalblessing': {'name': 'Revival Blessing', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Revives a fainted Pokemon to 50% HP.'},
    'roar': {'name': 'Roar', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': -6, 'description': 'Forces the target to switch to a random ally.'},
    'roost': {'name': 'Roost', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Heals 50% HP. Flying-type removed 'til turn ends."},
    'scald': {'name': 'Scald', 'type': 'Water', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'},
    'seismictoss': {'name': 'Seismic Toss', 'type': 'Fighting', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "Does damage equal to the user's level."},
    'shadowball': {'name': 'Shadow Ball', 'type': 'Ghost', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Sp. Def by 1."},
    'shellsmash': {'name': 'Shell Smash', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Def, SpD by 1; raises Atk, SpA, Spe by 2.'},
    'slackoff': {'name': 'Slack Off', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'},
    'sleeptalk': {'name': 'Sleep Talk', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User must be asleep. Uses another known move.'},
    'sludgebomb': {'name': 'Sludge Bomb', 'type': 'Poison', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'},
    'sludgewave': {'name': 'Sludge Wave', 'type': 'Poison', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': '10% chance to poison adjacent Pokemon.'},
    'softboiled': {'name': 'Soft-Boiled', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'},
    'spikes': {'name': 'Spikes', 'type': 'Ground', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts grounded foes on switch-in. Max 3 layers.'},
    'spiritshackle': {'name': 'Spirit Shackle', 'type': 'Ghost', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Prevents the target from switching out.'},
    'stealthrock': {'name': 'Stealth Rock', 'type': 'Rock', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts foes on switch-in. Factors Rock weakness.'},
    'steameruption': {'name': 'Steam Eruption', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 95, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'},
    'stickyweb': {'name': 'Sticky Web', 'type': 'Bug', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Speed of grounded foes by 1 on switch-in.'},
    'stoneedge': {'name': 'Stone Edge', 'type': 'Rock', 'category': 'Physical', 'base_power': 100, 'accuracy': 80, 'priority': 0, 'description': 'High critical hit ratio.'},
    'strengthsap': {'name': 'Strength Sap', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User heals HP=target's Atk stat. Lowers Atk by 1."},
    'substitute': {'name': 'Substitute', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User takes 1/4 its max HP to put in a substitute.'},
    'suckerpunch': {'name': 'Sucker Punch', 'type': 'Dark', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first. Fails if target is not attacking.'},
    'supercellslam': {'name': 'Supercell Slam', 'type': 'Electric', 'category': 'Physical', 'base_power': 100, 'accuracy': 95, 'priority': 0, 'description': 'User is hurt by 50% of its max HP if it misses.'},
    'superpower': {'name': 'Superpower', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Attack and Defense by 1."},
    'surf': {'name': 'Surf', 'type': 'Water', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dive.'},
    'switcheroo': {'name': 'Switcheroo', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."},
    'swordsdance': {'name': 'Swords Dance', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack by 2."},
    'synthesis': {'name': 'Synthesis', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'},
    'terablast': {'name': 'Tera Blast', 'type': 'Normal', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'If Terastallized: Phys. if Atk > SpA, type = Tera.'},
    'thunder': {'name': 'Thunder', 'type': 'Electric', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to paralyze. Can't miss in rain."},
    'thunderbolt': {'name': 'Thunderbolt', 'type': 'Electric', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to paralyze the target.'},
    'thunderwave': {'name': 'Thunder Wave', 'type': 'Electric', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': 'Paralyzes the target.'},
    'toxic': {'name': 'Toxic', 'type': 'Poison', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': "Badly poisons the target. Poison types can't miss."},
    'trailblaze': {'name': 'Trailblaze', 'type': 'Grass', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': "100% chance to raise the user's Speed by 1."},
    'trick': {'name': 'Trick', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."},
    'tripleaxel': {'name': 'Triple Axel', 'type': 'Ice', 'category': 'Physical', 'base_power': 20, 'accuracy': 90, 'priority': 0, 'description': 'Hits 3 times. Each hit can miss, but power rises.'},
    'uturn': {'name': 'U-turn', 'type': 'Bug', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'},
    'vacuumwave': {'name': 'Vacuum Wave', 'type': 'Fighting', 'category': 'Special', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'},
    'voltswitch': {'name': 'Volt Switch', 'type': 'Electric', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'},
    'waterspout': {'name': 'Water Spout', 'type': 'Water', 'category': 'Special', 'base_power': 150, 'accuracy': 100, 'priority': 0, 'description': "Less power as user's HP decreases. Hits foe(s)."},
    'wavecrash': {'name': 'Wave Crash', 'type': 'Water', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'willowisp': {'name': 'Will-O-Wisp', 'type': 'Fire', 'category': 'Status', 'base_power': 0, 'accuracy': 85, 'priority': 0, 'description': 'Burns the target.'},
    'woodhammer': {'name': 'Wood Hammer', 'type': 'Grass', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'yawn': {'name': 'Yawn', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Puts the target to sleep after 1 turn.'},
    'zenheadbutt': {'name': 'Zen Headbutt', 'type': 'Psychic', 'category': 'Physical', 'base_power': 80, 'accuracy': 90, 'priority': 0, 'description': '20% chance to make the target flinch.'},
}


def ident(value):
    return re.sub(r'[^a-z0-9]', '', str(value or '').lower())


def number(x, default=0.0):
    try:
        n = float(x)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def hp(p):
    c = str(p.get('condition', ''))
    if 'fnt' in c:
        return 0.0, 1.0
    m = re.match(r'(\d+)/(\d+)', c)
    if m:
        return float(m[1]), max(1., float(m[2]))
    return 100., 100.


def fraction(p):
    h, m = hp(p)
    return h/m


def status(p):
    return p.get('status') or next((x for x in str(p.get('condition','')).split() if x in ('brn','par','slp','frz','psn','tox')), '')


def effect(t, types):
    v = 1.
    for typ in types:
        v *= _CHART.get(t, {}).get(typ, 1.)
    return v


def stage(n):
    n = max(-6., min(6., number(n)))
    return (2+n)/2 if n >= 0 else 2/(2-n)


def species(p):
    return str(p.get('details', p.get('ident','').split(':')[-1])).split(',')[0].strip()


def level(p):
    m = re.search(r', L(\d+)', str(p.get('details','')))
    return int(m[1]) if m else 85


def parse_answer(text, legal):
    if not isinstance(text, str):
        return None, {}
    obj = {}
    try:
        start = text.find('{')
        if start >= 0:
            val, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(val, dict):
                obj = val
    except (ValueError, TypeError):
        pass
    action = str(obj.get('action','')).lower().strip()
    if action in legal:
        return action, obj
    # Match the entire Tera phrase, never silently downgrade an illegal Tera.
    for m in re.finditer(r'\b(?:move\s+[1-4](?:\s+terastallize)?|switch\s+[1-6])\b', text.lower()):
        action = re.sub(r'\s+', ' ', m[0])
        if action in legal:
            return action, {}
    return None, {}


class BaseBoard:
    def __init__(self, obs, memory):
        self.obs = obs
        self.req = obs.get('request') or {}
        self.team = (self.req.get('side') or {}).get('pokemon') or []
        self.own = next((p for p in self.team if p.get('active')), self.team[0] if self.team else {})
        self.foe = dict(next((p for p in obs.get('opponent_revealed',[]) if p.get('ident') == (obs.get('active') or {}).get('p2')), {}))
        for k,v in memory.get('reveals',{}).get(self.foe.get('ident'),{}).items():
            if not self.foe.get(k): self.foe[k]=v
        self.public = obs.get('public_summary') or {}
        self.dex = obs.get('species_dex') or {}
        self.moves = dict(_PUBLIC_MOVES)
        self.moves.update(memory.get('move_facts', {}))
        self.moves.update(obs.get('move_dex') or {})
        self.active = (self.req.get('active') or [{}])[0]
        self.legal = [a for a in obs.get('legal_actions',[]) if isinstance(a,str) and a != 'noop']
        self.memory = memory
        self.weather = ident(self.public.get('weather'))
        self.field = str(self.public.get('field',{})).lower()
        self.forced = not any(a.startswith('move ') for a in self.legal)

    def pub(self, p):
        return (self.public.get('pokemon') or {}).get(p.get('ident'), p)

    def ability(self, p):
        return ident(p.get('ability') or self.pub(p).get('ability') or p.get('baseAbility'))

    def item(self, p):
        # An empty private item means consumed/removed, not an unknown item.
        if 'item' in p: return ident(p['item'])
        return ident(self.pub(p).get('item') or self.memory.get('reveals',{}).get(p.get('ident'),{}).get('item'))

    def boost(self, p, stat):
        return number((self.pub(p).get('boosts') or {}).get(stat))

    def types(self, p, tera=False):
        tt = p.get('terastallized') or p.get('teraTypeRevealed') or self.pub(p).get('teraTypeRevealed')
        if tera:
            tt = p.get('teraType') or self.active.get('canTerastallize')
        if isinstance(tt,str) and tt and tt != 'Stellar':
            return [tt]
        return list((self.dex.get(species(p)) or {}).get('types',[]))

    def base_types(self, p):
        return list((self.dex.get(species(p)) or {}).get('types',[]))

    def stat(self, p, s, ignore_boost=False):
        value = number((p.get('stats') or {}).get(s))
        if not value:
            base = number((self.dex.get(species(p),{}).get('base_stats') or {}).get(s), 80)
            # Observed own requests mostly match neutral 31 IV / 84 EV stats.
            # For opponents this is an estimate, not knowledge of their spread.
            value = math.floor((2*base+52)*level(p)/100)+5
        value *= 1 if ignore_boost else stage(self.boost(p,s))
        item, ability = self.item(p), self.ability(p)
        if s == 'atk':
            if item == 'choiceband': value *= 1.5
            if ability in ('hugepower','purepower'): value *= 2
            if ability == 'guts' and status(p): value *= 1.5
        if s == 'spa' and item == 'choicespecs': value *= 1.5
        if s in ('def','spd') and item == 'eviolite': value *= 1.5
        if s == 'spd' and item == 'assaultvest': value *= 1.5
        if s == 'spd' and 'sand' in self.weather and 'Rock' in self.types(p): value *= 1.5
        if s == 'def' and 'snow' in self.weather and 'Ice' in self.types(p): value *= 1.5
        if s == 'spe':
            if item == 'choicescarf': value *= 1.5
            if status(p) == 'par' and ability != 'quickfeet': value *= .5
            if ability == 'quickfeet' and status(p): value *= 1.5
            if (ability == 'swiftswim' and 'rain' in self.weather or
                ability == 'chlorophyll' and ('sun' in self.weather or 'sunny' in self.weather) or
                ability == 'sandrush' and 'sand' in self.weather or
                ability == 'slushrush' and 'snow' in self.weather): value *= 2
            side = 'p1' if p.get('ident','').startswith('p1') else 'p2'
            if 'tailwind' in ident((self.public.get('side_conditions') or {}).get(side,{})): value *= 2
        return max(1., value)

    def maxhp(self, p):
        if p.get('stats'):
            return hp(p)[1]
        base = number((self.dex.get(species(p),{}).get('base_stats') or {}).get('hp'), 80)
        return math.floor((2*base+52)*level(p)/100)+level(p)+10

    def move(self, mid):
        return self.moves.get(ident(mid), {})

    def damage(self, atk, defend, mid, tera=False, defender_tera=False):
        """Approximate damage fraction at a middle roll, before accuracy."""
        m = self.move(mid) if isinstance(mid,str) else mid
        mid = ident(mid) if isinstance(mid,str) else ident(m.get('name'))
        bp = number(m.get('base_power'))
        typ, cat = m.get('type','Normal'), m.get('category','Status')
        ability, item = self.ability(atk), self.item(atk)
        da = self.ability(defend)
        if mid in ('seismictoss','nightshade'):
            return level(atk)/self.maxhp(defend) if effect(typ,self.types(defend,defender_tera)) else 0
        if mid in ('superfang','naturesmadness','ruination'):
            return fraction(defend)*.5 if effect(typ,self.types(defend,defender_tera)) else 0
        if mid in ('endeavor',): return max(0, fraction(defend)-hp(atk)[0]/self.maxhp(defend))
        if mid in ('eruption','waterspout','dragonenergy'): bp = 150*fraction(atk)
        if mid in ('electroball','gyroball','heavyslam','heatcrash','lowkick','grassknot'): bp = bp or 80
        if mid == 'ragefist': bp = 50 # Untracked hits: explicitly conservative.
        if mid in ('storedpower','powertrip'): bp = 20+20*sum(max(0,number(v)) for v in (self.pub(atk).get('boosts') or {}).values())
        if mid == 'lastrespects': bp = 50
        active_tera = (atk.get('teraType') if tera else atk.get('terastallized')) or atk.get('teraTypeRevealed') or self.pub(atk).get('teraTypeRevealed')
        if mid == 'terablast' and active_tera:
            typ = active_tera
            cat = 'Physical' if self.stat(atk,'atk') > self.stat(atk,'spa') else 'Special'
        if cat == 'Status' or bp <= 0: return 0.
        if mid == 'knockoff' and self.item(defend) not in ('','none'): bp *= 1.5
        if mid == 'facade' and status(atk) in ('par','brn','psn','tox'): bp *= 2
        if mid == 'hex' and status(defend): bp *= 2
        if mid == 'acrobatics' and not item: bp *= 2
        if mid in ('bulletseed','rockblast','iciclespear','pinmissile','tailslap','armthrust'):
            bp *= 5 if ability == 'skilllink' else 4.5 if item == 'loadeddice' else 3
        if mid in ('populationbomb',): bp *= 8
        if mid in ('doublehit','dualwingbeat','bonemerang','doubleironbash'): bp *= 2
        if mid in ('surgingstrikes','tripledive','tripleaxel'): bp = 75 if mid == 'surgingstrikes' else 90
        if ability == 'technician' and bp <= 60: bp *= 1.5
        if ability == 'strongjaw' and any(x in mid for x in ('bite','fang','crunch')): bp *= 1.5
        if ability == 'sharpness' and any(x in mid for x in ('blade','slash','cutter','cleave','sword','slic')): bp *= 1.5
        if ability == 'punkrock' and mid in ('boomburst','overdrive','hypervoice','snarl'): bp *= 1.3
        if ability in ('pixilate','aerilate','refrigerate','galvanize') and typ == 'Normal':
            typ = {'pixilate':'Fairy','aerilate':'Flying','refrigerate':'Ice','galvanize':'Electric'}[ability]; bp *= 1.2
        e = effect(typ, self.types(defend,defender_tera))
        if ability in ('scrappy','mindseye') and typ in ('Normal','Fighting') and 'Ghost' in self.types(defend,defender_tera):
            e = effect(typ,[x for x in self.types(defend,defender_tera) if x != 'Ghost'])
        if ability not in ('moldbreaker','turboblaze','teravolt'):
            if typ == 'Ground' and da == 'levitate': e = 0
            if typ == 'Water' and da in ('waterabsorb','stormdrain','dryskin'): e = 0
            if typ == 'Electric' and da in ('voltabsorb','lightningrod','motordrive'): e = 0
            if typ == 'Fire' and da in ('flashfire','wellbakedbody'): e = 0
            if typ == 'Grass' and da == 'sapsipper': e = 0
            if da == 'wonderguard' and e <= 1: e = 0
            if da == 'thickfat' and typ in ('Fire','Ice'): e *= .5
            if da in ('multiscale','shadowshield') and fraction(defend) >= .99: e *= .5
            if da in ('filter','solidrock','prismarmor') and e > 1: e *= .75
        if typ == 'Ground' and self.item(defend) == 'airballoon': e = 0
        if ability == 'tintedlens' and 0 < e < 1: e *= 2
        atstat = 'atk' if cat == 'Physical' else 'spa'
        destat = 'def' if cat == 'Physical' or mid in ('psyshock','psystrike','secretsword') else 'spd'
        a = self.stat(atk, 'def' if mid == 'bodypress' else atstat, da == 'unaware')
        if mid == 'foulplay': a = self.stat(defend,'atk')
        d = self.stat(defend,destat,ability == 'unaware' or mid in ('sacredsword','darkestlariat'))
        if cat == 'Physical' and status(atk) == 'brn' and ability != 'guts' and mid != 'facade': a *= .5
        stab = 1.5 if typ in self.base_types(atk) else 1.
        tt = active_tera
        if tt == typ: stab = 2. if typ in self.base_types(atk) else 1.5
        if ability == 'adaptability' and stab > 1: stab += .5
        mod = stab*e
        if item == 'lifeorb': mod *= 1.3
        if item == 'expertbelt' and e > 1: mod *= 1.2
        if 'rain' in self.weather: mod *= 1.5 if typ == 'Water' else .5 if typ == 'Fire' else 1
        if 'sun' in self.weather: mod *= 1.5 if typ == 'Fire' else .5 if typ == 'Water' else 1
        terrain=ident(self.public.get('field',{}))
        atk_grounded='Flying' not in self.types(atk,tera) and ability!='levitate' and item!='airballoon'
        def_grounded='Flying' not in self.types(defend,defender_tera) and da!='levitate' and self.item(defend)!='airballoon'
        if atk_grounded and ((typ=='Grass' and 'grassyterrain' in terrain) or (typ=='Electric' and 'electricterrain' in terrain) or (typ=='Psychic' and 'psychicterrain' in terrain)): mod*=1.3
        if def_grounded and 'grassyterrain' in terrain and mid in ('earthquake','bulldoze','magnitude'): mod*=.5
        if def_grounded and 'mistyterrain' in terrain and typ=='Dragon': mod*=.5
        side = 'p1' if defend.get('ident','').startswith('p1') else 'p2'
        screens = ident((self.public.get('side_conditions') or {}).get(side,{}))
        if 'auroraveil' in screens or (cat == 'Physical' and 'reflect' in screens) or (cat == 'Special' and 'lightscreen' in screens): mod *= .5
        result = (((2*level(atk)/5+2)*bp*a/d)/50+2)*mod*.925/self.maxhp(defend)
        return max(0., result)

    def priority(self,p,mid):
        m=self.move(mid)
        value=number(m.get('priority'))
        if ident(mid)=='grassyglide' and 'grassyterrain' in ident(self.public.get('field',{})):
            if 'Flying' not in self.types(p) and self.ability(p)!='levitate': value=1
        if self.ability(p)=='prankster' and m.get('category')=='Status': value+=1
        if self.ability(p)=='triage' and ident(mid) in ('drainingkiss','drainpunch','gigadrain','hornleech','floralhealing','synthesis'): value+=3
        if self.ability(p)=='galewings' and fraction(p)>=1 and m.get('type')=='Flying': value+=1
        return value

    def speed_first(self, p, mid):
        priority = self.priority(p,mid)
        known_priorities = [self.priority(self.foe,m) for m in self.foe.get('moves',[]) if self.damage(self.foe,p,m) > 0]
        enemy_priority = max(known_priorities,default=0)
        if priority != enemy_priority: return priority > enemy_priority
        pair = (p.get('ident'),self.foe.get('ident'))
        learned = self.memory.get('order',{}).get(pair)
        # Only use an observed ordering if boosts/status/field have not changed.
        if learned and learned[0] == self.speed_key(p): return learned[1]
        a,b = self.stat(p,'spe'), self.stat(self.foe,'spe')
        if 'trickroom' in ident(self.public.get('field',{})): return a < b*.95
        return a > b*1.05

    def speed_key(self,p):
        return (self.boost(p,'spe'),self.boost(self.foe,'spe'),status(p),status(self.foe),self.weather,self.field,self.item(p),self.item(self.foe))

    def threat(self, p, tera=False):
        # Revealed attacks plus explicitly hypothetical STAB coverage. Unknown
        # coverage is left to Astra; do not assert an unseen move or ability.
        vals = [self.damage(self.foe,p,m,defender_tera=tera) for m in self.foe.get('moves',[])]
        for typ in set(self.base_types(self.foe)+self.types(self.foe)):
            for cat in ('Physical','Special'):
                vals.append(self.damage(self.foe,p,{'name':'hypothetical STAB','type':typ,'category':cat,'base_power':85},defender_tera=tera)*.9)
        est = max(vals,default=.25)
        # Transfer evidence of a stronger-than-modeled attacker to replacement
        # ranking. This is a bounded empirical risk factor, not a hidden set.
        est *= self.memory.get('danger',{}).get(self.foe.get('ident'),1.)
        key = (p.get('ident'),self.foe.get('ident'),str(self.types(p)),str(self.types(self.foe)),self.boost(self.foe,'atk'),self.boost(self.foe,'spa'),self.boost(p,'def'),self.boost(p,'spd'))
        if not tera:
            est = max(est,self.memory.get('incoming',{}).get(key,0))
        return min(2.,est)

    def hazard(self,p):
        if self.item(p) == 'heavydutyboots' or self.ability(p) == 'magicguard': return 0.
        cs = (self.public.get('side_conditions') or {}).get('p1',{})
        types = self.types(p)
        n = .125*effect('Rock',types) if 'stealthrock' in ident(cs) else 0.
        grounded = 'Flying' not in types and self.ability(p) != 'levitate' and self.item(p) != 'airballoon'
        if grounded:
            for k,v in cs.items():
                if ident(k) in ('spikes','movespikes'):
                    layers = number(v.get('layers'),1) if isinstance(v,dict) else 1
                    n += .25 if layers >= 3 else 1/6 if layers >= 2 else .125
        return n

    def residual(self,p):
        n = .0625 if status(p) == 'brn' else .125 if status(p) in ('psn','tox') else 0.
        if 'leechseed' in ident(self.pub(p).get('volatiles',{})): n += .125
        if self.item(p) in ('leftovers','blacksludge'): n -= .0625
        return max(0,n)

    def attack_options(self,p):
        vals=[]
        for m in p.get('moves',[]):
            d = self.damage(p,self.foe,m)
            if d > 0: vals.append((d,m))
        return sorted(vals,reverse=True)

    def rank(self, recent):
        vals=[]
        incoming=self.threat(self.own)
        for action in self.legal:
            part=action.split(); idx=int(part[1])-1
            if part[0] == 'switch':
                if idx >= len(self.team): continue
                p=self.team[idx]
                incoming_p=self.threat(p)
                outgoing=self.attack_options(p)
                best=outgoing[0] if outgoing else (0.,'')
                hp_left=fraction(p)-self.hazard(p)
                if fraction(p)==0:
                    score=30+max((self.stat(p,'atk'),self.stat(p,'spa')))/50 # Revival request only.
                else:
                    first=self.speed_first(p,best[1])
                    hits=incoming_p*(1 if self.forced or first else 2)
                    score=35*min(best[0],fraction(self.foe))+10*hp_left-24*incoming_p
                    if hp_left <= hits*1.08: score -= 30
                    if best[0]*.85>=fraction(self.foe) and first: score+=28
                    if not self.forced:
                        score-=18+22*incoming_p+30*self.hazard(p)
                        if incoming*1.12 >= fraction(self.own): score+=22
                        if len(recent)>=2 and recent[-1].startswith('switch '): score-=18
                vals.append({'action':action,'value':round(score,2),'name':species(p),'incoming_est':round(incoming_p,3),'best_damage_est':round(best[0],3),'entry_hp_cost':round(self.hazard(p),3)})
                continue
            moves=self.active.get('moves') or []
            if idx>=len(moves): continue
            mid=ident(moves[idx].get('id') or moves[idx].get('move'))
            md=self.move(mid); tera=len(part)>2
            damage=self.damage(self.own,self.foe,mid,tera)
            threat=self.threat(self.own,tera)
            first=self.speed_first(self.own,mid)
            acc=md.get('accuracy',100); acc=1. if acc is True else number(acc,100)/100
            score=70*min(damage,fraction(self.foe))*acc
            if damage>0:
                ko=damage*.86>=fraction(self.foe)
                if ko: score += 36 if first else 17
                if not first and threat*1.1>=fraction(self.own): score=score*.1-20
                elif not ko or not first: score-=12*min(threat,fraction(self.own))
                if mid in ('uturn','voltswitch','flipturn') and damage>0: score+=5
                if mid in ('gigadrain','drainpunch','drainingkiss','oblivionwing','leechlife'): score+=12*min(1-fraction(self.own),damage/2)
                if mid in ('flareblitz','bravebird','woodhammer','wavecrash','headsmash') and fraction(self.own)<.35: score-=6
                if mid in ('hyperbeam','gigaimpact','rockwrecker','blastburn','hydrocannon','frenzyplant'): score-=15
                if mid in ('fakeout','firstimpression') and self.memory.get('active_age',0)>0: score=-50
            else:
                score=-8
                desc=str(md.get('description','')).lower()
                if mid in ('recover','roost','slackoff','softboiled','milkdrink','shoreup','synthesis','moonlight','morningsun','strengthsap','rest'):
                    gain=min(.5,1-fraction(self.own))
                    if gain>.18 and threat<.45 and (first or fraction(self.own)>threat*1.15): score=55*gain-15*threat+8
                    if mid=='rest' and status(self.own)=='slp': score=-30
                if mid=='sleeptalk': score=50 if status(self.own)=='slp' else -50
                if mid in ('stealthrock','spikes','stickyweb','toxicspikes') and fraction(self.own)>threat*2:
                    cs=ident((self.public.get('side_conditions') or {}).get('p2',{}))
                    if mid not in cs: score=14
                if mid in ('spore','sleeppowder','lovelykiss','hypnosis','yawn','toxic','willowisp','thunderwave') and not status(self.foe):
                    score=14 if fraction(self.foe)>.5 else 2
                    ft=self.types(self.foe)
                    if mid in ('spore','sleeppowder') and 'Grass' in ft: score=-30
                    if mid=='willowisp' and 'Fire' in ft: score=-30
                    if mid=='toxic' and ('Steel' in ft or 'Poison' in ft): score=-30
                    if mid=='thunderwave' and ('Ground' in ft or 'Electric' in ft): score=-30
                if mid in ('swordsdance','nastyplot','dragondance','quiverdance','calmmind','bulkup','shellsmash'):
                    if fraction(self.own)>threat*3+.15 and max(self.boost(self.own,'atk'),self.boost(self.own,'spa'))<2: score=18
                if mid in ('protect','banefulbunker','spikyshield','silktrap') and self.residual(self.foe)>.05: score=12
            if tera:
                score-=12
                if incoming*1.1>=fraction(self.own) and threat*1.2<fraction(self.own): score+=25
            if recent[-2:]==[action,action] and damage==0: score-=25
            if status(self.own)=='slp' and mid not in ('sleeptalk','snore'): score-=25
            vals.append({'action':action,'value':round(score,2),'name':mid,'damage_est':round(damage,3),'incoming_est':round(threat,3),'acts_first_est':first})
        return sorted(vals,key=lambda x:x['value'],reverse=True)


class Board(BaseBoard):
    """Move-specific public evidence, with uncertainty kept explicit."""
    def damage(self, atk, defend, mid, tera=False, defender_tera=False):
        value = super().damage(atk, defend, mid, tera, defender_tera)
        md = self.move(mid) if isinstance(mid, str) else mid
        # Flash Fire's public activation is a 1.5x Fire boost, not immunity only.
        if md.get('type') == 'Fire' and 'flashfire' in ident(self.pub(atk).get('volatiles', {})):
            value *= 1.5
        return value

    def evidence_context(self, p):
        return (species(p), tuple(self.types(p)), self.ability(p), self.item(p),
                status(p), tuple(sorted((self.pub(p).get('boosts') or {}).items())),
                json.dumps(self.pub(p).get('volatiles') or {},sort_keys=True))

    def damage_field(self):
        return (self.weather,self.field,json.dumps(self.public.get('side_conditions') or {},sort_keys=True))

    def move_threats(self, p, tera=False):
        rows=[]
        for name in self.foe.get('moves', []):
            mid=ident(name); md=self.move(mid)
            if md.get('category') == 'Status': continue
            est=self.damage(self.foe,p,mid,defender_tera=tera)
            samples=self.memory.get('move_evidence',{}).get((self.foe.get('ident'),mid),[])
            # A ratio is useful only for the same attacker context. It transfers
            # to a new defender through the type/stat model, never as raw HP loss.
            scale=1.; exact=0.; used=0
            for sample in samples:
                if sample['attacker_context'] != self.evidence_context(self.foe): continue
                if sample['field'] != self.damage_field(): continue
                if sample.get('prediction',0) > .025:
                    ratio=sample['loss']/sample['prediction']
                    # Criticals and zero-damage events are excluded on collection.
                    scale=max(scale,min(2.5,ratio))
                    used+=1
                if (not tera and sample['defender_context']==self.evidence_context(p)
                        and sample['defender']==p.get('ident')):
                    # HP-scaled moves must shrink after the attacker is weakened.
                    factor=1.
                    if mid in ('eruption','waterspout','dragonenergy'):
                        factor=fraction(self.foe)/max(.01,sample['attacker_hp'])
                    exact=max(exact,sample['loss']*factor)
            est=max(est*scale,exact)
            rows.append({'move':mid,'damage_est':round(est,3),
                         'unknown':not bool(md), 'empirical_samples':used,
                         'reference':'Astra inferred' if mid in self.memory.get('move_facts',{}) and mid not in (self.obs.get('move_dex') or {}) and mid not in _PUBLIC_MOVES else 'public dex'})
        return rows

    def threat(self,p,tera=False):
        rows=self.move_threats(p,tera)
        vals=[r['damage_est'] for r in rows]
        for typ in set(self.base_types(self.foe)+self.types(self.foe)):
            for cat in ('Physical','Special'):
                vals.append(self.damage(self.foe,p,{'name':'hypothetical STAB','type':typ,
                            'category':cat,'base_power':85},defender_tera=tera)*.9)
        return min(2.,max(vals,default=.25))

    def unknown_moves(self):
        return [ident(m) for m in self.foe.get('moves',[]) if not self.move(m)]


class Harness:
    def __init__(self,ctx):
        self.ctx=ctx
        self.memory={'incoming':{},'order':{},'reveals':{},'danger':{},'active_age':0,'move_facts':{},'move_evidence':{}}
        self.recent=deque(maxlen=12)
        self.policy=None
        self.note=''
        self.calls=0
        self.failures=0
        self.steps=0
        self.cooldown=0
        self.rejected=set()
        self.policy_hits=0
        self.model_pending=False
        self.prev_forced=False

    def log(self,event,**kw):
        try: self.ctx.log(event,**kw)
        except Exception: pass

    def ask(self,prompt,remaining):
        """Bound the wait including backend queue time, not only generation.

        If a call remains queued near the deadline, use the remaining time for
        legal code actions. The context cancels the in-flight model at game end.
        No extra model call is submitted while this one is outstanding.
        """
        timeout=min(145.,remaining-50)
        future=self.ctx.spawn(lambda: self.ctx.think(prompt,session_id=None,timeout_s=timeout))
        self.model_pending=True
        while not future.done():
            if self.ctx.done: return None
            if self.ctx.budget.remaining().get('wall_s',0)<100:
                self.log('model_wait_cutoff',reserve_s=100)
                return None
            time.sleep(.05)
        self.model_pending=False
        return future.result()

    def legacy_observe_transition(self,old,new,action):
        """Learn only from p1-visible events; no seed or referee fields."""
        lines=new.get('recent_events') or []
        if isinstance(lines,str): lines=lines.splitlines()
        for line in lines:
            parts=str(line).split('|')
            if len(parts)<4 or not parts[2].startswith('p2'): continue
            who=re.sub(r'^p2[a-z]:', 'p2:', parts[2])
            known=self.memory['reveals'].setdefault(who,{})
            if parts[1] in ('-item','-enditem'): known['item']=parts[3] if parts[1]=='-item' else ''
            if parts[1]=='-ability': known['ability']=parts[3]
            if '[of]' not in line:
                for attr in ('item','ability'):
                    m=re.search(r'\[from\] '+attr+r': ([^|]+)',line)
                    if m: known[attr]=m[1].strip()
        a,b=Board(old,self.memory),Board(new,self.memory)
        changed=a.own.get('ident')!=b.own.get('ident')
        self.memory['active_age']=0 if changed else self.memory['active_age']+1
        if a.foe.get('ident')!=b.foe.get('ident') or changed: return
        attacks=[]; damage=0.; last_side=''
        last_hp=hp(a.own)[0]
        for line in lines:
            parts=str(line).split('|')
            if len(parts)>3 and parts[1]=='move':
                side=parts[2][:2]
                attacks.append((side,ident(parts[3])))
                last_side=side
            if len(parts)>3 and parts[1]=='-damage' and parts[2].startswith('p1'):
                now=hp({'condition':parts[3]})[0]
                if last_side=='p2' and '[from]' not in line: damage+=max(0,last_hp-now)
                last_hp=now
            if len(parts)>3 and parts[1]=='-heal' and parts[2].startswith('p1'):
                last_hp=hp({'condition':parts[3]})[0]
        if damage>0:
            key=(a.own.get('ident'),a.foe.get('ident'),str(a.types(a.own)),str(a.types(a.foe)),a.boost(a.foe,'atk'),a.boost(a.foe,'spa'),a.boost(a.own,'def'),a.boost(a.own,'spd'))
            self.memory['incoming'][key]=max(self.memory['incoming'].get(key,0),damage/hp(a.own)[1])
            predicted=a.threat(a.own)
            prior=self.memory['danger'].get(a.foe.get('ident'),1.)
            if predicted>.03:
                factor=min(2.,prior*(damage/hp(a.own)[1])/predicted)
                self.memory['danger'][a.foe.get('ident')]=max(prior,factor)
        if len(attacks)==2 and {x[0] for x in attacks}=={'p1','p2'}:
            moves=[a.move(x[1]) for x in attacks]
            users=[a.own if x[0]=='p1' else a.foe for x in attacks]
            if all(moves) and a.priority(users[0],attacks[0][1])==a.priority(users[1],attacks[1][1]):
                self.memory['order'][(a.own.get('ident'),a.foe.get('ident'))]=(a.speed_key(a.own),attacks[0][0]=='p1')

    def save_move_facts(self,obj,b):
        facts=obj.get('move_facts')
        if not isinstance(facts,dict): return
        revealed={ident(m) for p in b.obs.get('opponent_revealed',[]) for m in p.get('moves',[])}
        updated=set()
        for name,card in list(facts.items())[:12]:
            mid=ident(name)
            if mid not in revealed or mid in _PUBLIC_MOVES or mid in (b.obs.get('move_dex') or {}): continue
            if not isinstance(card,list) or len(card)!=5: continue
            typ,cat,power,accuracy,priority=card
            if not isinstance(typ,str) or typ not in _CHART or cat not in ('Physical','Special','Status'): continue
            if isinstance(power,bool) or isinstance(priority,bool): continue
            if accuracy is True: accuracy=100
            if accuracy is False: continue
            power=number(power,-1);accuracy=number(accuracy,-1);priority=number(priority,-10)
            if cat=='Status' and any(s['loss']>0 for (foe,move),samples in self.memory['move_evidence'].items() if move==mid for s in samples): continue
            if not (0<=power<=250 and 1<=accuracy<=100 and -7<=priority<=5 and priority==int(priority)): continue
            self.memory['move_facts'][mid]={'name':name,'type':typ,'category':cat,
                'base_power':power,'accuracy':accuracy,'priority':priority,'description':'Public mechanics inferred by Astra; verify against events.'}
            updated.add(mid)
            self.log('move_fact',move=mid,card=card)
        # Raw observed losses remain useful if their move reference arrived later.
        for samples in self.memory['move_evidence'].values():
            for sample in samples:
                if (not sample.get('prediction') or sample['move'] in updated) and sample.get('snapshot'):
                    saved=Board(sample['snapshot'],self.memory)
                    sample['prediction']=saved.damage(saved.foe,saved.own,sample['move'])

    def threat_report(self,b):
        out=[]
        for p in b.team:
            if fraction(p)<=0: continue
            out.append({'pokemon':species(p),'speed_est':round(b.stat(p,'spe')),
                        'hp_fraction':round(fraction(p),3),'moves':b.move_threats(p)})
        return {'foe_speed_est':round(b.stat(b.foe,'spe')),'team':out}

    def evidence_report(self,b):
        out=[]
        for (foe,mid),samples in self.memory['move_evidence'].items():
            if foe!=b.foe.get('ident'): continue
            for s in samples[-2:]:
                out.append({'move':mid,'target':s['defender'],'hp_lost':round(s['loss'],3),
                            'at_least':s['censored'],'predicted':round(s.get('prediction',0),3),
                            'target_types':s['defender_context'][1],'attacker_hp':round(s['attacker_hp'],3),
                            'on_switch':s['on_switch']})
        return out[-12:]

    def observe_transition(self,old,new,action):
        # Parent retains item/ability revelations and speed-order bookkeeping.
        self.legacy_observe_transition(old,new,action)
        work=copy.deepcopy(old)
        work['species_dex']=new.get('species_dex') or old.get('species_dex') or {}
        lines=new.get('recent_events') or []
        if isinstance(lines,str): lines=lines.splitlines()
        req=work.get('request') or {};team=(req.get('side') or {}).get('pokemon') or []
        pub=work.setdefault('public_summary',{});pp=pub.setdefault('pokemon',{})
        opponents=work.setdefault('opponent_revealed',[])
        pending=None
        def who(raw): return re.sub(r'^(p[12])[a-z]:',r'\1:',raw)
        def pokemon(key):
            return next((p for p in (team if key.startswith('p1') else opponents) if p.get('ident')==key),None)
        def flush():
            nonlocal pending
            if pending and pending['loss']>0 and not pending.get('critical'):
                key=(pending.pop('attacker'),pending['move'])
                hist=self.memory['move_evidence'].setdefault(key,[])
                hist.append(pending)
                del hist[:-6]
                self.log('damage_evidence',move=key[1],target=pending['defender'],
                         loss=round(pending['loss'],3),censored=pending['censored'],on_switch=pending['on_switch'])
            pending=None
        for line in lines:
            parts=str(line).split('|')
            if len(parts)<2: continue
            event=parts[1];key=who(parts[2]) if len(parts)>2 else ''
            if event in ('move','switch','drag','turn'): flush()
            if event in ('switch','drag') and len(parts)>4:
                p=pokemon(key)
                if p is None:
                    p={'ident':key,'moves':[]};opponents.append(p)
                p.update(details=parts[3],condition=parts[4])
                for q in team:
                    if key.startswith('p1'): q['active']=q.get('ident')==key
                work.setdefault('active',{})[key[:2]]=key
                pp[key]=dict(pp.get(key,{}),ident=key,details=parts[3],condition=parts[4],boosts={},volatiles={})
                continue
            p=pokemon(key)
            if event=='move' and len(parts)>3 and key.startswith('p2'):
                board=Board(work,self.memory)
                mid=ident(parts[3])
                # Save just the visible snapshot for later calibration if Astra
                # supplies the missing public definition on the next request.
                snap={k:copy.deepcopy(work.get(k)) for k in ('request','active','opponent_revealed','species_dex','move_dex','public_summary','legal_actions')}
                pending={'attacker':key,'move':mid,'defender':board.own.get('ident'),
                    'attacker_context':board.evidence_context(board.foe),
                    'defender_context':board.evidence_context(board.own),
                    'field':board.damage_field(),'attacker_hp':fraction(board.foe),
                    'loss':0.,'prediction':board.damage(board.foe,board.own,mid),
                    'censored':False,'snapshot':snap,'on_switch':action.startswith('switch ')}
            if event=='-crit' and pending: pending['critical']=True
            if event in ('-damage','-heal') and len(parts)>3 and p:
                if event=='-damage' and pending and key==pending['defender'] and '[from]' not in line:
                    before,maximum=hp(p)
                    after=hp({'condition':parts[3]})[0]
                    pending['loss']+=max(0,before-after)/maximum
                    pending['censored']=pending['censored'] or 'fnt' in parts[3]
                p['condition']=parts[3]
                pp.setdefault(key,{})['condition']=parts[3]
            if event=='-terastallize' and len(parts)>3 and p:
                p['terastallized' if key.startswith('p1') else 'teraTypeRevealed']=parts[3]
                pp.setdefault(key,{})['teraTypeRevealed']=parts[3]
            if event in ('-boost','-unboost','-setboost') and len(parts)>4:
                boosts=pp.setdefault(key,{}).setdefault('boosts',{})
                n=number(parts[4]);stat=parts[3]
                boosts[stat]=n if event=='-setboost' else boosts.get(stat,0)+n*(1 if event=='-boost' else -1)
            if event in ('-status','-curestatus') and p:
                p['status']=parts[3] if event=='-status' and len(parts)>3 else ''
            if event in ('-ability','-item','-enditem') and p and len(parts)>3:
                attr='ability' if event=='-ability' else 'item'
                p[attr]=parts[3] if event!='-enditem' else ''
            if event=='-weather' and len(parts)>2: pub['weather']=parts[2]
            if event in ('-start','-end') and len(parts)>3 and p:
                vol=pp.setdefault(key,{}).setdefault('volatiles',{})
                if event=='-start': vol[parts[3]]=parts[4:]
                else:
                    for name in list(vol):
                        if ident(name)==ident(parts[3]): vol.pop(name,None)
            if event in ('-sidestart','-sideend') and len(parts)>3:
                conditions=pub.setdefault('side_conditions',{}).setdefault(key[:2],{})
                if event=='-sidestart': conditions.setdefault(parts[3],{'layers':1})
                else:
                    for name in list(conditions):
                        if ident(name)==ident(parts[3]): conditions.pop(name,None)
            if event in ('-fieldstart','-fieldend') and len(parts)>2:
                field=pub.setdefault('field',{})
                if isinstance(field,dict):
                    if event=='-fieldstart':field[parts[2]]=parts[3:]
                    else:
                        for name in list(field):
                            if ident(name)==ident(parts[2]):field.pop(name,None)
        flush()

    def fingerprint(self,b):
        return (b.foe.get('ident'),species(b.foe),str(b.types(b.foe)),
                json.dumps(b.pub(b.foe).get('boosts') or {},sort_keys=True),
                json.dumps(b.public.get('side_conditions') or {},sort_keys=True),
                b.weather,b.field)

    def save_policy(self,obj,b,action):
        self.policy=None
        self.save_move_facts(obj,b)
        note=obj.get('note')
        if isinstance(note,str): self.note=note[:700]
        pol=obj.get('policy')
        if not isinstance(pol,dict): return
        user=b.own
        if action.startswith('switch '):
            i=int(action.split()[1])-1
            if i>=len(b.team): return
            user=b.team[i]
        attack=ident(pol.get('attack'))
        heal=ident(pol.get('heal'))
        if attack not in [ident(m) for m in user.get('moves',[])]: return
        if b.move(attack).get('category')=='Status': return
        allowed_heal={'recover','roost','slackoff','softboiled','milkdrink','shoreup','synthesis','moonlight','morningsun','strengthsap'}
        if heal not in allowed_heal or heal not in user.get('moves',[]): heal=''
        self.policy={'user':user.get('ident'),'species':species(user),'foe':self.fingerprint(b),
                     'attack':attack,'heal':heal,'heal_below':min(.75,max(.25,number(pol.get('heal_below'),.5))),
                     'hp_floor':min(.8,max(.08,number(pol.get('hp_floor'),.2))),
                     'remaining':min(4,max(0,int(number(pol.get('max_steps'),3)))),
                     'status':status(user),'turn':b.obs.get('turn'),
                     'entry_pending':action.startswith('switch ')}

    def policy_action(self,b,ranking):
        p=self.policy
        if b.unknown_moves(): return None
        if not p or p['remaining']<=0 or b.forced or b.obs.get('last_error'): return None
        if p['user']!=b.own.get('ident') or p['species']!=species(b.own): return None
        current=self.fingerprint(b)
        if p.get('entry_pending'):
            p['entry_pending']=False
            # A policy for a terrain/weather setter already anticipates its
            # own entry effect. Accept only that specific expected change;
            # opponent identity, typing, boosts and side conditions must match.
            setters={'grassysurge':'grassyterrain','electricsurge':'electricterrain',
                     'psychicsurge':'psychicterrain','mistysurge':'mistyterrain',
                     'drizzle':'raindance','drought':'sunnyday',
                     'sandstream':'sandstorm','snowwarning':'snow'}
            expected=setters.get(b.ability(b.own))
            if expected and p['foe'][:5]==current[:5]:
                if expected in ident(b.weather+' '+b.field): p['foe']=current
        if p['foe']!=current: return None
        if p['status']!=status(b.own): return None
        events=b.obs.get('recent_events') or []
        if isinstance(events,str): events=events.splitlines()
        # Routine damage, healing, move reveals and our own boosts do not revoke
        # the policy. Binding effects, foe boosts, field changes and failure do.
        bad=('|-immune|','|-fail|','|-miss|','|-activate|','|-start|','|-end|','|replace|','|detailschange|','|-transform|','|-ability|','|-endability|','|-item|','|-enditem|')
        if any(any(x in str(line) for x in bad) for line in events): return None
        if any(x in ident(b.pub(b.own).get('volatiles',{})) for x in ('encore','disable','confusion','perish','yawn')): return None
        incoming=b.threat(b.own)*1.18+b.residual(b.own)
        hpnow=fraction(b.own)
        mid=p['attack']
        if p['heal'] and hpnow<p['heal_below'] and (b.speed_first(b.own,p['heal']) or hpnow>incoming):
            if incoming<.48: mid=p['heal']
        entry=None
        for r in ranking:
            if r.get('name')==mid and not r['action'].endswith('terastallize') and r['action'].startswith('move '):
                entry=r;break
        if not entry: return None
        fast_ko=(entry.get('acts_first_est') and entry.get('damage_est',0)*.80 >= fraction(b.foe)+.015)
        if (hpnow<p['hp_floor'] or hpnow<=incoming) and not fast_ko: return None
        # Do not continue a now-outclassed attack after stat drops or a changed
        # choice lock. Model policy remains preferred among comparable attacks.
        best_damage=max((r.get('damage_est',0) for r in ranking if not r['action'].endswith('terastallize')),default=0)
        if mid==p['attack'] and not fast_ko and best_damage>entry.get('damage_est',0)*1.6+.10: return None
        if mid==p['heal'] and list(self.recent)[-2:]==[entry['action'],entry['action']]: return None
        p['remaining']-=1
        self.policy_hits+=1
        return entry['action']

    def prompt(self,b,ranking):
        team=[]
        for i,p in enumerate(b.team):
            team.append({'slot':i+1,'species':species(p),'hp':p.get('condition'),'active':p.get('active'),
                         'types':b.types(p),'stats':p.get('stats'),'boosts':b.pub(p).get('boosts',{}),
                         'ability':p.get('ability'),'item':p.get('item'),'tera':p.get('teraType'),
                         'terastallized':p.get('terastallized'),'moves':p.get('moves')})
        dex={}
        for p in b.team:
            if fraction(p)>0:
                for mid in p.get('moves',[]):
                    m=b.move(mid)
                    dex[mid]=[m.get('type'),m.get('category'),m.get('base_power'),m.get('accuracy'),m.get('priority'),m.get('description')]
        events=b.obs.get('recent_events') or []
        if isinstance(events,str): events=events.splitlines()
        foe_roster=[]
        for p in b.obs.get('opponent_revealed',[]):
            q=dict(p)
            for k,v in self.memory['reveals'].get(p.get('ident'),{}).items():
                if not q.get(k): q[k]=v
            q['types_now']=b.types(p);q['base_stats']=(b.dex.get(species(p)) or {}).get('base_stats');foe_roster.append(q)
        state={'turn':b.obs.get('turn'),'legal':b.legal,'forced_switch':b.forced,'team':team,
               'active_request':b.active,'opponent_active':b.foe.get('ident'),'opponents':foe_roster,
               'field':b.public,'recent_protocol':events[-55:],'last_error':b.obs.get('last_error'),
               'estimates':ranking,'move_reference':dex,'previous_strategy':self.note,
               'recent_actions':list(self.recent)[-8:],'calls_remaining':max(0,22-self.calls),
               'revealed_attack_risks':self.threat_report(b),
               'unknown_revealed_moves':b.unknown_moves(),
               'damage_evidence':self.evidence_report(b)}
        instruction='''Play p1 in Gen 9 Random Battle. Choose a winning immediate action AND a short reactive policy for this matchup. No team preview. Opponent is a modest heuristic that often switches away from bad matchups. Avoid repeated switch cycles and capitalize on safe setup, status or hazards when they create progress. Preserve a useful check, but do not sacrifice multiple Pokemon to save Tera. Use Tera when it changes survival/KO or enables a sweep. Respect priority, speed, abilities, choice lock, hazards and actual revealed Tera; your original STAB remains after Tera.
All legal options are available: the estimates are rough aids, NOT recommendations or guaranteed damage. Opponent stats use a neutral balanced-spread estimate; unknown coverage/abilities/items are NOT modeled. Hypothetical STAB threats may overestimate; use revealed protocol damage and your Pokemon knowledge. Special move/ability mechanics may be omitted. Full own moves and stats are provided. Consider opponent coverage and likely switching before selecting a switch. Forced switch gives no immediate attack. A low-HP fast attacker can be better than a healthy slower victim.
Missing enemy move definitions are UNKNOWN, not harmless. First inspect revealed_attack_risks and damage_evidence. Damage on a switch is real evidence, including a KO's lower bound. For each unknown_revealed_moves entry, use your Pokemon rules knowledge to supply its public base data in move_facts; do not invent unseen moves or sets. Data format is moveID:[type,Physical/Special/Status,basePower,accuracyPercent,priority]. Use standard unmodified base power (e.g. HP-dependent moves are adjusted by code). Omit any fact you do not know. A revealed high-power move may make generic STAB estimates dangerously low. Compare replacements on speed, incoming damage, and ability to retaliate; do not call a neutral target a resist. Evaluate whether damaging an HP-dependent attack's user first prevents a KO. Do not set up when a known hit threatens a KO.
Return JSON, under 230 words, no analysis outside it:
{"action":"one exact current legal command","move_facts":{},"policy":{"attack":"move ID","heal":"recovery move ID or empty","heal_below":0.55,"hp_floor":0.2,"max_steps":3},"note":"short strategic memory: win condition, threats, next step"}
Policy is for the Pokemon that will be active AFTER your immediate action (including selected replacement); choose its main attacking move. Code may repeat that attack for up to max_steps (0-4) while the same foe stays active, or use heal below the threshold. It resolves slots fresh and cancels on dangerous damage, changed foe/types/boosts/field/status, failures or binding effects. Set max_steps=0 for a delicate prediction. Setup, Tera and switches are only the immediate action, never repeated blindly. The policy's attack must be a damaging move from that Pokemon's set. Normal damage/move reveals alone do not invalidate it. Avoid an extremely conservative hp_floor without reason. Note is carried into future independent calls.
STATE:
'''
        return instruction+json.dumps(state,ensure_ascii=False,separators=(',',':'),default=str)

    def guarded_model_action(self, b, ranking, action):
        """Accept model advice only when it survives a local tactical sanity check.

        Astra sees uncertainty and is useful for switches/setup, but a prose answer
        can miss that the current move is a guaranteed knockout or that a proposed
        replacement is immediately lost.  The deterministic evaluator is deliberately
        conservative: it vetoes only high-confidence tactical blunders and otherwise
        preserves Astra's strategic choice.
        """
        if action not in b.legal:
            return None, 'illegal'
        top = ranking[0] if ranking else {}
        if action.startswith('switch ') and not b.forced:
            # A second voluntary switch in a row is usually a momentum leak.
            # The parent ranked this pattern down, but a direct Astra answer
            # bypassed that ranking and produced switch loops in losing traces.
            if self.recent and self.recent[-1].startswith('switch ') and not self.prev_forced:
                cand=next((x for x in ranking if x.get('action')==action),{})
                if cand.get('incoming_est',1) > .45 or cand.get('best_damage_est',0) < .30 * max(.01,fraction(b.foe)):
                    attack=next((x for x in ranking if x.get('action','').startswith('move ') and not x['action'].endswith('terastallize')),None)
                    if attack:
                        return attack['action'], 'veto_switch_loop'
            # Never give up an estimated immediate KO for a voluntary switch.
            for r in ranking:
                if r.get('action','').startswith('move ') and not r['action'].endswith('terastallize'):
                    if r.get('acts_first') and r.get('damage_est',0) >= .82 * max(.01, fraction(b.foe)):
                        return r['action'], 'veto_switch_ko'
                    break
            try:
                idx=int(action.split()[1])-1
                if idx < len(b.team) and fraction(b.team[idx]) > 0:
                    incoming=r.get('incoming_est',0) if (r:=next((x for x in ranking if x.get('action')==action),{})) else 0
                    # A switch that likely faints to the revealed attack is worse
                    # than taking a meaningful hit with the active Pokémon.
                    if incoming >= .78 and top.get('damage_est',0) >= .28 * max(.01,fraction(b.foe)):
                        return top.get('action', action), 'veto_lethal_switch'
            except Exception:
                pass
        if 'terastallize' in action:
            if not (b.active.get('canTerastallize') and not b.own.get('terastallized')):
                return top.get('action', action), 'veto_spent_tera'
            # Tera is reserved for a decisive survival/KO swing.
            plain=next((x for x in ranking if x.get('action')==action.replace(' terastallize','')),None)
            chosen=next((x for x in ranking if x.get('action')==action),None)
            if plain and chosen and chosen.get('damage_est',0) < plain.get('damage_est',0)*1.12 and chosen.get('incoming_est',0) < fraction(b.own)*.85:
                return top.get('action', action), 'veto_small_tera'
        return action, 'accepted'

    def emergency_action(self, b, ranking):
        """Apply only high-confidence responses to a revealed lethal attack.

        The normal ranker is deliberately exploratory.  Once a move has dealt
        substantial damage, however, preserving a Tera that does not change the
        defensive matchup or repeatedly entering a near-certain KO is avoidable
        bookkeeping error.  This filter is used for code fallbacks as well as
        model answers.
        """
        if not ranking:
            return None, ''
        # Never spend Tera merely because it is the only high-ranked spelling of
        # a move.  Require a clear estimated survival or knockout swing.
        top = ranking[0]
        if str(top.get('action','')).endswith(' terastallize'):
            action = top['action']
            plain = next((r for r in ranking if r.get('action') == action[:-13]), None)
            if plain is not None:
                gain = b.threat(b.own, False) - b.threat(b.own, True)
                ko_swing = top.get('damage_est', 0) >= .86 * max(.01, fraction(b.foe)) and plain.get('damage_est', 0) < .86 * max(.01, fraction(b.foe))
                if gain < .14 and not ko_swing:
                    for r in ranking:
                        a = r.get('action','')
                        if a.startswith('move ') and not a.endswith(' terastallize'):
                            self.log('veto_fallback_tera',chosen=action,replacement=a,def_gain=round(gain,3))
                            return a, 'veto_fallback_tera'
        # If the active slot is in immediate danger, prefer a replacement with
        # the smallest revealed incoming damage, provided it can retaliate.
        incoming = b.threat(b.own) + b.residual(b.own)
        # A fast guaranteed knockout is safer than any switch, even when the
        # active slot is itself threatened.
        for r in ranking:
            a = r.get('action','')
            if (a.startswith('move ') and not a.endswith(' terastallize')
                    and r.get('acts_first') and r.get('damage_est',0) >= .86 * max(.01, fraction(b.foe))):
                return a, 'finish_ko'
        if not b.forced and fraction(b.own) > 0 and incoming >= max(.62, fraction(b.own)*.92):
            choices=[]
            for r in ranking:
                a=r.get('action','')
                if not a.startswith('switch '): continue
                try: idx=int(a.split()[1])-1
                except Exception: continue
                if idx < 0 or idx >= len(b.team) or fraction(b.team[idx]) <= 0: continue
                # Do not switch to a slot already estimated to lose immediately.
                if r.get('incoming_est',2.0) >= .88: continue
                choices.append(r)
            if choices:
                choices.sort(key=lambda r:(r.get('incoming_est',2),-r.get('best_damage_est',0),-r.get('value',0)))
                pick=choices[0]
                self.log('emergency_switch',action=pick.get('action'),incoming=round(incoming,3),replacement_incoming=pick.get('incoming_est'))
                return pick['action'], 'emergency_switch'
        return top.get('action'), 'ranked'

    def run_episode(self,game):
        ctx=self.ctx
        try: obs=ctx.observe()
        except Exception:
            try: obs=game.observe()
            except Exception: return
        while not ctx.done and not getattr(game,'done',False) and self.steps<880:
            if not isinstance(obs,dict):
                try: obs=ctx.observe()
                except Exception: return
                if not isinstance(obs,dict): return
            b=Board(obs,self.memory)
            if not b.legal: return
            try: ranking=b.rank(list(self.recent))
            except Exception as exc:
                self.log('rank_error',error=type(exc).__name__)
                ranking=[{'action':a,'value':0} for a in b.legal]
            ranking=[r for r in ranking if r['action'] not in self.rejected]
            if not ranking:
                self.rejected.clear();ranking=[{'action':a,'value':0} for a in b.legal]
            action=None; source='fallback'
            try: action=self.policy_action(b,ranking)
            except Exception: self.policy=None
            if action: source='policy'
            else:
                self.policy=None
                try: rem=ctx.budget.remaining()
                except Exception: rem={'system2_calls':0,'wall_s':0}
                # Reserve wall time to finish with code; never consume the final
                # call, since budget exhaustion itself closes an episode.
                can_ask=(not self.model_pending and self.calls<22 and rem.get('system2_calls',0)>1 and rem.get('wall_s',0)>185 and self.cooldown<=0)
                if can_ask and len(ranking)>1:
                    self.calls+=1
                    try:
                        res=self.ask(self.prompt(b,ranking),rem['wall_s'])
                        action,obj=parse_answer(getattr(res,'text',''),[r['action'] for r in ranking])
                        if action:
                            action,guard=self.guarded_model_action(b,ranking,action)
                            source='astra' if guard=='accepted' else 'astra_guard'
                            self.log('model_guard',decision=guard,action=action)
                            self.save_policy(obj,b,action);self.failures=0
                        else: self.cooldown=2
                    except Exception as exc:
                        self.log('model_failure',error=type(exc).__name__)
                        self.failures+=1;self.cooldown=min(4,self.failures)
                elif self.cooldown>0: self.cooldown-=1
            if ctx.done: return
            if action is None:
                action, emergency_reason = self.emergency_action(b, ranking)
                if emergency_reason not in ('', 'ranked'):
                    source = 'safety'
            if action not in b.legal:
                action, _ = self.emergency_action(b, ranking)
                if action not in b.legal: action=ranking[0]['action']
            self.log('choice',source=source,action=action,turn=obs.get('turn'),calls=self.calls,policy_hits=self.policy_hits)
            old=obs
            self.prev_forced=b.forced
            try:
                nxt=ctx.act(action)
                obs=nxt if isinstance(nxt,dict) else ctx.observe()
            except Exception as exc:
                if ctx.done: return
                self.log('act_rejected',error=type(exc).__name__,action=action)
                self.rejected.add(action);self.policy=None
                try: obs=ctx.observe()
                except Exception: return
                if len(self.rejected)>12: return
                continue
            self.steps+=1
            if isinstance(obs,dict) and obs.get('last_error'):
                self.rejected.add(action);self.policy=None
                if len(self.rejected)>12: return
                continue
            self.rejected.clear()
            self.recent.append(action)
            if isinstance(obs,dict):
                try: self.observe_transition(old,obs,action)
                except Exception: pass
        self.log('harness_complete',steps=self.steps,calls=self.calls,policy_hits=self.policy_hits)


def build(ctx):
    return Harness(ctx)

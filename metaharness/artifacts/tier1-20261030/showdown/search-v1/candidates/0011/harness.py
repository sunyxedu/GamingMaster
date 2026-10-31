"""Two-turn exchange search with model-proposed, explicitly uncertain threats.

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
_PUBLIC_MOVES = {'flamethrower': {'name': 'Flamethrower', 'type': 'Fire', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to burn the target.'}, 'sludgebomb': {'name': 'Sludge Bomb', 'type': 'Poison', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'}, 'steameruption': {'name': 'Steam Eruption', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 95, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'}, 'haze': {'name': 'Haze', 'type': 'Ice', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Eliminates all stat changes.'}, 'icespinner': {'name': 'Ice Spinner', 'type': 'Ice', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Ends the effects of terrain.'}, 'bellydrum': {'name': 'Belly Drum', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User loses 50% max HP. Maximizes Attack.'}, 'terablast': {'name': 'Tera Blast', 'type': 'Normal', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'If Terastallized: Phys. if Atk > SpA, type = Tera.'}, 'liquidation': {'name': 'Liquidation', 'type': 'Water', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Defense by 1."}, 'meteorbeam': {'name': 'Meteor Beam', 'type': 'Rock', 'category': 'Special', 'base_power': 120, 'accuracy': 90, 'priority': 0, 'description': "Raises user's Sp. Atk by 1 on turn 1. Hits turn 2."}, 'darkpulse': {'name': 'Dark Pulse', 'type': 'Dark', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '20% chance to make the target flinch.'}, 'hurricane': {'name': 'Hurricane', 'type': 'Flying', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to confuse target. Can't miss in rain."}, 'earthpower': {'name': 'Earth Power', 'type': 'Ground', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'thunderbolt': {'name': 'Thunderbolt', 'type': 'Electric', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to paralyze the target.'}, 'nuzzle': {'name': 'Nuzzle', 'type': 'Electric', 'category': 'Physical', 'base_power': 20, 'accuracy': 100, 'priority': 0, 'description': '100% chance to paralyze the target.'}, 'grassknot': {'name': 'Grass Knot', 'type': 'Grass', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the target.'}, 'focusblast': {'name': 'Focus Blast', 'type': 'Fighting', 'category': 'Special', 'base_power': 120, 'accuracy': 70, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'recover': {'name': 'Recover', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'}, 'dazzlinggleam': {'name': 'Dazzling Gleam', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'}, 'psyshock': {'name': 'Psyshock', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Damages target based on Defense, not Sp. Def.'}, 'calmmind': {'name': 'Calm Mind', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk and Sp. Def by 1."}, 'fireblast': {'name': 'Fire Blast', 'type': 'Fire', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': '10% chance to burn the target.'}, 'moonlight': {'name': 'Moonlight', 'type': 'Fairy', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'}, 'moonblast': {'name': 'Moonblast', 'type': 'Fairy', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': "30% chance to lower the target's Sp. Atk by 1."}, 'shadowball': {'name': 'Shadow Ball', 'type': 'Ghost', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Sp. Def by 1."}, 'mirrorcoat': {'name': 'Mirror Coat', 'type': 'Psychic', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': -5, 'description': 'If hit by special attack, returns double damage.'}, 'stickyweb': {'name': 'Sticky Web', 'type': 'Bug', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Speed of grounded foes by 1 on switch-in.'}, 'leechlife': {'name': 'Leech Life', 'type': 'Bug', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'}, 'seismictoss': {'name': 'Seismic Toss', 'type': 'Fighting', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "Does damage equal to the user's level."}, 'softboiled': {'name': 'Soft-Boiled', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'}, 'thunderwave': {'name': 'Thunder Wave', 'type': 'Electric', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': 'Paralyzes the target.'}, 'healbell': {'name': 'Heal Bell', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Cures the user's party of all status conditions."}, 'stoneedge': {'name': 'Stone Edge', 'type': 'Rock', 'category': 'Physical', 'base_power': 100, 'accuracy': 80, 'priority': 0, 'description': 'High critical hit ratio.'}, 'closecombat': {'name': 'Close Combat', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Defense and Sp. Def by 1."}, 'knockoff': {'name': 'Knock Off', 'type': 'Dark', 'category': 'Physical', 'base_power': 65, 'accuracy': 100, 'priority': 0, 'description': '1.5× damage if foe holds an item. Removes item.'}, 'swordsdance': {'name': 'Swords Dance', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack by 2."}, 'scald': {'name': 'Scald', 'type': 'Water', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'}, 'icebeam': {'name': 'Ice Beam', 'type': 'Ice', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze the target.'}, 'substitute': {'name': 'Substitute', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User takes 1/4 its max HP to put in a substitute.'}, 'nastyplot': {'name': 'Nasty Plot', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk by 2."}, 'drainingkiss': {'name': 'Draining Kiss', 'type': 'Fairy', 'category': 'Special', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 75% of the damage dealt.'}, 'stealthrock': {'name': 'Stealth Rock', 'type': 'Rock', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts foes on switch-in. Factors Rock weakness.'}, 'earthquake': {'name': 'Earthquake', 'type': 'Ground', 'category': 'Physical', 'base_power': 100, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dig.'}, 'flipturn': {'name': 'Flip Turn', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'}, 'bugbite': {'name': 'Bug Bite', 'type': 'Bug', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': "User steals and eats the target's Berry."}, 'dualwingbeat': {'name': 'Dual Wingbeat', 'type': 'Flying', 'category': 'Physical', 'base_power': 40, 'accuracy': 90, 'priority': 0, 'description': 'Hits 2 times in one turn.'}, 'roost': {'name': 'Roost', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Heals 50% HP. Flying-type removed 'til turn ends."}, 'outrage': {'name': 'Outrage', 'type': 'Dragon', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Lasts 2-3 turns. Confuses the user afterwards.'}, 'dragondance': {'name': 'Dragon Dance', 'type': 'Dragon', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Speed by 1."}, 'voltswitch': {'name': 'Volt Switch', 'type': 'Electric', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'}, 'psychicnoise': {'name': 'Psychic Noise', 'type': 'Psychic', 'category': 'Special', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'For 2 turns, the target is prevented from healing.'}, 'uturn': {'name': 'U-turn', 'type': 'Bug', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'}, 'sludgewave': {'name': 'Sludge Wave', 'type': 'Poison', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': '10% chance to poison adjacent Pokemon.'}, 'roar': {'name': 'Roar', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': -6, 'description': 'Forces the target to switch to a random ally.'}, 'iciclecrash': {'name': 'Icicle Crash', 'type': 'Ice', 'category': 'Physical', 'base_power': 85, 'accuracy': 90, 'priority': 0, 'description': '30% chance to make the target flinch.'}, 'iceshard': {'name': 'Ice Shard', 'type': 'Ice', 'category': 'Physical', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'}, 'surf': {'name': 'Surf', 'type': 'Water', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dive.'}, 'yawn': {'name': 'Yawn', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Puts the target to sleep after 1 turn.'}, 'vacuumwave': {'name': 'Vacuum Wave', 'type': 'Fighting', 'category': 'Special', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'}, 'esperwing': {'name': 'Esper Wing', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% chance to raise user Speed by 1. High crit.'}, 'synthesis': {'name': 'Synthesis', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'}, 'bodypress': {'name': 'Body Press', 'type': 'Fighting', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "Uses user's Def stat as Atk in damage calculation."}, 'woodhammer': {'name': 'Wood Hammer', 'type': 'Grass', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'spikes': {'name': 'Spikes', 'type': 'Ground', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts grounded foes on switch-in. Max 3 layers.'}, 'leafstorm': {'name': 'Leaf Storm', 'type': 'Grass', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."}, 'superpower': {'name': 'Superpower', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Attack and Defense by 1."}, 'hydropump': {'name': 'Hydro Pump', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 80, 'priority': 0, 'description': 'No additional effect.'}, 'encore': {'name': 'Encore', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'Target repeats its last move for its next 3 turns.'}, 'alluringvoice': {'name': 'Alluring Voice', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% confuse target that had a stat rise this turn.'}, 'flareblitz': {'name': 'Flare Blitz', 'type': 'Fire', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil. 10% chance to burn. Thaws user.'}, 'megahorn': {'name': 'Megahorn', 'type': 'Bug', 'category': 'Physical', 'base_power': 120, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect.'}, 'poisonjab': {'name': 'Poison Jab', 'type': 'Poison', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'}, 'toxic': {'name': 'Toxic', 'type': 'Poison', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': "Badly poisons the target. Poison types can't miss."}, 'protect': {'name': 'Protect', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 4, 'description': 'Prevents moves from affecting the user this turn.'}, 'rapidspin': {'name': 'Rapid Spin', 'type': 'Normal', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'Free user from hazards/bind/Leech Seed; +1 Spe.'}, 'bodyslam': {'name': 'Body Slam', 'type': 'Normal', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': '30% chance to paralyze the target.'}, 'bravebird': {'name': 'Brave Bird', 'type': 'Flying', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'acrobatics': {'name': 'Acrobatics', 'type': 'Flying', 'category': 'Physical', 'base_power': 55, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if the user has no held item.'}, 'leafblade': {'name': 'Leaf Blade', 'type': 'Grass', 'category': 'Physical', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'High critical hit ratio.'}, 'doubleedge': {'name': 'Double-Edge', 'type': 'Normal', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'originpulse': {'name': 'Origin Pulse', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'}, 'waterspout': {'name': 'Water Spout', 'type': 'Water', 'category': 'Special', 'base_power': 150, 'accuracy': 100, 'priority': 0, 'description': "Less power as user's HP decreases. Hits foe(s)."}, 'thunder': {'name': 'Thunder', 'type': 'Electric', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to paralyze. Can't miss in rain."}, 'psychoboost': {'name': 'Psycho Boost', 'type': 'Psychic', 'category': 'Special', 'base_power': 140, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."}, 'sleeptalk': {'name': 'Sleep Talk', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User must be asleep. Uses another known move.'}, 'rest': {'name': 'Rest', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User sleeps 2 turns and restores HP and status.'}, 'wavecrash': {'name': 'Wave Crash', 'type': 'Water', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'curse': {'name': 'Curse', 'type': 'Ghost', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Curses if Ghost, else -1 Spe, +1 Atk, +1 Def.'}, 'revivalblessing': {'name': 'Revival Blessing', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Revives a fainted Pokemon to 50% HP.'}, 'psychic': {'name': 'Psychic', 'type': 'Psychic', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'bugbuzz': {'name': 'Bug Buzz', 'type': 'Bug', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'facade': {'name': 'Facade', 'type': 'Normal', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if user is burn/poison/paralyzed.'}, 'trailblaze': {'name': 'Trailblaze', 'type': 'Grass', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': "100% chance to raise the user's Speed by 1."}, 'willowisp': {'name': 'Will-O-Wisp', 'type': 'Fire', 'category': 'Status', 'base_power': 0, 'accuracy': 85, 'priority': 0, 'description': 'Burns the target.'}, 'slackoff': {'name': 'Slack Off', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'}, 'bulkup': {'name': 'Bulk Up', 'type': 'Fighting', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Defense by 1."}, 'heatcrash': {'name': 'Heat Crash', 'type': 'Fire', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the user than the target.'}, 'overheat': {'name': 'Overheat', 'type': 'Fire', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."}, 'highhorsepower': {'name': 'High Horsepower', 'type': 'Ground', 'category': 'Physical', 'base_power': 95, 'accuracy': 95, 'priority': 0, 'description': 'No additional effect.'}, 'supercellslam': {'name': 'Supercell Slam', 'type': 'Electric', 'category': 'Physical', 'base_power': 100, 'accuracy': 95, 'priority': 0, 'description': 'User is hurt by 50% of its max HP if it misses.'}, 'suckerpunch': {'name': 'Sucker Punch', 'type': 'Dark', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first. Fails if target is not attacking.'}, 'ironhead': {'name': 'Iron Head', 'type': 'Steel', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to make the target flinch.'}, 'drainpunch': {'name': 'Drain Punch', 'type': 'Fighting', 'category': 'Physical', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'}, 'jetpunch': {'name': 'Jet Punch', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'}, 'strengthsap': {'name': 'Strength Sap', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User heals HP=target's Atk stat. Lowers Atk by 1."}, 'hypervoice': {'name': 'Hyper Voice', 'type': 'Normal', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'}, 'energyball': {'name': 'Energy Ball', 'type': 'Grass', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'spiritshackle': {'name': 'Spirit Shackle', 'type': 'Ghost', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Prevents the target from switching out.'}, 'defog': {'name': 'Defog', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': '-1 evasion; ends user and target hazards/terrain.'}, 'switcheroo': {'name': 'Switcheroo', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."}, 'shellsmash': {'name': 'Shell Smash', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Def, SpD by 1; raises Atk, SpA, Spe by 2.'}, 'tripleaxel': {'name': 'Triple Axel', 'type': 'Ice', 'category': 'Physical', 'base_power': 20, 'accuracy': 90, 'priority': 0, 'description': 'Hits 3 times. Each hit can miss, but power rises.'}, 'bulletseed': {'name': 'Bullet Seed', 'type': 'Grass', 'category': 'Physical', 'base_power': 25, 'accuracy': 100, 'priority': 0, 'description': 'Hits 2-5 times in one turn.'}, 'trick': {'name': 'Trick', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."}, 'zenheadbutt': {'name': 'Zen Headbutt', 'type': 'Psychic', 'category': 'Physical', 'base_power': 80, 'accuracy': 90, 'priority': 0, 'description': '20% chance to make the target flinch.'}, 'freezedry': {'name': 'Freeze-Dry', 'type': 'Ice', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze. Super effective on Water.'}}


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


class Board:
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
        if ability == 'technician' and (bp <= 60 or mid in ('bulletseed','rockblast','iciclespear','pinmissile','tailslap','armthrust','dualwingbeat','doublehit','bonemerang','doubleironbash','populationbomb','surgingstrikes','tripledive','tripleaxel')): bp *= 1.5
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


# The model proposes uncertain threats; only observations certify a move or ability.
_HEAL = {'recover','roost','slackoff','softboiled','milkdrink','shoreup','synthesis','moonlight','morningsun','strengthsap','rest'}
_SETUP = {'swordsdance':{'atk':2}, 'nastyplot':{'spa':2}, 'dragondance':{'atk':1,'spe':1},
          'quiverdance':{'spa':1,'spd':1,'spe':1}, 'calmmind':{'spa':1,'spd':1},
          'bulkup':{'atk':1,'def':1}, 'shellsmash':{'atk':2,'spa':2,'spe':2,'def':-1,'spd':-1},
          'coil':{'atk':1,'def':1,'accuracy':1},'agility':{'spe':2}}
_PIVOT = {'uturn','voltswitch','flipturn'}


def sethp(p, val):
    p['_hp'] = max(0., min(1., val))
    # Preserve sufficient precision and exact own denominator for damage math.
    maximum = p.get('_maxhp', hp(p)[1])
    p['_maxhp'] = maximum
    p['condition'] = ('0 fnt' if val <= 0 else str(max(1,round(maximum*min(1,val))))+'/'+str(round(maximum)))
    if p.get('status') and val > 0: p['condition'] += ' '+p['status']


def sh(p):
    return p.get('_hp', fraction(p))


class SearchBoard(Board):
    def __init__(self,obs,memory):
        super().__init__(obs,memory)
        for hypothesis in memory.get('hypotheses',{}).values():
            for m in hypothesis.get('moves',[]):
                key=ident(m.get('name'))
                if key not in self.moves:self.moves[key]=m

    def pub(self,p):
        if p.get('_sim'): return p
        return super().pub(p)

    def stat(self,p,s,ignore_boost=False):
        v=super().stat(p,s,ignore_boost)
        vol=ident(self.pub(p).get('volatiles',{}))
        if ('protosynthesis'+s in vol or 'quarkdrive'+s in vol):v*=1.5 if s=='spe' else 1.3
        if s=='spe' and p.get('_speed_factor'): v*=p['_speed_factor']
        if s in ('atk','spa') and p.get('_power_factor'): v*=p['_power_factor']
        return v

    def damage(self,atk,defend,mid,tera=False,defender_tera=False):
        d=super().damage(atk,defend,mid,tera,defender_tera)
        if isinstance(mid,str):
            md=ident(mid)
            # Explicit revealed Flash Fire is a public volatile, never a species assumption.
            if self.move(md).get('type')=='Fire' and 'flashfire' in ident(self.pub(atk).get('volatiles',{})): d*=1.5
            if md in ('solarbeam','solarblade') and 'sun' not in self.weather: d*=.50
            if md in ('meteorbeam','electroshot') and self.item(atk)!='powerherb' and not (md=='electroshot' and 'rain' in self.weather): d*=.5
        return d

    def clone_p(self,p):
        q=copy.deepcopy(p)
        pub=self.pub(p)
        for k in ('boosts','volatiles','status','ability','teraTypeRevealed'):
            if k in pub and k not in q:q[k]=copy.deepcopy(pub[k])
        q['boosts']=copy.deepcopy(pub.get('boosts') or {})
        q['volatiles']=copy.deepcopy(pub.get('volatiles') or {})
        q['_sim']=True
        q['_maxhp']=self.maxhp(p)
        q['condition']=str(round(fraction(p)*q['_maxhp']))+'/'+str(round(q['_maxhp']))
        q['_hp']=fraction(p)
        q['status']=status(p)
        q['ability']=self.ability(p)
        q['item']=self.item(p)
        return q

    def hypothesis_moves(self,p):
        known=[ident(m) for m in p.get('moves',[])]
        hyp=self.memory.get('hypotheses',{}).get(species(p),{})
        moves=list(known)
        for m in hyp.get('moves',[]):
            mid=ident(m.get('name'))
            if mid and mid not in moves and len(moves)<4:moves.append(mid)
        # Missing moves are uncertainty, not zero damage. Generic STAB is marked synthetic.
        if len(known)<4:
            for typ in self.base_types(p):
                if any(self.move(m).get('type')==typ and self.move(m).get('category') in ('Physical','Special') for m in moves):continue
                mid='hyp'+ident(typ)
                cat='Physical' if self.stat(p,'atk')>self.stat(p,'spa') else 'Special'
                self.moves[mid]={'name':mid,'type':typ,'category':cat,'base_power':85,'priority':0,'accuracy':100}
                moves.append(mid)
        return list(dict.fromkeys(moves))

    def legal_moves(self,p,root=False):
        if root:
            return [dict(kind='move',mid=ident(self.active['moves'][int(a.split()[1])-1].get('id') or self.active['moves'][int(a.split()[1])-1].get('move')),tera=a.endswith('terastallize'),action=a)
                    for a in self.legal if a.startswith('move ') and int(a.split()[1])<=len(self.active.get('moves',[]))]
        mids=p.get('moves',[])
        locked=p.get('_locked')
        if locked and self.item(p) in ('choiceband','choicespecs','choicescarf'):mids=[locked]
        ans=[]
        for mid in mids:
            mid=ident(mid)
            if mid in ('fakeout','firstimpression') and p.get('_age',1)>0: continue
            if mid==p.get('_last') and mid=='gigatonhammer':continue
            ans.append(dict(kind='move',mid=mid,tera=False))
        return ans

    def replies(self,p,q,allow_switch=False):
        candidates=[]
        known={ident(x) for x in q.get('moves',[])}
        hyp=self.memory.get('hypotheses',{}).get(species(q),{})
        for mid in self.hypothesis_moves(q):
            m=self.move(mid)
            if not m:continue
            if q.get('_locked') and self.item(q) in ('choicescarf','choicespecs','choiceband') and mid!=q['_locked']:continue
            d=self.damage(q,p,mid)
            if d>0:
                # Opponent chooses against the current active, independent of our hidden choice.
                v=min(sh(p),d)*90+12*(d>=sh(p))
                if self.priority(q,mid)>0:v+=4
                if mid.startswith('hyp'):v-=7
                if mid not in known:v-=3
            elif mid in _HEAL and sh(q)<.58:v=30*(1-sh(q))
            elif mid in _SETUP and sh(q)>.65 and max((self.damage(p,q,x) for x in p.get('moves',[])),default=0)<.35:v=16
            else:continue
            candidates.append((v,dict(kind='move',mid=mid,tera=False)))
        if not candidates:candidates=[(0,dict(kind='move',mid='hypnormal',tera=False))]
        candidates.sort(key=lambda x:x[0],reverse=True)
        selected=candidates[:3]
        top=selected[0][0]
        # Adapt the portfolio to this opponent's public, active-species policy.
        # The seeded heuristic often repeats a revealed attack while its matchup
        # remains unchanged; empirical counts are only a small prior, never a fact
        # about hidden moves.
        foe_key = str(q.get('ident',''))
        counts = self.memory.get('foe_actions',{}).get(foe_key) or self.memory.get('foe_actions',{}).get(foe_key.split(':',1)[0],{})
        weighted=[]
        for v,a in selected:
            mid=ident(a.get('mid',''))
            repeat=float(counts.get(mid,0))
            # At most a 35% relative boost; unknown/hypothetical moves retain
            # their conservative penalty and cannot dominate solely by repetition.
            boost=min(0.35, 0.08*repeat)
            weighted.append((math.exp(max(-8,(v-top)/13))*(1.0+boost),a))
        if allow_switch:
            # Only revealed, living bench targets. Never instantiate an unknown team member.
            own_hit=max((self.damage(p,q,m) for m in p.get('moves',[])),default=0)
            switches=[]
            for target in self.obs.get('opponent_revealed',[]):
                if target.get('ident')==q.get('ident') or fraction(target)<=0:continue
                r=self.clone_p(target)
                hit=max((self.damage(p,r,m) for m in p.get('moves',[])),default=.3)
                retaliate=max((self.damage(r,p,m) for m in self.hypothesis_moves(r)),default=.2)
                gain=own_hit-hit+.3*retaliate
                switches.append((gain,r))
            if switches:
                gain,r=max(switches,key=lambda x:x[0])
                if gain>.18:
                    chance=min(.62,max(.15,gain*.65))
                    total=sum(w for w,a in weighted)
                    weighted=[(w*(1-chance)/total,a) for w,a in weighted]
                    weighted.append((chance,dict(kind='switch',p=r)))
        total=sum(w for w,a in weighted)
        return [(w/total,a) for w,a in weighted]

    def order_first(self,p,q,a,b):
        pa=self.priority(p,a.get('mid',''));pb=self.priority(q,b.get('mid',''))
        if pa!=pb:return pa>pb
        ownsp=self.stat(p,'spe');foesp=self.stat(q,'spe')
        # Speed evidence from real equal-priority moves overrides the base-stat estimate.
        entry=self.memory.get('speed',{}).get((p.get('ident'),q.get('ident')))
        key=(self.boost(p,'spe'),self.boost(q,'spe'),status(p),status(q),self.weather,self.field)
        if entry and entry[0]==key:return entry[1]
        return ownsp<foesp if 'trickroom' in self.field else ownsp>foesp

    def weight(self,p):
        return self.memory.get('weights',{}).get(species(p),1.)

    def change_boosts(self,p,changes):
        for stat,delta in changes.items():
            if self.ability(p)=='contrary':delta=-delta
            if self.ability(p)=='simple':delta*=2
            p['boosts'][stat]=max(-6,min(6,p['boosts'].get(stat,0)+delta))
        if self.item(p)=='whiteherb' and any(n<0 for n in p['boosts'].values()):
            p['boosts']={s:max(0,n) for s,n in p['boosts'].items()}
            p['item']=''

    def transition(self,p,q,a,b,hit=True):
        p=copy.deepcopy(p);q=copy.deepcopy(q)
        reward=0.;origp=sh(p);origq=sh(q)
        pswitch=a['kind']=='switch';qswitch=b['kind']=='switch'
        if pswitch:
            p=copy.deepcopy(a['p']);origp=sh(p)
            p['boosts']={};p['volatiles']={};p['_age']=0
            sethp(p,sh(p)-self.hazard(p))
            if self.ability(p)=='intimidate':q['boosts']['atk']=max(-6,q['boosts'].get('atk',0)-1)
            reward-=.035 # Opportunity cost is primarily the lost action, not a large hand penalty.
        if qswitch:
            q=copy.deepcopy(b['p']);origq=sh(q);q['boosts']={};q['volatiles']={}
            if self.ability(q)=='intimidate':p['boosts']['atk']=max(-6,p['boosts'].get('atk',0)-1)
        if a.get('tera'):
            p['terastallized']=p.get('teraType');reward-=.12
        if pswitch and qswitch:order=[]
        elif pswitch:order=[False]
        elif qswitch:order=[True]
        else:order=[True,False] if self.order_first(p,q,a,b) else [False,True]
        for own in order:
            u,v,act=(p,q,a) if own else (q,p,b)
            if sh(u)<=0 or sh(v)<=0:continue
            if own and not hit:continue
            mid=act.get('mid','');m=self.move(mid)
            if v.get('_protect') and m.get('target','normal') not in ('self','allySide'):
                if self.damage(u,v,mid)>0 or mid in ('toxic','thunderwave','willowisp','leechseed','spore','sleeppowder','hypnosis'):continue
            if status(u)=='slp' and mid not in ('sleeptalk','snore'):continue
            if mid=='sleeptalk':
                attacks=[x for x in u.get('moves',[]) if self.damage(u,v,x)>0]
                if status(u)=='slp' and attacks:mid=max(attacks,key=lambda x:self.damage(u,v,x));m=self.move(mid)
                else:continue
            if mid in ('fakeout','firstimpression') and u.get('_age',1)>0:continue
            if mid=='suckerpunch' and self.move((b if own else a).get('mid','')).get('category')=='Status':continue
            d=self.damage(u,v,mid)
            if d>0:
                # Root own misses are explicit branches; later/foe misses use expected damage.
                if not own:
                    acc=m.get('accuracy',100);d*=1 if acc is True else number(acc,100)/100
                if status(u)=='par':d*=.75
                d*=1.0 if own else 1.04
                if self.priority(u,mid)>0 and 'psychicterrain' in ident(self.public.get('field',{})) and 'Flying' not in self.types(v) and self.ability(v)!='levitate':d=0
                before=sh(v)
                if before>=.999 and (self.item(v)=='focussash' or self.ability(v)=='sturdy') and d>=before and mid not in ('dualwingbeat','bulletseed','rockblast','iciclespear','populationbomb','surgingstrikes'):
                    d=max(0,before-1/self.maxhp(v))
                sethp(v,before-d)
                dealt=min(before,d)
                if mid in ('gigadrain','drainpunch','drainingkiss','oblivionwing','leechlife','hornleech'):
                    sethp(u,sh(u)+dealt*self.maxhp(v)/self.maxhp(u)*(.75 if mid in ('drainingkiss','oblivionwing') else .5))
                if mid in ('flareblitz','bravebird','woodhammer','wavecrash','doubleedge','headsmash'):
                    sethp(u,sh(u)-dealt*self.maxhp(v)/self.maxhp(u)*(.5 if mid=='headsmash' else 1/3))
                if self.item(u)=='lifeorb' and self.ability(u) not in ('magicguard','sheerforce'):sethp(u,sh(u)-.1)
                if mid in ('closecombat','headlongrush'):
                    self.change_boosts(u,{'def':-1,'spd':-1})
                if mid in ('overheat','dracometeor','leafstorm','fleurcannon','psychoboost'):self.change_boosts(u,{'spa':-2})
                if mid=='superpower':
                    self.change_boosts(u,{'atk':-1,'def':-1})
                if mid in ('rapidspin','flamecharge','trailblaze'):self.change_boosts(u,{'spe':1})
                if mid in ('nuzzle','thunderwave') and not status(v) and not set(self.types(v))&{'Electric','Ground'}:v['status']='par'
                if mid in _PIVOT:reward+=.045 if own else -.045
                if mid in ('hyperbeam','gigaimpact','rockwrecker','blastburn','hydrocannon','frenzyplant'):u['_recharge']=True
            elif mid in _HEAL:
                if mid=='rest':u['status']='slp';sethp(u,1.)
                else:sethp(u,sh(u)+.5)
            elif mid in _SETUP:
                self.change_boosts(u,_SETUP[mid])
            elif mid in ('thunderwave','toxic','willowisp','spore','sleeppowder','hypnosis','yawn') and not status(v):
                types=self.types(v);newstatus=''
                if mid=='thunderwave' and not set(types)&{'Ground','Electric'}:newstatus='par'
                if mid=='toxic' and not set(types)&{'Poison','Steel'}:newstatus='tox'
                if mid=='willowisp' and 'Fire' not in types:newstatus='brn'
                if mid in ('spore','sleeppowder','hypnosis') and 'Grass' not in types:
                    sleeping=any(status(x)=='slp' and fraction(x)>0 for x in self.obs.get('opponent_revealed',[]))
                    if not own or not sleeping:newstatus='slp'
                if newstatus:
                    v['status']=newstatus;reward+=(.13 if own else -.13)
            elif mid in ('stealthrock','spikes','toxicspikes','stickyweb'):
                side='p2' if own else 'p1'
                if mid not in ident((self.public.get('side_conditions') or {}).get(side,{})):
                    reward+=(.18 if own else -.18)
            elif mid in ('reflect','lightscreen','auroraveil'):reward+=(.12 if own else -.12)
            elif mid in ('protect','banefulbunker','spikyshield'):
                # Protect is represented by skipping the attack below on second action.
                u['_protect']=True
            elif mid in ('leechseed',) and 'Grass' not in self.types(v):v['volatiles']['leechseed']=True;reward+=(.10 if own else -.10)
            u['_locked']=mid;u['_last']=mid
        for u in (p,q):
            u.pop('_protect',None)
            if sh(u)>0:
                residual=self.residual(u)
                if status(u)=='tox':residual=max(residual,.125)
                sethp(u,sh(u)-residual)
                if self.item(u) in ('leftovers','blacksludge') and residual==0:sethp(u,sh(u)+.0625)
                u['_age']=u.get('_age',0)+1
        reward+=(origq-sh(q))-(origp-sh(p))*self.weight(p)
        if sh(q)<=0 and origq>0:reward+=1.35
        if sh(p)<=0 and origp>0:reward-=1.35*self.weight(p)
        return p,q,reward

    def value_second(self,p,q):
        if sh(p)<=0 or sh(q)<=0:return 0.
        actions=self.legal_moves(p)
        if p.get('_recharge'):return -min(sh(p),max((self.damage(q,p,m) for m in self.hypothesis_moves(q)),default=.2))
        if not actions:return 0.
        replies=self.replies(p,q,False)[:2]
        total=sum(w for w,a in replies)
        best=-9.
        for a in actions:
            vals=[]
            for w,b in replies:
                _,_,v=self.transition(p,q,a,b)
                acc=self.move(a.get('mid','')).get('accuracy',100)
                accuracy=1 if acc is True else number(acc,100)/100
                if accuracy<1:
                    _,_,miss=self.transition(p,q,a,b,False);v=accuracy*v+(1-accuracy)*miss
                vals.append((w/total,v))
            val=.85*sum(w*v for w,v in vals)+.15*min(v for w,v in vals)
            best=max(best,val)
        return best

    def search(self,recent):
        p=self.clone_p(self.own);q=self.clone_p(self.foe)
        p['_age']=self.memory.get('active_age',0)
        for hypothesis in self.memory.get('hypotheses',{}).values():
            for m in hypothesis.get('moves',[]):
                key=ident(m.get('name'))
                if key not in self.moves:self.moves[key]=m
        roots=self.legal_moves(p,True)
        for a in self.legal:
            if a.startswith('switch '):
                i=int(a.split()[1])-1
                if i<len(self.team):roots.append(dict(kind='switch',p=self.clone_p(self.team[i]),action=a))
        replies=self.replies(p,q,True)
        values=[]
        for a in roots:
            if self.forced:
                r=a.get('p',p)
                if sh(r)<=0:
                    val=.7+self.stat(r,'spe')/500 # legal fainted target means Revival Blessing
                else:
                    r['boosts']={};r['_age']=0
                    cost=self.hazard(r);sethp(r,sh(r)-cost)
                    val=self.value_second(r,q)-cost*self.weight(r)
                    if sh(r)<=0:val-=2
                values.append({'action':a['action'],'value':round(val,4)});continue
            vals=[]
            for w,b in replies:
                pp,qq,v=self.transition(p,q,a,b)
                v+=.86*self.value_second(pp,qq)
                acc=self.move(a.get('mid','')).get('accuracy',100)
                accuracy=1 if acc is True else max(.01,min(1,number(acc,100)/100))
                if a['kind']=='move' and accuracy<1:
                    pp,qq,miss=self.transition(p,q,a,b,False)
                    miss+=.86*self.value_second(pp,qq)
                    v=accuracy*v+(1-accuracy)*miss
                vals.append((w,v))
            val=.82*sum(w*v for w,v in vals)+.18*min(v for w,v in vals)
            if a['kind']=='switch' and recent and recent[-1].startswith('switch '):val-=.13
            if a['kind']=='move' and a.get('mid') in _SETUP:
                # Modest terminal value for useful setup beyond a two-turn horizon.
                if self.boost(p,'atk')<2 and self.boost(p,'spa')<2:val+=.06
            if len(recent)>=2 and recent[-2:]==[a['action']]*2 and self.move(a.get('mid','')).get('category')=='Status':val-=.24
            values.append({'action':a['action'],'value':round(val,4)})
        return sorted(values,key=lambda x:x['value'],reverse=True)


class Harness:
    def __init__(self,ctx):
        self.ctx=ctx
        self.memory={'hypotheses':{},'weights':{},'reveals':{},'speed':{},'active_age':0,'foe_actions':{},'foe_switches':{}}
        self.calls=0;self.steps=0;self.pending=False;self.recent=deque(maxlen=10)
        self.asked=set();self.rejected=set();self.note=''

    def log(self,event,**kw):
        try:self.ctx.log(event,**kw)
        except Exception:pass

    def ask(self,prompt,remaining):
        timeout=min(140.,remaining-70)
        future=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=None,timeout_s=timeout))
        self.pending=True
        while not future.done():
            if self.ctx.done:return None
            if self.ctx.budget.remaining().get('wall_s',0)<120:return None
            time.sleep(.05)
        self.pending=False
        return future.result()

    def prompt(self,b):
        state={'team':b.team,'active_request':b.active,'opponent':b.foe,
               'revealed_roster':b.obs.get('opponent_revealed',[]),'species_reference':b.dex,
               'field':b.public,'recent_protocol':b.obs.get('recent_events',[])[-55:],
               'own_move_reference':b.obs.get('move_dex',{}),'previous_note':self.note,'observed_opponent_policy':self.memory.get('foe_actions',{}).get(str(b.foe.get('ident','')), {})}
        return '''You advise a Gen 9 Random Battle search controller. Do NOT choose an action. Code compares two-turn exchanges, switch damage, priority, speed and Tera. Supply up to FOUR plausible attacking/recovery/setup moves for the active opponent, prioritizing revealed moves and dangerous likely coverage. Unknown moves are HYPOTHESES, not facts. Use actual move names and base power (variable-power moves may use 0); type, category, priority, and accuracy. Do not invent species-specific sets or claim a hidden item is known. The code already has the own exact stats/moves/items. Identify at most two valuable teammates that must remain useful against revealed threats, but do not protect a teammate at the cost of losing the battle. Note critical mechanics briefly. Opponent is a modest heuristic and often switches to a revealed resistance. Keep output under 220 words and return only JSON:
{"moves":[{"name":"Earthquake","type":"Ground","category":"Physical","base_power":100,"accuracy":100,"priority":0}],"preserve":["own species"],"expendable":["own species"],"note":"one brief tactical warning"}
STATE:\n'''+json.dumps(state,separators=(',',':'),default=str)

    def parse(self,text,b):
        if not isinstance(text,str):return False
        try:
            start=text.index('{');obj,_=json.JSONDecoder().raw_decode(text[start:])
        except (ValueError,TypeError):return False
        if not isinstance(obj,dict):return False
        moves=[]
        raw=obj.get('moves',[])
        if not isinstance(raw,list):raw=[]
        for m in raw[:4]:
            if not isinstance(m,dict):continue
            name=str(m.get('name',''))[:40]
            if not ident(name) or not isinstance(m.get('type'),str) or m.get('type') not in _CHART or m.get('category') not in ('Physical','Special','Status'):continue
            moves.append({'name':name,'type':m['type'],'category':m['category'],
                          'base_power':max(0,min(200,number(m.get('base_power')))),
                          'accuracy':max(25,min(100,100 if m.get('accuracy') is True else number(m.get('accuracy'),100))),
                          'priority':max(-7,min(5,number(m.get('priority'))))})
        self.memory['hypotheses'][species(b.foe)]={'moves':moves}
        for p in b.team:
            name=species(p)
            preserve=obj.get('preserve',[]);expend=obj.get('expendable',[])
            self.memory['weights'][name]=1.16 if isinstance(preserve,list) and name in preserve else .90 if isinstance(expend,list) and name in expend else 1.
        if isinstance(obj.get('note'),str):self.note=obj['note'][:450]
        self.log('hypothesis',foe=species(b.foe),moves=moves,note=self.note)
        return True

    def learn(self,old,new):
        a=SearchBoard(old,self.memory);b=SearchBoard(new,self.memory)
        self.memory['active_age']=self.memory.get('active_age',0)+1 if a.own.get('ident')==b.own.get('ident') else 0
        attacks=[]
        events=new.get('recent_events') or []
        if isinstance(events,str):events=events.splitlines()
        for line in events:
            x=str(line).split('|')
            if len(x)<4:continue
            if x[1]=='move':attacks.append((x[2][:2],ident(x[3])))
            who=re.sub(r'^(p[12])[a-z]:',r'\1:',x[2])
            if x[2].startswith('p2'):
                fact=self.memory['reveals'].setdefault(who,{})
                if x[1]=='-ability':fact['ability']=x[3]
                if x[1] in ('-item','-enditem'):fact['item']=x[3] if x[1]=='-item' else ''
                for attr in ('item','ability'):
                    m=re.search(r'\[from\] '+attr+r': ([^|]+)',line)
                    if m and '[of]' not in line:fact[attr]=m[1]
            if x[1]=='move' and str(x[2]).startswith('p2') and len(x)>3:
                # Protocol move lines expose the opponent's actual public choice.
                key=who
                mid=ident(x[3])
                bucket=self.memory.setdefault('foe_actions',{}).setdefault(key,{})
                bucket[mid]=int(bucket.get(mid,0))+1
            if x[1]=='switch' and str(x[2]).startswith('p2') and len(x)>3:
                key=str(x[2]).split(':',1)[0]
                bucket=self.memory.setdefault('foe_switches',{}).setdefault(key,0)
                self.memory['foe_switches'][key]=int(bucket)+1
        if a.own.get('ident')==b.own.get('ident') and a.foe.get('ident')==b.foe.get('ident') and len(attacks)==2 and {x[0] for x in attacks}=={'p1','p2'}:
            ownmid=next(x[1] for x in attacks if x[0]=='p1');foemid=next(x[1] for x in attacks if x[0]=='p2')
            if a.move(ownmid) and a.move(foemid) and a.priority(a.own,ownmid)==a.priority(a.foe,foemid):
                key=(a.boost(a.own,'spe'),a.boost(a.foe,'spe'),status(a.own),status(a.foe),a.weather,a.field)
                self.memory['speed'][(a.own.get('ident'),a.foe.get('ident'))]=(key,attacks[0][0]=='p1')

    def run_episode(self,game):
        ctx=self.ctx
        try:obs=ctx.observe()
        except Exception:return
        while not ctx.done and not getattr(game,'done',False) and self.steps<870:
            if not isinstance(obs,dict):return
            b=SearchBoard(obs,self.memory)
            if not b.legal:return
            try:rem=ctx.budget.remaining()
            except Exception:rem={}
            missing=tuple(sorted(ident(m) for m in b.foe.get('moves',[]) if not b.move(m)))
            fingerprint=(species(b.foe),str(b.types(b.foe)),missing)
            # One scouting call per opposing species/type; room for two later corrections.
            if fingerprint not in self.asked and self.calls<9 and not self.pending and rem.get('system2_calls',0)>1 and rem.get('wall_s',0)>210:
                self.asked.add(fingerprint);self.calls+=1
                try:
                    result=self.ask(self.prompt(b),rem['wall_s'])
                    if self.parse(getattr(result,'text',''),b):
                        self.asked.add((species(b.foe),str(b.types(b.foe)),()))
                except Exception as exc:self.log('model_failure',error=type(exc).__name__)
            if ctx.done:return
            try:ranking=b.search(list(self.recent))
            except Exception as exc:
                self.log('search_failure',error=type(exc).__name__,detail=str(exc)[:200])
                try:ranking=b.rank(list(self.recent))
                except Exception:ranking=[{'action':a,'value':0} for a in b.legal]
            ranking=[r for r in ranking if r['action'] in b.legal and r['action'] not in self.rejected]
            if not ranking:
                remaining=[a for a in b.legal if a not in self.rejected]
                if not remaining:return
                ranking=[{'action':a,'value':0} for a in remaining]
            action=ranking[0]['action']
            self.log('choice',action=action,turn=obs.get('turn'),calls=self.calls,ranking=ranking[:8])
            old=obs
            try:
                nxt=ctx.act(action);obs=nxt if isinstance(nxt,dict) else ctx.observe()
            except Exception as exc:
                if ctx.done:return
                self.rejected.add(action);self.log('act_rejected',action=action,error=type(exc).__name__)
                try:obs=ctx.observe()
                except Exception:return
                continue
            self.steps+=1
            if not isinstance(obs,dict):return
            if obs.get('last_error'):
                self.rejected.add(action);continue
            self.rejected.clear();self.recent.append(action)
            try:self.learn(old,obs)
            except Exception as exc:self.log('learn_failure',error=type(exc).__name__)
        self.log('complete',steps=self.steps,calls=self.calls)


def build(ctx):
    return Harness(ctx)

# NetHack (NetHack Learning Environment)

## What the game is

NetHack is a single-player, turn-based dungeon crawler. The canonical goal is to descend to
the bottom of the dungeon, obtain the Amulet of Yendor, and return to the surface to ascend
and offer it (the “ascension”). A run ends in death, quitting, or successful ascension. The
official guide describes the objective and game commands in detail: [NetHack Guide, Playing the
Game](https://nethack.org/v366/Guidebook.html#_playing_the_game), [The Dungeons of Doom](https://nethack.org/v366/Guidebook.html#_the_dungeons_of_doom).

NLE is Facebook Research's open-source wrapper around the NetHack 3.6.6 engine. It exposes
NetHack as a Gym environment for reinforcement-learning agents, with deterministic seeding
available through Gym's reset API. Project documentation and source are at the [NLE
repository](https://github.com/facebookresearch/nle) and [NLE API reference](https://github.com/facebookresearch/nle/tree/main/nle).

## State and progress

The player's persistent objectives are survival, useful equipment and experience, exploration,
and eventually ascension. Practical progress signals are dungeon depth (the `dungeon depth`
field in `blstats`), player level/HP, inventory and identified objects, and game score. The
engine's score is a cumulative integer shown by `#score`; it rewards gold, experience, dungeon
progress and major achievements (see the [Guidebook scoring section](https://nethack.org/v366/Guidebook.html#_scoring)), and is the natural scalar to report. For learning benchmarks,
NLE also supplies task-specific environments (`NetHackScore`, `NetHackStaircase`, `NetHackGold`,
etc.) whose rewards are documented in [NLE environments](https://github.com/facebookresearch/nle#environments).

Death gives a final score and ends the episode. Reaching a staircase objective or the terminal
ascension gives a useful sparse success signal; report both final score and maximum depth so a
good surviving run is distinguishable from an early lucky score.

## Controls and actions

NetHack is keyboard-command driven. Movement uses `h j k l y u b n` (vi directions), `.` waits
one turn, `s` searches, `o` opens, `c` closes, `e` eats, `q` quaffs, `r` reads, `w` wields,
`W` wears armour, `P` puts on rings/amulets, `d` drops, `,` picks up, `i` inventories, `x`
examines, `>` descends stairs, `<` ascends stairs, `:` looks at a square, `Ctrl-P` recalls
messages, and `#` selects extended commands. Exact key semantics are listed in the
[NetHack command reference](https://nethack.org/v366/Guidebook.html#_commands).

NLE converts these keys to integer actions in `nle.nethack.ACTIONS`; the adaptor should send one
integer per environment step. Some commands open a yes/no, menu, or text prompt. The next
observation contains the prompt/message, and the adaptor must send the corresponding key(s) as
additional turns.

## Pace and tier

This is tier 1. NetHack advances only when an input command is accepted; monsters take their
turns during that same step. There is no wall-clock deadline between steps, so Astra can spend
seconds or a minute planning while the emulator is paused. A decision is strategically urgent
only in the game-time sense (for example, a monster adjacent to the player); waiting in the
harness does not consume turns. Set `decision_latency_s` to about 60 seconds as an engineering
budget, with zero real-time penalty, and keep action latency low once a command is chosen.

## Sources

* [NetHack Guidebook (3.6.6)](https://nethack.org/v366/Guidebook.html)
* [NetHack 3.6.6 source and documentation](https://github.com/NetHack/NetHack)
* [NLE README and environments](https://github.com/facebookresearch/nle)
* [NLE Python API source](https://github.com/facebookresearch/nle/tree/main/nle)

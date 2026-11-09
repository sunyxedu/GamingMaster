# Flappy Bird study brief

## Scope and identity

This brief treats **Flappy Bird** as the original one-button arcade game and, where an executable is needed on the offline server, its faithful open-source Gymnasium implementation. The original game was made by Dong Nguyen/.Gears; it is not an online service and has no supported public game API. The Gymnasium implementation is therefore the recommended experimental target, not an assertion that its constants are the proprietary original's exact constants.

## Objective and rules

- The bird continuously moves through a fixed-width playfield while gravity makes it descend.
- A flap/tap gives the bird an upward velocity impulse; no flap is the other action.
- Green pipe pairs scroll from right to left. The bird must pass through each pair's gap without touching either pipe or the ground. In the reference implementation, going above y=0 is penalized (`-0.5`) but is not itself a terminal collision; pipe and ground collisions end the episode.
- A pipe pair passed awards one point. Collision ends the run. The conventional medal thresholds are 10/20/30/40 for bronze/silver/gold/platinum, but the adaptor should report the integer score rather than a medal.
- There is no inventory, opponent, account, or persistent progression in the core game. The meaningful objective is maximize score / number of consecutive pipe pairs survived.

The original control and scoring description is corroborated by [How to Play Flappy Bird](https://flappybird.io/how-to-play) and the historical overview in [Wikipedia](https://en.wikipedia.org/wiki/Flappy_Bird). The implementation-level API and reward description are in the [Gymnasium README](https://github.com/markub3327/flappy-bird-gymnasium#readme).

## Controls and actions

Canonical action alphabet:

| Action | Meaning |
|---|---|
| `0` | do nothing for one simulation frame |
| `1` | flap once for one simulation frame |

Human versions commonly bind tap, mouse click, Space, or Arrow Up. The [FlapPyBird README](https://github.com/sourabhv/FlapPyBird#readme) documents Up/Space and Escape. In the chosen environment, use `env.step(0)` or `env.step(1)`; do not synthesize keyboard events.

## State and scoring

The best observation for an agent is the 12-feature numerical state (`use_lidar=False`): three ordered pipe records, each with pipe x, upper-gap edge y, and lower-gap edge y, followed by bird y, vertical velocity, and rotation. With normalization enabled, x/y are divided by screen dimensions, velocity by maximum vertical velocity, and rotation by 90. The alternative is 180 normalized LIDAR distances. The source code defines these fields in [`_get_observation_features`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py#L470-L526).

For the reference implementation, the default screen is 288x512, pipe x velocity is -4 pixels per simulation frame, gravity is +1, flap velocity is -9, maximum downward velocity is +10, bird rectangle is 34x24, pipe width is 52, and ground y is 79% of screen height. These are clone constants, not claimed original-game measurements; see [`constants.py`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/constants.py). The default pipe gap is 100 pixels in the current Gymnasium constructor, and the eight possible gap offsets are selected by the seeded NumPy RNG; see [`_get_random_pipe`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py#L388-L403).

The environment reward is +0.1 for surviving a frame, +1.0 for passing a pipe, -0.5 for touching the top, and -1.0 for death. For evaluation, use `info["score"]` / pipe count as the primary score: survival reward is shaped training feedback, while pipe count directly measures the task. The environment returns Gymnasium's `(observation, reward, terminated, truncated, info)` from `step`, and `(observation, info)` from `reset`; see the [Gymnasium Env API](https://gymnasium.farama.org/api/env/).

## Pace and tier

This is **tier 2**. The world advances every call to `step`, and the reference implementation is configured for 30 rendered frames per second (`metadata["render_fps"] = 30`), although a non-rendered harness can advance faster than wall clock. Pipes move 4 px/frame, so a pipe can traverse the 288 px screen in roughly 72 frames (2.4 s at 30 fps). A flap impulse of -9 followed by +1 gravity gives a roughly 19-frame arc before the bird returns to comparable vertical velocity; decisions should therefore be made at frame or short-burst resolution, not after Astra spends seconds planning. In the live game, a 0.5 s delay consumes about 15 frames and can remove most of the safe correction window; use Jev for immediate flap/no-flap decisions and Astra only for strategy calibration or recovery planning.

A practical initial control deadline is **0.10–0.20 s** (3–6 frames at 30 fps) when the bird is near a pipe, with at most ~0.33 s (10 frames) in a wide-open region. `decision_latency_s` in `study.json` is a conservative 0.15. This is an engineering estimate derived from the documented 30 FPS metadata and constants, not a claim about the original game's undocumented wall-clock scheduler.

## Reset and reproducibility

Call `obs, info = env.reset(seed=seed)`. The current implementation invokes `super().reset(seed=seed)`, uses its `np_random` to choose each pipe's gap offset, and creates three pipes at x = 288, 432, and 576 before play. Use held-out integer seeds to test generalization; record seed, final score, termination cause, and frame count. `reset()` without a seed continues the RNG stream as specified by Gymnasium.

## Sources

- [Flappy Bird Gymnasium README](https://github.com/markub3327/flappy-bird-gymnasium#readme)
- [Flappy Bird Gymnasium environment source](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py)
- [Flappy Bird Gymnasium constants](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/constants.py)
- [Gymnasium API](https://gymnasium.farama.org/api/env/)
- [FlapPyBird clone](https://github.com/sourabhv/FlapPyBird)
- [How to Play Flappy Bird](https://flappybird.io/how-to-play)
- [Historical overview](https://en.wikipedia.org/wiki/Flappy_Bird)

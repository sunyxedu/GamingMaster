# MiniGrid / BabyAI adaptor

This is a Tier 1 adaptor for the official `minigrid==3.0.0` Gymnasium package. Importing
`minigrid` registers the local MiniGrid and BabyAI environments; no server, account, GUI,
network connection, or credentials are needed. The adaptor constructs a fresh environment on
each `reset(seed)` and closes it at episode end.

When the evaluator passes `env_id`, `environment_id`, or `id`, that ID is used directly. If it
does not, the external seed is mapped with SHA-256 domains `minigrid-adaptor-v1:scenario:<seed>`
and `minigrid-adaptor-v1:reset:<seed>`. The first digest selects uniformly from this ordered,
audited allowlist: `MiniGrid-Empty-Random-6x6-v0`, `MiniGrid-DoorKey-6x6-v0`,
`MiniGrid-MultiRoom-N2-S4-v0`, `BabyAI-GoToRedBallGrey-v0`, `BabyAI-GoToObj-v0`, and
`BabyAI-PickupLoc-v0`. The second digest is the Gymnasium reset seed. This avoids Python's
process-randomized `hash()` and registry order. Reset and finish records are appended under
`runtime/episodes.jsonl` with the external seed, actual reset seed, ID, kwargs, and package
versions.

The model receives the mission, facing direction, carried object when visible, and a decoded
egocentric symbolic grid from the public Gymnasium observation. `unknown` cells remain unknown;
the hidden world and verifier state are never exposed. Grid cells are reported in the image's
`[x,y,channel]` convention, with relative x increasing right and y increasing toward the rear.
The action vocabulary is `left`, `right`, `forward`, `pickup`, `drop`, `toggle`, and `done`.
Malformed adaptor actions are ignored and reported in the next observation; valid but blocked
actions are passed through and consume a normal environment step.

For audited sparse tasks, success is a positive terminal reward or explicit `info["success"]`.
The primary score is zero on failure and `0.1 + 0.9 * remaining_steps / max_steps` on success.
`info()` and `score_breakdown()` retain success, steps, budget, remaining fraction, raw reward and
return, terminal flags, ID, and both seeds. This preserves headroom for faster successful play.

Admission checks covered each selected family: Empty-Random (goal reach), DoorKey (key, door,
goal), MultiRoom (room traversal), BabyAI GoToRedBallGrey, BabyAI GoToObj, and BabyAI PickupLoc.
All six use the installed sparse verifier/reward convention: failure and timeout return zero,
while the successful verifier transition returns a positive reward; no shaping or custom wrapper
is applied. The train, held-out, and validation seed groups use the same family mix (two Empty,
two DoorKey, and one of each other family).

The adaptor has no intrinsic wall-clock deadline: Gymnasium changes state only when `step()` is
called. Rendering is deliberately disabled. Package versions are pinned in `requirements.txt`;
install it in the evaluator interpreter before validation.

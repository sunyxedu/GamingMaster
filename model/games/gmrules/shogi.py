"""Shogi — the gmrules game ``shogi``: the SIMULATOR of the existing shogi adapter (the
IL side is unchanged):

    rules + frames    data_processing/rs/gmrules/src/games/shogi/ — the converter's own
                      engine (python-shogi semantics, incl. its pawn-drop-mate
                      approximation) and its frozen frames (converter_version 2, schema
                      "shogi.v2"): obs board_seq [28, 9, 9] / hand_seq [14, 4] /
                      globals_seq [7] (move number, mover, check, repetition counts),
                      legal mode [2] / from [81] / to_cond [81, 81] / promo_cond
                      [81, 81, 2] / drop_piece [14] / drop_to_cond [14, 81], action keys
                      dropped / from_idx / to_idx / promoted / piece_idx / drop_to_idx
                      (mover-POV squares; the other branch's keys are ignored)
    data              `gmdata shogi fetch|convert` (floodgate CSA): the converter builds
                      its frames through the same gmrules code, so an IL frame is the RL
                      frame of the same position (shard meta {"global_features": 7})
    simulator         gmsim.Sim("shogi") → env/vec_gmsim.py (interleaved: one row per
                      game, every frame from the mover's view, lockstep)
    model             the shogi adapter IL trains: model/games/registry.py
                      ``_build_shogi`` (ShogiEncoder + the move-or-drop ShogiDecoder; the
                      registry keeps it, the entry below names it) — train_il_game's
                      LABEL_KEYS / META_ARCH_KEYS / META_LEGACY keep their own shogi
                      entries, which the tables below repeat
    RL                train_rl_joint's ShogiSpec (mode / from / to / piece teacher-forced
                      in the replay, the promotion and drop-square leaves under the masks
                      the rollout sampled with)

Game ends (the env's defaults; ``--env_kw shogi:key=value`` sets the gmrules config):
sennichite (the 4th occurrence of a position: a draw, perpetual check loses), no legal
move (a loss), ``max_plies`` (256, floodgate's Max_Moves: an adjudicated draw) and the
CSA rule-27 entering-king declaration (``entering_king = "csa27"``, automatic).
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="shogi",
    sim_config={},
    description="Shogi (floodgate / CSA rules, the converter's engine) on its frozen v2 frames",
    builder="model.games.registry:_build_shogi",
    label_keys=("dropped", "from_idx", "to_idx", "piece_idx"),
    meta_arch={"global_features": ("shogi_global_features",)},
    sample_keys={"action_dropped": "dropped", "action_from": "from_idx",
                 "action_to": "to_idx", "action_promoted": "promoted",
                 "action_piece": "piece_idx", "action_drop_to": "drop_to_idx"},
    rl_spec="training.train_rl_joint:ShogiSpec",
)

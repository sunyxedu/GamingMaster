"""Standard chess — the gmrules game ``chess``: the SIMULATOR of the existing chess
adapter (the IL side is unchanged):

    rules + frames    data_processing/rs/gmrules/src/games/chess/ — shakmaty's FIDE
                      rules and the chess converter's frozen frames (converter_version 3):
                      obs board_seq [14, 8, 8] / globals_seq [10] (castling rights, side,
                      clocks, check, "position seen >= 1 / >= 2 times before"), legal
                      from [64] / to_cond [64, 64] / promotion_cond [64, 64, 5], action
                      keys from_idx / to_idx / promotion (POV squares, ranks mirrored for
                      Black; promotion 0 none, 1 N, 2 B, 3 R, 4 Q; castling = king -> g/c)
    data              `gmdata chess fetch|convert` (Lichess PGN): the converter builds its
                      frames through the same gmrules game, so an IL frame is the RL frame
                      of the same position, byte for byte (shard meta
                      {"global_features": 10})
    simulator         gmsim.Sim("chess") → env/vec_gmsim.py (interleaved: one row per
                      game, every frame from the mover's view, lockstep)
    model             the chess adapter IL trains: model/games/registry.py
                      ``_build_chess`` (BoardEncoder + FromToDecoder; the registry keeps
                      it, the entry below names it) — train_il_game's LABEL_KEYS /
                      META_ARCH_KEYS / META_LEGACY keep their own chess entries, which the
                      tables below repeat
    RL                train_rl_joint's FromToSpec (from / to teacher-forced in the replay,
                      the promotion head under the rollout's promotion mask)

Game ends (the env's defaults; ``--env_kw chess:key=value`` sets the gmrules config):
checkmate, stalemate, insufficient material, the fifty-move rule (``fifty_move_draw``)
and threefold repetition (``threefold_draw``) as AUTOMATIC draws — FIDE makes the
last two claims, the side that profits always claims — and ``max_plies`` (1024) an
adjudicated draw.  The network needs ``board_global_features=10`` (the v3 shards'
meta; the registry default 8 is the legacy v2 width): FromToSpec.check_env refuses a
mismatch.
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="chess",
    sim_config={},
    description="Standard chess (FIDE rules by shakmaty) on the chess converter's frozen v3 frames",
    builder="model.games.registry:_build_chess",
    label_keys=("from_idx", "to_idx"),
    meta_arch={"global_features": ("board_global_features",)},
    sample_keys={"action_from": "from_idx", "action_to": "to_idx",
                 "action_promotion": "promotion"},
    rl_spec="training.train_rl_joint:FromToSpec",
)

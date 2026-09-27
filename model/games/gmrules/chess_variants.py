"""Chess variants (lichess: chess960, crazyhouse, atomic, antichess, three-check, king of
the hill, horde, racing kings) — ONE gmrules game with a variant chance event:

    rules + frames    data_processing/rs/gmrules/src/games/chess_variants.rs
                      (shakmaty's variant rules + lichess' automatic ends; schema
                      "chessvar.v1": obs board_seq [17, 8, 8] / hand_seq [10, 8] /
                      globals_seq [24], legal from / to_cond / promotion_cond over 74
                      tokens, targets from_idx / to_idx / promotion + the IL masks)
    converter         data_processing/rs/gmdata/src/chess_variants.rs
                      (`gmdata chess-variants convert`, lichess monthly dumps)
    fetch             scripts/fetch_chess_variants.sh (download + convert, full scale)
    simulator         gmsim.Sim("chess_variants") → env/vec_gmsim.py
    model             model/games/chess_variants.py (shogi-style encoder: 64 square +
                      10 pocket-slot tokens; the board adapter's FromToDecoder, a drop =
                      "from the pocket slot to the square")
    RL                train_rl_joint's FromToSpec (teacher-forced from / to replay)

The RL env draws the variant uniformly per episode (``--env_kw
chess_variants:variants=[...]`` / ``weights=[...]`` narrows / reweights it) and ends
games at a threefold repetition (``auto_threefold``: a claim on lichess, which the side
that profits always makes); the converter keeps lichess' own automatic ends only.
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="chess_variants",
    sim_config={"auto_threefold": True},
    description="Lichess chess variants (chess960, crazyhouse, atomic, antichess, three-check, "
                "king of the hill, horde, racing kings), one game with a variant chance event",
    builder="model.games.chess_variants:build_chess_variants_network",
    label_keys=("from_idx", "to_idx"),
    meta_arch={"board_h": ("board_h",), "board_w": ("board_w",),
               "piece_channels": ("piece_channels",),
               "global_features": ("board_global_features",),
               "hand_piece_types": ("n_hand_piece_types",),
               "hand_features": ("hand_feature_dim",),
               "n_promotions": ("n_promotions",)},
    sample_keys={"action_from": "from_idx", "action_to": "to_idx",
                 "action_promotion": "promotion"},
    rl_spec="training.train_rl_joint:FromToSpec",
)

"""Go — the gmrules game ``go``: the SIMULATOR of the existing go adapter (the IL side
is unchanged):

    rules + frames    data_processing/rs/gmrules/src/games/go/ — the KGS converter's rules
                      (captures, suicide illegal, simple ko; a pass is always legal) and
                      its frozen frames (converter_version 2): obs board_seq [4, N, N]
                      (own / opponent / empty, the ko point) / globals_seq [6] (mover,
                      komi from the mover's view, komi known, area scoring, handicap, the
                      previous move passed), legal place [N·N] / pass [], action keys
                      place_idx / passed
    data              `gmdata go fetch|convert` (KGS SGF): the converter builds its frames
                      through the same gmrules code, so an IL frame is the RL frame of the
                      same position (shard meta {"global_features": 6, "board_size": N})
    simulator         gmsim.Sim("go") → env/vec_gmsim.py (interleaved: one row per game,
                      every frame from the mover's view, lockstep)
    model             the go adapter IL trains: model/games/registry.py ``_build_go``
                      (BoardEncoder + the pass-gated PlacementDecoder; the registry keeps
                      it, the entry below names it) — train_il_game's LABEL_KEYS /
                      META_ARCH_KEYS / META_LEGACY keep their own go entries, which the
                      tables below repeat
    RL                train_rl_joint's GoSpec (the pass-gated log-prob, no conditioning
                      labels; full-head KL; the env built on the network's board)

Game ends (the env's defaults; ``--env_kw go:key=value`` sets the gmrules config): two
consecutive passes, ``max_moves`` (2·N², passes included) or a positional-superko
repetition (``superko``); every end is scored by Tromp-Taylor area with ``komi`` 7.5
for White (the KGS 4d+ corpus is essentially Chinese rules with 7.5).
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="go",
    sim_config={},
    description="Go (KGS converter rules: simple ko, no suicide; Tromp-Taylor area count) "
                "on its frozen v2 frames",
    builder="model.games.registry:_build_go",
    label_keys=(),
    meta_arch={"global_features": ("board_global_features",),
               "board_size": ("board_h", "board_w")},
    sample_keys={"action_place": "place_idx", "action_pass": "passed"},
    rl_spec="training.train_rl_joint:GoSpec",
)

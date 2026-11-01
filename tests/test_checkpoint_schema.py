"""model/checkpoint_migrations.py: the schema stamp and the legacy rules it sits next to.

Pins (a) what the stamp is and that an unstamped checkpoint reads as legacy, (b) that the
stamp changes nothing a loader computes — the legacy rules give the same config dict with
and without it, and are the identity on the complete config dict a stamped checkpoint
carries — and (c) that the stamped file round-trips through torch.save / torch.load,
including weights_only=True, so code that predates the stamp reads it as one more key.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model.config as model_config  # noqa: E402
from model.checkpoint_migrations import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CONFIG_DEFAULTS,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    checkpoint_config_dict,
    legacy_memory_vision,
    schema_version,
    stamp_schema_version,
    upgrade_config_dict,
)
from model.config import ModelConfig, checkpoint_config  # noqa: E402


def test_stamp_and_version():
    ck = {"config": {}, "iteration": 3}
    assert schema_version(ck) == LEGACY_SCHEMA_VERSION == 0
    assert stamp_schema_version(ck) is ck
    assert ck[SCHEMA_VERSION_KEY] == CHECKPOINT_SCHEMA_VERSION == schema_version(ck)
    assert set(ck) == {"config", "iteration", SCHEMA_VERSION_KEY}
    stamp_schema_version(ck)                       # idempotent
    assert ck[SCHEMA_VERSION_KEY] == CHECKPOINT_SCHEMA_VERSION


@pytest.mark.parametrize("bad", [-1, 1.0, True, "1"])
def test_version_must_be_a_non_negative_int(bad):
    with pytest.raises(ValueError):
        schema_version({SCHEMA_VERSION_KEY: bad})


def test_config_module_re_exports_the_legacy_rules():
    for name in ("LEGACY_CONFIG_DEFAULTS", "LEGACY_CONFIG_SCOPE", "MEMORY_VISION_ERA_FIELD",
                 "checkpoint_config_dict", "is_train_rl_checkpoint", "legacy_memory_vision",
                 "upgrade_config_dict"):
        import model.checkpoint_migrations as m
        assert getattr(model_config, name) is getattr(m, name), name


# the checkpoint shapes legacy_memory_vision tells apart, each unstamped
LEGACY_CHECKPOINTS = [
    {"config": {"grid_h": 25}, "iteration": 10},                              # train_rl, pre-memory
    {"config": {"grid_h": 25, "joint_decoder": True}, "iteration": 10},       # train_rl, memory era
    {"stage": "rl_joint", "games": ["generals"], "configs": {"generals": {"grid_h": 25}},
     "args": {"env_kw": ["generals:memory_vision=True"]}},
    {"stage": "rl_joint", "games": ["generals"], "configs": {"generals": {"grid_h": 25}},
     "args": {"env_kw": []}},
    {"games": ["generals", "chess"], "configs": {"generals": {"grid_h": 25}, "chess": {}}},
    {"game": "generals", "config": {"grid_h": 25}, "epoch": 2},
]


@pytest.mark.parametrize("ck", LEGACY_CHECKPOINTS)
def test_stamp_does_not_change_what_a_loader_computes(ck):
    stamped = stamp_schema_version({k: v for k, v in ck.items()})
    assert legacy_memory_vision(stamped) == legacy_memory_vision(ck)
    if "configs" in ck:
        for g in ck["games"]:
            assert checkpoint_config_dict(stamped, g) == checkpoint_config_dict(ck, g)
    else:
        assert checkpoint_config_dict(stamped) == checkpoint_config_dict(ck)
        assert checkpoint_config(stamped) == checkpoint_config(ck)


def test_legacy_rules_are_identity_on_a_complete_config_dict():
    """What version 1 promises: the trainers write cfg.__dict__, every field present, so
    neither upgrade_config_dict nor the memory_vision inference changes it."""
    for cfg in (ModelConfig(), ModelConfig(memory_vision=True, core_type="transformer")):
        d = dict(cfg.__dict__)
        assert set(LEGACY_CONFIG_DEFAULTS) & set(ModelConfig.__dataclass_fields__) <= set(d)
        assert upgrade_config_dict(d) == d
        ck = stamp_schema_version({"config": d, "iteration": 1})
        assert checkpoint_config_dict(ck) == d
        assert checkpoint_config(ck) == cfg


def test_stamped_checkpoint_round_trips_for_old_readers(tmp_path):
    ck = stamp_schema_version({"config": ModelConfig().__dict__, "iteration": 1,
                               "model_state": {"w": torch.zeros(2)}})
    torch.save(ck, tmp_path / "ck.pt")
    strict = torch.load(tmp_path / "ck.pt", map_location="cpu", weights_only=True)
    assert strict[SCHEMA_VERSION_KEY] == CHECKPOINT_SCHEMA_VERSION
    assert strict["config"] == ck["config"] and strict["iteration"] == 1
    assert checkpoint_config(strict) == ModelConfig()


def test_multi_game_export_is_stamped():
    from training.train_multi_game import game_checkpoint
    ck = {"games": ["chess"], "configs": {"chess": {"grid_h": 8}},
          "model_state": {"nets.chess.w": torch.zeros(1), "nets.chess.core.x": torch.ones(1)},
          "step": 7}
    out = game_checkpoint(ck, "chess")
    assert schema_version(out) == CHECKPOINT_SCHEMA_VERSION
    assert out["game"] == "chess" and set(out["model_state"]) == {"w", "core.x"}

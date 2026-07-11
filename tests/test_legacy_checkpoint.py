"""Legacy (pre-refactor) checkpoint compatibility.

The game-adapter refactor moved the generals modules into
``obs_encoder.*`` / ``action_decoder.*``. Checkpoints saved before it
(e.g. IL phase 1 trained on main) use the old flat key names and are
remapped on load via ``GeneralsNetwork.LEGACY_KEY_MAP``. These tests pin
that migration — the exact operation ``train_il.py --finetune_from
<phase-1 ckpt>`` performs — plus the reinit-prefix handling and the
optimizer-state resume guards around it.

Legacy-format state dicts are synthesized with
``GeneralsNetwork.remap_to_legacy_keys`` (the exact inverse of the load
remap), so no checkpoint fixture is needed.
"""

import copy
import hashlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.network import GeneralsNetwork
from model.rl_utils import (
    is_legacy_state_dict,
    load_model_state,
    optim_param_groups,
    optim_state_mismatch,
)

# Mirror the phase-1 lineage: recurrent temporal mixer + world-model head.
SMALL_CFG = dict(
    d_model=32, n_heads=4, d_ff=64, encoder_channels=16, n_st_layers=1,
    trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1, dropout=0.0,
    temporal_mixer="recurrent", predict_next_obs=True,
)


def small_net(seed=0, **overrides):
    torch.manual_seed(seed)
    return GeneralsNetwork(ModelConfig(**{**SMALL_CFG, **overrides}))


def legacy_state_dict(net):
    """Synthesize a main-format state dict from a current-format net."""
    return GeneralsNetwork.remap_to_legacy_keys(net.state_dict())


def forward_fixture(B=2, T=3):
    torch.manual_seed(123)
    return dict(
        grid_seq=torch.randn(B, T, 12, 25, 25),
        globals_seq=torch.randn(B, T, 4),
        legal_mask=(torch.rand(B, T, 1, 25, 25) < 0.3),
        # Fully teacher-forced source → no sampling → deterministic forward
        source_y=torch.randint(0, 25, (B, T)),
        source_x=torch.randint(0, 25, (B, T)),
    )


# ======================================================================
# Key mapping
# ======================================================================

def test_legacy_key_map_is_invertible():
    net = small_net()
    sd = net.state_dict()
    legacy = GeneralsNetwork.remap_to_legacy_keys(sd)
    # No collisions in either direction
    assert len(legacy) == len(sd)
    back = GeneralsNetwork.remap_legacy_keys(legacy)
    assert set(back) == set(sd)
    # Round trip maps every key back to itself (same tensor objects)
    for k, v in back.items():
        assert v is sd[k]


def test_legacy_format_renames_expected_families():
    net = small_net()
    legacy = legacy_state_dict(net)
    # Every adapter key got the old flat spelling...
    assert not any(k.startswith(("obs_encoder.", "action_decoder."))
                   for k in legacy)
    for old_prefix in GeneralsNetwork.LEGACY_KEY_MAP:
        assert any(k.startswith(old_prefix) for k in legacy), old_prefix
    # ...and the shared-core keys kept their names
    for k in legacy:
        if not k.startswith(tuple(GeneralsNetwork.LEGACY_KEY_MAP)):
            assert k in net.state_dict(), k


def test_remap_is_identity_on_new_format_keys():
    net = small_net()
    sd = net.state_dict()
    remapped = GeneralsNetwork.remap_legacy_keys(dict(sd))
    assert set(remapped) == set(sd)
    for k, v in remapped.items():
        assert v is sd[k]


# ======================================================================
# Loading a legacy checkpoint
# ======================================================================

def test_legacy_state_dict_loads_strict_and_exact():
    net_a = small_net(seed=0)
    net_b = small_net(seed=1)  # different init
    net_b.load_state_dict(legacy_state_dict(net_a), strict=True)
    for k, v in net_b.state_dict().items():
        assert torch.equal(v, net_a.state_dict()[k]), k


def test_legacy_load_via_load_model_state_matches_forward():
    """The exact --finetune_from path: load_model_state on a legacy dict
    (with a DDP 'module.' prefix on top) reproduces the source model's
    forward outputs bit-for-bit."""
    net_a = small_net(seed=0)
    net_b = small_net(seed=1)
    wrapped = {"module." + k: v for k, v in legacy_state_dict(net_a).items()}
    load_model_state(net_b, wrapped, ctx="test legacy+module prefix")

    net_a.eval()
    net_b.eval()
    fx = forward_fixture()
    with torch.no_grad():
        out_a = net_a(**fx, act_steps=1)
        out_b = net_b(**fx, act_steps=1)
    assert set(out_a) == set(out_b)
    for k in out_a:
        assert torch.equal(out_a[k], out_b[k]), k


def test_legacy_load_matches_step():
    """step() (sampling path incl. the shared masked_sample guard) agrees
    between the source net and a legacy-loaded copy under the same seed."""
    net_a = small_net(seed=0)
    net_b = small_net(seed=1)
    net_b.load_state_dict(legacy_state_dict(net_a), strict=True)
    net_a.eval()
    net_b.eval()

    torch.manual_seed(11)
    grid = torch.randn(2, 12, 25, 25)
    globals_vec = torch.randn(2, 4)
    legal = torch.rand(2, 1, 25, 25) < 0.3

    with torch.no_grad():
        torch.manual_seed(77)
        out_a = net_a.step(grid, globals_vec, legal_mask=legal, act_steps=1)
        torch.manual_seed(77)
        out_b = net_b.step(grid, globals_vec, legal_mask=legal, act_steps=1)
    for k in out_a:
        if k == "st_cache":
            for ca, cb in zip(out_a[k], out_b[k]):
                assert torch.equal(ca, cb)
        else:
            assert torch.equal(out_a[k], out_b[k]), k


def test_new_format_load_still_works_through_override():
    net_a = small_net(seed=0)
    net_b = small_net(seed=1)
    load_model_state(net_b, net_a.state_dict(), ctx="test new format")
    for k, v in net_b.state_dict().items():
        assert torch.equal(v, net_a.state_dict()[k]), k


# ======================================================================
# reinit_prefixes on a legacy checkpoint — both spellings must work
# ======================================================================

@pytest.mark.parametrize("prefix", [
    "direction_head.",                  # old spelling
    "action_decoder.direction_head.",   # new spelling
])
def test_reinit_prefix_spellings_on_legacy_checkpoint(prefix):
    net_a = small_net(seed=0)
    net_b = small_net(seed=1)
    fresh = {k: v.clone() for k, v in net_b.state_dict().items()}

    load_model_state(net_b, legacy_state_dict(net_a),
                     ctx=f"test reinit {prefix!r}",
                     reinit_prefixes=(prefix,))

    reinit_keys = [k for k in net_b.state_dict()
                   if k.startswith("action_decoder.direction_head.")]
    assert reinit_keys
    for k, v in net_b.state_dict().items():
        if k in reinit_keys:
            assert torch.equal(v, fresh[k]), f"{k} was not kept fresh"
        else:
            assert torch.equal(v, net_a.state_dict()[k]), f"{k} not loaded"


@pytest.mark.parametrize("prefix", [
    "next_obs_head.",
    "obs_encoder.next_obs_head.",
])
def test_reinit_prefix_for_module_absent_from_checkpoint(prefix):
    """Legitimate zero-drop case: the checkpoint predates the module; the
    model keeps its fresh init and the load succeeds."""
    net_old = small_net(seed=0, predict_next_obs=False)
    net_new = small_net(seed=1, predict_next_obs=True)
    fresh = {k: v.clone() for k, v in net_new.state_dict().items()}

    load_model_state(net_new, legacy_state_dict(net_old),
                     ctx="test reinit absent module",
                     reinit_prefixes=(prefix,))

    for k, v in net_new.state_dict().items():
        if k.startswith("obs_encoder.next_obs_head."):
            assert torch.equal(v, fresh[k])
        else:
            assert torch.equal(v, net_old.state_dict()[k]), k


def test_reinit_prefix_typo_raises():
    net_a = small_net(seed=0)
    net_b = small_net(seed=1)
    with pytest.raises(ValueError, match="typo"):
        load_model_state(net_b, legacy_state_dict(net_a),
                         ctx="test reinit typo",
                         reinit_prefixes=("nonexistent_head.",))


# ======================================================================
# Legacy-format detection
# ======================================================================

def test_is_legacy_state_dict():
    net = small_net()
    assert not is_legacy_state_dict(net.state_dict(), net)
    legacy = legacy_state_dict(net)
    assert is_legacy_state_dict(legacy, net)
    wrapped = {"module." + k: v for k, v in legacy.items()}
    assert is_legacy_state_dict(wrapped, net)


# ======================================================================
# Optimizer-state resume guard
# ======================================================================

def stepped_optimizer(net):
    """AdamW over optim_param_groups with populated moment state."""
    opt = torch.optim.AdamW(optim_param_groups(net, 1e-4), lr=1e-3)
    loss = sum(p.sum() for p in net.parameters())
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return opt


def test_optim_state_matching_layout_accepted():
    net = small_net(seed=0)
    saved = stepped_optimizer(net).state_dict()
    fresh_opt = torch.optim.AdamW(optim_param_groups(net, 1e-4), lr=1e-3)
    assert optim_state_mismatch(fresh_opt, saved) is None
    fresh_opt.load_state_dict(saved)  # must not raise


def test_optim_state_detects_positional_reorder():
    """A permutation between differently-shaped parameters with identical
    counts (the cross-branch --resume hazard) must be rejected, not
    silently loaded. Same-shape swaps are invisible to the shape guard —
    test_parameter_registration_order_is_pinned covers those by pinning
    the order itself."""
    net = small_net(seed=0)
    saved = copy.deepcopy(stepped_optimizer(net).state_dict())

    # Swap two saved positions whose parameters have different shapes —
    # exactly what a registration-order change does.
    group0 = saved["param_groups"][0]["params"]
    shapes = {i: tuple(saved["state"][i]["exp_avg"].shape)
              for i in group0 if i in saved["state"]}
    i, j = next((a, b) for a in group0 for b in group0
                if a in shapes and b in shapes and shapes[a] != shapes[b])
    pi, pj = group0.index(i), group0.index(j)
    group0[pi], group0[pj] = group0[pj], group0[pi]

    fresh_opt = torch.optim.AdamW(optim_param_groups(net, 1e-4), lr=1e-3)
    reason = optim_state_mismatch(fresh_opt, saved)
    assert reason is not None and "order" in reason


def test_optim_state_detects_group_count_mismatch():
    net = small_net(seed=0)
    saved = copy.deepcopy(stepped_optimizer(net).state_dict())
    saved["param_groups"].append({"params": []})
    fresh_opt = torch.optim.AdamW(optim_param_groups(net, 1e-4), lr=1e-3)
    assert optim_state_mismatch(fresh_opt, saved) is not None


def test_optim_state_detects_param_count_mismatch():
    net = small_net(seed=0)
    saved = copy.deepcopy(stepped_optimizer(net).state_dict())
    saved["param_groups"][0]["params"] = saved["param_groups"][0]["params"][:-1]
    fresh_opt = torch.optim.AdamW(optim_param_groups(net, 1e-4), lr=1e-3)
    assert optim_state_mismatch(fresh_opt, saved) is not None


def test_optim_state_unstepped_state_accepted():
    """A state dict whose ``state`` is empty (no step taken yet) has nothing
    to misalign and must be accepted."""
    net = small_net(seed=0)
    opt = torch.optim.AdamW(optim_param_groups(net, 1e-4), lr=1e-3)
    assert optim_state_mismatch(opt, opt.state_dict()) is None


def test_parameter_registration_order_is_pinned():
    """Optimizer state dicts are positional, and permutations between
    SAME-shaped parameters evade the shape guard entirely (torch loads
    and steps them without a symptom). Pinning the registration order
    turns any future reorder into a test failure instead of silent
    optimizer-state corruption in existing checkpoints.

    If this fails because you intentionally reordered parameters: bump
    the digest AND make sure --resume rejects optimizer_state from
    checkpoints saved before the change (see is_legacy_state_dict for
    the rename-based precedent).
    """
    net = small_net(seed=0)
    names = [n for n, _ in net.named_parameters()]
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    assert digest == "1949922d1663c575", (
        f"parameter registration order changed (digest {digest}); "
        f"optimizer states in existing checkpoints are now misaligned"
    )


# ======================================================================
# On-disk round trip — the exact --finetune_from file seam
# ======================================================================

def test_phase1_checkpoint_file_round_trip(tmp_path):
    """Save a synthetic phase-1-style checkpoint file (legacy keys, config
    dict, stepped optimizer state) and drive it through the same steps
    train_il.py performs: torch.load, config reconstruction, legacy
    detection, weight load — and the optimizer-state rejection the
    --resume guard applies to pre-refactor checkpoints."""
    net_a = small_net(seed=0)
    opt = stepped_optimizer(net_a)
    path = tmp_path / "checkpoint_epoch40.pt"
    torch.save({
        "epoch": 40,
        "step": 0,
        "model_state": legacy_state_dict(net_a),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": {},
        "config": net_a.cfg.__dict__,
        "loss": 0.5,
    }, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Config reconstruction as train_il.py does — the recurrent mixer and
    # the world-model head must survive the file round trip.
    cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    assert cfg.temporal_mixer == "recurrent"
    assert cfg.predict_next_obs is True

    torch.manual_seed(99)
    net_b = GeneralsNetwork(cfg)
    assert is_legacy_state_dict(ckpt["model_state"], net_b)

    load_model_state(net_b, ckpt["model_state"], ctx="phase-1 file")
    for k, v in net_b.state_dict().items():
        assert torch.equal(v, net_a.state_dict()[k]), k

    # --resume on this branch must refuse the pre-refactor optimizer state
    # purely from the legacy model_state detection, regardless of shapes.
    assert is_legacy_state_dict(ckpt["model_state"], net_b) is True

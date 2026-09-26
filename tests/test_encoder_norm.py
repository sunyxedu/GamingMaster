"""Conv observation-encoder normalisation (``cfg.encoder_norm``,
model/components/encoder.py).

* "group" (the default): per-frame GroupNorm, every encoder norm layer from
  one factory, ``encoder_norm_groups`` groups (default 1 = LayerNorm over
  (C, H, W)) that must divide every width.  A frame's features are a function
  of that frame alone in train mode, so the audit's BatchNorm leak is gone
  (tests/test_audit_model_causality.py runs the full causality /
  sample-independence audit on both layouts); padded frames need no exclusion
  and the encoder path has no host sync;
* "batch": the legacy BatchNorm2d layout.  Configs without the field (every
  checkpoint written before it) rebuild it, load strictly and compute exactly
  what they computed, through every production loader;
* guards: a BatchNorm checkpoint never loads silently into a GroupNorm
  encoder (same weight / bias keys);
* multi-game builds: encoder_norm is per game, not a core field;
* ``FrameGroupNorm`` keeps the input dtype under autocast, like BatchNorm.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torch.nn.functional as F

from model.components.encoder import (
    GN_EPS,
    EncoderNorm,
    FrameGroupNorm,
    ResBlock,
    encode_frames,
    encoder_norm_of,
    make_encoder_norm,
)
from model.config import ModelConfig, config_from_dict, upgrade_config_dict
from model.games import GAME_REGISTRY, build_game_network
from model.multi_game import (
    build_multi_game,
    core_config_overrides,
    core_state_dict,
    is_core_key,
    load_core,
)
from model.rl_utils import load_model_state
from tests.test_audit_model_causality import (
    BASE,
    CONV_ENCODER_GAMES,
    CORES,
    GAME_OVERRIDES,
    _labels,
    _legal,
    _obs,
)
from tests.test_audit_model_causality import B as AB
from tests.test_audit_model_causality import T as AT

CONV_GAMES = sorted(CONV_ENCODER_GAMES)
_BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def _build(game: str, norm=None, core: str = "trm", seed: int = 0, **kw) -> nn.Module:
    torch.manual_seed(seed)
    extra = {} if norm is None else {"encoder_norm": norm}
    return build_game_network(game, **BASE, **CORES[core], temporal_mixer="recurrent",
                              **GAME_OVERRIDES[game], **extra, **kw)


def _batch(game: str, net: nn.Module, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    obs = _obs(game, net.cfg, g)
    return obs, _legal(game, net.cfg, obs, g), _labels(game, net.cfg, g)


def _outputs(net, obs, legal, labels, mode: str, frame_mask=None):
    net.train(mode == "train")
    torch.manual_seed(1234)
    with torch.no_grad():
        out = net(obs, legal=legal, labels=labels, act_steps=2, frame_mask=frame_mask)
    return {k: v for k, v in out.items() if torch.is_tensor(v)}


def _assert_same(a, b, what):
    assert a.keys() == b.keys(), what
    for k in a:
        assert torch.equal(a[k], b[k]), f"{what}: {k}"


# ---------------------------------------------------------------------------
# the factory and its group counts
# ---------------------------------------------------------------------------

def test_factory_layers():
    gn = make_encoder_norm("group", 192)
    assert type(gn) is FrameGroupNorm and gn.num_groups == 1 and gn.num_channels == 192
    assert gn.eps == GN_EPS == 1e-3
    assert gn.affine and torch.equal(gn.weight, torch.ones(192)) and not gn.bias.any()
    assert make_encoder_norm("group", 192, groups=32).num_groups == 32
    bn = make_encoder_norm("batch", 192)
    assert type(bn) is nn.BatchNorm2d and bn.num_features == 192
    for bad in ("layer", "instance", "Group", ""):
        with pytest.raises(ValueError, match="encoder_norm"):
            make_encoder_norm(bad, 8)
    for groups in (0, 5, 16):                 # must divide the width
        with pytest.raises(ValueError, match="encoder_norm_groups"):
            make_encoder_norm("group", 24, groups=groups)
    with pytest.raises(ValueError, match="encoder_norm"):
        build_game_network("chess", **BASE, encoder_norm="layer")
    with pytest.raises(ValueError, match="encoder_norm_groups"):   # 3 does not divide 8
        build_game_network("chess", **BASE, encoder_norm_groups=3)
    blk = ResBlock(16)                        # the component default = the config default
    assert all(type(m) is FrameGroupNorm and m.num_groups == 1
               for m in blk.net if isinstance(m, nn.GroupNorm))
    assert encoder_norm_of(ModelConfig()) == EncoderNorm("group", 1)
    assert encoder_norm_of(object()) == EncoderNorm("batch", 1)       # legacy objects


def test_one_group_is_layer_norm_over_chw():
    """The default: GroupNorm with one group = LayerNorm over (C, H, W) of
    every frame, then a per-channel affine."""
    torch.manual_seed(0)
    gn = make_encoder_norm("group", 12)
    with torch.no_grad():
        gn.weight.uniform_(0.5, 1.5)
        gn.bias.uniform_(-0.5, 0.5)
    x = torch.randn(5, 12, 7, 6) * 3 + 1
    ref = F.layer_norm(x, (12, 7, 6), eps=GN_EPS) * gn.weight.view(1, -1, 1, 1) \
        + gn.bias.view(1, -1, 1, 1)
    assert torch.allclose(gn(x), ref, atol=1e-5, rtol=1e-5)


def _norm_modules(net: nn.Module):
    return {n: m for n, m in net.named_modules()
            if isinstance(m, (nn.GroupNorm, nn.modules.batchnorm._BatchNorm))}


@pytest.mark.parametrize("game", sorted(GAME_REGISTRY))
def test_every_encoder_norm_comes_from_the_factory(game):
    """Default build: every normalisation layer of the conv encoders is a
    FrameGroupNorm with cfg.encoder_norm_groups groups (1) and nothing else in
    the network is a GroupNorm / BatchNorm; the legacy build puts a BatchNorm2d
    at exactly the same module paths.  Both builds have identical parameters
    (the rest of the random init included) and differ only by the BN buffers."""
    grp, bat = _build(game), _build(game, "batch")
    assert grp.cfg.encoder_norm == "group" and bat.cfg.encoder_norm == "batch"
    g_norms, b_norms = _norm_modules(grp), _norm_modules(bat)
    assert g_norms.keys() == b_norms.keys()
    assert bool(g_norms) == (game in CONV_ENCODER_GAMES)
    four = _norm_modules(_build(game, encoder_norm_groups=4))
    for name, m in g_norms.items():
        assert name.startswith("obs_encoder."), name
        assert type(m) is FrameGroupNorm and m.num_groups == 1 and m.eps == GN_EPS
        assert type(four[name]) is FrameGroupNorm and four[name].num_groups == 4
        assert type(b_norms[name]) is nn.BatchNorm2d
        assert b_norms[name].num_features == m.num_channels
    gp, bp = dict(grp.named_parameters()), dict(bat.named_parameters())
    assert gp.keys() == bp.keys()
    for k in gp:
        assert torch.equal(gp[k], bp[k]), k
    gs, bs = grp.state_dict(), bat.state_dict()
    extra = set(bs) - set(gs)
    assert set(gs) <= set(bs)
    assert extra == {f"{n}.{b}" for n in b_norms for b in _BN_BUFFERS}


# ---------------------------------------------------------------------------
# config: new default, legacy value for configs predating the field
# ---------------------------------------------------------------------------

def test_config_default_and_legacy_upgrade():
    assert ModelConfig().encoder_norm == "group" and ModelConfig().encoder_norm_groups == 1
    for game in GAME_REGISTRY:
        assert build_game_network(game, **BASE).cfg.encoder_norm == "group", game
    assert upgrade_config_dict({})["encoder_norm"] == "batch"
    assert upgrade_config_dict({"encoder_norm": "group"})["encoder_norm"] == "group"
    assert config_from_dict({"d_model": 32, "n_heads": 4}).encoder_norm == "batch"
    assert config_from_dict({"encoder_norm": "group"}).encoder_norm == "group"
    # an encoder property, not a core field: a new game around an old brain
    # gets the current default
    assert "encoder_norm" not in core_config_overrides({"d_model": 32})
    assert "encoder_norm_groups" not in core_config_overrides({"encoder_norm_groups": 4})


# ---------------------------------------------------------------------------
# legacy checkpoints: rebuilt with BatchNorm, strict load, identical outputs
# ---------------------------------------------------------------------------

def _legacy_checkpoint(game: str):
    """A network as the pre-field code built it (BatchNorm, verified
    bit-identical to it) with live running statistics, and its checkpoint:
    the config dict carries no encoder_norm key."""
    net = _build(game, "batch")
    obs, legal, labels = _batch(game, net, seed=3)
    net.train()
    with torch.no_grad():
        for _ in range(3):                     # move the running statistics
            net(obs, legal=legal, labels=labels, act_steps=2)
    old_cfg = {k: v for k, v in net.cfg.__dict__.items() if k != "encoder_norm"}
    return net, {"game": game, "config": old_cfg, "model_state": net.state_dict()}


@pytest.mark.parametrize("game", CONV_GAMES)
def test_legacy_checkpoint_rebuilds_and_computes_the_same(game):
    from training.train_il_game import build_net_tolerant, resume_arch_overrides

    net, ckpt = _legacy_checkpoint(game)
    obs, legal, labels = _batch(game, net)
    fm = torch.ones(AB, AT, dtype=torch.bool)
    fm[1, AT - 2:] = False
    rebuilt = {
        # train_il_game / train_rl_game (resume, load_game_net)
        "resume_arch_overrides": build_net_tolerant(game, resume_arch_overrides(ckpt, {}))[0],
        # train_multi_game resume / eval_multi_game.load_multi
        "upgrade_config_dict": build_net_tolerant(game, upgrade_config_dict(ckpt["config"]))[0],
    }
    if game == "generals":                    # train_il.py / train_rl.py / scripts
        from model.network import GeneralsNetwork
        rebuilt["config_from_dict"] = GeneralsNetwork(config_from_dict(ckpt["config"]))
    for how, net2 in rebuilt.items():
        assert net2.cfg.encoder_norm == "batch", how
        net2.load_state_dict(ckpt["model_state"], strict=True)
        for mode in ("eval", "train"):
            for mask in (None, fm):
                a = _outputs(net, obs, legal, labels, mode, mask)
                b = _outputs(net2, obs, legal, labels, mode, mask)
                _assert_same(a, b, f"{game} via {how}, {mode}, mask={mask is not None}")
        # and the running statistics evolve identically (train-mode forwards)
        for k, v in net.state_dict().items():
            assert torch.equal(v, net2.state_dict()[k]), (how, k)

    # the new default does not accept it: strict load fails, and the lenient
    # loader refuses to drop the BatchNorm statistics
    new = build_net_tolerant(game, dict(ckpt["config"]))[0]
    assert new.cfg.encoder_norm == "group"
    with pytest.raises(RuntimeError, match="running_mean"):
        new.load_state_dict(ckpt["model_state"], strict=True)
    with pytest.raises(RuntimeError, match="encoder_norm"):
        load_model_state(new, ckpt["model_state"], ctx="legacy")
    # ...and a GroupNorm checkpoint in a BatchNorm model misses the statistics
    with pytest.raises(RuntimeError, match="missing"):
        load_model_state(net, new.state_dict(), ctx="group into batch")


def test_old_multi_game_checkpoint_loads_with_batchnorm(tmp_path):
    """eval_multi_game.load_multi (and train_multi_game's resume, the same
    upgrade) on a multi-game checkpoint saved before encoder_norm existed."""
    from training.eval_multi_game import load_multi

    torch.manual_seed(0)
    core = dict(BASE, **CORES["rd_moe"], temporal_mixer="recurrent", encoder_norm="batch")
    games = ["othello", "chess", "generals"]
    over = {g: GAME_OVERRIDES[g] for g in games}
    mg = build_multi_game(games, core, over)
    strip = lambda d: {k: v for k, v in d.items() if k != "encoder_norm"}   # noqa: E731
    first = mg[games[0]]
    torch.save({
        "games": games,
        "configs": {g: strip(mg[g].cfg.__dict__) for g in games},
        "core_config": strip(core_config_overrides(first.cfg)),
        "core_state": core_state_dict(first),
        "model_state": mg.state_dict(),
        "train_model_state": mg.state_dict(),
    }, tmp_path / "multi_old.pt")
    loaded = load_multi(str(tmp_path / "multi_old.pt"), torch.device("cpu"))
    for g in games:
        assert loaded[g].cfg.encoder_norm == "batch"
        assert any(isinstance(m, nn.BatchNorm2d) for m in loaded[g].modules())
    sd = loaded.state_dict()
    for k, v in mg.state_dict().items():
        assert torch.equal(sd[k], v), k


# ---------------------------------------------------------------------------
# multi-game builds
# ---------------------------------------------------------------------------

def test_multi_game_encoder_norm_is_per_game_and_core_is_shared():
    core = dict(BASE, **CORES["rd_moe"], temporal_mixer="recurrent")
    games = ["othello", "chess", "generals", "mahjong"]
    over = {g: dict(GAME_OVERRIDES[g]) for g in games}
    over["othello"]["encoder_norm"] = "batch"            # e.g. one legacy adapter
    torch.manual_seed(0)
    mg = build_multi_game(games, core, over)
    kinds = {g: {type(m) for m in mg[g].obs_encoder.modules()
                 if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d))} for g in games}
    assert kinds == {"othello": {nn.BatchNorm2d}, "chess": {FrameGroupNorm},
                     "generals": {FrameGroupNorm}, "mahjong": set()}
    for g in games[1:]:                                  # one core object
        assert mg[g].st_layers is mg["othello"].st_layers and mg[g].core is mg["othello"].core
    # the brain moves between encoder layouts: the core keys hold no norm buffers
    cs = core_state_dict(mg["othello"])
    assert not any(k.rsplit(".", 1)[-1] in _BN_BUFFERS for k in cs)
    torch.manual_seed(1)
    fresh = build_game_network("go", **core, **GAME_OVERRIDES["go"])
    load_core(fresh, cs)
    for k, v in cs.items():
        assert torch.equal(fresh.state_dict()[k], v), k
    assert all(is_core_key(k) for k in cs)
    # all-default build: no BatchNorm anywhere (nothing for BatchNormStatSync)
    from training.distributed import BatchNormStatSync
    assert not BatchNormStatSync(build_multi_game(games, core, GAME_OVERRIDES)).bns
    assert len(BatchNormStatSync(mg).bns) == 6          # othello's legacy encoder


# ---------------------------------------------------------------------------
# padded frames, host syncs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["group", "batch"])
@pytest.mark.parametrize("game", CONV_GAMES)
def test_padded_frames_never_reach_real_frames(game, norm):
    """Train mode, tail-padded rows: whatever the padded frames hold, every
    output at a real frame is bit-identical.  "group" needs no exclusion (it
    runs every frame), "batch" keeps them out of its statistics."""
    net = _build(game, norm)
    obs, legal, labels = _batch(game, net)
    fm = torch.ones(AB, AT, dtype=torch.bool)
    fm[1, AT - 2:] = False
    fm[2, 3:] = False
    garbage = {k: v.clone() for k, v in obs.items()}
    g = torch.Generator().manual_seed(5)
    for k, v in garbage.items():
        if v.is_floating_point():
            v[~fm] = torch.randn(v[~fm].shape, generator=g) * 50.0
    a = _outputs(net, obs, legal, labels, "train", fm)
    b = _outputs(net, garbage, legal, labels, "train", fm)
    for k in a:
        if a[k].dim() >= 2 and a[k].shape[:2] == (AB, AT):
            assert torch.equal(a[k][fm], b[k][fm]), (game, norm, k)


def test_encode_frames_mask_handling():
    stack = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), make_encoder_norm("batch", 8)).train()
    frames = torch.randn(2 * 4, 3, 5, 5)
    fm = torch.tensor([[True] * 4, [True, True, False, False]])
    masked = encode_frames(stack, frames, 2, 4, fm, batch_stats=True)
    assert masked.shape == (2, 4, 8, 5, 5)
    assert not masked[~fm].any()                         # zero features, excluded
    real = encode_frames(stack, frames[fm.reshape(-1)], 6, 1, None, batch_stats=True)
    assert torch.equal(masked[fm], real[:, 0])
    # a per-frame stack runs every frame: the mask changes nothing
    gstack = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), make_encoder_norm("group", 8))
    full = encode_frames(gstack, frames, 2, 4, None, batch_stats=False)
    assert torch.equal(encode_frames(gstack, frames, 2, 4, fm, batch_stats=False), full)
    assert torch.equal(full.reshape(8, 8, 5, 5)[5], gstack(frames[5:6])[0])


@pytest.mark.parametrize("game", CONV_GAMES)
def test_group_encoder_forward_has_no_host_sync(game, monkeypatch):
    """The train-mode encoder forward with a frame_mask: the GroupNorm stack
    reads nothing on the host and keeps static shapes (the legacy BatchNorm
    exclusion needs ``bool(frame_mask.all())`` and boolean indexing)."""
    from tests.test_inference_graph import _forbid_host_syncs

    fm = torch.ones(AB, AT, dtype=torch.bool)
    fm[1, AT - 2:] = False
    for norm in ("group", "batch"):
        net = _build(game, norm)
        obs, _, _ = _batch(game, net)
        enc = net.obs_encoder.train()
        with monkeypatch.context() as mp:
            _forbid_host_syncs(mp)
            if norm == "group":
                enc(obs, fm)
            else:
                with pytest.raises(AssertionError, match="host sync|boolean-mask"):
                    enc(obs, fm)


# ---------------------------------------------------------------------------
# FrameGroupNorm dtype flow
# ---------------------------------------------------------------------------

def _gn_pair(C=32, G=8):
    torch.manual_seed(0)
    a, b = FrameGroupNorm(G, C), nn.GroupNorm(G, C)
    with torch.no_grad():
        a.weight.uniform_(0.5, 1.5)
        a.bias.uniform_(-0.5, 0.5)
    b.load_state_dict(a.state_dict())
    return a, b


def test_frame_group_norm_is_group_norm_outside_autocast():
    a, b = _gn_pair()
    x = torch.randn(4, 32, 6, 5)
    assert torch.equal(a(x), b(x))
    # per frame: a row's output ignores the other rows
    assert torch.equal(a(x)[1:2], a(x[1:2]))


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_frame_group_norm_keeps_input_dtype_under_autocast(device):
    """CUDA autocast would run group_norm in float32 (CPU autocast keeps the
    input dtype, MPS casts to float32): FrameGroupNorm stays in bf16 on all."""
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    a, b = _gn_pair()
    a, b = a.to(device), b.to(device)
    x = torch.randn(4, 32, 6, 5, device=device)
    ref = b(x)
    conv = nn.Conv2d(32, 32, 1, bias=False).to(device)
    with torch.autocast(device, dtype=torch.bfloat16):
        h = conv(x)
        assert h.dtype == torch.bfloat16
        y = a(h)
        assert y.dtype == torch.bfloat16                 # BatchNorm's dtype flow
        y.float().square().sum().backward()
    assert a.weight.grad is not None and a.weight.grad.dtype == torch.float32
    assert a.bias.grad is not None and torch.isfinite(a.weight.grad).all()
    # statistics accumulated in float: only the output rounding differs
    with torch.no_grad():
        exact = b(conv(x))
    assert torch.allclose(y.float(), exact, atol=0.05, rtol=0.02)
    assert ref.dtype == torch.float32


# ---------------------------------------------------------------------------
# trainers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override,label,kind", [((), "group/1", FrameGroupNorm),
                                                  (("encoder_norm=batch",), "batch",
                                                   nn.BatchNorm2d)])
def test_train_multi_game_logs_saves_and_reloads_encoder_norm(tmp_path, capsys, override,
                                                              label, kind):
    """train_multi_game logs the encoder norm, writes it into the checkpoint
    config and eval_multi_game.load_multi rebuilds the same layers."""
    from data_processing.games.dataset import save_shard
    from tests.test_core_moe import _board_samples
    from training import train_multi_game
    from training.eval_multi_game import load_multi

    d = tmp_path / "othello"
    d.mkdir()
    save_shard(str(d / "a.pt"), "othello", _board_samples("othello", 4, 6))
    core = dict(BASE, **CORES["rd_moe"], temporal_mixer="recurrent")
    train_multi_game.main([
        "--data", f"othello={d}", "--steps", "2", "--batch_size", "2", "--warmup_steps", "1",
        "--log_every", "1", "--save_dir", str(tmp_path / "ck"), "--device", "cpu",
        "--override", *[f"{k}={v}" for k, v in core.items()], *override])
    init = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[init]")]
    assert len(init) == 1 and f"encoder_norm={label} " in init[0]
    path = tmp_path / "ck" / "multi_final.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["configs"]["othello"]["encoder_norm"] == ("batch" if override else "group")
    assert ck["configs"]["othello"]["encoder_norm_groups"] == 1
    net = load_multi(str(path), torch.device("cpu"))["othello"]
    assert {type(m) for m in net.obs_encoder.modules()
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d))} == {kind}

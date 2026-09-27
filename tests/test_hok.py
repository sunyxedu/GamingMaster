"""Honor of Kings 1v1 in the tier-3 pipeline: the adapter (model/games/hok.py).

Adapter (no gamecore needed): the encoder's 16 entity tokens; the rollout's
μ = sample()'s log-prob equals the replay's log π of the stored action (the stored
button teacher-forced, the compact masks the rollout sampled with); every sampled
argument is legal; greedy decoding is deterministic; il_head_terms decomposes
il_loss, and the heads a button does not use are neither sampled into μ nor
supervised.  IL (the gmdata hok convert schema): train_multi_game on whole records
longer than the chunk length, with the shards' legal masks.
"""

from __future__ import annotations

import pytest
import torch

from model.games import build_game_network
from model.games.hok import (
    ARG_HEADS,
    BUTTONS,
    DIR_BINS,
    N_FEATURES,
    N_TARGETS,
    N_TOKENS,
    hok_replay_legal,
)

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent")


def _obs(B, T, g):
    return {"features": torch.rand(B, T, N_FEATURES, generator=g)}


def _legal(B, T, g):
    button = torch.rand(B, T, BUTTONS, generator=g) < 0.5
    button[..., 2] = True                                         # Move always legal
    out = {"button": button}
    for h in ARG_HEADS[:4]:
        m = torch.rand(B, T, DIR_BINS, generator=g) < 0.6
        m[..., 0] = True
        out[h] = m
    target = torch.rand(B, T, BUTTONS, N_TARGETS, generator=g) < 0.5
    target[..., 0] = True
    sub = torch.rand(B, T, BUTTONS, 1 + len(ARG_HEADS), generator=g) < 0.5
    sub[..., 0] = True
    out.update(target=target, sub=sub)
    return out


def _net():
    torch.manual_seed(0)
    return build_game_network("hok", **TINY).eval()


def test_encoder_tokens_and_shapes():
    net = _net()
    g = torch.Generator().manual_seed(1)
    out = net(_obs(2, 3, g), legal=_legal(2, 3, g))
    assert net.obs_encoder.n_tokens == N_TOKENS
    assert out["button_logits"].shape == (2, 3, BUTTONS)
    assert out["target_logits"].shape == (2, 3, N_TARGETS)
    assert out["target_mask"].shape == (2, 3, N_TARGETS) and out["sub_mask"].shape == (2, 3, 6)
    with pytest.raises(ValueError, match="705"):
        net({"features": torch.rand(1, 1, 725)})


def test_rollout_logprob_is_the_replay_logprob_and_samples_are_legal():
    net = _net()
    g = torch.Generator().manual_seed(2)
    B = 16
    obs = _obs(B, 1, g)
    legal = _legal(B, 1, g)
    with torch.no_grad():
        out = net.step({k: v[:, 0] for k, v in obs.items()},
                       legal={k: v[:, 0] for k, v in legal.items()})
    b = out["action_button"]
    assert bool(legal["button"][torch.arange(B), 0, b].all())
    uses = legal["sub"][torch.arange(B), 0, b, 1:]
    for j, h in enumerate(ARG_HEADS[:4]):
        ok = legal[h][torch.arange(B), 0, out[f"action_{h}"]]
        assert bool((ok | ~uses[:, j]).all()), h
    tgt_ok = legal["target"][torch.arange(B), 0, b, out["action_target"]]
    assert bool((tgt_ok | ~uses[:, 4]).all())
    replay = {k: v.unsqueeze(1) for k, v in hok_replay_legal(out).items()}
    assert {"target_of", "sub_of", "button"} <= set(replay)
    acts = {"button_idx": b.unsqueeze(1)}
    acts.update({f"{h}_idx": out[f"action_{h}"].unsqueeze(1) for h in ARG_HEADS})
    with torch.no_grad():
        pred = net(obs, legal=replay, labels={"button_idx": acts["button_idx"]}, act_steps=None)
        lp, ent, _ = net.action_decoder.log_prob_entropy(pred, acts, replay)
    torch.testing.assert_close(lp[:, 0], out["action_log_prob"].float(), rtol=1e-4, atol=1e-5)
    assert torch.isfinite(ent).all()
    # an unused head's value does not enter the log-prob
    acts2 = dict(acts)
    unused = ~replay["sub_of"][:, 0, 1]
    acts2["move_x_idx"] = torch.where(unused.unsqueeze(1), (acts["move_x_idx"] + 1) % DIR_BINS,
                                      acts["move_x_idx"])
    with torch.no_grad():
        lp2, _, _ = net.action_decoder.log_prob_entropy(pred, acts2, replay)
    torch.testing.assert_close(lp2, lp)


def test_greedy_is_deterministic():
    net = _net()
    net.action_decoder.greedy = True
    g = torch.Generator().manual_seed(3)
    obs = {k: v[:, 0] for k, v in _obs(4, 1, g).items()}
    legal = {k: v[:, 0] for k, v in _legal(4, 1, g).items()}
    with torch.no_grad():
        a, b = net.step(obs, legal=legal), net.step(obs, legal=legal)
    for k in ("action_button",) + tuple(f"action_{h}" for h in ARG_HEADS):
        assert torch.equal(a[k], b[k]), k


def test_il_head_terms_decompose_il_loss_and_skip_unused_heads():
    torch.manual_seed(0)
    net = build_game_network("hok", **TINY).train()
    g = torch.Generator().manual_seed(4)
    B, T = 3, 5
    obs = _obs(B, T, g)
    legal = _legal(B, T, g)
    tgt = {"button_idx": torch.full((B, T), 2)}
    for h, n in zip(ARG_HEADS, (DIR_BINS,) * 4 + (N_TARGETS,)):
        tgt[f"{h}_idx"] = torch.randint(0, n, (B, T), generator=g)
    tgt["skill_x_idx"][:, :2] = -1                                # not used by the label
    compact = {k: legal[k] for k in ("button",) + ARG_HEADS[:4]}
    compact["target_of"] = legal["target"][:, :, 2]
    compact["sub_of"] = legal["sub"][:, :, 2]
    pred = net(obs, legal=compact, labels={"button_idx": tgt["button_idx"]})
    mask = torch.ones(B, T, dtype=torch.bool)
    dec = net.action_decoder
    loss = dec.il_loss(pred, tgt, mask)
    terms = dec.il_head_terms(pred, tgt, mask)
    total = sum((t["ce"] * t["weight"]).sum() / t["norm"] for t in terms.values())
    torch.testing.assert_close(loss["loss"], total)
    assert bool((terms["skill_x"]["weight"][:, :2] == 0).all())
    used = compact["sub_of"][..., 1:].float()
    for j, h in enumerate(ARG_HEADS):
        w = terms[h]["weight"]
        assert bool((w <= used[..., j]).all()), h
    loss["loss"].backward()
    assert net.action_decoder.target_head.net[0].weight.grad is not None


def _hok_records(n, lengths, g):
    """IL samples in the gmdata hok convert schema."""
    out = []
    for i, T in zip(range(n), lengths):
        tgt = {"button_idx": torch.full((T,), 2), "legal_button": torch.ones(T, BUTTONS, dtype=torch.bool)}
        for h in ARG_HEADS:
            n_h = N_TARGETS if h == "target" else DIR_BINS
            tgt[f"{h}_idx"] = torch.randint(0, n_h, (T,), generator=g)
            if h != "target":
                tgt[f"legal_{h}"] = torch.ones(T, DIR_BINS, dtype=torch.bool)
        tgt["legal_target"] = torch.ones(T, BUTTONS, N_TARGETS, dtype=torch.bool)
        sub = torch.zeros(T, 6, dtype=torch.bool)
        sub[:, :3] = True                                         # Move: button + move_x/z
        tgt["legal_sub_of"] = sub
        for h in ("skill_x", "skill_z", "target"):
            tgt[f"{h}_idx"][:] = -1
        out.append({"obs": {"features": torch.rand(T, N_FEATURES, generator=g)}, "tgt": tgt,
                    "meta": {"episode": i}})
    return out


def test_train_multi_game_learns_hok_shards_in_chunks(tmp_path):
    """The converter's schema through train_multi_game: whole records longer than the
    chunk length (trained chunk by chunk), the shards' legal masks (--il_legal_mask)."""
    from data_processing.games.dataset import save_shard
    from training import train_multi_game

    g = torch.Generator().manual_seed(5)
    (tmp_path / "hok").mkdir()
    save_shard(str(tmp_path / "hok" / "a.pt"), "hok", _hok_records(3, (7, 20, 13), g),
               meta={"features": N_FEATURES})
    train_multi_game.main([
        "--data", f"hok={tmp_path / 'hok'}", "--steps", "3", "--warmup_steps", "1",
        "--eval_every", "0", "--device", "cpu", "--frames_per_batch", "16", "--chunk_len", "8",
        "--max_len", "64", "--il_legal_mask", "--save_dir", str(tmp_path / "ck"),
        "--override", *[f"{k}={v}" for k, v in TINY.items()]])
    assert (tmp_path / "ck" / "multi_final.pt").is_file()

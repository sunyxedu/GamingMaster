"""The candidate-action adapter (model/games/candidate.py): token layout, legality
masking of padded / illegal candidates, the IL contract (il_loss = Σ il_head_terms,
soft targets over candidate indices through train_il_game.batch_loss), rollout /
replay log-prob agreement, permutation equivariance over candidates, the capturable
step, and the shard-meta architecture keys."""

from __future__ import annotations

import gmsim
import pytest
import torch

from model.games import build_game_network
from model.games.candidate import CandidateConfig, CandidateDecoder, CandidateEncoder
from model.games.gmrules import CANDIDATE_META_ARCH, candidate_games
from training.train_il_game import (
    LABEL_KEYS,
    META_ARCH_KEYS,
    apply_meta_arch_defaults,
    batch_loss,
    check_meta_arch,
)

CPU = torch.device("cpu")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1)
DIMS = dict(cand_max_entities=4, cand_entity_features=3, cand_entity_types=2,
            cand_global_features=2, cand_max_actions=5, cand_action_features=3)


def _net(**kw):
    torch.manual_seed(0)
    return build_game_network("kuhn_poker", **{**TINY, **DIMS, **kw}).eval()


def _obs(B=2, T=3, n_legal=(3, 5), seed=0):
    g = torch.Generator().manual_seed(seed)
    E, A = DIMS["cand_max_entities"], DIMS["cand_max_actions"]
    am = torch.zeros(B, T, A, dtype=torch.bool)
    for b in range(B):
        am[b, :, :n_legal[b % len(n_legal)]] = True
    em = torch.rand(B, T, E, generator=g) > 0.3
    return {
        "entities": torch.randn(B, T, E, 3, generator=g) * em.unsqueeze(-1),
        "entity_mask": em,
        "entity_type": torch.randint(0, 2, (B, T, E), generator=g).int() * em,
        "globals": torch.randn(B, T, 2, generator=g),
        "actions": torch.randn(B, T, A, 3, generator=g) * am.unsqueeze(-1),
        "action_mask": am,
    }


def test_registration_everywhere():
    for g in ("kuhn_poker", "goofspiel"):
        assert g in candidate_games()
        assert LABEL_KEYS[g] == ()
        assert META_ARCH_KEYS[g] == CANDIDATE_META_ARCH
    from training.train_rl_joint import GAME_SPECS, CandidateSpec
    assert isinstance(GAME_SPECS["kuhn_poker"], CandidateSpec) and GAME_SPECS["kuhn_poker"].full_kl


def test_dims_default_to_the_simulator_spec():
    net = build_game_network("goofspiel", **TINY)
    spec = gmsim.game_info("goofspiel")["spec"]
    for meta_key, (field,) in CANDIDATE_META_ARCH.items():
        assert getattr(net.cfg, field) == spec["meta"][meta_key]
    assert net.obs_encoder.n_tokens == spec["meta"]["max_entities"] + 1 + spec["meta"]["max_actions"]


def test_token_layout_and_masking():
    net = _net()
    enc = net.obs_encoder
    assert isinstance(enc, CandidateEncoder) and isinstance(net.action_decoder, CandidateDecoder)
    assert enc.n_tokens == 4 + 1 + 5 and enc.positional_embedding().shape == (1, 10, 32)
    obs = _obs()
    pred = net(obs)
    assert pred["action_logits"].shape == (2, 3, 5)
    assert torch.equal(pred["action_mask"], obs["action_mask"])
    # garbage in absent entity slots / padded candidate slots changes nothing
    obs2 = {k: v.clone() for k, v in obs.items()}
    obs2["entities"][~obs["entity_mask"]] = 7.0
    obs2["entity_type"][~obs["entity_mask"]] = 1
    obs2["actions"][~obs["action_mask"]] = -3.0
    pred2 = net(obs2)
    assert torch.allclose(pred["action_logits"], pred2["action_logits"], atol=1e-6)
    # sampling never picks a padded / illegal candidate (also with a narrower legal mask)
    step_obs = {k: v[:, 0] for k, v in obs.items()}
    legal = step_obs["action_mask"].clone()
    legal[:, 0] = False
    torch.manual_seed(1)
    for _ in range(20):
        out = net.step(step_obs, legal={"action": legal})
        idx = out["action_idx"]
        assert bool(legal.gather(1, idx.unsqueeze(1)).all())


def test_il_loss_is_the_sum_of_head_terms_and_soft_targets():
    net = _net().train()
    obs = _obs()
    tgt = {"action_idx": torch.tensor([[0, 2, 1], [4, -1, 3]]),
           "value": torch.zeros(2, 3)}
    mask = torch.ones(2, 3, dtype=torch.bool)
    pred = net(obs)
    dec = net.action_decoder
    il = dec.il_loss(pred, tgt, mask)
    terms = dec.il_head_terms(pred, tgt, mask)
    t = terms["action"]
    assert torch.allclose(il["loss"], (t["ce"] * t["weight"]).sum() / t["norm"])
    assert t["weight"].tolist() == [[1, 1, 1], [1, 0, 1]]           # the −1 label is skipped
    assert il["answer_mask"].tolist() == [[True] * 3, [True, False, True]]
    # a label on a padded candidate is never supervised
    bad = {**tgt, "action_idx": torch.tensor([[4, 2, 1], [4, 0, 3]])}
    assert dec.il_head_terms(pred, bad, mask)["action"]["weight"][0, 0] == 0
    # soft targets: sparse == dense, one-hot teacher == hard label
    mb = {"obs": obs, "tgt": tgt, "frame_mask": mask, "labels": None, "legal": None}
    hard, _, _ = batch_loss(net, mb, CPU, 0.2, 0.0, False)
    onehot = torch.zeros(2, 3, 5)
    onehot.scatter_(-1, tgt["action_idx"].clamp(min=0).unsqueeze(-1), 1.0)
    onehot[1, 1] = 0.0
    mb1 = {**mb, "tgt": {**tgt, "action_probs": onehot}}
    same, stats, _ = batch_loss(net, mb1, CPU, 0.2, 0.0, False)
    assert torch.allclose(hard, same, atol=1e-5) and "action_teacher_kl" in stats
    teacher_idx = torch.tensor([[[0, 1], [2, 0], [1, 2]], [[4, 3], [0, 1], [3, -1]]])
    teacher_val = torch.tensor([[[0.5, 0.5], [0.9, 0.1], [0.3, 0.7]],
                                [[0.6, 0.4], [0.5, 0.5], [1.0, 0.0]]])
    dense = torch.zeros(2, 3, 5)
    dense.scatter_add_(-1, teacher_idx.clamp(min=0), teacher_val * (teacher_idx >= 0))
    sparse = {**tgt, "action_probs_idx": teacher_idx, "action_probs_val": teacher_val}
    ls, _, _ = batch_loss(net, {**mb, "tgt": sparse}, CPU, 0.2, 0.0, False)
    ld, _, _ = batch_loss(net, {**mb, "tgt": {**tgt, "action_probs": dense}}, CPU, 0.2, 0.0, False)
    assert torch.allclose(ls, ld, atol=1e-6) and not torch.allclose(ls, hard)
    ls.backward()
    assert net.action_decoder.pointer.net[0].weight.grad is not None


def test_sample_and_replay_define_the_same_policy():
    net = _net()
    obs = {k: v[:, 0] for k, v in _obs().items()}
    torch.manual_seed(0)
    out = net.step(obs, legal={"action": obs["action_mask"]})
    pred = net({k: v.unsqueeze(1) for k, v in obs.items()},
               legal={"action": obs["action_mask"].unsqueeze(1)})
    lp, ent, heads = net.action_decoder.log_prob_entropy(
        pred, {"action_idx": out["action_idx"].unsqueeze(1)},
        {"action": obs["action_mask"].unsqueeze(1)})
    assert torch.allclose(lp.squeeze(1), out["action_log_prob"], atol=1e-5)
    assert bool((ent >= 0).all()) and "ent_action" in heads
    # greedy = the argmax legal candidate
    net.action_decoder.greedy = True
    g = net.step(obs, legal={"action": obs["action_mask"]})
    logits = pred["action_logits"].squeeze(1).masked_fill(~obs["action_mask"], -1e9)
    assert torch.equal(g["action_idx"], logits.argmax(-1))


def test_candidate_permutation_equivariance():
    """Candidates share one positional embedding: permuting them permutes the logits."""
    net = _net(cand_entity_types=2)
    obs = {k: v[:, :1] for k, v in _obs(n_legal=(5,)).items()}
    perm = torch.tensor([3, 0, 4, 1, 2])
    pobs = dict(obs, actions=obs["actions"][:, :, perm], action_mask=obs["action_mask"][:, :, perm])
    a = net(obs)["action_logits"]
    b = net(pobs)["action_logits"]
    assert torch.allclose(a[:, :, perm], b, atol=1e-5)


def test_capturable_step_with_recurrent_cache():
    net = _net(temporal_mixer="recurrent")
    obs = {k: v[:, 0] for k, v in _obs().items()}
    cache = net.initial_st_cache(2)
    net.prepare_inference()
    with torch.no_grad():
        for t in range(3):
            out = net.step(obs, legal={"action": obs["action_mask"]}, st_cache=cache,
                           step_idx=torch.tensor([t]))
            cache = out["st_cache"]
    assert out["action_idx"].shape == (2,)


def test_meta_arch_from_shards():
    meta = {"max_entities": 6, "entity_features": 9, "entity_types": 0, "global_features": 3,
            "max_actions": 7, "action_features": 2, "n_seats": 2, "layout": "per_seat"}
    kw = apply_meta_arch_defaults("kuhn_poker", dict(TINY), meta)
    net = build_game_network("kuhn_poker", **kw)
    cfg = net.cfg
    assert (cfg.cand_max_entities, cfg.cand_max_actions, cfg.cand_entity_types) == (6, 7, 0)
    assert net.obs_encoder.type_embed is None
    check_meta_arch("kuhn_poker", net.cfg, meta)
    with pytest.raises(ValueError, match="max_actions=9"):
        check_meta_arch("kuhn_poker", net.cfg, {**meta, "max_actions": 9})
    with pytest.raises(ValueError, match=">= 1"):
        CandidateEncoder(CandidateConfig(cand_global_features=0))

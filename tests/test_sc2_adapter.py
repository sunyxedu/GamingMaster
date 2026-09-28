"""StarCraft II adapter (model/games/sc2.py) on synthetic gmsc2 frames: shapes, the
IL-loss / head-term contract, the rollout log-prob replayed exactly, and teacher forcing."""

from __future__ import annotations

import pytest
import torch

from model.games.registry import build_game_network
from model.games.sc2 import (
    DONE,
    F_UNIT,
    G,
    KIND_POINT,
    KIND_UNIT,
    LOC,
    MAP,
    MAP_BITS,
    coarse_of,
    fine_of,
    point_of,
    target_kind_table,
)

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0, encoder_channels=8,
            core_type="transformer", core_layers=1, sc2_units=40, sc2_unit_vocab=30, sc2_action_vocab=20,
            sc2_buff_vocab=10, sc2_upgrades=6)


def _obs(B, T, cfg, g):
    U = cfg.sc2_units
    mask = torch.rand(B, T, U, generator=g) < 0.6
    own = mask & (torch.rand(B, T, U, generator=g) < 0.5)
    units = torch.randint(0, 256, (B, T, U, F_UNIT), generator=g).to(torch.uint8)
    units[..., 0] = torch.where(own, 0, torch.randint(2, 4, (B, T, U), generator=g)).to(torch.uint8)
    units[..., 1:3] = torch.randint(0, 4, (B, T, U, 2), generator=g).to(torch.uint8)
    units = units * mask.unsqueeze(-1)
    return {
        "units": units,
        "unit_xy": torch.randint(0, 65536, (B, T, U, 2), generator=g).to(torch.int32),
        "unit_type": (torch.randint(2, cfg.sc2_unit_vocab, (B, T, U), generator=g) * mask).to(torch.int32),
        "unit_order": torch.randint(0, cfg.sc2_action_vocab, (B, T, U), generator=g).to(torch.int32),
        "unit_buffs": torch.randint(0, cfg.sc2_buff_vocab, (B, T, U, 2), generator=g).to(torch.int32),
        "unit_mask": mask,
        "unit_own": own,
        "unit_commanded": (torch.rand(B, T, U, generator=g) < 0.1).float() * own,
        "map_bits": (torch.rand(B, T, MAP_BITS, MAP, MAP, generator=g) < 0.3).float(),
        "globals": torch.rand(B, T, G, generator=g),
        "upgrades": (torch.rand(B, T, cfg.sc2_upgrades, generator=g) < 0.2).float(),
        "last_action": torch.randint(0, cfg.sc2_action_vocab, (B, T), generator=g),
        "window_start": torch.rand(B, T, generator=g) < 0.5,
    }


def _tgt(obs, cfg, g):
    B, T, U = obs["unit_mask"].shape
    kinds = target_kind_table(cfg.sc2_action_vocab)
    typ = torch.randint(2, cfg.sc2_action_vocab, (B, T), generator=g)
    typ = torch.where(torch.rand(B, T, generator=g) < 0.4, torch.zeros_like(typ), typ)
    kind = torch.randint(0, 3, (B, T), generator=g)
    # a kind the type takes (the first allowed one), none for done
    allowed = kinds[typ]
    kind = torch.where(allowed.gather(-1, kind.unsqueeze(-1)).squeeze(-1), kind, allowed.float().argmax(-1))
    kind = torch.where(typ == DONE, torch.zeros_like(kind), kind)
    present = obs["unit_mask"]
    unit = torch.multinomial(present.reshape(-1, U).float() + 1e-6, 1, generator=g).reshape(B, T)
    point = torch.randint(0, LOC * LOC, (B, T), generator=g)
    actors = (torch.rand(B, T, U, generator=g) < 0.3) & obs["unit_own"] & (typ >= 2).unsqueeze(-1)
    return {"action_type": typ, "actors": actors.float(), "target_kind": kind,
            "target_unit": torch.where(kind == KIND_UNIT, unit, torch.full_like(unit, -1)),
            "target_point": torch.where(kind == KIND_POINT, point, torch.full_like(point, -1)),
            "queued": torch.rand(B, T, generator=g) < 0.2, "value": torch.ones(B, T)}


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_game_network("sc2", **TINY).eval()


def _forward(net, obs, labels=None):
    return net(obs, labels=labels)


def test_point_grid_round_trips():
    cfg = build_game_network("sc2", **TINY).cfg
    p = torch.arange(0, LOC * LOC, 97)
    assert torch.equal(point_of(coarse_of(p, cfg), fine_of(p, cfg), cfg), p)
    assert int(coarse_of(torch.tensor(LOC * LOC - 1), cfg)) == cfg.sc2_map_tokens - 1


def test_target_kinds_come_from_the_vocabulary():
    tab = target_kind_table(1773)
    assert tab[DONE].tolist() == [True, False, False]
    # Smart (ability 1, the vocabulary's first action → id 2) takes a point or a unit
    assert tab[2].tolist() == [False, True, True]


def test_il_loss_is_the_sum_of_its_head_terms(net):
    g = torch.Generator().manual_seed(1)
    obs = _obs(2, 3, net.cfg, g)
    tgt = _tgt(obs, net.cfg, g)
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask[1, 2] = False
    labels = {k: tgt[k] for k in ("action_type", "actors", "target_point")}
    out = net(obs, labels=labels)
    pred = out["pred"] if "pred" in out else out
    dec = net.action_decoder
    il = dec.il_loss(pred, tgt, mask)
    terms = dec.il_head_terms(pred, tgt, mask)
    assert set(terms) == {"type", "kind", "actors", "queued", "unit", "coarse", "fine"}
    total = 0.0
    for h, t in terms.items():
        kind = t.get("kind", "categorical")
        wshape = t["logits"].shape[:-1] if kind == "categorical" else t["logits"].shape
        assert t["weight"].shape == wshape and t["ce"].shape == wshape, h
        assert not t["weight"].requires_grad and t["ce"].requires_grad, h
        assert torch.equal(t["norm"], t["weight"].sum().clamp(min=1)), h
        part = (t["ce"] * t["weight"]).sum() / t["norm"]
        assert float(part) == pytest.approx(float(il[f"{h}_loss"]), abs=1e-6), h
        total = total + part
    assert float(total) == pytest.approx(float(il["loss"]), abs=1e-5)
    assert torch.isfinite(il["loss"])
    il["loss"].backward()


def test_rollout_log_prob_replays_exactly(net):
    g = torch.Generator().manual_seed(2)
    obs = _obs(4, 1, net.cfg, g)
    torch.manual_seed(3)
    with torch.no_grad():
        out = net(obs)
        pred = out["pred"] if "pred" in out else out
        s = net.action_decoder.sample({k: v[:, 0] for k, v in pred.items()})
    actions = {"action_type": s["action_type"], "actors": s["action_actors"], "target_kind": s["action_target_kind"],
               "target_unit": s["action_target_unit"], "target_point": s["action_target_point"], "queued": s["action_queued"]}
    # the replay teacher-forces the committed type, actors and coarse cell
    labels = {"action_type": actions["action_type"].unsqueeze(1), "actors": actions["actors"].float().unsqueeze(1),
              "target_point": actions["target_point"].unsqueeze(1)}
    with torch.no_grad():
        out2 = net(obs, labels=labels)
        pred2 = out2["pred"] if "pred" in out2 else out2
        lp, ent, head = net.action_decoder.log_prob_entropy(pred2, {k: v.unsqueeze(1) for k, v in actions.items()})
    torch.testing.assert_close(lp[:, 0], s["action_log_prob"], rtol=0, atol=1e-4)
    assert torch.isfinite(ent).all()
    # actors are own units only, and none on a done decision
    own = obs["unit_own"][:, 0]
    assert not (s["action_actors"] & ~own).any()
    assert not s["action_actors"][s["action_type"] == DONE].any()


def test_teacher_forcing_commits_the_labels(net):
    g = torch.Generator().manual_seed(4)
    obs = _obs(2, 2, net.cfg, g)
    tgt = _tgt(obs, net.cfg, g)
    labels = {k: tgt[k] for k in ("action_type", "actors", "target_point")}
    with torch.no_grad():
        out = net(obs, labels=labels)
        pred = out["pred"] if "pred" in out else out
    assert torch.equal(pred["type_idx"], tgt["action_type"])
    assert torch.equal(pred["actors_used"], tgt["actors"] > 0.5)
    has_pt = tgt["target_point"] >= 0
    assert torch.equal(pred["coarse_idx"][has_pt], coarse_of(tgt["target_point"][has_pt], net.cfg))
    # a −1 label (dropped by the teacher-forcing schedule) samples instead
    labels["actors"] = torch.full_like(labels["actors"], -1)
    with torch.no_grad():
        out = net(obs, labels=labels)
        pred = out["pred"] if "pred" in out else out
    assert not (pred["actors_used"] & ~obs["unit_own"]).any()

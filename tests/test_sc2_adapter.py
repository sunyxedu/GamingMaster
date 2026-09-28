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


def test_fine_cell_features_locate_units_and_pixels():
    from model.games.sc2 import N_FINE_FEATS, Sc2Config, fine_cell_features

    cfg = Sc2Config(**{k: v for k, v in TINY.items() if k.startswith("sc2_")})
    n, cell = MAP // cfg.sc2_map_patch, LOC // (MAP // cfg.sc2_map_patch)
    U = cfg.sc2_units
    xy = torch.zeros(1, U, 2, dtype=torch.int32)
    # unit 0 (own, an actor) at LOC point (x=37, y=21): coarse token (1, 2), fine cell (5, 5)
    xy[0, 0] = torch.tensor([round((37 + 0.5) / LOC * 65535), round((21 + 0.5) / LOC * 65535)])
    # unit 1 (enemy) in another token
    xy[0, 1] = torch.tensor([60000, 60000])
    present = torch.zeros(1, U, dtype=torch.bool)
    present[0, :2] = True
    own = torch.zeros(1, U, dtype=torch.bool)
    own[0, 0] = True
    bits = torch.zeros(1, MAP_BITS, MAP, MAP)
    bits[0, 3, 21 * MAP // LOC, 37 * MAP // LOC] = 1.0         # pathable under the unit's pixel
    ctx = {"unit_xy": xy, "present": present, "own": own, "map_bits": bits}
    coarse = coarse_of(torch.tensor([21 * LOC + 37]), cfg)
    assert int(coarse) == (21 // cell) * n + 37 // cell
    feats = fine_cell_features(ctx, coarse, own.clone(), cfg)
    assert feats.shape == (1, cell * cell, N_FINE_FEATS)
    f = int(fine_of(torch.tensor([21 * LOC + 37]), cfg))
    assert feats[0, f, 0] == 0.5 and feats[0, f, 2] == 0.5 and feats[0, :, 1].sum() == 0
    assert feats[0, :, :3].sum() == 1.0                              # only that cell
    assert feats[0, f, 3 + 3] == 1.0                                 # its pixel's pathable bit
    # the 4 x 4 fine cells of that pixel share it, no other cell has it
    assert feats[0, :, 3 + 3].sum() == (cell // cfg.sc2_map_patch) ** 2
    assert feats[0, f, -1] == 1.0 and feats[0, f, 3 + MAP_BITS:3 + MAP_BITS + 2].abs().max() < 0.1   # at the centroid


def test_fine_addon_starts_at_zero_and_learns_the_actor_cell():
    """The fine head alone cannot place a point inside its map token (the patch
    embedding blurs sub-token positions: SC2 val fine_loss stayed at ln 256); with the
    add-on it learns a target at the commanding unit's own position."""
    torch.manual_seed(0)
    net = build_game_network("sc2", **TINY)
    dec = net.action_decoder
    assert float(dec.addon_fine.out.weight.abs().sum() + dec.addon_fine.out.bias.abs().sum()) == 0.0
    g = torch.Generator().manual_seed(3)
    B, T = 4, 3
    obs = _obs(B, T, net.cfg, g)
    tgt = _tgt(obs, net.cfg, g)
    # every frame: a point command of ONE own unit whose target is the unit's own position
    U = net.cfg.sc2_units
    first = torch.argmax(obs["unit_own"].float(), dim=-1)            # [B, T]
    actors = torch.zeros(B, T, U)
    actors.scatter_(-1, first.unsqueeze(-1), 1.0)
    uxy = obs["unit_xy"].gather(2, first[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    px = (uxy[..., 0].float() / 65535 * LOC).long().clamp(0, LOC - 1)
    py = (uxy[..., 1].float() / 65535 * LOC).long().clamp(0, LOC - 1)
    kinds = target_kind_table(net.cfg.sc2_action_vocab)
    typ = int(torch.nonzero(kinds[2:, KIND_POINT]).reshape(-1)[0]) + 2
    tgt.update(action_type=torch.full((B, T), typ), target_kind=torch.full((B, T), KIND_POINT),
               target_point=py * LOC + px, actors=actors, target_unit=torch.full((B, T), -1))
    labels = {k: tgt[k] for k in ("action_type", "actors", "target_point")}
    mask = torch.ones(B, T, dtype=torch.bool)

    def fine_loss(params, steps):
        opt = torch.optim.Adam(params, lr=3e-2)
        out = []
        for _ in range(steps):
            res = net(obs, labels=labels)
            t = dec.il_head_terms(res["pred"] if "pred" in res else res, tgt, mask)["fine"]
            loss = (t["ce"] * t["weight"]).sum() / t["norm"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            out.append(float(loss))
        return out

    net.train()
    with_addon = fine_loss([p for n, p in net.named_parameters() if "addon_fine" in n], 80)
    assert with_addon[0] > 5.0 and with_addon[-1] < 1.5, with_addon[::10]


def test_checkpoints_without_the_addon_load():
    """A checkpoint older than an add-on (no addon_* keys) loads strictly through
    with_addon_defaults and resumes through load_resume_weights; a PARTLY present add-on
    is refused."""
    from model.multi_game import MultiGameNetwork, missing_addon_keys, with_addon_defaults
    from training.train_multi_game import load_resume_weights

    torch.manual_seed(0)
    model = MultiGameNetwork({"sc2": build_game_network("sc2", **TINY)})
    old = {k: v.clone() for k, v in model.state_dict().items() if ".addon_" not in k}
    assert missing_addon_keys(model.state_dict(), old) == {k for k in model.state_dict() if ".addon_" in k}
    model.load_state_dict(with_addon_defaults(model, old))
    load_resume_weights(model, old, [])
    partial = dict(old)
    k = next(k for k in model.state_dict() if ".addon_fine.out.weight" in k)
    partial[k] = model.state_dict()[k]
    with pytest.raises(SystemExit):
        load_resume_weights(model, partial, [])

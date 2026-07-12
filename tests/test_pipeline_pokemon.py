"""End-to-end tests for the Pokémon Showdown data pipeline
(replay JSON sim-protocol logs → per-turn observer-view shards).

Covers:
  1. protocol parser state evolution on an inline mini log: switch/move/
     damage/faint/boost/status/weather/hazards/tera/win → hp fracs,
     active slots, revealed-move labels, boosts, statuses
  2. label conventions: move_idx = revealed-move index, switch_idx = own
     team slot, tera flag, skips (forced replacement / cant / struggle)
  3. value is ±1 from |win| for the sample's side; no winner → no samples
  4. the 3 real gen9ou replays (tests/fixtures/showdown) parse without crash
     and produce ≥1 sample per side
  5. shard → GameShardDataset → collate → GameNetwork(PokemonConfig tiny)
     forward + il_loss finite + backward
  6. c8f7b28 pins: obs["move_ids"] labeled-slot identity invariant,
     forced-pivot switches (Emergency Exit / Eject Button) never labeled,
     a 5th revealed move labels −1, shard meta carries the bucket config
"""

import json
import os

import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences
from data_processing.games.pokemon.convert import convert
from data_processing.games.pokemon.protocol import (
    POKE_FEAT_ACTIVE,
    POKE_FEAT_BOOSTS,
    POKE_FEAT_FAINTED,
    POKE_FEAT_HP,
    POKE_FEAT_REVEALED,
    POKE_FEAT_STATUS,
    POKE_FEAT_TERA,
    move_to_id,
    parse_log,
    replay_to_samples,
    species_to_id,
)
from model.game_network import GameNetwork
from model.games.pokemon import PokemonConfig, PokemonDecoder, PokemonEncoder

REAL_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "showdown")

# Inline mini battle: team preview, leads, switch/move/damage/boost/
# status/weather/hazards/faint/tera/win — hand-checkable line by line.
MINI_LOG = "\n".join([
    "|player|p1|Alice|1|1500",
    "|player|p2|Bob|2|1490",
    "|gen|9",
    "|clearpoke",
    "|poke|p1|Garchomp, M|",
    "|poke|p1|Rotom-Wash|",
    "|poke|p1|Kingambit, F|",
    "|poke|p2|Heatran, M|",
    "|poke|p2|Weavile, F|",
    "|teampreview",
    "|start",
    "|switch|p1a: Chompy|Garchomp, M|100/100",
    "|switch|p2a: Heatran|Heatran, M|100/100",
    "|turn|1",
    # T1: p1 Swords Dance (move 0, boosts +2 atk), p2 switches to Weavile (slot 1)
    "|switch|p2a: Weavile|Weavile, F|100/100",
    "|move|p1a: Chompy|Swords Dance|p1a: Chompy",
    "|-boost|p1a: Chompy|atk|2",
    "|upkeep",
    "|turn|2",
    # T2: p1 Earthquake (move 1) damages Weavile; p2 Ice Shard burns? no — status via Scald below
    "|move|p2a: Weavile|Ice Shard|p1a: Chompy",
    "|-supereffective|p1a: Chompy",
    "|-damage|p1a: Chompy|20/100",
    "|move|p1a: Chompy|Earthquake|p2a: Weavile",
    "|-damage|p2a: Weavile|30/100",
    "|-weather|Snow",
    "|upkeep",
    "|turn|3",
    # T3: p1 switches to Rotom (slot 1); p2 sets Spikes (move 1)
    "|switch|p1a: Wash|Rotom-Wash|100/100",
    "|move|p2a: Weavile|Spikes|p1a: Wash",
    "|-sidestart|p1: Alice|Spikes",
    "|upkeep",
    "|turn|4",
    # T4: p2 terastallizes + Ice Shard (move 0); p1 Scald (move 0) burns Weavile
    "|-terastallize|p2a: Weavile|Ice",
    "|move|p2a: Weavile|Ice Shard|p1a: Wash",
    "|-damage|p1a: Wash|70/100",
    "|move|p1a: Wash|Scald|p2a: Weavile",
    "|-damage|p2a: Weavile|5/100",
    "|-status|p2a: Weavile|brn",
    "|-damage|p2a: Weavile|0 fnt",
    "|faint|p2a: Weavile",
    "|upkeep",
    "|switch|p2a: Heatran|Heatran, M|100/100",   # forced replacement — skipped
    "|turn|5",
    # T5: p1 Scald again (move 0 — already revealed); p2 Magma Storm (move 0)
    "|move|p1a: Wash|Scald|p2a: Heatran",
    "|-damage|p2a: Heatran|60/100",
    "|move|p2a: Heatran|Magma Storm|p1a: Wash",
    "|-damage|p1a: Wash|10/100",
    "|upkeep",
    "|turn|6",
    # T6: p1 cant (frz? say par); p2 Magma Storm
    "|cant|p1a: Wash|par",
    "|move|p2a: Heatran|Magma Storm|p1a: Wash",
    "|-damage|p1a: Wash|0 fnt",
    "|faint|p1a: Wash",
    "|win|Bob",
])


def mini_replay():
    return {"id": "test-mini-1", "log": MINI_LOG, "rating": 1500,
            "formatid": "gen9ou", "players": ["Alice", "Bob"]}


# ======================================================================
# 1+2. Protocol parsing: state evolution + labels on the mini log
# ======================================================================

def test_mini_log_labels_and_counts():
    res = parse_log(MINI_LOG)
    assert res.winner == "p2"
    assert res.players == {"p1": "Alice", "p2": "Bob"}
    assert res.turns == 6

    p1 = res.records["p1"]
    # p1 acts visibly on turns 1-5 (turn 6 is |cant|)
    assert [r.turn for r in p1] == [1, 2, 3, 4, 5]
    assert [r.action_type for r in p1] == [0, 0, 1, 0, 0]
    # Swords Dance → 0, Earthquake → 1 (revealed order), Scald → 0 for Rotom
    assert [r.move_idx for r in p1] == [0, 1, -1, 0, 0]
    assert p1[2].switch_idx == 1                      # Rotom-Wash = own slot 1
    assert all(r.tera == 0 for r in p1)

    p2 = res.records["p2"]
    assert [r.turn for r in p2] == [1, 2, 3, 4, 5, 6]
    assert [r.action_type for r in p2] == [1, 0, 0, 0, 0, 0]
    assert p2[0].switch_idx == 1                      # Weavile = p2 slot 1
    # Ice Shard → 0, Spikes → 1, Ice Shard → 0, Magma Storm → 0 (Heatran)
    assert [r.move_idx for r in p2] == [-1, 0, 1, 0, 0, 0]
    assert p2[3].tera == 1 and p2[4].tera == 0        # tera exactly on turn 4

    # skips: p2's forced replacement after Weavile fainted, p1's |cant|
    assert res.skipped.get("forced_replacement_switch") == 1
    assert res.skipped.get("cant") == 1
    # handled lines produce no unknown entries for the state-relevant set
    assert "-damage" not in res.unknown and "switch" not in res.unknown


def test_mini_log_state_evolution():
    res = parse_log(MINI_LOG)
    p1 = res.records["p1"]
    p2 = res.records["p2"]

    # p1 frame @T1: own Garchomp (slot 0) active, full hp, all revealed=own lead only
    f1 = p1[0].frame
    feats = f1["poke_feats"]
    assert f1["active_idx"].item() == 0
    assert feats[0, POKE_FEAT_ACTIVE] == 1.0
    assert feats[0, POKE_FEAT_HP] == 1.0
    assert feats[0, POKE_FEAT_REVEALED] == 1.0
    assert feats[1, POKE_FEAT_REVEALED] == 0.0        # Rotom not yet seen
    # opponent side: 2 known species (mini teams of 3/2), Heatran active
    assert f1["slot_mask"][:3].all() and f1["slot_mask"][6:8].all()
    assert not f1["slot_mask"][3:6].any() and not f1["slot_mask"][8:12].any()
    assert feats[6, POKE_FEAT_ACTIVE] == 1.0          # Heatran = opp slot 0

    # p1 frame @T2: own atk boost +2/6 from Swords Dance; opp active = Weavile
    f2 = p1[1].frame
    assert f2["poke_feats"][0, POKE_FEAT_BOOSTS + 0] == pytest.approx(2 / 6)
    assert f2["poke_feats"][7, POKE_FEAT_ACTIVE] == 1.0

    # p1 frame @T3: Garchomp hp 20/100; weather=snow one-hot idx 3
    f3 = p1[2].frame
    assert f3["poke_feats"][0, POKE_FEAT_HP] == pytest.approx(0.20)
    assert f3["field_seq"][3] == 1.0

    # p1 frame @T4: boosts reset after Garchomp switched out; own spikes layer 1
    f4 = p1[3].frame
    assert f4["active_idx"].item() == 1               # Rotom now active
    assert f4["poke_feats"][0, POKE_FEAT_BOOSTS + 0] == 0.0
    assert f4["field_seq"][10] == pytest.approx(1 / 3)   # own spikes/3

    # p1 frame @T5: opp Weavile fainted + burned + tera'd; Heatran back active
    f5 = p1[4].frame
    assert f5["poke_feats"][7, POKE_FEAT_FAINTED] == 1.0
    assert f5["poke_feats"][7, POKE_FEAT_HP] == 0.0
    assert f5["poke_feats"][7, POKE_FEAT_STATUS + 1] == 1.0   # brn
    assert f5["poke_feats"][7, POKE_FEAT_TERA] == 1.0
    assert f5["poke_feats"][6, POKE_FEAT_ACTIVE] == 1.0

    # p2 frame @T5 (their own perspective mirrors: own slot 0 = Heatran)
    g5 = p2[4].frame
    assert g5["active_idx"].item() == 0
    assert g5["poke_feats"][1, POKE_FEAT_FAINTED] == 1.0      # own Weavile


def test_mini_replay_samples_value_and_tensors():
    samples, res = replay_to_samples(mini_replay(), species_buckets=64)
    assert len(samples) == 2
    by_side = {s["meta"]["side"]: s for s in samples}
    p1, p2 = by_side["p1"], by_side["p2"]

    assert p1["obs"]["poke_feats"].shape == (5, 12, 17)
    assert p2["obs"]["poke_feats"].shape == (6, 12, 17)
    assert (p1["tgt"]["value"] == -1.0).all()          # Alice lost
    assert (p2["tgt"]["value"] == 1.0).all()           # Bob won
    assert p1["tgt"]["action_type"].tolist() == [0, 0, 1, 0, 0]
    assert p1["tgt"]["active_idx"].tolist() == [0, 0, 0, 1, 1]
    assert p2["tgt"]["tera"].tolist() == [0, 0, 0, 1, 0, 0]

    # species ids: stable hash, own slot 0 = Garchomp for p1 and opp slot 6 for p2
    gid = species_to_id("Garchomp", 64)
    assert gid != 0
    assert (p1["obs"]["species_ids"][:, 0] == gid).all()
    assert (p2["obs"]["species_ids"][:, 6] == gid).all()


def test_no_winner_no_samples():
    log = MINI_LOG.rsplit("\n|win|Bob", 1)[0]
    samples, res = replay_to_samples({"id": "x", "log": log})
    assert res.winner is None
    assert samples == []


# ======================================================================
# 6. c8f7b28 pins: move_ids identity / forced pivots / 5th reveal / meta
# ======================================================================

def _first_moves_by_turn(log):
    """(side, turn) → the side's first genuine ``|move|`` name that turn —
    the same line the parser labels ([from]-called moves skipped)."""
    used, turn = {}, 0
    for line in log.split("\n"):
        parts = line.split("|")
        cmd = parts[1] if len(parts) > 1 else ""
        if cmd == "turn":
            turn = int(parts[2])
        elif cmd == "move" and turn > 0 and len(parts) >= 4:
            if any("[from]" in a for a in parts[4:]):
                continue
            used.setdefault((parts[2][:2], turn), parts[3])
    return used


def test_move_ids_labeled_slot_identity():
    """The by-identity scoring contract: at every move-labeled frame the
    observed id at the labeled slot is either 0 (first reveal — the label
    points at the next free slot) or exactly the hash-id of the move the
    expert used, never a different move's identity."""
    logs = [("mini", MINI_LOG)]
    for name in sorted(os.listdir(REAL_DIR)):
        with open(os.path.join(REAL_DIR, name), encoding="utf-8") as fh:
            logs.append((name, json.load(fh)["log"]))

    n_repeat_use = 0
    for name, log in logs:
        used = _first_moves_by_turn(log)
        samples, res = replay_to_samples(
            {"id": name, "log": log}, species_buckets=64, move_buckets=256)
        for s in samples:
            side = s["meta"]["side"]
            recs = res.records[side]
            at, mi = s["tgt"]["action_type"], s["tgt"]["move_idx"]
            assert len(recs) == at.shape[0]
            for t, rec in enumerate(recs):
                if at[t].item() != 0 or mi[t].item() < 0:
                    continue
                slot_ids = s["obs"]["move_ids"][t]
                labeled = slot_ids[mi[t]].item()
                expect = move_to_id(used[(side, rec.turn)], 256)
                assert labeled in (0, expect), (name, side, rec.turn)
                if labeled == 0:
                    # first reveal: all slots before the labeled one filled
                    assert (slot_ids[: mi[t]] != 0).all(), (name, side, rec.turn)
                else:
                    n_repeat_use += 1
    assert n_repeat_use > 0        # exercised on genuine repeat uses


@pytest.mark.parametrize("trigger", [
    "|-activate|p2a: Goli|ability: Emergency Exit",
    "|-enditem|p2a: Goli|Eject Button",
])
def test_forced_pivot_switch_not_labeled(trigger):
    """A mid-turn forced pivot (Emergency Exit / Eject Button) must not be
    emitted as a labeled switch decision; the surrounding genuine
    decisions must all be labeled."""
    log = "\n".join([
        "|player|p1|Alice|1|1500",
        "|player|p2|Bob|2|1490",
        "|gen|9",
        "|poke|p1|Garchomp, M|",
        "|poke|p1|Rotom-Wash|",
        "|poke|p2|Golisopod, M|",
        "|poke|p2|Heatran, M|",
        "|teampreview",
        "|start",
        "|switch|p1a: Chompy|Garchomp, M|100/100",
        "|switch|p2a: Goli|Golisopod, M|100/100",
        "|turn|1",
        # T1: p1 Earthquake triggers the pivot; p2's replacement switch is
        # forced (their real decision — a cancelled move — never logged)
        "|move|p1a: Chompy|Earthquake|p2a: Goli",
        "|-damage|p2a: Goli|40/100",
        trigger,
        "|switch|p2a: Tran|Heatran, M|100/100",
        "|upkeep",
        "|turn|2",
        # T2: both sides act normally — genuine decisions
        "|move|p1a: Chompy|Earthquake|p2a: Tran",
        "|-damage|p2a: Tran|60/100",
        "|move|p2a: Tran|Magma Storm|p1a: Chompy",
        "|-damage|p1a: Chompy|50/100",
        "|upkeep",
        "|turn|3",
        "|move|p1a: Chompy|Earthquake|p2a: Tran",
        "|-damage|p2a: Tran|0 fnt",
        "|faint|p2a: Tran",
        "|win|Alice",
    ])
    res = parse_log(log)
    assert res.skipped.get("forced_pivot_switch") == 1
    # p2's only labeled decision is the genuine turn-2 move
    p2 = res.records["p2"]
    assert [r.turn for r in p2] == [2]
    assert p2[0].action_type == 0 and p2[0].move_idx == 0
    # p1's genuine decisions all present
    p1 = res.records["p1"]
    assert [r.turn for r in p1] == [1, 2, 3]
    assert all(r.action_type == 0 for r in p1)


FIFTH_MOVES = ["Swords Dance", "Earthquake", "Scale Shot", "Fire Fang",
               "Stone Edge"]


def test_fifth_revealed_move_labels_unsupervisable():
    """A 5th revealed move (Transform-style dynamic movesets) has no
    observed slot: the label must be −1, not clamped to 3 — slot 3's
    observed identity is the 4th move, not the one used."""
    lines = [
        "|player|p1|Alice|1|1500",
        "|player|p2|Bob|2|1490",
        "|gen|9",
        "|poke|p1|Garchomp, M|",
        "|poke|p2|Corviknight, M|",
        "|teampreview",
        "|start",
        "|switch|p1a: Chompy|Garchomp, M|100/100",
        "|switch|p2a: Corv|Corviknight, M|100/100",
    ]
    for t, mv in enumerate(FIFTH_MOVES, start=1):
        lines += [
            f"|turn|{t}",
            f"|move|p1a: Chompy|{mv}|p2a: Corv",
            "|move|p2a: Corv|Roost|p2a: Corv",
            "|upkeep",
        ]
    lines.append("|win|Alice")
    log = "\n".join(lines)

    res = parse_log(log)
    p1 = res.records["p1"]
    assert [r.action_type for r in p1] == [0] * 5
    assert [r.move_idx for r in p1] == [0, 1, 2, 3, -1]

    samples, _ = replay_to_samples(
        {"id": "fifth", "log": log}, species_buckets=64, move_buckets=256)
    p1s = next(s for s in samples if s["meta"]["side"] == "p1")
    assert p1s["tgt"]["action_type"].tolist() == [0] * 5
    assert p1s["tgt"]["move_idx"].tolist() == [0, 1, 2, 3, -1]
    # the −1 frame observes the FIRST four moves; the used move's id is
    # in none of the slots, so any slot label would be a wrong identity
    ids5 = p1s["obs"]["move_ids"][4].tolist()
    assert ids5 == [move_to_id(m, 256) for m in FIFTH_MOVES[:4]]
    assert move_to_id(FIFTH_MOVES[4], 256) not in ids5


def test_shard_meta_pins_bucket_config(real_shard_dir):
    """save_shard meta must record the hash-bucket sizes so trainer-side
    validation against the model config is possible."""
    expected = {"move_buckets": 256, "species_buckets": 64}
    names = [n for n in sorted(os.listdir(real_shard_dir)) if n.endswith(".pt")]
    assert names
    shard = torch.load(os.path.join(real_shard_dir, names[0]),
                       map_location="cpu", weights_only=False)
    assert shard["meta"] == expected
    assert GameShardDataset(real_shard_dir, game="pokemon").shard_meta == expected


# ======================================================================
# 3+4. Real replays parse and convert
# ======================================================================

@pytest.fixture(scope="module")
def real_shard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("pokemon_shards")
    stats = convert(REAL_DIR, str(out), shard_size=256, species_buckets=64)
    assert stats["seen"] == 3 and stats["kept"] == 3
    assert stats["samples"] == 6 and stats["shards"] == 1
    assert stats["frames"] > 0
    return str(out)


def test_real_replays_parse():
    paths = sorted(os.listdir(REAL_DIR))
    assert len(paths) == 3
    for name in paths:
        with open(os.path.join(REAL_DIR, name), encoding="utf-8") as fh:
            replay = json.load(fh)
        samples, res = replay_to_samples(replay, species_buckets=512)
        assert res.winner in ("p1", "p2"), name
        assert len(samples) >= 1, name
        for s in samples:
            T = s["tgt"]["action_type"].shape[0]
            assert T >= 1
            assert s["obs"]["slot_mask"].all()          # team preview: all known
            hp = s["obs"]["poke_feats"][..., 0]
            assert ((hp >= 0) & (hp <= 1)).all()
            at = s["tgt"]["action_type"]
            mi, si = s["tgt"]["move_idx"], s["tgt"]["switch_idx"]
            assert ((at == 0) | (at == 1)).all()
            assert ((mi[at == 0] >= 0) & (mi[at == 0] <= 3)).all()
            assert ((si[at == 1] >= 0) & (si[at == 1] <= 5)).all()
            assert ((s["tgt"]["active_idx"] >= 0) & (s["tgt"]["active_idx"] <= 5)).all()


def test_real_replay_min_rating_filter(tmp_path):
    # ratings of the three replays: 1293 / 1183 / 1073
    stats = convert(REAL_DIR, str(tmp_path / "o"), min_rating=1200)
    assert stats["kept"] == 1 and stats["low_rating"] == 2


# ======================================================================
# 5. E2E: shard → dataset → collate → network → il_loss → backward
# ======================================================================

def test_e2e_dataset_to_gradients(real_shard_dir):
    torch.manual_seed(0)
    ds = GameShardDataset(real_shard_dir, game="pokemon")
    assert len(ds) == 6
    batch = collate_game_sequences([ds[i] for i in range(len(ds))])
    frame_mask = batch["meta"]["frame_mask"]
    B, T = frame_mask.shape
    assert batch["obs"]["poke_feats"].shape == (B, T, 12, 17)
    assert batch["obs"]["species_ids"].shape == (B, T, 12)
    assert batch["obs"]["field_seq"].shape == (B, T, 20)
    assert frame_mask.sum() == sum(batch["meta"]["lengths"])

    cfg = PokemonConfig(
        d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, dropout=0.0, pokemon_species_buckets=64,
    )
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))
    net.train()
    labels = {"action_type": batch["tgt"]["action_type"],
              "active_idx": batch["tgt"]["active_idx"]}
    pred = net(batch["obs"], labels=labels, frame_mask=frame_mask)
    losses = net.action_decoder.il_loss(pred, batch["tgt"], frame_mask)
    mask_f = frame_mask.float()
    v_loss = (torch.nn.functional.mse_loss(
        pred["value"], batch["tgt"]["value"], reduction="none",
    ) * mask_f).sum() / mask_f.sum()
    loss = losses["loss"] + 0.2 * v_loss

    assert torch.isfinite(loss), losses
    for k in ("type_loss", "move_loss", "switch_loss", "tera_loss"):
        assert torch.isfinite(losses[k]), (k, losses[k])
    loss.backward()

    got_grad = False
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
            got_grad = got_grad or p.grad.abs().sum() > 0
    assert got_grad

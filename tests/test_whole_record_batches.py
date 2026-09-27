"""Whole-record training (WholeRecordBatchSampler, train_multi_game --frames_per_batch):
every trained frame carries its record's whole history — batches of whole records from
their first frame, sized by a frame budget, as in play (the network's only memory is
its recurrent state, which a cropped window would restart mid-game); a record longer
than --chunk_len trains chunk by chunk with that state carried (truncated BPTT)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data_processing.games.dataset import (GameShardDataset, WholeRecordBatchSampler,
                                           save_shard, whole_record_batches)


def _records(lengths, start_id=0):
    out = []
    for i, T in enumerate(lengths):
        out.append({"obs": {"board_seq": torch.rand(T, 3, 8, 8)},
                    "tgt": {"place_idx": torch.arange(T) % 64,
                            "passed": torch.zeros(T, dtype=torch.bool),
                            "value": torch.full((T,), float(start_id + i))},
                    "meta": {"id": start_id + i}})
    return out


@pytest.fixture()
def shards(tmp_path):
    rng = np.random.default_rng(0)
    lengths = [rng.integers(1, 120, 23).tolist(), rng.integers(1, 40, 17).tolist(),
               rng.integers(50, 200, 9).tolist()]
    start = 0
    for i, ls in enumerate(lengths):
        # served as they are (layout per_seat: no seat split, dataset.SEAT_SPLIT_GAMES)
        save_shard(str(tmp_path / f"s{i}.pt"), "othello", _records(ls, start),
                   meta={"layout": "per_seat"})
        start += len(ls)
    return tmp_path, [x for ls in lengths for x in ls]


def test_batches_cover_every_record_once_within_budget(shards):
    root, lengths = shards
    ds = GameShardDataset(str(root), game="othello")
    torch.manual_seed(0)
    sampler = WholeRecordBatchSampler(ds, frames=128, max_batch=10)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(ds)))                  # every record, once
    for b in batches:
        shard = {ds.locate(i)[0] for i in b}
        assert len(shard) == 1                                   # shard-local
        longest = max(lengths[i] for i in b)
        assert len(b) <= 10 and (len(b) * longest <= 128 or len(b) == 1)
    assert any(len(b) == 1 and lengths[b[0]] > 128 for b in batches)   # over budget alone
    assert len(sampler) == len(batches) and sampler.records_per_batch() == pytest.approx(
        len(ds) / len(batches))
    torch.manual_seed(1)
    assert list(sampler) != batches                              # reshuffled per epoch


def test_records_are_served_whole_from_their_first_frame(shards):
    root, lengths = shards
    full = GameShardDataset(str(root), game="othello")
    capped = GameShardDataset(str(root), game="othello", max_len=64, crop="prefix")
    assert (capped.sample_lengths() == np.minimum(lengths, 64)).all()
    for i in range(len(full)):
        a, b = full[i], capped[i]
        n = min(lengths[i], 64)
        assert torch.equal(b["obs"]["board_seq"], a["obs"]["board_seq"][:n])
    with pytest.raises(ValueError, match="prefix"):
        WholeRecordBatchSampler(GameShardDataset(str(root), game="othello", max_len=64,
                                                 crop="balanced"), frames=128)


def test_similar_lengths_share_a_batch():
    L = np.array([5, 60, 3, 30, 31, 100, 2, 50])
    b = whole_record_batches(L, 64, 100, np.arange(8))
    assert b == [[6, 2, 0], [3, 4], [7], [1], [5]]
    padded = sum(len(x) * max(L[i] for i in x) for x in b)
    assert padded <= int(L.sum()) + 64                           # little padding


def test_train_multi_game_trains_whole_records(shards, tmp_path):
    from training import train_multi_game

    root, _ = shards
    res = train_multi_game.main([
        "--data", f"othello={root}", "--steps", "2", "--batch_size", "2",
        "--warmup_steps", "1", "--eval_every", "0", "--device", "cpu",
        "--frames_per_batch", "160", "--max_len", "150", "--save_dir", str(tmp_path / "ck"),
        "--override", "d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "core_type=transformer", "core_layers=1", "moe_experts=4",
        "temporal_mixer=recurrent"])
    assert (tmp_path / "ck" / "multi_final.pt").is_file()
    assert res is None or isinstance(res, dict)


def test_train_multi_game_trains_long_records_in_chunks(shards, tmp_path, monkeypatch):
    """--chunk_len: a micro-batch longer than it (here records up to 199 frames in
    batches of <= 64) trains chunk by chunk, each continuing from the previous one's
    memory state (train_il_game.chunked_batch_loss)."""
    from training import train_il_game, train_multi_game

    root, _ = shards
    starts = []
    orig = train_il_game.batch_loss

    def spy(*a, **kw):
        starts.append(kw.get("t0", 0))
        return orig(*a, **kw)

    monkeypatch.setattr(train_il_game, "batch_loss", spy)
    torch.manual_seed(0)
    train_multi_game.main([
        "--data", f"othello={root}", "--steps", "6", "--games_per_step", "2",
        "--warmup_steps", "1", "--eval_every", "0", "--device", "cpu",
        "--frames_per_batch", "64", "--max_len", "200", "--chunk_len", "48",
        "--save_dir", str(tmp_path / "ck"),
        "--override", "d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "core_type=transformer", "core_layers=2", "moe_experts=4",
        "temporal_mixer=recurrent", "core_delta_memory=1", "core_delta_heads=2",
        "core_delta_dk=8", "core_delta_writes=2"])
    assert (tmp_path / "ck" / "multi_final.pt").is_file()
    assert any(s > 0 for s in starts) and all(s % 48 == 0 for s in starts)


def test_alternating_records_are_served_as_seat_streams(tmp_path):
    """A two-player perfect-information record (both movers' frames, alternating) is
    served as its two seat streams: frames 0, 2, … and 1, 3, … of the record, whole
    from their first frame, with their own lengths in the whole-record sampler; a
    one-frame record's second stream repeats its frame."""
    lengths = [7, 1, 12, 2]
    save_shard(str(tmp_path / "a.pt"), "othello", _records(lengths))
    ds = GameShardDataset(str(tmp_path), game="othello")
    assert ds.seat_split and len(ds) == 2 * len(lengths)
    want = [(7 + 1) // 2, 7 // 2, 1, 1, 6, 6, 1, 1]
    assert ds.sample_lengths().tolist() == want
    assert ds.frame_stats() == {"samples": 8, "frames": sum(want), "raw_frames": sum(want)}
    for r, T in enumerate(lengths):
        for seat in (0, 1):
            s = ds[2 * r + seat]
            frames = list(range(seat, T, 2)) or [0]
            assert s["tgt"]["place_idx"].tolist() == [f % 64 for f in frames]
            assert s["tgt"]["value"].tolist() == [float(r)] * len(frames)
            assert s["meta"] == {"id": r, "seat_stream": seat}
    capped = GameShardDataset(str(tmp_path), game="othello", max_len=4, crop="prefix")
    assert capped.sample_lengths().tolist() == [min(L, 4) for L in want]
    assert capped[4]["tgt"]["place_idx"].tolist() == [0, 2, 4, 6]
    batches = list(WholeRecordBatchSampler(ds, frames=8, max_batch=4))
    assert sorted(i for b in batches for i in b) == list(range(8))
    # the switch: shards that are per-seat already, or seat_split=False
    plain = GameShardDataset(str(tmp_path), game="othello", seat_split=False)
    assert not plain.seat_split and len(plain) == len(lengths)

"""env/hok5v5*.py — the Honor of Kings 5v5 simulator interface — without the SDK.

- env/hok5v5_config.py on a synthetic archive in the SDK's ``config.dat`` layout (the
  six text files the 5v5 patch touches, with their stock 3v3 fields, and a binary
  entry): the archive codec round-trips (payloads XORed in whole four-byte words, the
  tail bytes as they are), the patch sets exactly the verified 5v5 fields and nothing
  else, is idempotent, and refuses a layout it does not know;
- env/hok5v5.py against tests/fixtures/hok5v5_fake_worker.py (the worker protocol,
  no gamecore): one controlled camp (``next_decision`` / ``step_action``), both camps
  (``step_round`` in camp order), the game-over frame, and the guards — a stale frame,
  malformed probabilities, an uncontrolled camp, a worker reporting another ABI.

The real match (licensed gamecore, a Linux node): scripts/hok/smoke_5v5.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from env import hok5v5_config as cfg
from env.hok5v5 import (
    ACTION_HEAD_SIZES,
    HEROES_PER_CAMP,
    OBSERVATION_SIZE,
    POLICY_SIZE,
    Hok5v5Env,
)

FAKE_WORKER = Path(__file__).resolve().parent / "fixtures" / "hok5v5_fake_worker.py"
BINARY = "AILab/ai_config/5v5/model/weights.bin"


def stock_entries():
    """The files the patch reads, with the stock 3v3 values (and a stale 5v5 comment)."""
    team = "\n".join([
        "feature_name_list=" + cfg._hero_feature_names(3),
        "feature_size_list=289;2852;289;2852;289;2852",
        "prob_name_list=" + cfg._hero_probability_names(3),
        "output_size_list=13;25;42;42;3;39;1", ""])
    tactics = "\n".join([
        "game_type=3v3", "feature_size_list=1734;2852",
        "feature_size_list_in_one_frame=1734;2852", "frame_gap=3", ""])
    return [
        cfg.ConfigEntry(cfg.MODEL_CONFIG, b"hero_cnt_each_camp=3\nmodel_name=team\n"),
        cfg.ConfigEntry(BINARY, bytes(range(1, 8))),              # 7 bytes: a 3-byte tail
        cfg.ConfigEntry(cfg.COMBINE_CONFIG, b"num_unit=6,20,6,20,0\n"),
        cfg.ConfigEntry(cfg.COMMON_CONFIG,
                        b"#ten_hero_vec_size=2364\nten_hero_vec_size=1422\n"),
        cfg.ConfigEntry(cfg.PRIVATE_CONFIG, b"ten_hero_vec_size=84\n"),
        cfg.ConfigEntry(cfg.TACTICS_CONFIG, tactics.encode()),
        cfg.ConfigEntry(cfg.TEAM_CONFIG, team.encode()),
    ]


def test_archive_codec_round_trips_and_keeps_the_tail_bytes():
    entries = stock_entries()
    blob = cfg.dumps_archive(entries)
    assert cfg.loads_archive(blob) == entries
    assert cfg.dumps_archive(cfg.loads_archive(blob)) == blob
    # the 7-byte payload: its first four bytes XORed with ef cd ab 00, the last three plain
    plain = bytes(range(1, 8))
    at = blob.rindex(bytes(a ^ b for a, b in zip(plain[:4], bytes.fromhex("efcdab00"))))
    assert blob[at + 4: at + 7] == plain[4:]
    with pytest.raises(cfg.ConfigArchiveError, match="truncated"):
        cfg.loads_archive(blob[:-2])
    with pytest.raises(cfg.ConfigArchiveError, match="trailing"):
        cfg.loads_archive(blob + b"\0")
    with pytest.raises(cfg.ConfigArchiveError, match="entry count"):
        cfg.loads_archive(b"\0\0\0\0")
    with pytest.raises(cfg.ConfigArchiveError, match="duplicate"):
        cfg.dumps_archive(entries + entries[:1])


def test_patch_sets_the_5v5_fields_and_nothing_else():
    entries = stock_entries()
    patched = cfg.patch_5v5(entries)
    assert [e.name for e in patched] == [e.name for e in entries]          # order kept
    by = {e.name: e.data for e in patched}
    assert by[BINARY] == bytes(range(1, 8))                                # untouched
    assert by[cfg.MODEL_CONFIG] == b"hero_cnt_each_camp=5\nmodel_name=team\n"
    assert by[cfg.COMBINE_CONFIG] == b"num_unit=10,20,6,20,0\n"
    # the stale comment line stays; the active line takes the measured width
    assert by[cfg.COMMON_CONFIG] == b"#ten_hero_vec_size=2364\nten_hero_vec_size=2410\n"
    assert by[cfg.PRIVATE_CONFIG] == b"ten_hero_vec_size=140\n"
    tactics = by[cfg.TACTICS_CONFIG].decode().splitlines()
    assert tactics == ["game_type=5v5", "feature_size_list=1734;3896",
                       "feature_size_list_in_one_frame=1734;3896", "frame_gap=3"]
    team = dict(line.split("=", 1) for line in by[cfg.TEAM_CONFIG].decode().splitlines())
    assert team["feature_name_list"].count("_feature_img;") == 5
    assert team["feature_size_list"] == ";".join(["289;3896"] * 5)
    assert len(team["prob_name_list"].split(";")) == 5 * 7
    assert team["output_size_list"] == "13;25;42;42;3;39;1"               # the native ABI
    cfg.validate_5v5(patched)
    assert cfg.patch_5v5(patched) == patched                               # idempotent
    assert (cfg.FEATURE_IMAGE_SIZE + cfg.FEATURE_VECTOR_SIZE, cfg.POLICY_SIZE) == (5630, 162)


def _with(entries, name, data):
    return [cfg.ConfigEntry(e.name, data if e.name == name else e.data) for e in entries]


def test_patch_refuses_a_layout_it_does_not_know():
    entries = stock_entries()
    with pytest.raises(cfg.ConfigArchiveError, match="unexpected hero_cnt_each_camp"):
        cfg.patch_5v5(_with(entries, cfg.MODEL_CONFIG, b"hero_cnt_each_camp=4\n"))
    with pytest.raises(cfg.ConfigArchiveError, match="exactly one active num_unit"):
        cfg.patch_5v5(_with(entries, cfg.COMBINE_CONFIG, b"num_unit=6,20,6,20,0\nnum_unit=1\n"))
    with pytest.raises(cfg.ConfigArchiveError, match="missing"):
        cfg.patch_5v5([e for e in entries if e.name != cfg.PRIVATE_CONFIG])
    team = next(e.data for e in entries if e.name == cfg.TEAM_CONFIG)
    with pytest.raises(cfg.ConfigArchiveError, match="output_size_list"):
        cfg.patch_5v5(_with(entries, cfg.TEAM_CONFIG, team.replace(b";3;39;1", b";5;39;1")))
    with pytest.raises(cfg.ConfigArchiveError, match="hero_cnt_each_camp='3', expected '5'"):
        cfg.validate_5v5(entries)                                          # still 3v3


@pytest.fixture
def sdk_config(tmp_path):
    path = tmp_path / "sdk" / "config.dat"
    cfg.write_archive(stock_entries(), path)
    return path


def test_build_writes_a_separate_archive(sdk_config, tmp_path):
    source_blob = sdk_config.read_bytes()
    target = tmp_path / "out" / "config.5v5.dat"
    report = cfg.build_5v5_config(sdk_config, target)
    assert sdk_config.read_bytes() == source_blob            # the SDK's file is not touched
    cfg.validate_5v5(cfg.load_archive(target))
    assert report.files == 7 and not report.tested_source                  # a synthetic archive
    assert report.as_dict()["observation_size"] == 5630
    with pytest.raises(FileExistsError):
        cfg.build_5v5_config(sdk_config, target)
    assert cfg.build_5v5_config(sdk_config, target, overwrite=True).target_sha256 \
        == report.target_sha256
    with pytest.raises(ValueError, match="different files"):
        cfg.build_5v5_config(sdk_config, sdk_config, overwrite=True)
    assert cfg.main([str(sdk_config), str(tmp_path / "cli.dat")]) == 0


def _env(sdk_config, tmp_path, name, **kw):
    return Hok5v5Env(hok_root=str(tmp_path), sdk_config=str(sdk_config),
                     worker_cmd=[sys.executable, str(FAKE_WORKER)],
                     workdir=str(tmp_path / "work" / name), runtime_id=name, **kw)


def test_one_controlled_camp_plays_to_the_game_over_frame(sdk_config, tmp_path):
    with _env(sdk_config, tmp_path, "one", max_frame_num=8) as env:
        assert env.metadata["observation_size"] == OBSERVATION_SIZE
        assert not env.config_report.tested_source
        env.reset(use_common_ai=(True, False))
        assert env.controlled_agents == (1,)
        with pytest.raises(ValueError, match="not controlled"):
            env.step_feature(0)
        decisions = []
        while True:
            frame = env.next_decision(1)
            if frame.gameover:
                break
            assert frame.continue_process and frame.observations.shape == (5, OBSERVATION_SIZE)
            assert frame.observations.dtype == np.float32
            assert [h.camp_id for h in frame.heroes] == [2] * HEROES_PER_CAMP
            with pytest.raises(RuntimeError, match="pending prediction"):
                env.step_feature(1)                    # the frame must be answered first
            probs = env.random_probabilities(seed=frame.frame_no)
            results = env.step_action(1, probs, frame)
            assert len(results) == HEROES_PER_CAMP
            offset = 0
            for head, width in enumerate(ACTION_HEAD_SIZES[:-1]):
                want = probs[:, offset: offset + width].argmax(1)
                assert [r.actions[head] for r in results] == want.tolist()
                assert all(r.legal_action[head].shape == (width,) for r in results)
                offset += width
            assert all(r.is_train and r.sub_actions.shape == (5,) for r in results)
            with pytest.raises(RuntimeError, match="no pending"):
                env.step_action(1, probs, frame)       # answered once
            decisions.append(frame.frame_no)
        assert decisions == [3, 6] and frame.frame_no == 8           # every third frame
        assert not frame.continue_process and len(frame.heroes) == HEROES_PER_CAMP
        assert [h.reward for h in frame.heroes] == [-1.0] * HEROES_PER_CAMP
        # the interface serves another match after close_game
        env.close_game()
        env.reset(camp=((190, 173, 117, 169, 112), (141, 111, 107, 128, 163)),
                  use_common_ai=(False, True))
        assert env.controlled_agents == (0,) and env.next_decision(0).frame_no == 3


def test_both_camps_are_served_in_camp_order(sdk_config, tmp_path):
    seen = []

    def policy(tag):
        def act(frame):
            seen.append((tag, frame.frame_no, frame.token))
            return Hok5v5Env.random_probabilities(seed=frame.frame_no)
        return act

    with _env(sdk_config, tmp_path, "two", max_frame_num=7) as env:
        env.reset(use_common_ai=(False, False))
        with pytest.raises(RuntimeError, match="one controlled camp"):
            env.next_decision(0)
        with pytest.raises(KeyError, match="missing policies"):
            env.step_round({0: policy("red")})
        rounds = 0
        while True:
            camps = env.step_round({0: policy("red"), 1: policy("blue")})
            rounds += 1
            assert set(camps) == {0, 1}
            for frame, results in camps.values():
                assert (results is not None) == frame.continue_process
            if any(frame.gameover for frame, _ in camps.values()):
                break
        assert rounds == 7
        # camp 0's prediction is answered before camp 1's frame is requested
        assert [(tag, n) for tag, n, _ in seen] == [("red", 3), ("blue", 3),
                                                    ("red", 6), ("blue", 6)]
        assert [t for _, _, t in seen] == [1, 2, 3, 4]


def test_guards(sdk_config, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="runtime_id"):
        Hok5v5Env(hok_root=str(tmp_path), sdk_config=str(sdk_config), runtime_id="a b",
                  worker_cmd=[sys.executable, str(FAKE_WORKER)])
    with pytest.raises(ValueError, match="sdk_config is required"):
        Hok5v5Env(hok_root=str(tmp_path), worker_cmd=[sys.executable, str(FAKE_WORKER)],
                  workdir=str(tmp_path / "w"))
    with _env(sdk_config, tmp_path, "guards", max_frame_num=20) as env:
        with pytest.raises(ValueError, match="exactly five heroes"):
            env.reset(camp=((1, 2, 3), (4, 5, 6)))
        with pytest.raises(ValueError, match="repeats hero_id"):
            env.reset(camp=((1, 1, 2, 3, 4), (5, 6, 7, 8, 9)))
        with pytest.raises(ValueError, match="at least one camp"):
            env.reset(use_common_ai=(True, True))
        env.reset(use_common_ai=(True, False))
        frame = env.next_decision(1)
        good = env.random_probabilities(seed=0)
        assert good.shape == (HEROES_PER_CAMP, POLICY_SIZE)
        with pytest.raises(ValueError, match="shape"):
            env.step_action(1, good[:, :-1], frame)
        bad = good.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            env.step_action(1, bad, frame)
        zero = good.copy()
        zero[2, :13] = 0.0
        with pytest.raises(ValueError, match="nonzero mass"):
            env.step_action(1, zero, frame)
        scaled = env.normalise_probabilities(good * 3.0)
        np.testing.assert_allclose(scaled[:, :13].sum(1), 1.0, rtol=1e-5)
        np.testing.assert_allclose(scaled[:, -1], good[:, -1] * 3.0)      # the value scalar
        env.step_action(1, good, frame)
        later = env.next_decision(1)
        with pytest.raises(ValueError, match="stale"):
            env.step_action(1, good, frame)            # an earlier frame's token
        env.step_action(1, good, later)
    monkeypatch.setenv("HOK5V5_FAKE_OBSERVATION_SIZE", "4586")
    with pytest.raises(RuntimeError, match="observation ABI mismatch"):
        _env(sdk_config, tmp_path, "abi")

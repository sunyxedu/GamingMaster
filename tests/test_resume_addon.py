"""Resuming a joint run whose game gained an ADD-ON module (model.multi_game.ADDON_PREFIX:
zero output at construction) after its checkpoint was written — the SC2 decoder's
addon_fine joining the running Stage-1 lines at a handover.  The checkpoint (othello +
chess, no add-on) resumes into a run where chess has ``addon_probe``: the old weights,
moments and EMA shadows load exactly, the add-on starts at its fresh weights without
optimizer state, training continues and the new checkpoint (now with the add-on)
resumes again; a checkpoint with only PART of an add-on is refused."""

import pytest
import torch

from tests.test_resume_add_games import OLD, STEPS, _args, _load, _probe, shards  # noqa: F401
from training import train_multi_game as tmg


class Addon(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((4,), 3.0))
        self.b = torch.nn.Parameter(torch.zeros(2))

    def reset_addon(self):
        with torch.no_grad():
            self.w.fill_(3.0)


def _with_addon(mp):
    build = tmg.build_net_tolerant

    def build_with(game, arch):
        net, dropped = build(game, arch)
        if game == "chess":
            net.addon_probe = Addon()
        return net, dropped

    mp.setattr(tmg, "build_net_tolerant", build_with)


def test_resume_with_an_addon(shards, tmp_path):  # noqa: F811
    src = tmp_path / "src"
    with pytest.MonkeyPatch.context() as mp:
        _probe(mp)
        tmg.main(_args(shards, src, OLD, "--steps", str(STEPS), "--ckpt_every", str(STEPS)))
    ck = _load(src / "multi_latest.pt")
    assert not any(".addon_" in k for k in ck["train_model_state"])
    out = tmp_path / "resumed"
    with pytest.MonkeyPatch.context() as mp:
        _probe(mp)
        _with_addon(mp)
        tmg.main(_args(shards, out, OLD, "--steps", str(STEPS + 2), "--ckpt_every", "1",
                       "--resume", str(src / "multi_latest.pt")))
    new = _load(out / "multi_latest.pt")
    assert new["step"] == STEPS + 2
    st = new["train_model_state"]
    assert torch.equal(st["nets.chess.addon_probe.w"], torch.full((4,), 3.0))    # no gradient: unchanged
    # every old weight trained on (the run continued), the old keys all there
    assert set(ck["train_model_state"]) <= set(st)
    # the new checkpoint (with the add-on) resumes as usual
    again = tmp_path / "again"
    with pytest.MonkeyPatch.context() as mp:
        _probe(mp)
        _with_addon(mp)
        tmg.main(_args(shards, again, OLD, "--steps", str(STEPS + 3), "--ckpt_every", "1",
                       "--resume", str(out / "multi_latest.pt")))
    assert _load(again / "multi_latest.pt")["step"] == STEPS + 3
    # part of an add-on: refused
    part = dict(ck)
    part["train_model_state"] = {**ck["train_model_state"], "nets.chess.addon_probe.w": torch.zeros(4)}
    torch.save(part, tmp_path / "part.pt")
    with pytest.MonkeyPatch.context() as mp:
        _probe(mp)
        _with_addon(mp)
        with pytest.raises(SystemExit):
            tmg.main(_args(shards, tmp_path / "bad", OLD, "--steps", str(STEPS + 1),
                           "--resume", str(tmp_path / "part.pt")))

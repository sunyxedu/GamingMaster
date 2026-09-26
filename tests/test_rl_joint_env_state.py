"""training/train_rl_joint.py — the simulators' state in the RL checkpoint
(``JointRLTrainer._env_states`` / ``_load_env_states``): every rank's envs are
saved (one ``all_gather_object`` posted by every rank's ``save``) and a resume
restores this rank's.  Lux waves continue the envs' 5-match games, so without
it a resume restarted them (end to end: tests/test_vec_lux.py); a different
world size or env count restarts the games with a note."""

from types import SimpleNamespace

from env.vec_game import make_vec_env
from training import train_rl_joint
from training.train_rl_joint import JointRLTrainer


def _trainer(envs, *, rank=0, world=1, enabled=False, logs=None):
    """The trainer attributes the two methods read."""
    return SimpleNamespace(rl={g: SimpleNamespace(env=e) for g, e in envs.items()},
                           dist_env=SimpleNamespace(enabled=enabled), world=world, rank=rank,
                           rank0=rank == 0, log=logs.append if logs is not None else print)


def _rng(env):
    return env._rng.bit_generator.state


def test_single_process_roundtrip():
    env = make_vec_env("othello", 3, seed=1)
    env._rng.integers(0, 10, size=5)                         # a non-initial state
    states = JointRLTrainer._env_states(_trainer({"othello": env}))
    assert len(states) == 1 and states[0]["othello"]["num_envs"] == 3
    fresh = make_vec_env("othello", 3, seed=99)
    JointRLTrainer._load_env_states(_trainer({"othello": fresh}), states, 1)
    assert _rng(fresh) == _rng(env)
    # no checkpointed envs (an older checkpoint): nothing happens
    JointRLTrainer._load_env_states(_trainer({"othello": fresh}), None, 1)


def test_ranks_are_gathered_and_each_restores_its_own(monkeypatch):
    envs = [make_vec_env("othello", 2, seed=s) for s in (1, 2)]

    def all_gather_object(out, obj):                         # a 2-rank collective
        out[0] = obj if obj["othello"]["state"]["rng"] == _rng(envs[0]) else None
        out[1] = {"othello": {"num_envs": 2, "state": envs[1].state_dict()}}

    monkeypatch.setattr(train_rl_joint.dist, "all_gather_object", all_gather_object)
    states = JointRLTrainer._env_states(_trainer({"othello": envs[0]}, world=2, enabled=True))
    assert [s["othello"]["state"]["rng"] for s in states] == [_rng(e) for e in envs]
    # a non-zero rank posts the collective but keeps nothing (rank 0 writes)
    assert JointRLTrainer._env_states(
        _trainer({"othello": envs[1]}, rank=1, world=2, enabled=True)) is None
    fresh = make_vec_env("othello", 2, seed=7)
    JointRLTrainer._load_env_states(_trainer({"othello": fresh}, rank=1, world=2), states, 2)
    assert _rng(fresh) == _rng(envs[1])
    # another world size / env count: nothing restored, a note
    logs = []
    for t in (_trainer({"othello": make_vec_env("othello", 2, seed=7)}, logs=logs),
              _trainer({"othello": make_vec_env("othello", 3, seed=7)}, rank=1, world=2,
                       logs=logs)):
        before = _rng(t.rl["othello"].env)
        JointRLTrainer._load_env_states(t, states, 2)
        assert _rng(t.rl["othello"].env) == before
    assert len(logs) == 2 and all("restart" in m for m in logs)


def test_adapters_without_state_dict_are_skipped():
    bare = SimpleNamespace(num_envs=1)                       # the protocol does not require it
    assert JointRLTrainer._env_states(_trainer({"x": bare})) == [{}]
    JointRLTrainer._load_env_states(_trainer({"x": bare}),
                                    [{"x": {"num_envs": 1, "state": {}}}], 1)

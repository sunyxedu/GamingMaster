"""Resuming a joint run with games added (training/train_multi_game.py
``--resume`` with a superset of the checkpoint's games).

A 2-game checkpoint (othello, chess; a chess probe parameter no forward uses,
so one optimizer parameter never has state) resumes into 3-game runs with
gomoku inserted in the middle of the --data order and in front of it (then
the shared core's canonical names move under gomoku: nothing may be matched
by position).  In-process, single process:

- after loading (a resume at the checkpoint's step trains nothing) the old
  games' weights and buffers (core aliases, MoE balance biases and BatchNorm
  statistics included), EMA shadows, optimizer moments and step counters are
  bit-identical to the checkpoint's, the optimizer's param-group hyper-
  parameters are the checkpoint's (decay and QK-norm no-decay groups); the
  added game's weights and buffers equal a fresh build's, its EMA shadows
  its fresh weights, and it has no optimizer state; training then continues
  (every game drawn every step: the new game's step counters count from 1,
  the old ones continue) and the 3-game checkpoint resumes again;
- all of it also from a checkpoint without saved parameter names (an older
  one: the parameter order is rebuilt on the meta device), also for a frozen
  core with trainable norms; a reconstruction that does not reproduce the
  saved optimizer is refused; the checkpoint's games alone load the same
  way, in its --data order (an older checkpoint then is not rebuilt: its
  layout is the run's own) or another;
- refused: removing a game, an --override that contradicts the checkpoint's
  core, an added game whose built core differs, changed --freeze_core.

Data parallel (spawned gloo ranks drawing the games in rotation; CUDA tensors
over gloo where available): 2-rank sharded and unsharded checkpoints of the 2
games resume on 3 ranks with gomoku added, sharded <-> unsharded — the two
resumed runs are bit-identical, continue the old step counters and count the
new game's from 1, and the sharded ranks hold disjoint partitions of the
gathered state; and (CPU) a sharded 3-rank resume of the older-format
checkpoint loads on every rank exactly its partition of the remapped state.
"""

import random

import pytest
import torch

from data_processing.games.dataset import save_shard
from model.multi_game import is_core_key
from tests.gloo_devices import DEVICES, local_ranks
from tests.test_core_moe import _board_samples
from tests.test_zero1 import (
    CORE_TF,
    _assert_runs_identical,
    _assert_same,
    _assert_same_checkpoint,
    _free_port,
    _run_procs,
    _train_worker,
)
from training import train_multi_game as tmg
from training.train_multi_game import OPT_NAMES_KEY

OLD = ["othello", "chess"]                  # the checkpoint's games
NEW = "gomoku"                              # the game added on resume
ORDERS = {"middle": ["othello", "gomoku", "chess"], "first": ["gomoku", "othello", "chess"]}
STEPS = 3                                   # the checkpoint's step
MORE = 5                                    # resumed runs train on to this step


def _args(root, save_dir, games, *extra, kind="cpu"):
    dev = ["--device", "cpu"] if kind == "cpu" else ["--device", "cuda", "--dist_backend", "gloo"]
    return [*(a for g in games for a in ("--data", f"{g}={root}/{g}")),
            "--val", f"othello={root}/othello", "--batch_size", "2", "--warmup_steps", "2",
            "--eval_every", "2", "--log_every", "1", "--ema_rate", "0.9",
            "--moe_balance_rate", "0.01", "--clip_grad", "0.5", "--seed", "3", *dev,
            "--save_dir", str(save_dir), *extra,
            # legacy BatchNorm encoders: their running statistics are the
            # buffers a resume must carry exactly (the default per-frame
            # GroupNorm has none)
            "--override", *[f"{k}={v}" for k, v in {**CORE_TF, "encoder_norm": "batch"}.items()]]


def _probe(mp) -> None:
    """chess gets a trainable parameter outside every forward (never a
    gradient, never optimizer state) — as in the ZeRO-1 trainer tests."""
    build = tmg.build_net_tolerant

    def build_with_probe(game, arch):
        net, dropped = build(game, arch)
        if game == "chess":
            net.zero_probe = torch.nn.Parameter(torch.full((3,), 2.0))
        return net, dropped

    mp.setattr(tmg, "build_net_tolerant", build_with_probe)
    mp.delenv("WORLD_SIZE", raising=False)


@pytest.fixture(autouse=True)
def _probed(monkeypatch):
    _probe(monkeypatch)


class _AllGames:
    """train_multi_game's ``random``: every step draws every game once."""

    seed = staticmethod(random.seed)

    @staticmethod
    def choices(names, weights=None, k=1):
        assert k == len(names), (names, k)
        return list(names)


def _own(name: str, game: str) -> bool:
    """A parameter / buffer of ``game``'s own modules (not the shared core)."""
    pre = f"nets.{game}."
    return name.startswith(pre) and not is_core_key(name[len(pre):])


def _canon(name: str, games) -> str:
    """The canonical name (first in named_parameters) in a run of ``games``
    of a checkpoint parameter: a core tensor's lies under the first game."""
    _, _, rest = name.split(".", 2)
    return f"nets.{games[0]}.{rest}" if is_core_key(rest) else name


def _flat(names):
    return [n for group in names for n in group]


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _strip_names(ck, path):
    """An older checkpoint: the same, without the saved parameter names."""
    old = {k: v for k, v in ck.items() if k != OPT_NAMES_KEY}
    torch.save(old, path)
    return old


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    root = tmp_path_factory.mktemp("add_games_data")
    for g in ORDERS["middle"]:
        (root / g).mkdir()
        kw = {"board": 9} if g == "gomoku" else {}
        save_shard(str(root / g / "a.pt"), g, _board_samples(g, 6, 6, **kw),
                   **({"meta": {"board_size": 9}} if g == "gomoku" else {}))
    return root


def _source(shards, out, *extra):
    """A 2-game run to step STEPS: (multi_latest.pt, an older-format copy)."""
    with pytest.MonkeyPatch.context() as mp:
        _probe(mp)
        tmg.main(_args(shards, out, OLD, "--steps", str(STEPS), "--ckpt_every", str(STEPS),
                       *extra))
    ck = _load(out / "multi_latest.pt")
    assert ck["games"] == OLD and ck["step"] == STEPS and OPT_NAMES_KEY in ck
    _strip_names(ck, out / "old_format.pt")
    return {"names": out / "multi_latest.pt", "old_format": out / "old_format.pt", "ck": ck}


@pytest.fixture(scope="module")
def source(shards, tmp_path_factory):
    return _source(shards, tmp_path_factory.mktemp("add_games_src"))


@pytest.fixture(scope="module")
def frozen_source(shards, tmp_path_factory):
    """The same with a frozen core whose norms train."""
    return _source(shards, tmp_path_factory.mktemp("add_games_frozen"), "--freeze_core",
                   "--train_core_norms")


def _assert_loaded(out, ck, games, fresh=None):
    """``out`` (a resume at the checkpoint's step: nothing trained) holds the
    checkpoint's state by name for the old games and the core, a fresh
    build's (``fresh``; None: no game added) for the added game."""
    model, ema, opt = out["model"], out["ema"], out["optimizer"]
    state = model.state_dict()
    assert (NEW in games) == (fresh is not None)
    # weights and buffers: every checkpoint key exactly (all core aliases,
    # MoE balance biases, BatchNorm statistics), the added game's core names
    # are the same core, its own modules a fresh build's
    assert {k for k in state if not k.startswith(f"nets.{NEW}.")} == set(ck["train_model_state"])
    for k, v in ck["train_model_state"].items():
        assert torch.equal(state[k], v), k
    assert any(k.endswith("balance_bias") for k in ck["train_model_state"])
    assert any(k.endswith("running_var") for k in ck["train_model_state"])
    own = [k for k in state if _own(k, NEW)]
    assert fresh is None or any(k.endswith("running_mean") for k in own)
    for k in state:
        if k.startswith(f"nets.{NEW}.") and k not in own:
            assert torch.equal(state[k], ck["train_model_state"][k.replace(NEW, "othello", 1)]), k
    for k in own:
        assert torch.equal(state[k], fresh["model"].state_dict()[k]), k

    # EMA: the checkpoint's shadows under this run's canonical names; the
    # added game's from its fresh weights
    new_own = sorted(n for n, p in model.named_parameters() if p.requires_grad and _own(n, NEW))
    assert fresh is None or new_own
    fresh_params = dict(fresh["model"].named_parameters()) if fresh is not None else {}
    assert set(ema.shadow) == {_canon(n, games) for n in ck["ema_state"]} | set(new_own)
    for n, v in ck["ema_state"].items():
        assert torch.equal(ema.shadow[_canon(n, games)], v), n
    for n in new_own:
        assert torch.equal(ema.shadow[n], fresh_params[n]), n

    # optimizer: moments and step counters moved by name; no state for the
    # added game; the checkpoint's group hyper-parameters, groups keep their role
    names = tmg.optimizer_param_names(model, opt)
    got, want = opt.state_dict(), ck["optimizer_state"]
    at = {_canon(n, games): i for i, n in enumerate(_flat(ck[OPT_NAMES_KEY]))}
    assert len(at) == len(_flat(ck[OPT_NAMES_KEY]))
    assert set(_flat(names)) == set(at) | set(new_own)
    stateless = 0
    for i, n in enumerate(_flat(names)):
        if n in at:
            if at[n] in want["state"]:
                _assert_same(got["state"][i], want["state"][at[n]], n)
            else:
                assert i not in got["state"], n
                stateless += 1
        else:
            assert i not in got["state"], n
    assert stateless >= 1                       # the probe (and frozen / undrawn tensors)
    assert "nets.chess.zero_probe" in at and at["nets.chess.zero_probe"] not in want["state"]
    assert len(got["param_groups"]) == len(want["param_groups"]) == 2   # QK-norm: no decay
    for gg, wg, gn, wn in zip(got["param_groups"], want["param_groups"], names,
                              ck[OPT_NAMES_KEY]):
        assert ({k: v for k, v in gg.items() if k != "params"}
                == {k: v for k, v in wg.items() if k != "params"})
        assert {_canon(n, games) for n in wn} <= set(gn)
    assert want["param_groups"][1]["weight_decay"] == 0.0


@pytest.mark.parametrize("fmt", ["names", "old_format"])
@pytest.mark.parametrize("where", ["middle", "first"])
def test_resume_adds_a_game(source, shards, tmp_path, monkeypatch, where, fmt):
    games = ORDERS[where]
    ck = source["ck"]
    assert any(k.endswith("balance_bias") and v.abs().sum() > 0
               for k, v in ck["train_model_state"].items()), "balance bias never moved"
    fresh = tmg.main(_args(shards, tmp_path / "fresh", games, "--steps", "0"))
    loaded = tmg.main(_args(shards, tmp_path / "load", games, "--steps", str(STEPS),
                            "--resume", str(source[fmt])))
    _assert_loaded(loaded, ck, games, fresh)

    # train on, every game every step: the added game's counters start at 1
    monkeypatch.setattr(tmg, "random", _AllGames())
    more = tmg.main(_args(shards, tmp_path / "more", games, "--steps", str(MORE),
                          "--ckpt_every", str(MORE), "--resume", str(source[fmt])))
    model, opt = more["model"], more["optimizer"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    steps = {n: opt.state_dict()["state"].get(i, {}).get("step")
             for i, n in enumerate(_flat(tmg.optimizer_param_names(model, opt)))}
    ck_steps = {_canon(n, games): ck["optimizer_state"]["state"].get(i, {}).get("step")
                for i, n in enumerate(_flat(ck[OPT_NAMES_KEY]))}
    core = [n for n in steps if is_core_key(n.split(".", 2)[2]) and ck_steps[n]]
    assert core and all(n.startswith(f"nets.{games[0]}.") for n in core)
    assert all(steps[n] == ck_steps[n] + MORE - STEPS for n in core)    # continued
    new = [steps[n] for n in steps if _own(n, NEW)]
    assert set(new) <= {None, MORE - STEPS} and new.count(MORE - STEPS) > len(new) // 2
    assert steps["nets.chess.zero_probe"] is None
    latest = _load(tmp_path / "more" / "multi_latest.pt")
    assert latest["games"] == games and latest["step"] == MORE
    assert latest["scheduler_state"]["last_epoch"] == MORE
    assert _flat(latest[OPT_NAMES_KEY]) == list(steps)
    assert set(latest["mixture"]) == set(games)

    # the 3-game checkpoint resumes as usual (a plain resume)
    again = tmg.main(_args(shards, tmp_path / "again", games, "--steps", str(MORE + 1),
                           "--resume", str(tmp_path / "more" / "multi_latest.pt")))
    assert all(torch.isfinite(p).all() for p in again["model"].parameters())


@pytest.mark.parametrize("fmt", ["names", "old_format"])
@pytest.mark.parametrize("games", [OLD, OLD[::-1]], ids=["same_order", "reordered"])
def test_resume_without_added_games(source, shards, tmp_path, monkeypatch, games, fmt):
    """No game added: the same state by name, also with the checkpoint's
    games in another --data order (the core's canonical names move under
    chess).  An older checkpoint of the same games in the same order is not
    rebuilt: its layout is the run's own (what the resume always assumed)."""
    if fmt == "old_format" and games == OLD:
        def no_rebuild(*a, **kw):
            raise AssertionError("rebuilt the layout of a same-order checkpoint")

        monkeypatch.setattr(tmg, "checkpoint_param_names", no_rebuild)
    loaded = tmg.main(_args(shards, tmp_path, games, "--steps", str(STEPS),
                            "--resume", str(source[fmt])))
    _assert_loaded(loaded, source["ck"], games)


def test_old_checkpoint_parameter_order_is_rebuilt(source, frozen_source):
    """Without saved names the checkpoint's optimizer order is rebuilt from
    its configs and run flags — exactly the names the trainer now saves; a
    rebuild that cannot reproduce the saved optimizer is refused."""
    for src in (source, frozen_source):
        ck = src["ck"]
        old = {k: v for k, v in ck.items() if k != OPT_NAMES_KEY}
        assert tmg.checkpoint_param_names(old) == ck[OPT_NAMES_KEY]
    n_all = len(_flat(source["ck"][OPT_NAMES_KEY]))
    frozen = frozen_source["ck"]
    assert 0 < len(_flat(frozen[OPT_NAMES_KEY])) < n_all
    old = {k: v for k, v in frozen.items() if k != OPT_NAMES_KEY}
    for flags in ({"freeze_core": False}, {"train_core_norms": False}):
        bad = {**old, "args": {**old["args"], **flags}}
        with pytest.raises(SystemExit, match="does not reproduce"):
            tmg.checkpoint_param_names(bad)
    # an EMA whose names are not the rebuilt model's trainable parameters
    ema = dict(old["ema_state"])
    first = next(iter(ema))
    ema = {**{k: v for k, v in ema.items() if k != first}, first: ema[first]}   # reordered
    with pytest.raises(SystemExit, match="EMA"):
        tmg.checkpoint_param_names({**old, "ema_state": ema})
    with pytest.raises(SystemExit, match="does not match"):
        tmg.checkpoint_param_names({**frozen, OPT_NAMES_KEY: frozen[OPT_NAMES_KEY][:1]})


@pytest.mark.parametrize("fmt", ["names", "old_format"])
def test_frozen_core_resume_adds_a_game(frozen_source, shards, tmp_path, fmt):
    games = ORDERS["first"]
    extra = ("--freeze_core", "--train_core_norms")
    fresh = tmg.main(_args(shards, tmp_path / "fresh", games, "--steps", "0", *extra))
    loaded = tmg.main(_args(shards, tmp_path / "load", games, "--steps", str(STEPS), *extra,
                            "--resume", str(frozen_source[fmt])))
    _assert_loaded(loaded, frozen_source["ck"], games, fresh)
    trainable = {n for n, p in loaded["model"].named_parameters() if p.requires_grad}
    assert {n for n in trainable if not _own(n, NEW)} == {
        _canon(n, games) for n in _flat(frozen_source["ck"][OPT_NAMES_KEY])}


def test_removing_a_game_is_refused(source, shards, tmp_path):
    with pytest.raises(SystemExit, match=r"lacks \['chess'\].*not remove"):
        tmg.main(_args(shards, tmp_path, ["othello", NEW], "--steps", str(MORE),
                       "--resume", str(source["names"])))


def test_added_game_with_another_core_is_refused(source, shards, tmp_path, monkeypatch):
    # an --override contradicting the checkpoint's core (as with --init_core)
    argv = _args(shards, tmp_path, ORDERS["middle"], "--steps", str(MORE),
                 "--resume", str(source["names"]))
    with pytest.raises(SystemExit, match="contradicts the core"):
        tmg.main([*argv, "moe_experts=2"])
    # a game whose network comes out with another core anyway
    build = tmg.build_net_tolerant

    def other_core(game, arch):
        net, dropped = build(game, arch)
        if game == NEW:
            net.cfg.dropout = 0.25
        return net, dropped

    monkeypatch.setattr(tmg, "build_net_tolerant", other_core)
    with pytest.raises(SystemExit, match="core differs from the checkpoint's"):
        tmg.main(argv)


@pytest.mark.parametrize("fmt", ["current", "older_core_config"])
def test_added_game_takes_every_core_field_of_the_checkpoint(shards, tmp_path, fmt):
    """moe_bias_clip is kept by the shared MoE modules at construction, and the
    core the games share is the FIRST game's: a game added in front of the
    --data order must be built with the checkpoint's value even when the resume
    command line does not repeat the --override that set it (else the old
    games' balance biases silently lose their clip), a contradicting --override
    is refused, and --init_core takes it from the checkpoint the same way —
    also from a checkpoint whose core_config predates the field (its games'
    full configs carry it)."""
    from model.components.core import moe_layers

    src = tmp_path / "src"
    tmg.main([*_args(shards, src, OLD, "--steps", "2", "--ckpt_every", "2"),
              "moe_bias_clip=0.05"])
    path = src / "multi_latest.pt"
    if fmt == "older_core_config":
        ck = _load(path)
        ck["core_config"] = {k: v for k, v in ck["core_config"].items()
                             if k not in ("moe_bias_clip", "trm_rms_eps")}
        path = tmp_path / "older.pt"
        torch.save(ck, path)
    games = ORDERS["first"]
    out = tmg.main(_args(shards, tmp_path / "res", games, "--steps", "2",
                         "--resume", str(path)))
    model = out["model"]
    assert {g: model[g].cfg.moe_bias_clip for g in games} == dict.fromkeys(games, 0.05)
    assert {m.bias_clip for m in moe_layers(model)} == {0.05}
    with pytest.raises(SystemExit, match="contradicts the core"):
        tmg.main([*_args(shards, tmp_path / "bad", games, "--steps", "2",
                         "--resume", str(path)), "moe_bias_clip=0.2"])
    # a new game around the brain (--init_core): the checkpoint's core fields
    init = tmg.main(_args(shards, tmp_path / "init", [NEW], "--steps", "1",
                          "--init_core", str(path)))["model"]
    assert init[NEW].cfg.moe_bias_clip == 0.05
    assert {m.bias_clip for m in moe_layers(init)} == {0.05}


def test_changed_freeze_flags_are_refused(source, frozen_source, shards, tmp_path):
    games = ORDERS["middle"]
    with pytest.raises(SystemExit, match="frozen in this run"):
        tmg.main(_args(shards, tmp_path / "a", games, "--steps", str(MORE), "--freeze_core",
                       "--train_core_norms", "--resume", str(source["names"])))
    with pytest.raises(SystemExit, match="not in its optimizer"):
        tmg.main(_args(shards, tmp_path / "b", games, "--steps", str(MORE),
                       "--resume", str(frozen_source["names"])))
    # an older checkpoint of the same games, same order (not rebuilt)
    with pytest.raises(SystemExit, match="does not fit this run's parameters"):
        tmg.main(_args(shards, tmp_path / "c", OLD, "--steps", str(MORE),
                       "--resume", str(frozen_source["old_format"])))


# ----------------------------------------------------------------------
# data parallel: sharded <-> unsharded, 2 -> 3 ranks
# ----------------------------------------------------------------------

class _Rotation:
    """train_multi_game's ``random`` on data-parallel rank ``rank``: its
    i-th draw is game (rank + i) mod n — with as many ranks as games every
    game is drawn at every step, deterministically."""

    def __init__(self, rank):
        self.rank, self.t, self.seed = rank, 0, random.seed

    def choices(self, names, weights=None, k=1):
        out = [names[(self.rank + self.t + j) % len(names)] for j in range(k)]
        self.t += k
        return out


def _rotating_worker(rank, world, port, root, tag, argv, kind, lrs):
    tmg.random = _Rotation(rank)
    _train_worker(rank, world, port, root, tag, argv, kind, lrs, False, False)


def _spawn(root, tag, world, argv, kind):
    """One trainer run on ``world`` spawned ranks (test_zero1's worker —
    probe on chess — with rotating game draws): (per-rank outputs,
    multi_latest.pt or None, multi_final.pt)."""
    ctx = torch.multiprocessing.get_context("spawn")
    port, lrs = _free_port(), local_ranks(world, kind)
    _run_procs([ctx.Process(target=_rotating_worker,
                            args=(r, world, port, str(root), tag, argv, kind, lrs))
                for r in range(world)])
    ck = root / f"ck_{tag}"
    latest = ck / "multi_latest.pt"
    return ([_load(root / f"{tag}_rank{r}.pt") for r in range(world)],
            _load(latest) if latest.exists() else None, _load(ck / "multi_final.pt"))


@pytest.fixture(scope="module", params=DEVICES)
def dist_runs(request, shards, tmp_path_factory):
    """2-rank checkpoints of the 2 games (unsharded u2, sharded s2) resumed
    on 3 ranks with gomoku in front: s2 -> unsharded, u2 -> sharded; on the
    CPU also a sharded 3-rank resume of u2 without saved names that trains
    nothing (what each rank loaded)."""
    kind = request.param
    root = tmp_path_factory.mktemp(f"add_games_dist_{kind}")
    games = ORDERS["first"]

    def run(tag, world, run_games, steps, *extra):
        argv = _args(shards, root / f"ck_{tag}", run_games, "--steps", str(steps),
                     "--ckpt_every", str(steps), "--games_per_step", "1", "--bucket_mb", "0.01",
                     "--dist_timeout", "2", *extra, kind=kind)
        return _spawn(root, tag, world, argv, kind)

    def src(tag):
        return str(root / f"ck_{tag}" / "multi_latest.pt")

    runs = {"u2": run("u2", 2, OLD, STEPS), "s2": run("s2", 2, OLD, STEPS, "--shard_optimizer")}
    runs["s2>u3"] = run("s2_u3", 3, games, MORE, "--resume", src("s2"))
    runs["u2>s3"] = run("u2_s3", 3, games, MORE, "--resume", src("u2"), "--shard_optimizer")
    if kind == "cpu":
        _strip_names(runs["u2"][1], root / "u2_old.pt")
        runs["old>s3@load"] = run("old_s3_load", 3, games, STEPS, "--resume",
                                  str(root / "u2_old.pt"), "--shard_optimizer")
    return runs


def test_dist_resume_adds_a_game_across_sharding_and_world_sizes(dist_runs):
    """s2 -> 3 ranks unsharded and u2 -> 3 ranks sharded agree bit for bit
    (the sources do: ZeRO-1 checkpoints are unsharded ones); the resumed
    runs continue the old step counters and count the added game's from 1
    (every game is drawn at every step), and the sharded ranks hold disjoint
    partitions of the gathered state."""
    games = ORDERS["first"]
    _assert_same_checkpoint(dist_runs["u2"][1], dist_runs["s2"][1], "sources")
    _assert_runs_identical(dist_runs["s2>u3"], dist_runs["u2>s3"], "s2>u3 vs u2>s3")
    ranks, latest, _ = dist_runs["u2>s3"]
    assert latest["games"] == games and latest["step"] == MORE and latest["world_size"] == 3
    names = _flat(latest[OPT_NAMES_KEY])
    opt_state, ema_state = latest["optimizer_state"]["state"], latest["ema_state"]
    src = dist_runs["u2"][1]
    ck_steps = {_canon(n, games): src["optimizer_state"]["state"].get(i, {}).get("step")
                for i, n in enumerate(_flat(src[OPT_NAMES_KEY]))}
    steps = {n: opt_state.get(i, {}).get("step") for i, n in enumerate(names)}
    new = [s for n, s in steps.items() if _own(n, NEW)]
    assert set(new) <= {None, MORE - STEPS} and new.count(MORE - STEPS) > len(new) // 2
    old = [n for n in steps if not _own(n, NEW) and ck_steps[n]]
    assert len(old) > len(new)
    assert all(steps[n] == ck_steps[n] + MORE - STEPS for n in old)
    assert steps["nets.chess.zero_probe"] is None
    seen_opt, seen_ema = set(), set()
    for r in ranks:
        mine = set(r["owned"])
        assert set(r["opt_state"]) <= mine and not seen_opt & set(r["opt_state"])
        assert set(r["ema"]) == {r["ema_names"][i] for i in mine}
        _assert_same(r["opt_state"], {i: opt_state[i] for i in r["opt_state"]}, "opt part")
        _assert_same(r["ema"], {n: ema_state[n] for n in r["ema"]}, "ema part")
        seen_opt |= set(r["opt_state"])
        seen_ema |= set(r["ema"])
    assert seen_opt == set(opt_state) and seen_ema == set(ema_state)


def test_dist_sharded_ranks_load_their_share_of_an_old_checkpoint(dist_runs, shards, tmp_path):
    """A sharded 3-rank resume (nothing trained) of the 2-rank checkpoint
    without saved names: every rank holds exactly its partition — of the NEW
    model's parameters — of the checkpoint's optimizer state and EMA,
    remapped by name; the added game has no optimizer state and its shadows
    are its weights, a single-process fresh build's (rank 0's init)."""
    if "old>s3@load" not in dist_runs:
        pytest.skip("CPU only")
    games = ORDERS["first"]
    ranks, _, _ = dist_runs["old>s3@load"]
    ck = dist_runs["u2"][1]
    at = {_canon(n, games): i for i, n in enumerate(_flat(ck[OPT_NAMES_KEY]))}
    ck_ema = {_canon(n, games): v for n, v in ck["ema_state"].items()}
    fresh = tmg.main(_args(shards, tmp_path, games, "--steps", "0"))["model"].state_dict()
    owned = []
    for r in ranks:
        names = r["ema_names"]                  # canonical names in optimizer order
        owned += r["owned"]
        for i in r["owned"]:
            n = names[i]
            if n in at and at[n] in ck["optimizer_state"]["state"]:
                _assert_same(r["opt_state"][i], ck["optimizer_state"]["state"][at[n]], n)
            else:
                assert i not in r["opt_state"], n
                assert n in at or _own(n, NEW), n
        assert set(r["opt_state"]) <= set(r["owned"])
        for n, v in r["ema"].items():
            if _own(n, NEW):
                assert torch.equal(v, r["params"][n]) and torch.equal(v, fresh[n]), n
            else:
                assert torch.equal(v, ck_ema[n]), n
        for k, v in ck["train_model_state"].items():
            if k in r["params"] or k in r["buffers"]:
                assert torch.equal(r["params"].get(k, r["buffers"].get(k)), v), k
    assert sorted(owned) == list(range(len(ranks[0]["ema_names"])))


@pytest.mark.parametrize("reinit", ["othello", "chess"])
def test_resume_reinitialises_a_game(source, shards, tmp_path, reinit):
    """--reinit_game G: G's own modules (encoder, decoder, value head) start as a fresh
    build's around the checkpoint's core — no optimizer state, EMA = the fresh weights —
    while the other game and the whole core load exactly; othello is the checkpoint's
    first game, so the core's canonical names (optimizer state, EMA) are its own names
    and must survive the reinit."""
    ck = source["ck"]
    fresh = tmg.main(_args(shards, tmp_path / "fresh", OLD, "--steps", "0"))
    out = tmg.main(_args(shards, tmp_path / "load", OLD, "--steps", str(STEPS),
                         "--resume", str(source["names"]), "--reinit_game", reinit))
    model, ema, opt = out["model"], out["ema"], out["optimizer"]
    keep = [g for g in OLD if g != reinit][0]
    state, fresh_state = model.state_dict(), fresh["model"].state_dict()
    for k, v in state.items():
        if _own(k, reinit):
            assert torch.equal(v, fresh_state[k]), k                   # a fresh build's
        else:
            assert torch.equal(v, ck["train_model_state"][k]), k       # the checkpoint's
    assert any(_own(k, reinit) for k in state) and any(_own(k, keep) for k in state)
    # optimizer: no state for the reinitialised game's own parameters; the core's and the
    # other game's moments by name
    names = _flat(tmg.optimizer_param_names(model, opt))
    got = opt.state_dict()["state"]
    at = {_canon(n, OLD): i for i, n in enumerate(_flat(ck[OPT_NAMES_KEY]))}
    moved = 0
    for i, n in enumerate(names):
        if _own(n, reinit):
            assert i not in got, n
        elif at.get(n) in ck["optimizer_state"]["state"]:
            _assert_same(got[i], ck["optimizer_state"]["state"][at[n]], n)
            moved += 1
    assert moved > 0 and any(is_core_key(n.split(".", 2)[2]) and i in got
                             for i, n in enumerate(names))
    fresh_params = dict(fresh["model"].named_parameters())
    for n, v in ema.shadow.items():
        want = fresh_params[n] if _own(n, reinit) else ck["ema_state"][n]
        assert torch.equal(v, want), n
    for bad, match in ((["go"], "not trained by"), (OLD, "at least one game")):
        with pytest.raises(SystemExit, match=match):
            tmg.forget_games(ck, bad)

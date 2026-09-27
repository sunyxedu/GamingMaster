"""ZeRO-1 sharding of the optimizer state and the EMA (training/zero.py,
training/train_multi_game.py --shard_optimizer).

- The partition is deterministic, covers every tensor once, is balanced by
  numel and depends only on the tensor sizes and the world size; the
  broadcast bucket plan is the same on every rank.
- Memory: the per-rank optimizer + EMA bytes of the production config
  (10 games, d=384, 8-layer transformer, MoE 32 / 64 experts, 12 ranks),
  counted on the meta device, and the expected per-rank peaks
  (scripts/zero1_memory.py; run pytest with -s to see the table).
- Optimizer level, 2 and 3 real gloo ranks (CPU; CUDA tensors over gloo
  where available): fed the same synced gradients, the sharded step equals a
  single-process AdamATan2 + ModelEMA step bit for bit — the weights after
  every step, the gathered optimizer state (the unsharded state_dict) and the
  gathered EMA — with a parameter that never gets a gradient, a game adapter
  without gradient on every rank at every step (skipped: no state, no step
  count) and a non-finite gradient norm on one rank (dropped everywhere); the
  evaluation swap restores the weights and buffers exactly; a full state dict
  loads back sharded; per-bucket, per-tensor and one-bucket broadcasts agree.
  The same on a toy model with a weight tied inside the model, a module
  registered twice, a non-contiguous and a float64 parameter.
- Trainer level: sharded 2- and 3-rank runs are bit-identical to the
  unsharded runs (weights, buffers, validation metrics — the EMA swapped into
  the live model vs the unsharded deep copy — per-rank partitions against the
  gathered state, both checkpoints), also with a frozen core and without an
  EMA; resumes across sharded <-> unsharded and world
  sizes (2 -> 3, 3 -> 2, 1 -> 2) are bit-identical to each other; and a
  2-rank run fed the single-process micro-batches (scripted game draws on
  single-record datasets) is bit-identical to single-process
  --games_per_step 2, sharded or not.  (With 3 ranks no such run exists: the
  single process scales each micro-batch loss by 1/3, the data-parallel run
  divides the sum of 3 gradients by 3, which rounds differently — so
  3-rank equivalence to single-process is pinned at the optimizer level.)
  Where CUDA is available the 2- and 3-rank sharded/unsharded comparisons
  also run on CUDA tensors over gloo (deterministic kernels).
- --shard_optimizer is refused at world size 1; a partial EMA refuses the
  whole-model views.
"""

import copy
import math
import os
import socket
import time
from multiprocessing.connection import wait

import pytest
import torch
import torch.nn as nn

from data_processing.games.dataset import save_shard
from tests.gloo_devices import DEVICES, hide_gpus_if_cpu, local_ranks, needs_cuda
from tests.test_core_moe import _board_samples

#: the production core type (transformer + MoE), tiny
CORE_TF = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0, encoder_channels=16,
               core_type="transformer", core_layers=2, moe_experts=4)
GAMES = ("othello", "chess", "gomoku")
STEPS = 5
NAN_STEP = 3                # the last rank poisons its synced gradient at this step
RESUME_STEPS = 7            # resumed runs continue from step STEPS to this one
TIMEOUT_S = 300


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_procs(procs):
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT_S
    try:
        # a rank that dies leaves its peers blocked in a collective: stop at
        # the first failure instead of waiting for their timeout
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            wait([p.sentinel for p in procs if p.is_alive()], timeout=1.0)
            bad = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            assert not bad, f"rank failed with exit code {bad[0]}"
        assert not any(p.is_alive() for p in procs), "ranks hung (collective desync?)"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)


def _assert_same(a, b, path=""):
    """Recursive bit-exact equality of checkpoint-like objects."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), (path, set(a) ^ set(b))
        for k in a:
            _assert_same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a.cpu(), b.cpu()), (
            f"{path}: max |diff| {(a.cpu().double() - b.cpu().double()).abs().max()}")
    else:
        assert a == b, (path, a, b)


#: checkpoint entries that must agree bit for bit ("args", "world_size" differ)
CKPT_KEYS = ("games", "configs", "core_config", "core_state", "model_state",
             "train_model_state", "mixture", "step", "optimizer_state", "scheduler_state",
             "ema_state", "optimizer_param_names")


def _assert_same_checkpoint(a, b, what=""):
    assert {k for k in CKPT_KEYS if k in a} == {k for k in CKPT_KEYS if k in b}, what
    for k in CKPT_KEYS:
        if k in a:
            _assert_same(a[k], b[k], f"{what}:{k}")


# ----------------------------------------------------------------------
# partition, buckets, refusals (in-process)
# ----------------------------------------------------------------------

def test_partition_is_greedy_largest_first():
    from training.zero import partition_by_numel, rank_loads

    # 5 -> r0, 4 -> r1, 3 -> r1 (7), 3 -> r0 (8), 3 -> r1 (10)
    assert partition_by_numel([5, 4, 3, 3, 3], 2) == [0, 1, 1, 0, 1]
    assert rank_loads([5, 4, 3, 3, 3], [0, 1, 1, 0, 1], 2) == [8, 10]
    assert partition_by_numel([7], 3) == [0]            # ranks may own nothing
    assert partition_by_numel([], 2) == []
    with pytest.raises(ValueError):
        partition_by_numel([1], 0)


@pytest.mark.parametrize("world", [2, 3, 12])
def test_partition_is_deterministic_and_balanced(world):
    """Every tensor exactly once; each rank's load at most the mean plus the
    largest tensor; the same answer every time (no RNG, no dict order)."""
    from model.multi_game import build_multi_game
    from training.zero import partition_by_numel, rank_loads

    with torch.device("meta"):
        model = build_multi_game(list(GAMES), CORE_TF)
    numels = [p.numel() for p in model.parameters()]      # shared core tensors once
    owner = partition_by_numel(numels, world)
    assert owner == partition_by_numel(list(numels), world)
    assert len(owner) == len(numels) and set(owner) == set(range(world))
    loads = rank_loads(numels, owner, world)
    assert sum(loads) == sum(numels)
    assert max(loads) <= sum(numels) / world + max(numels)


def test_bucket_plan_is_rank_invariant_and_bounded():
    """The broadcast buckets depend only on the indices and the partition:
    identical on every rank, every index once, owned by the bucket's rank,
    at most cap bytes unless a single tensor."""
    from model.multi_game import build_multi_game
    from model.optim import AdamATan2
    from training.train_multi_game import multi_game_param_groups
    from training.zero import Zero1

    torch.manual_seed(0)
    model = build_multi_game(list(GAMES), CORE_TF)
    opt = AdamATan2(multi_game_param_groups(model, 0.1))
    world = 3
    zs = [Zero1(model, opt, r, world, bucket_mb=0.01) for r in range(world)]
    idxs = list(range(0, len(zs[0].params), 2))
    plans = [z._buckets(reversed(idxs)) for z in zs]
    assert plans[0] == plans[1] == plans[2]
    flat = [i for _, b in plans[0] for i in b]
    assert sorted(flat) == idxs
    cap = zs[0].cap_bytes
    for r, b in plans[0]:
        assert all(zs[0].owner[i] == r for i in b)
        sizes = [zs[0].params[i].numel() * 4 for i in b]
        assert len(b) == 1 or sum(sizes) <= cap
    assert any(len(b) > 1 for _, b in plans[0]) and any(len(b) == 1 for _, b in plans[0])
    per_tensor = Zero1(model, opt, 0, world, bucket_mb=0)._buckets(idxs)
    assert all(len(b) == 1 for _, b in per_tensor) and len(per_tensor) == len(idxs)
    assert zs[0].owned != zs[1].owned
    assert sorted(i for z in zs for i in z.owned) == list(range(len(zs[0].params)))


def test_partial_ema_refuses_whole_model_views():
    from model.multi_game import build_multi_game
    from model.optim import ModelEMA

    model = build_multi_game(["othello", "chess"], CORE_TF)
    some = {id(p) for p in list(model.parameters())[::2]}
    ema = ModelEMA(model, 0.9, only=some)
    assert ema.partial and 0 < len(ema.shadow) < len(ModelEMA(model, 0.9).shadow)
    for call in (lambda: ema.ema_copy(model), lambda: ema.ema_model_state(model),
                 lambda: ema.copy_to(model)):
        with pytest.raises(RuntimeError, match="partition"):
            call()


def test_shard_optimizer_is_refused_at_world_size_one(monkeypatch):
    from training import train_multi_game as tmg

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(SystemExit, match="more than one rank"):
        tmg.main(["--data", "othello=unused", "--device", "cpu", "--shard_optimizer"])


def test_production_memory_plan(capsys):
    """The production config, 12 ranks: per-rank optimizer + EMA bytes
    sharded vs unsharded and the expected per-rank peak at 32 and 64 experts
    (from the measured 11.9 GiB peak of the unsharded 32-expert run).  64
    experts fit a ~16 GiB 1g.18gb slice only sharded."""
    from scripts.zero1_memory import GIB, format_plan, memory_plan

    rows, a = memory_plan(experts=(32, 64), world=12)
    r32, r64 = rows
    # registry-default adapters: pokemon's exact Showdown vocabularies (converter v3: 1493
    # species / 956 move ids, was 512 / 256 hash buckets) add 646,204 of the 213.48M
    assert 213e6 < r32.params < 214e6 and r32.expert_params == 8 * 32 * (384 * 1024 + 512 * 384)
    assert 364e6 < r64.params < 365e6 and r64.expert_params == 2 * r32.expert_params
    for r in rows:
        assert r.state_unsharded == 12 * r.params           # m + v + EMA, fp32
        ideal = r.state_unsharded / 12
        assert ideal <= r.state_sharded_max <= 1.02 * ideal  # whole tensors balance
        assert r.weights == r.grads == 4 * r.params
    assert r32.peak_unsharded == pytest.approx(11.9 * GIB)
    assert r64.peak_sharded < r32.peak_unsharded < r64.peak_unsharded
    assert r64.peak_unsharded > 15 * GIB and r64.peak_sharded < 12 * GIB
    with capsys.disabled():
        print("\n" + format_plan(rows, a, 12, 11.9, 32, 10))


# ----------------------------------------------------------------------
# optimizer level: sharded step == single-process step, bit for bit
# ----------------------------------------------------------------------

def _opt_worker(rank, world, init_file, root, kind, lrs, bucket_mb):
    import torch.distributed as dist

    from tests.gloo_devices import worker_device

    dev = worker_device(kind, lrs[rank])          # pins the MIG slice first
    torch.set_num_threads(1)
    from model.multi_game import build_multi_game, is_core_key
    from model.optim import AdamATan2, ModelEMA, trm_lr_lambda
    from training.distributed import collective_clip_grad_norm_
    from training.train_multi_game import multi_game_param_groups
    from training.zero import Zero1

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world)
    checks = {}
    try:
        torch.manual_seed(0)                      # the same model on every rank
        model = build_multi_game(list(GAMES), CORE_TF)
        model["chess"].zero_probe = nn.Parameter(torch.full((3,), 2.0))   # never a gradient
        model = model.to(dev)
        ref = copy.deepcopy(model)                # the single-process reference

        def make(m):
            opt = AdamATan2(multi_game_param_groups(m, 0.1), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
            return opt, torch.optim.lr_scheduler.LambdaLR(opt, trm_lr_lambda(3, STEPS, 1.0))

        opt, sched = make(model)
        ref_opt, ref_sched = make(ref)
        zero = Zero1(model, opt, rank, world, bucket_mb=bucket_mb)
        ema, ref_ema = zero.make_ema(model, 0.9), ModelEMA(ref, 0.9)
        params = [p for p in model.parameters() if p.requires_grad]
        ref_params = [p for p in ref.parameters() if p.requires_grad]
        same_weights, dropped = [], []
        for step in range(1, STEPS + 1):
            gen = torch.Generator().manual_seed(1000 + step)     # the same on every rank
            skip = f"nets.{GAMES[step % len(GAMES)]}."           # no rank drew this game
            for (n, p), q in zip(model.named_parameters(), ref.parameters()):
                if n.endswith("zero_probe") or (n.startswith(skip)
                                                and not is_core_key(n[len(skip):])):
                    continue
                g = (torch.randn(p.shape, generator=gen) * 0.02).to(dev)
                p.grad, q.grad = g.clone(), g.clone()
            if step == NAN_STEP and rank == world - 1:
                next(p for p in params if p.grad is not None).grad.fill_(math.nan)
            norm = collective_clip_grad_norm_(params, 0.5, dev)
            dropped.append(not bool(torch.isfinite(norm)))
            if not dropped[-1]:
                zero.step()
                sched.step()
                ema.update(model)
            if step != NAN_STEP:
                torch.nn.utils.clip_grad_norm_(ref_params, 0.5)
                ref_opt.step()
                ref_sched.step()
                ref_ema.update(ref)
            opt.zero_grad(set_to_none=True)
            ref_opt.zero_grad(set_to_none=True)
            same_weights.append(all(torch.equal(p, q) for p, q in
                                    zip(model.parameters(), ref.parameters())))
        checks["weights equal the single-process ones after every step"] = all(same_weights)
        checks["only the non-finite step was dropped"] = (
            dropped == [s == NAN_STEP for s in range(1, STEPS + 1)])

        # the state is partitioned: this rank holds its own parameters' only
        mine = set(zero.owned)
        local = opt.state_dict()["state"]
        checks["optimizer state only for owned parameters"] = set(local) <= mine
        checks["EMA shadows = the owned parameters"] = (
            set(ema.shadow) == {zero.names[i] for i in mine})
        nbytes = torch.tensor([zero.local_state_bytes(ema)], dtype=torch.int64, device=dev)
        dist.all_reduce(nbytes)
        full = sum(t.numel() * t.element_size() for st in ref_opt.state.values()
                   for t in st.values() if torch.is_tensor(t))
        full += sum(t.numel() * t.element_size() for t in ref_ema.shadow.values())
        checks["the partitions add up to the unsharded state"] = int(nbytes) == full

        # gathered state == the single-process state (on rank 0 and on the last rank)
        want_opt, want_ema = ref_opt.state_dict(), ref_ema.state_dict()
        for dst in sorted({0, world - 1}):
            got_opt = zero.gather_optimizer_state(dst=dst)
            got_ema = zero.gather_ema(ema, dst=dst)
            if rank != dst:
                checks[f"dst {dst}: None elsewhere"] = got_opt is None and got_ema is None
                continue
            try:
                _assert_same(got_opt["param_groups"], want_opt["param_groups"])
                _assert_same(dict(got_opt["state"]), dict(want_opt["state"]))
                assert list(got_ema) == list(want_ema)
                _assert_same(got_ema, want_ema)
                assert all(v.device.type == "cpu" for v in got_ema.values())
                checks[f"dst {dst}: gathered state == single-process state"] = True
            except AssertionError as e:
                checks[f"dst {dst}: gathered state == single-process state ({e})"] = False
        probe = zero.index[id(model["chess"].zero_probe)]
        checks["the probe was never stepped"] = (
            probe not in want_opt["state"]
            and torch.equal(model["chess"].zero_probe.cpu(), torch.full((3,), 2.0)))
        counts = [st["step"] for st in want_opt["state"].values()]
        checks["the undrawn adapters were skipped on some steps"] = (
            0 < min(counts) < max(counts) == STEPS - 1)

        # evaluation: the full EMA swapped in, weights and buffers restored
        train_w = [p.detach().clone() for p in model.parameters()]
        bufs = [b.detach().clone() for b in model.buffers()]
        ema_ref_model = ref_ema.ema_copy(ref)
        with zero.ema_weights(model, ema) as m:
            checks["the EMA weights inside the block"] = m is model and all(
                torch.equal(p, q) for p, q in zip(model.parameters(),
                                                  ema_ref_model.parameters()))
            for b in model.buffers():
                b.add_(1)                          # eval-side buffer churn
        checks["training weights restored"] = all(
            torch.equal(p, q) for p, q in zip(model.parameters(), train_w))
        checks["buffers restored"] = all(torch.equal(b, s) for b, s in zip(model.buffers(), bufs))

        # a full (unsharded-layout) state dict loads back into a sharded optimizer
        opt2, _ = make(model)
        zero2 = Zero1(model, opt2, rank, world, bucket_mb=bucket_mb)
        zero2.load_optimizer_state(want_opt)
        back = opt2.state_dict()
        try:
            assert set(back["state"]) == {i for i in want_opt["state"] if i in mine}
            _assert_same({i: back["state"][i] for i in back["state"]},
                         {i: want_opt["state"][i] for i in back["state"]})
            checks["a full state dict loads back sharded"] = True
        except AssertionError as e:
            checks[f"a full state dict loads back sharded ({e})"] = False
        torch.save(checks, os.path.join(root, f"opt{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", DEVICES)
@pytest.mark.parametrize("world, bucket_mb", [(2, 0.01), (3, 0.01), (3, 0.0), (2, 1000.0)])
def test_sharded_step_equals_single_process(tmp_path, kind, world, bucket_mb):
    ctx = torch.multiprocessing.get_context("spawn")
    lrs = local_ranks(world, kind)
    _run_procs([ctx.Process(target=_opt_worker,
                            args=(r, world, str(tmp_path / "store"), str(tmp_path), kind, lrs,
                                  bucket_mb))
                for r in range(world)])
    want = {"weights equal the single-process ones after every step",
            "only the non-finite step was dropped", "optimizer state only for owned parameters",
            "EMA shadows = the owned parameters", "the partitions add up to the unsharded state",
            "the probe was never stepped", "the undrawn adapters were skipped on some steps",
            "the EMA weights inside the block", "training weights restored", "buffers restored",
            "a full state dict loads back sharded"}
    for r in range(world):
        checks = torch.load(tmp_path / f"opt{r}.pt")
        failed = [k for k, ok in checks.items() if not ok]
        assert not failed, f"rank {r}: {failed}"
        assert want <= set(checks), want - set(checks)
        gathered = [k for k in checks if "gathered state == single-process state" in k]
        assert len(gathered) == (1 if r in (0, world - 1) else 0)


# ----------------------------------------------------------------------
# trainer level
# ----------------------------------------------------------------------

#: the scripted game draws of the aligned runs: step s's micro-batches (two
#: different games, so a BatchNorm layer of their legacy encoders never updates
#: on both ranks, and one game undrawn); rank r draws entry r, the single
#: process both
SCRIPT = [("othello", "chess"), ("chess", "gomoku"), ("gomoku", "othello"),
          ("othello", "gomoku"), ("chess", "othello")]


class _ScriptedRandom:
    """Stands in for train_multi_game's ``random`` module."""

    def __init__(self, rank):
        import random

        self.rank, self.t, self.seed = rank, 0, random.seed

    def choices(self, names, weights=None, k=1):
        pair = SCRIPT[self.t % len(SCRIPT)]
        self.t += 1
        out = list(pair) if self.rank is None else [pair[self.rank]]
        assert len(out) == k, (out, k)
        return out


def _common(data: str, save_dir: str, kind: str, overrides=()):
    dev = ["--device", "cpu"] if kind == "cpu" else ["--device", "cuda", "--dist_backend", "gloo"]
    return ["--data", f"othello={data}/othello", "--data", f"chess={data}/chess",
            "--data", f"gomoku={data}/gomoku", "--val", f"othello={data}/othello",
            "--batch_size", "2", "--warmup_steps", "3", "--eval_every", "2", "--log_every", "2",
            "--ema_rate", "0.9", "--moe_balance_rate", "0.01", "--clip_grad", "0.5",
            "--seed", "3", "--bucket_mb", "0.01", *dev, "--save_dir", save_dir,
            "--override", *[f"{k}={v}" for k, v in CORE_TF.items()], *overrides]


def _train_worker(rank, world, port, root, tag, argv, kind, lrs, poison, scripted):
    hide_gpus_if_cpu(kind)
    if world > 1:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                          LOCAL_RANK=str(lrs[rank]), WORLD_SIZE=str(world))
    else:
        for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            os.environ.pop(k, None)
    if kind == "cuda":
        # deterministic kernels: two separate runs must agree bit for bit
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_num_threads(1)
    from training import train_multi_game as tmg

    build, sync = tmg.build_net_tolerant, tmg.sync_gradients
    record = []         # per step: (games with a local adapter grad, games with one on ANY rank)

    def build_with_probe(game, arch):
        net, dropped = build(game, arch)
        if game == "chess":
            # trainable, but outside every forward: never a gradient
            net.zero_probe = torch.nn.Parameter(torch.full((3,), 2.0))
        return net, dropped

    def adapters(model):
        return sorted(g for g in model.games
                      if any(p.grad is not None for p in model[g].obs_encoder.parameters()))

    def recording_sync(model, world_, **kw):
        local = adapters(model)
        sync(model, world_, **kw)
        record.append((local, adapters(model)))
        if poison and rank == world - 1 and len(record) == NAN_STEP:
            next(p for p in model.parameters() if p.grad is not None).grad.fill_(math.nan)

    vals = []           # every validation's global per-game metrics
    evaluate_sharded = tmg.evaluate_sharded

    def recording_eval(*a, **kw):
        vals.append(evaluate_sharded(*a, **kw))
        return vals[-1]

    tmg.build_net_tolerant, tmg.sync_gradients = build_with_probe, recording_sync
    tmg.evaluate_sharded = recording_eval
    if scripted:
        tmg.random = _ScriptedRandom(rank if world > 1 else None)
    if poison and world == 1:
        clip, calls = torch.nn.utils.clip_grad_norm_, [0]

        def poisoning_clip(params, max_norm, *a, **kw):
            params = list(params)
            calls[0] += 1
            if calls[0] == NAN_STEP:
                next(p for p in params if p.grad is not None).grad.fill_(math.nan)
            return clip(params, max_norm, *a, **kw)

        torch.nn.utils.clip_grad_norm_ = poisoning_clip
    out = tmg.main(argv)
    model, ema, opt, zero = out["model"], out["ema"], out["optimizer"], out["zero"]
    probe = model["chess"].zero_probe
    shadows = ema.shadow if ema is not None else {}
    torch.save({
        "probe_index": next(i for i, p in enumerate(p for g in opt.param_groups
                                                    for p in g["params"]) if p is probe),
        "params": {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
        "buffers": {n: b.detach().cpu().clone() for n, b in model.named_buffers()},
        "ema": {n: v.detach().cpu().clone() for n, v in shadows.items()},
        "opt_state": {i: {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                          for k, v in st.items()}
                      for i, st in opt.state_dict()["state"].items()},
        "owned": list(zero.owned) if zero is not None else None,
        "ema_names": list(zero.names) if zero is not None else None,
        "state_bytes": (zero.local_state_bytes(ema) if zero is not None else
                        sum(t.numel() * t.element_size() for st in opt.state.values()
                            for t in st.values() if torch.is_tensor(t))
                        + sum(t.numel() * t.element_size() for t in shadows.values())),
        "record": record,
        "vals": vals,
        "trainable": sorted(n for n, p in model.named_parameters() if p.requires_grad),
        "device": str(next(model.parameters()).device),
    }, os.path.join(root, f"{tag}_rank{rank}.pt"))


def _train(root, tag, world, *extra, kind="cpu", data="data", poison=False, scripted=False,
           steps=STEPS, overrides=()):
    """One train_multi_game run on ``world`` spawned ranks (1: a single
    process); returns (per-rank outputs, multi_latest.pt, multi_final.pt).
    ``overrides``: config KEY=VALUE pairs on top of CORE_TF."""
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    lrs = local_ranks(world, kind)
    argv = [*_common(str(root / data), str(root / f"ck_{tag}"), kind, overrides),
            "--steps", str(steps), "--ckpt_every", str(steps), *extra, "--dist_timeout", "2"]
    _run_procs([ctx.Process(target=_train_worker,
                            args=(r, world, port, str(root), tag, argv, kind, lrs, poison,
                                  scripted))
                for r in range(world)])

    def load(p):
        return torch.load(p, weights_only=False, map_location="cpu")

    outs = [load(root / f"{tag}_rank{r}.pt") for r in range(world)]
    return (outs, load(root / f"ck_{tag}" / "multi_latest.pt"),
            load(root / f"ck_{tag}" / "multi_final.pt"))


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("zero1")
    for data, n in (("data", 6), ("aligned", 1)):
        for g in GAMES:
            (root / data / g).mkdir(parents=True)
            kw = {"board": 9} if g == "gomoku" else {}
            save_shard(str(root / data / g / "a.pt"), g, _board_samples(g, n, 6, **kw),
                       **({"meta": {"board_size": 9}} if g == "gomoku" else {}))
    return root


@pytest.fixture(scope="module")
def base(root):
    """Unsharded and sharded runs on 2 and 3 ranks: one micro-batch per rank
    per step (3 games: some adapter always lacks a gradient on every rank),
    a probe parameter no forward uses, a non-finite gradient norm on the
    last rank at step NAN_STEP."""
    kw = dict(poison=True)
    return {tag: _train(root, tag, world, "--games_per_step", "1",
                        *(["--shard_optimizer"] if tag[0] == "s" else []), **kw)
            for tag, world in (("u2", 2), ("s2", 2), ("u3", 3), ("s3", 3))}


def _assert_runs_identical(a, b, what):
    (ra, la, fa), (rb, lb, fb) = a, b
    _assert_same_checkpoint(la, lb, f"{what} latest")
    _assert_same_checkpoint(fa, fb, f"{what} final")
    for r, (x, y) in enumerate(zip(ra, rb)):
        _assert_same(x["params"], y["params"], f"{what} rank {r} params")
        _assert_same(x["buffers"], y["buffers"], f"{what} rank {r} buffers")
        # validation: the sharded run evaluates the live model with the
        # gathered EMA swapped in, the unsharded one a deep copy (ema_copy)
        _assert_same(x["vals"], y["vals"], f"{what} rank {r} validation")


@pytest.mark.parametrize("world", [2, 3])
def test_sharded_run_is_bit_identical_to_unsharded(base, world):
    _assert_runs_identical(base[f"u{world}"], base[f"s{world}"], f"world {world}")
    ranks, latest, _ = base[f"s{world}"]
    # the default per-frame GroupNorm encoders: no running statistics
    assert not any(n.endswith("running_mean") for n in ranks[0]["buffers"])
    assert len(ranks[0]["vals"]) == 3                # validation at steps 2, 4 and 5
    for r in ranks[1:]:                              # identical full weights everywhere
        _assert_same(ranks[0]["params"], r["params"], "rank params")
        _assert_same(ranks[0]["buffers"], r["buffers"], "rank buffers")
    assert latest["step"] == STEPS and latest["world_size"] == world
    assert latest["args"]["shard_optimizer"]


@pytest.mark.parametrize("world", [2, 3])
def test_ranks_hold_disjoint_partitions_of_the_gathered_state(base, world):
    """Every rank's optimizer state and EMA shadows are exactly its own
    partition of the checkpoint's (gathered) state; together they are all of
    it, and their bytes add up to the unsharded rank's."""
    ranks, latest, _ = base[f"s{world}"]
    unsharded = base[f"u{world}"][0]
    opt_state, ema_state = latest["optimizer_state"]["state"], latest["ema_state"]
    seen_opt, seen_ema = set(), set()
    for r in ranks:
        mine = set(r["owned"])
        assert set(r["opt_state"]) <= mine
        assert set(r["ema"]) == {r["ema_names"][i] for i in mine}
        _assert_same(r["opt_state"], {i: opt_state[i] for i in r["opt_state"]}, "opt part")
        _assert_same(r["ema"], {n: ema_state[n] for n in r["ema"]}, "ema part")
        assert not seen_opt & set(r["opt_state"]) and not seen_ema & set(r["ema"])
        seen_opt |= set(r["opt_state"])
        seen_ema |= set(r["ema"])
    assert seen_opt == set(opt_state) and seen_ema == set(ema_state)
    assert sum(r["state_bytes"] for r in ranks) == unsharded[0]["state_bytes"]
    assert max(r["state_bytes"] for r in ranks) < unsharded[0]["state_bytes"] * 0.75


@pytest.mark.parametrize("world", [2, 3])
def test_skipped_parameters_and_dropped_steps_stay_exact(base, world):
    """Sharded runs keep the contract of the unsharded ones: an adapter no
    rank drew and the unused probe stay grad-None (never stepped: no
    optimizer state, value untouched); the non-finite step is dropped on
    every rank (core step counters at STEPS - 1)."""
    ranks, latest, _ = base[f"s{world}"]
    record = ranks[0]["record"]
    assert len(record) == STEPS
    assert any(len(anywhere) < len(GAMES) for _, anywhere in record)
    for r in ranks:
        assert [a for _, a in r["record"]] == [a for _, a in record]    # agreed on every rank
        assert torch.equal(r["params"]["nets.chess.zero_probe"], torch.full((3,), 2.0))
    steps = [st["step"] for st in latest["optimizer_state"]["state"].values()]
    assert max(steps) == STEPS - 1 and min(steps) < STEPS - 1
    n_opt = sum(len(g["params"]) for g in latest["optimizer_state"]["param_groups"])
    assert len(latest["ema_state"]) == n_opt           # every trainable parameter
    assert ranks[0]["probe_index"] not in latest["optimizer_state"]["state"]


@pytest.mark.parametrize("extra", [("--freeze_core", "--train_core_norms"), ("--ema_rate", "0")],
                         ids=["frozen_core", "no_ema"])
def test_sharded_variants_are_bit_identical_to_unsharded(root, extra):
    """The paths the base runs do not take, on 3 ranks: a frozen core with
    trainable norms (frozen tensors are in the model but outside the
    optimizer, the partition and the EMA) and no EMA at all (validation on
    the training weights, checkpoints without ema_state)."""
    tag = "frozen" if "--freeze_core" in extra else "noema"
    runs = [_train(root, f"{mode}_{tag}", 3, "--games_per_step", "1", *extra, *shard,
                   poison=True)
            for mode, shard in (("u", ()), ("s", ("--shard_optimizer",)))]
    _assert_runs_identical(*runs, tag)
    ranks, latest, _ = runs[1]
    assert ranks[0]["vals"] and latest["args"]["shard_optimizer"]
    if tag == "frozen":
        trainable = set(ranks[0]["trainable"])
        assert len(trainable) < len(ranks[0]["params"]), "nothing frozen"
        assert set(latest["ema_state"]) == trainable == set(ranks[0]["ema_names"])
    else:
        assert latest["ema_state"] is None and not any(r["ema"] for r in ranks)


def _toy_worker(rank, world, init_file, root, bucket_mb):
    """Zero1 on a model with the parameter shapes the multi-game model lacks:
    a weight tied inside the model, a module registered under two names, a
    non-contiguous parameter, a float64 one (a second dtype bucket), one
    stepped every other step and one never — against single-process
    AdamATan2 + ModelEMA, bit for bit."""
    import torch.distributed as dist

    from model.optim import AdamATan2, ModelEMA
    from training.zero import Zero1

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world)
    try:
        torch.manual_seed(0)
        model = nn.Module()
        model.a, model.b = nn.Linear(8, 8), nn.Linear(8, 8)
        model.b.weight = model.a.weight
        model.sub = nn.Module()
        model.sub.alias = model.a
        model.t = nn.Parameter(torch.randn(6, 5).t())
        model.h = nn.Parameter(torch.randn(7, dtype=torch.float64))
        model.big = nn.Parameter(torch.randn(300))
        model.unused = nn.Parameter(torch.randn(4))
        model.bn = nn.BatchNorm1d(8)
        assert not model.t.is_contiguous()
        ref = copy.deepcopy(model)
        opt, ref_opt = AdamATan2(model.parameters(), lr=1e-2), AdamATan2(ref.parameters(), lr=1e-2)
        zero = Zero1(model, opt, rank, world, bucket_mb=bucket_mb)
        ema, ref_ema = zero.make_ema(model, 0.8), ModelEMA(ref, 0.8)
        for step in range(4):
            gen = torch.Generator().manual_seed(step)
            for (n, p), q in zip(model.named_parameters(), ref.parameters()):
                if n == "unused" or (n == "h" and step % 2):
                    continue
                g = torch.randn(p.shape, generator=gen, dtype=torch.float64).to(p.dtype)
                p.grad, q.grad = g.clone(), g.clone()
            zero.step()
            ema.update(model)
            ref_opt.step()
            ref_ema.update(ref)
            opt.zero_grad(set_to_none=True)
            ref_opt.zero_grad(set_to_none=True)
            assert all(torch.equal(p, q) for p, q in zip(model.parameters(), ref.parameters()))
        got_opt, got_ema = zero.gather_optimizer_state(), zero.gather_ema(ema)
        if rank == 0:
            want = ref_opt.state_dict()
            _assert_same(got_opt["param_groups"], want["param_groups"])
            _assert_same(dict(got_opt["state"]), dict(want["state"]))
            assert list(got_ema) == list(ref_ema.state_dict())
            _assert_same(got_ema, ref_ema.state_dict())
            _assert_same(zero.ema_model_state(model, got_ema), ref_ema.ema_model_state(ref))
        with zero.ema_weights(model, ema):
            ema_ref = ref_ema.ema_copy(ref)
            assert all(torch.equal(p, q) for p, q in zip(model.parameters(),
                                                          ema_ref.parameters()))
        assert all(torch.equal(p, q) for p, q in zip(model.parameters(), ref.parameters()))
        assert model.t.stride() == ref.t.stride()
        torch.save(True, os.path.join(root, f"toy{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world, bucket_mb", [(2, 0.0), (3, 1e-3)])
def test_tied_aliased_noncontiguous_and_mixed_dtype_parameters(tmp_path, world, bucket_mb):
    ctx = torch.multiprocessing.get_context("spawn")
    _run_procs([ctx.Process(target=_toy_worker,
                            args=(r, world, str(tmp_path / "store"), str(tmp_path), bucket_mb))
                for r in range(world)])
    assert all(torch.load(tmp_path / f"toy{r}.pt") for r in range(world))


@pytest.fixture(scope="module")
def resumed(root, base):
    """Resumes from the step-STEPS checkpoints to RESUME_STEPS: across
    sharded <-> unsharded and world sizes 2 -> 3, 3 -> 2, 1 -> 2."""
    sp = _train(root, "sp", 1, "--games_per_step", "2")      # a single-process checkpoint

    def res(tag, world, src, shard):
        return _train(root, tag, world, "--games_per_step", "1", "--resume",
                      str(root / f"ck_{src}" / "multi_latest.pt"),
                      *(["--shard_optimizer"] if shard else []), steps=RESUME_STEPS)

    return {"sp": sp,
            "s2>u3": res("s2_u3", 3, "s2", False), "u2>s3": res("u2_s3", 3, "u2", True),
            "u3>s2": res("u3_s2", 2, "u3", True), "s3>u2": res("s3_u2", 2, "s3", False),
            "sp>s2": res("sp_s2", 2, "sp", True), "sp>u2": res("sp_u2", 2, "sp", False)}


@pytest.mark.parametrize("a, b", [("s2>u3", "u2>s3"), ("u3>s2", "s3>u2"), ("sp>s2", "sp>u2")])
def test_resume_across_sharding_and_world_sizes(resumed, a, b):
    """Sharded checkpoints are unsharded ones (the base checkpoints agree bit
    for bit), so any mix of the two, at any world size, continues the same:
    the resumed runs agree bit for bit, and continue the step counters."""
    _assert_runs_identical(resumed[a], resumed[b], f"{a} vs {b}")
    for tag in (a, b):
        _, latest, _ = resumed[tag]
        assert latest["step"] == RESUME_STEPS
        steps = [st["step"] for st in latest["optimizer_state"]["state"].values()]
        # the base runs dropped their step NAN_STEP, the single-process prefix none
        assert max(steps) == RESUME_STEPS - (0 if tag.startswith("sp") else 1)


@pytest.fixture(scope="module")
def aligned(root):
    """The same micro-batches three ways: a single process with 2 per step,
    and 2 ranks with 1 each (rank r takes micro-batch r of the step), sharded
    and not; single-record datasets make every draw of a game the same batch,
    the game draws are scripted (SCRIPT).  Legacy BatchNorm encoders
    (encoder_norm=batch, the layout of checkpoints predating the field) so
    that the runs also pin the trainer's BatchNorm statistics merge; the other
    trainer-level runs build the default per-frame GroupNorm."""
    kw = dict(data="aligned", poison=True, scripted=True, overrides=("encoder_norm=batch",))
    return {"sp": _train(root, "al_sp", 1, "--games_per_step", "2", **kw),
            "u2": _train(root, "al_u2", 2, "--games_per_step", "1", **kw),
            "s2": _train(root, "al_s2", 2, "--games_per_step", "1", "--shard_optimizer", **kw)}


def test_aligned_two_rank_runs_are_bit_identical_to_single_process(aligned):
    """2 ranks x 1 micro-batch == 1 process x 2 micro-batches, bit for bit:
    (g_a + g_b) / 2 is exactly the single process's g_a/2 + g_b/2 (halving is
    exact), each BatchNorm layer updates on one rank only (merged verbatim),
    MoE loads add as integers — so weights, buffers, EMA, optimizer state and
    both checkpoints agree, sharded or not."""
    (sp, _, _), (u2, _, _) = aligned["sp"], aligned["u2"]
    assert [a for _, a in u2[0]["record"]] == [sorted(p) for p in SCRIPT]
    # every game's legacy encoder has BatchNorm statistics, and they moved
    for g in GAMES:
        counts = [b for n, b in sp[0]["buffers"].items()
                  if n.startswith(f"nets.{g}.obs_encoder") and n.endswith("num_batches_tracked")]
        assert counts and all(int(c) > 0 for c in counts), g
    _assert_runs_identical(aligned["u2"], aligned["s2"], "aligned u2 vs s2")
    for tag in ("u2", "s2"):
        ranks, latest, final = aligned[tag]
        _, sp_latest, sp_final = aligned["sp"]
        _assert_same_checkpoint(sp_latest, latest, f"single-process vs {tag} latest")
        _assert_same_checkpoint(sp_final, final, f"single-process vs {tag} final")
        _assert_same(sp[0]["params"], ranks[0]["params"], f"single-process vs {tag} params")
        _assert_same(sp[0]["buffers"], ranks[0]["buffers"], f"single-process vs {tag} buffers")
    steps = [st["step"] for st in aligned["sp"][1]["optimizer_state"]["state"].values()]
    assert max(steps) == STEPS - 1                                   # NAN_STEP dropped


@pytest.fixture(scope="module")
def cuda_base(root):
    kw = dict(kind="cuda", poison=True)
    return {tag: _train(root, f"cuda_{tag}", int(tag[1]), "--games_per_step", "1",
                        *(["--shard_optimizer"] if tag[0] == "s" else []), **kw)
            for tag in ("u2", "s2", "u3", "s3")}


@needs_cuda
@pytest.mark.parametrize("world", [2, 3])
def test_cuda_tensors_over_gloo_sharded_run_is_bit_identical(cuda_base, world):
    """--device cuda --dist_backend gloo (the MIG-slice mode): the sharded
    run equals the unsharded one bit for bit (deterministic kernels), and its
    ranks hold disjoint partitions of the gathered state."""
    u, s = f"u{world}", f"s{world}"
    for tag in (u, s):
        assert all(r["device"].startswith("cuda") for r in cuda_base[tag][0])
    _assert_runs_identical(cuda_base[u], cuda_base[s], f"cuda world {world}")
    ranks, latest, _ = cuda_base[s]
    seen = set()
    for r in ranks:
        assert set(r["opt_state"]) <= set(r["owned"]) and not seen & set(r["opt_state"])
        seen |= set(r["opt_state"])
        _assert_same(r["ema"], {n: latest["ema_state"][n] for n in r["ema"]}, "cuda ema part")
    assert seen == set(latest["optimizer_state"]["state"])

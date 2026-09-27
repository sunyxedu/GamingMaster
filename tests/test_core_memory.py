"""Episode memory inside the core (model/components/memory.py): gated per-position
recurrences and the global gated-delta associative memory between core blocks.

* added to a trained network they change nothing (zero-initialised outputs): the
  network with memory computes exactly what it computed without;
* the full-sequence forward equals chained single-frame steps (the rollout / replay
  identity RL needs), with the states carried in the Stage-1 cache list;
* causal: a frame never sees a later one; the memory does carry information from
  earlier frames once trained (its output depends on the history);
* a sequence in chunks — forward from a carried state (``init_cache``, ``t0``,
  ``return_cache``) — computes what the whole forward does, and chunked_batch_loss
  trains a long batch chunk by chunk (truncated BPTT);
* gradients reach the memory; StreamStepper and the fixed-shape cache handle the
  new state shapes; games share the core's memory.
"""

from __future__ import annotations

import pytest
import torch

from env.league import StreamStepper
from model.games import build_game_network
from model.multi_game import build_multi_game

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=3, temporal_mixer="recurrent",
            encoder_channels=16, moe_experts=4)
MEM = dict(core_gated_memory="0,2", core_delta_memory="1", core_delta_heads=2,
           core_delta_dk=8, core_delta_writes=2)


def _obs(B, T, g):
    return {"board_seq": (torch.rand(B, T, 3, 8, 8, generator=g) < 0.3).float()}


def _randomize_memory(net):
    torch.manual_seed(3)
    with torch.no_grad():
        for n, p in net.core.memory.named_parameters():
            p.add_(0.2 * torch.randn_like(p))


def test_memory_joins_a_trained_network_without_changing_it():
    torch.manual_seed(0)
    base = build_game_network("othello", **TINY).eval()
    torch.manual_seed(0)
    mem = build_game_network("othello", **TINY, **MEM).eval()
    missing, unexpected = mem.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("core.memory.") for k in missing)
    g = torch.Generator().manual_seed(1)
    obs = _obs(3, 7, g)
    with torch.no_grad():
        a, b = base(obs), mem(obs)
    for k in ("place_logits", "value"):
        assert torch.equal(a[k], b[k]), k
    n_mem = sum(p.numel() for p in mem.core.memory.parameters())
    assert n_mem > 0 and len(mem.initial_st_cache(2)) == len(mem.st_layers) + 3


def test_forward_equals_chained_steps_and_is_causal():
    torch.manual_seed(0)
    net = build_game_network("othello", **TINY, **MEM).eval()
    _randomize_memory(net)
    g = torch.Generator().manual_seed(2)
    B, T = 2, 6
    obs = _obs(B, T, g)
    with torch.no_grad():
        full = net(obs)
        cache = None
        vals, logits = [], []
        for t in range(T):
            out = net.step({k: v[:, t] for k, v in obs.items()}, st_cache=cache, step_idx=t)
            cache = out["st_cache"]
            vals.append(out["value"])
            logits.append(out["place_logits"])
    torch.testing.assert_close(torch.stack(vals, 1), full["value"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.stack(logits, 1), full["place_logits"], rtol=1e-5, atol=1e-5)
    # causality: changing the last frames leaves the earlier outputs alone
    obs2 = {k: v.clone() for k, v in obs.items()}
    obs2["board_seq"][:, 4:] = 1.0 - obs2["board_seq"][:, 4:]
    with torch.no_grad():
        full2 = net(obs2)
    assert torch.equal(full2["value"][:, :4], full["value"][:, :4])
    # the memory carries history: an earlier change moves a later output
    obs3 = {k: v.clone() for k, v in obs.items()}
    obs3["board_seq"][:, 0] = 1.0 - obs3["board_seq"][:, 0]
    with torch.no_grad():
        full3 = net(obs3)
    assert not torch.allclose(full3["value"][:, 5], full["value"][:, 5])


def test_delta_memory_alone_carries_history():
    """Only the global delta memory crosses frames in the core: the Stage-1 recurrence
    is bypassed by comparing a net whose memory is live against the same net with the
    delta memory's output switched off."""
    torch.manual_seed(0)
    net = build_game_network("othello", **TINY, core_delta_memory="2", core_delta_heads=2,
                             core_delta_dk=8, core_delta_writes=2).eval()
    _randomize_memory(net)
    g = torch.Generator().manual_seed(4)
    obs = _obs(2, 5, g)
    with torch.no_grad():
        live = net(obs)["value"]
        net.core.memory.delta["2"].out_proj.weight.zero_()
        off = net(obs)["value"]
    assert torch.equal(live[:, 0], off[:, 0])            # the first frame reads an empty memory
    assert not torch.allclose(live[:, 1:], off[:, 1:])


@pytest.mark.parametrize("mem", [{}, MEM], ids=["stage1_only", "core_memory"])
def test_chunked_forward_equals_the_whole_sequence(mem):
    """forward over frames [0, s) with return_cache, then over [s, T) from that cache
    at t0=s, computes what one forward over [0, T) does (what chunked truncated-BPTT
    training relies on); the chunk's cache also seeds step() at frame s."""
    torch.manual_seed(0)
    net = build_game_network("othello", **TINY, **mem).eval()
    if mem:
        _randomize_memory(net)
    g = torch.Generator().manual_seed(7)
    B, T, s = 2, 9, 4
    obs = _obs(B, T, g)
    with torch.no_grad():
        full = net(obs)
        a = net({k: v[:, :s] for k, v in obs.items()}, return_cache=True)
        b = net({k: v[:, s:] for k, v in obs.items()}, init_cache=a["st_cache"], t0=s,
                return_cache=True)
        nxt = net.step({k: v[:, s] for k, v in obs.items()}, st_cache=a["st_cache"],
                       step_idx=s)
    assert len(a["st_cache"]) == len(net.initial_st_cache(B))
    for k in ("value", "place_logits"):
        torch.testing.assert_close(torch.cat([a[k], b[k]], 1), full[k], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(nxt["value"], full["value"][:, s], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="recurrent temporal mixer"):
        build_game_network("othello", **{**TINY, "temporal_mixer": "attention"})(
            obs, return_cache=True)


def test_chunked_forward_passes_absolute_time_to_the_decoder():
    """clash_royale's frame-time wait PE reads the absolute frame index: a chunk decodes
    at t0 = its first frame."""
    from model.games.clash_royale import CardPlacementDecoder, ClashRoyaleConfig, ClashRoyaleEncoder
    from model.game_network import GameNetwork
    from tests.test_games_clash_royale import _make_obs

    cfg = ClashRoyaleConfig(d_model=32, n_heads=4, d_ff=64, dropout=0.0, n_st_layers=1,
                            core_type="transformer", core_layers=3, temporal_mixer="recurrent",
                            encoder_channels=16, arena_h=8, arena_w=6, arena_channels=5,
                            hand_size=4, n_cards=20, cr_global_features=6,
                            cr_card_tokens="deck", card_feature_dim=9, cr_wait_time="frame",
                            **MEM)
    torch.manual_seed(0)
    net = GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg)).eval()
    _randomize_memory(net)
    obs = _make_obs(cfg, B=2, T=7, seed=3)
    with torch.no_grad():
        full = net(obs)["wait_logits"]
        a = net({k: v[:, :3] for k, v in obs.items()}, return_cache=True)
        b = net({k: v[:, 3:] for k, v in obs.items()}, init_cache=a["st_cache"], t0=3)
        b0 = net({k: v[:, 3:] for k, v in obs.items()}, init_cache=a["st_cache"])
    torch.testing.assert_close(b["wait_logits"], full[:, 3:], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(b0["wait_logits"], full[:, 3:])


def test_chunked_batch_loss_trains_long_batches_in_chunks():
    """chunked_batch_loss: the chunks' losses weigh like the whole batch's (same stats
    as batch_loss on it); the decoder's gradient — no path through the carried state —
    is the whole batch's; the chunks' backward calls hold at most one chunk."""
    from training.train_il_game import batch_loss, chunked_batch_loss

    torch.manual_seed(0)
    net = build_game_network("othello", **TINY, **MEM).train()
    _randomize_memory(net)
    g = torch.Generator().manual_seed(8)
    B, T = 2, 10
    obs = _obs(B, T, g)
    fm = torch.ones(B, T, dtype=torch.bool)
    fm[1, 7:] = False                                      # tail padding of the shorter row
    tgt = {"place_idx": torch.randint(0, 64, (B, T), generator=g),
           "passed": torch.zeros(B, T, dtype=torch.bool), "value": torch.rand(B, T) * 2 - 1}
    mb = dict(obs=obs, tgt=tgt, frame_mask=fm, labels=None, legal=None)
    args = (torch.device("cpu"), 1.0, 0.0, False)

    def grads():
        return {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    net.zero_grad()
    loss, whole_stats, _ = batch_loss(net, mb, *args)
    loss.backward()
    whole = grads()
    net.zero_grad()
    lengths = []
    stats, pred = chunked_batch_loss(net, mb, args[0], 4,
                                     lambda l: (lengths.append(1), l.backward()), *args[1:])
    chunked = grads()
    assert len(lengths) == 3 and pred["q_halt"].shape == (B, T)
    for k in ("loss", "value_loss"):
        assert stats[k] == pytest.approx(whole_stats[k], rel=1e-4), k
    dec = [n for n in whole if n.startswith("action_decoder.")]
    assert dec
    for n in dec:
        torch.testing.assert_close(chunked[n], whole[n], rtol=1e-4, atol=1e-6)
    # the memory's own parameters train in both (their gradient differs: it is cut at
    # the chunk boundaries)
    mem_grads = [n for n in chunked if ".memory." in n and n.endswith("out_proj.weight")]
    assert mem_grads and all(chunked[n].abs().sum() > 0 for n in mem_grads)
    # short batches take the plain path: one backward
    net.zero_grad()
    calls = []
    chunked_batch_loss(net, mb, args[0], 16, lambda l: (calls.append(1), l.backward()),
                       *args[1:])
    assert len(calls) == 1 and grads().keys() == whole.keys()


def test_gradients_reach_the_memory():
    torch.manual_seed(0)
    net = build_game_network("othello", **TINY, **MEM).train()
    g = torch.Generator().manual_seed(5)
    out = net(_obs(2, 4, g))
    (out["value"].sum() + out["place_logits"].logsumexp(-1).sum()).backward()
    for name, p in net.core.memory.named_parameters():
        if "out_proj" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_stream_stepper_and_fixed_cache_carry_the_memory_states():
    torch.manual_seed(0)
    net = build_game_network("othello", **TINY, **MEM).eval()
    _randomize_memory(net)
    g = torch.Generator().manual_seed(6)
    B, T = 3, 4
    obs = _obs(B, T, g)
    with torch.no_grad():
        full = net(obs)["value"]
        # fixed-shape cache (the CUDA-graph / fixed actor path)
        cache = net.initial_st_cache(B)
        shapes = [tuple(c.shape) for c in cache]
        for t in range(T):
            out = net.step({k: v[:, t] for k, v in obs.items()}, st_cache=cache,
                           step_idx=torch.tensor([t]))
            cache = out["st_cache"]
            assert [tuple(c.shape) for c in cache] == shapes
        torch.testing.assert_close(out["value"], full[:, -1], rtol=1e-5, atol=1e-5)
        # per-stream stepping with streams in different orders
        st = StreamStepper(net, act_steps=None, device=torch.device("cpu"))
        got = torch.zeros(B, T)
        for t in range(T):
            order = [2, 0, 1] if t % 2 else [0, 1, 2]
            o = {k: v[order, t] for k, v in obs.items()}
            res = st.step(o, None, order, [t] * B)
            got[order, t] = res["value"]
    torch.testing.assert_close(got, full, rtol=1e-5, atol=1e-5)


def test_games_share_the_core_memory():
    mg = build_multi_game(["othello", "chess"], {**TINY, **MEM})
    a, b = mg["othello"], mg["chess"]
    assert a.core.memory is b.core.memory
    with pytest.raises(ValueError, match="transformer"):
        build_game_network("othello", **{**TINY, "core_type": "recurrent_depth"}, **MEM)


def test_resume_adds_core_memory_to_a_trained_run(tmp_path):
    """train_multi_game --resume with core memory sub-layers the checkpoint lacks: the
    checkpoint's weights, optimizer state and EMA load, the memory starts fresh (zero
    outputs, no optimizer state) and trains."""
    from data_processing.games.dataset import save_shard
    from tests.test_core_moe import _board_samples
    from training import train_multi_game

    for g in ("othello", "chess"):
        (tmp_path / g).mkdir()
        save_shard(str(tmp_path / g / "a.pt"), g, _board_samples(g, 4, 6))
    core = [f"{k}={v}" for k, v in TINY.items()]
    base = ["--data", f"othello={tmp_path / 'othello'}", "--data", f"chess={tmp_path / 'chess'}",
            "--batch_size", "2", "--warmup_steps", "1", "--eval_every", "0", "--device", "cpu"]
    train_multi_game.main(base + ["--steps", "2", "--ckpt_every", "2",
                                  "--save_dir", str(tmp_path / "a"), "--override", *core])
    ck = torch.load(tmp_path / "a" / "multi_latest.pt", weights_only=False)
    mem = [f"{k}={v}" for k, v in MEM.items()]
    train_multi_game.main(base + ["--steps", "4", "--save_dir", str(tmp_path / "b"),
                                  "--resume", str(tmp_path / "a" / "multi_latest.pt"),
                                  "--override", *core, *mem])
    ck2 = torch.load(tmp_path / "b" / "multi_final.pt", weights_only=False)
    st = ck2["train_model_state"]
    mem_keys = [k for k in st if ".core.memory." in k]
    assert mem_keys and not any(".core.memory." in k for k in ck["train_model_state"])
    # the memory's zero output trained away from zero (it gets gradients)
    outs = [k for k in mem_keys if k.endswith("out_proj.weight")]
    assert outs and any(st[k].abs().sum() > 0 for k in outs)
    assert (ck2.get("configs") or {})["othello"]["core_delta_memory"] == "1"

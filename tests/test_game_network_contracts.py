"""Contracts at the game-adapter boundary.

- ``GeneralsDecoder._sample_source`` now delegates to the shared
  ``masked_sample`` guard; the equivalence test pins it bit-for-bit
  against the previous hand-rolled implementation (same RNG stream, same
  fallbacks for all-masked rows and NaN logits).
- ``GameNetwork`` validates that the encoder produces exactly the
  ``n_tokens`` it declares — the positional embedding and the TRM q_halt
  index are built from that constant, so a lying encoder must fail with a
  clear error instead of a broadcast traceback (or a silent q-index
  fallback if the shapes happen to line up).
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.game_network import GameNetwork
from model.games.generals import GeneralsDecoder, GeneralsEncoder

SMALL_CFG = dict(
    d_model=32, n_heads=4, d_ff=64, encoder_channels=16, n_st_layers=1,
    core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1, dropout=0.0,
)


# ======================================================================
# _sample_source ≡ previous hand-rolled implementation
# ======================================================================

def _reference_sample_source(source_logits, legal_mask, BT, N):
    """Verbatim copy of GeneralsDecoder._sample_source before it delegated
    to base.masked_sample — the behavior contract."""
    with torch.no_grad():
        logits_safe = source_logits.detach().clone().float()
        if legal_mask is not None:
            lm = legal_mask.reshape(BT, N)
            logits_safe = logits_safe.masked_fill(~lm, float("-inf"))
            all_masked = logits_safe.isinf() & (logits_safe < 0)
            bad_rows = all_masked.all(dim=-1)
            if bad_rows.any():
                logits_safe[bad_rows, 0] = 0.0
        nan_rows = logits_safe.isnan().any(dim=-1)
        if nan_rows.any():
            logits_safe[nan_rows] = 0.0
        probs = F.softmax(logits_safe, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


def _cases():
    torch.manual_seed(0)
    BT, N = 6, 25
    logits = torch.randn(BT, N) * 3

    mask = torch.rand(BT, N) < 0.4
    mask[:, 0] = True  # ≥1 legal per row

    fully_masked = mask.clone()
    fully_masked[2] = False  # row 2: nothing legal → fallback to index 0

    nan_logits = logits.clone()
    nan_logits[4, 7] = float("nan")  # row 4: NaN → uniform fallback

    # NaN at column 0, which mask[:, 0] = True keeps legal — the NaN
    # SURVIVES masked_fill, so this is the case that actually reaches the
    # uniform fallback under a mask (a NaN at a masked position is
    # overwritten with -inf before the NaN check and never triggers it).
    nan_at_legal = logits.clone()
    nan_at_legal[4, 0] = float("nan")

    return BT, N, [
        ("masked", logits, mask),
        ("all-masked-row", logits, fully_masked),
        ("nan-row-masked", nan_logits, mask),
        ("nan-at-legal-position-masked", nan_at_legal, mask),
        ("nan-row-unmasked", nan_logits, None),
        ("unmasked", logits, None),
    ]


@pytest.mark.parametrize("name,logits,mask",
                         _cases()[2], ids=[c[0] for c in _cases()[2]])
def test_sample_source_matches_reference(name, logits, mask):
    BT, N, _ = _cases()
    decoder = GeneralsDecoder(ModelConfig(**SMALL_CFG))
    # legal_mask arrives shaped like [B, T, 1, H, W]; any reshape(BT, N)-able
    # layout is accepted — feed it as [BT, 1, N] to exercise the reshape.
    legal = mask.reshape(BT, 1, N) if mask is not None else None

    torch.manual_seed(42)
    expected = _reference_sample_source(logits, legal, BT, N)
    torch.manual_seed(42)
    actual = decoder._sample_source(logits, legal, BT, N)

    assert torch.equal(actual, expected), name
    if mask is not None:
        for r in range(BT):
            # Rows with ≥1 legal source and finite logits must sample legally
            # (the all-masked and NaN fallbacks intentionally ignore the mask).
            if mask[r].any() and not logits[r].isnan().any():
                assert mask[r, actual[r]], f"row {r} sampled an illegal source"


def test_sample_source_no_grad_and_dtype():
    decoder = GeneralsDecoder(ModelConfig(**SMALL_CFG))
    logits = torch.randn(4, 25, requires_grad=True)
    out = decoder._sample_source(logits, None, 4, 25)
    assert out.dtype == torch.long
    assert not out.requires_grad
    assert logits.grad is None  # sampling never touches the graph


# ======================================================================
# Encoder n_tokens contract
# ======================================================================

class _LyingEncoder(GeneralsEncoder):
    """Declares more tokens than it produces (forward emits the real
    grid_h*grid_w; encode_step is overridden because the stock one builds
    its reshape from n_tokens and would fail before reaching the core)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_tokens = cfg.grid_h * cfg.grid_w + 3

    def encode_step(self, obs):
        grid = obs["grid_seq"]
        B, d = grid.shape[0], self.cfg.d_model
        spatial = self.spatial_encoder(grid)
        spatial = spatial + self.global_proj(
            obs["globals_seq"]).unsqueeze(-1).unsqueeze(-1)
        n_real = self.cfg.grid_h * self.cfg.grid_w
        return spatial.reshape(B, d, n_real).permute(0, 2, 1)


def test_forward_rejects_encoder_token_count_mismatch():
    cfg = ModelConfig(**SMALL_CFG)
    net = GameNetwork(cfg, _LyingEncoder(cfg), GeneralsDecoder(cfg))
    obs = {"grid_seq": torch.randn(1, 2, 12, 25, 25),
           "globals_seq": torch.randn(1, 2, 4)}
    with pytest.raises(RuntimeError, match="n_tokens"):
        net(obs, act_steps=1)


def test_step_rejects_encoder_token_count_mismatch():
    cfg = ModelConfig(**SMALL_CFG)
    net = GameNetwork(cfg, _LyingEncoder(cfg), GeneralsDecoder(cfg))
    obs = {"grid_seq": torch.randn(1, 12, 25, 25),
           "globals_seq": torch.randn(1, 4)}
    with pytest.raises(RuntimeError, match="n_tokens"):
        net.step(obs)


def test_forward_accepts_conforming_encoder():
    cfg = ModelConfig(**SMALL_CFG)
    net = GameNetwork(cfg, GeneralsEncoder(cfg), GeneralsDecoder(cfg))
    net.eval()
    obs = {"grid_seq": torch.randn(1, 2, 12, 25, 25),
           "globals_seq": torch.randn(1, 2, 4)}
    with torch.no_grad():
        out = net(obs, act_steps=1)
    assert out["source_logits"].shape == (1, 2, 625)
    assert out["value"].shape == (1, 2)


# ======================================================================
# q_head gradient contract across ACT depths (autocast cache poisoning)
# ======================================================================

@pytest.mark.parametrize("act_steps", [1, 2, 5])
@pytest.mark.parametrize("use_autocast", [False, True])
def test_qhead_receives_gradient_at_every_act_depth(act_steps, use_autocast):
    """Every parameter — q_head above all — must receive a gradient at
    every TRM ACT depth, with and without bf16 autocast.

    On torch 2.5.0a0 (NGC 24.09) the autocast weight-cast cache was
    poisoned by the no_grad ponder loop when act_steps > 1: the cached
    bf16 copy of q_head.weight carried no graph edge, q_halt still
    required grad via its input, and q_head.grad came back None — which
    desynced the manual NCCL gradient sync in production.
    ``_trm_ponder`` now clears the autocast cache after the no_grad loop.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(**SMALL_CFG, temporal_mixer="recurrent")
    net = GameNetwork(cfg, GeneralsEncoder(cfg), GeneralsDecoder(cfg))
    net.train()
    obs = {"grid_seq": torch.randn(1, 3, 12, 25, 25),
           "globals_seq": torch.randn(1, 3, 4)}

    ctx = torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=use_autocast)
    with ctx:
        out = net(obs, act_steps=act_steps)
        # Touch every differentiable output, as the real loss does — heads
        # excluded from the loss legitimately have no gradient.
        loss = sum(v.float().sum() for v in out.values()
                   if torch.is_tensor(v) and v.requires_grad)
    loss.backward()

    assert out["q_halt"].requires_grad
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"act_steps={act_steps} autocast={use_autocast}: {missing}"


def test_sampling_temperature_greedy_sharpened_and_restored():
    """masked_sample at temperature 0 is the masked argmax; at 1 (the default, and again after
    the block) it is bit-identical to the plain softmax draw; a lower temperature sharpens the
    draw; masked entries are never drawn at any temperature; a negative temperature is refused."""
    from model.games import base

    torch.manual_seed(0)
    logits = torch.randn(512, 9)
    mask = torch.rand(512, 9) > 0.3
    mask[:, 0] = True
    masked = logits.masked_fill(~mask, float("-inf"))
    with base.sampling_temperature(0.0):
        g = base.masked_sample(logits, mask)
    assert torch.equal(g, masked.argmax(-1))
    gen_state = torch.get_rng_state()
    plain = base.categorical_sample(F.softmax(base.safe_masked_logits(logits, mask), -1))
    torch.set_rng_state(gen_state)
    assert torch.equal(base.masked_sample(logits, mask), plain)          # temperature 1: unchanged
    top = masked.argmax(-1)
    hits = {}
    for t in (1.0, 0.25):
        n = 0
        with base.sampling_temperature(t):
            for _ in range(40):
                s = base.masked_sample(logits, mask)
                assert bool(mask.gather(1, s.unsqueeze(1)).all())          # never an illegal entry
                n += int((s == top).sum())
        hits[t] = n
    assert hits[0.25] > hits[1.0]
    assert base.current_sampling_temperature() == 1.0                    # restored after the block
    with pytest.raises(ValueError):
        with base.sampling_temperature(-0.5):
            pass
    try:
        with base.sampling_temperature(0.0):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert base.current_sampling_temperature() == 1.0                    # restored on an error too


def test_sampling_temperature_is_thread_local():
    """A sampling_temperature block does not leak into another thread: an evaluation game
    drawn greedily on one thread leaves a concurrent rollout thread sampling at 1."""
    import threading

    from model.games import base

    entered, release = threading.Event(), threading.Event()
    seen = {}

    def evaluator():
        with base.sampling_temperature(0.0):
            seen["inside"] = base.current_sampling_temperature()
            entered.set()
            release.wait(5.0)
        seen["after"] = base.current_sampling_temperature()

    t = threading.Thread(target=evaluator)
    t.start()
    assert entered.wait(5.0)
    seen["other_thread"] = base.current_sampling_temperature()
    release.set()
    t.join(5.0)
    assert seen == {"inside": 0.0, "other_thread": 1.0, "after": 1.0}

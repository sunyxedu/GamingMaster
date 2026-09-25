"""RL legal-masking tests.

The PPO-exactness invariant must hold on the MASKED path exactly as it
does on the unmasked one: step(legal_mask, direction_mask) during rollout
and forward(legal_mask, direction_mask) + compute_log_prob during the
update must produce identical log-probs (ratio == 1).

Also checks that masked sampling never emits an illegal source/direction
and that the env's dir_mask matches get_valid_directions.
"""

import pytest
import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.games.base import masked_log_prob, safe_masked_logits
from model.games.board import _masked_policy_logits
from model.games.clash_royale import _mask_logits as _cr_mask_logits
from model.games.hearthstone import _mask_logits as _hs_mask_logits
from model.games.mahjong import _mask_logits as _mj_mask_logits
from model.games.pokemon import PokemonDecoder
from model.network import GeneralsNetwork
from model.rl_utils import compute_policy_kl_heads


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
                       trm_block_layers=1)


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    m = GeneralsNetwork(cfg)
    m.eval()
    return m


def _rollout(model, env, T=6, act_steps=1, zero_legal_at=None):
    """Masked rollout via step(); returns stacked tensors for replay.

    ``zero_legal_at`` presents the model with an all-False source mask at
    that step — the trainer's no-legal-source case (turn 0, all armies
    == 1): the buffer records the sampled fallback action verbatim while
    the env receives a pass.
    """
    obs = env.reset(seed=7)
    cache = None
    rec = {k: [] for k in ("grid", "glb", "legal", "dirm", "src", "dir",
                           "spl", "passed", "lp")}
    for t in range(T):
        grid = obs[0]["grid"].unsqueeze(0)
        glb = obs[0]["globals"].unsqueeze(0)
        legal = obs[0]["legal_mask"].unsqueeze(0)
        if t == zero_legal_at:
            legal = torch.zeros_like(legal)
        dirm = obs[0]["dir_mask"].unsqueeze(0)
        with torch.no_grad():
            out = model.step(grid, glb, legal_mask=legal, st_cache=cache,
                             step_idx=t, act_steps=act_steps,
                             direction_mask=dirm)
        cache = out["st_cache"]
        rec["grid"].append(grid[0])
        rec["glb"].append(glb[0])
        rec["legal"].append(legal[0])
        rec["dirm"].append(dirm[0])
        rec["src"].append(out["action_source"].item())
        rec["dir"].append(out["action_direction"].item())
        rec["spl"].append(out["action_split"].item())
        rec["passed"].append(bool(out["action_pass"].item()))
        rec["lp"].append(out["action_log_prob"].item())

        action = None
        if not rec["passed"][-1] and bool(legal.any()):
            action = {"source_idx": rec["src"][-1],
                      "direction": rec["dir"][-1],
                      "split": rec["spl"][-1]}
        obs, _, done, _ = env.step(action, None)
        if done:
            break
    return rec


def _forward(model, cfg, rec):
    """Sequence forward over a recorded rollout (the ppo_update forward)."""
    grid_seq = torch.stack(rec["grid"]).unsqueeze(0)
    glb_seq = torch.stack(rec["glb"]).unsqueeze(0)
    legal_seq = torch.stack(rec["legal"]).unsqueeze(0)   # [1,T,1,H,W]
    dirm_seq = torch.stack(rec["dirm"]).unsqueeze(0)     # [1,T,4,H,W]
    actions = {
        "source_idx": torch.tensor(rec["src"]).unsqueeze(0),
        "direction": torch.tensor(rec["dir"]).unsqueeze(0),
        "split": torch.tensor(rec["spl"]).unsqueeze(0),
        "passed": torch.tensor(rec["passed"]).unsqueeze(0),
    }
    with torch.no_grad():
        pred = model(
            grid_seq, glb_seq,
            legal_mask=legal_seq,
            source_y=actions["source_idx"] // cfg.grid_w,
            source_x=actions["source_idx"] % cfg.grid_w,
            act_steps=1,
            direction_mask=dirm_seq,
        )
    return pred, actions, legal_seq


def _replay(model, cfg, rec):
    """Sequence forward + compute_log_prob over a recorded rollout."""
    pred, actions, legal_seq = _forward(model, cfg, rec)
    with torch.no_grad():
        lp, ent, _ = model.compute_log_prob(
            pred, actions, legal={"source": legal_seq})
    return lp, ent


class TestMaskedPPOExactness:
    def test_masked_rollout_replay_ratio_is_one(self, model, cfg):
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        rec = _rollout(model, env, T=6)
        lp, ent = _replay(model, cfg, rec)

        old = torch.tensor(rec["lp"]).unsqueeze(0)
        ratio = (lp - old).exp()
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), \
            f"masked rollout/replay mismatch: max|ratio-1|={(ratio - 1).abs().max()}"
        assert torch.isfinite(ent).all()

    def test_all_masked_step_replay_ratio_is_one(self, model, cfg):
        """A step with NO legal source books through the one-hot-index-0
        fallback; the replay must hit the same fallback on the [B, T, N]
        path and land ratio == 1.  Regression: the replay fallback wrote
        via mask+integer advanced indexing, which is invalid for the 2-D
        row mask and raised IndexError on the first real no-source step."""
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        rec = _rollout(model, env, T=6, zero_legal_at=2)
        assert len(rec["lp"]) > 2 and not rec["legal"][2].any()
        assert rec["src"][2] == 0  # all-masked fallback source

        lp, ent = _replay(model, cfg, rec)
        old = torch.tensor(rec["lp"]).unsqueeze(0)
        ratio = (lp - old).exp()
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), \
            f"all-masked step replay mismatch: max|ratio-1|={(ratio - 1).abs().max()}"
        assert torch.isfinite(ent).all()

    def test_all_masked_step_kl_anchor(self, model, cfg):
        """The KL anchor must survive an all-masked step (regression:
        compute_policy_kl's fallback used the same invalid indexing) and
        be exactly 0 against an identical reference distribution."""
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        rec = _rollout(model, env, T=6, zero_legal_at=2)
        assert not rec["legal"][2].any()

        pred, actions, legal_seq = _forward(model, cfg, rec)
        kls = compute_policy_kl_heads(
            pred, pred, legal_mask=legal_seq, passed=actions["passed"])
        for k, v in kls.items():
            assert torch.isfinite(v), f"{k} not finite"
            assert v.abs() < 1e-6, f"{k} != 0 against identical ref: {v}"

    def test_masked_sampling_never_illegal(self, model, cfg):
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        obs = env.reset(seed=11)
        legal = obs[0]["legal_mask"]
        dirm = obs[0]["dir_mask"]
        cache = None
        for t in range(12):
            with torch.no_grad():
                out = model.step(
                    obs[0]["grid"].unsqueeze(0),
                    obs[0]["globals"].unsqueeze(0),
                    legal_mask=obs[0]["legal_mask"].unsqueeze(0),
                    st_cache=cache, step_idx=t, act_steps=1,
                    direction_mask=obs[0]["dir_mask"].unsqueeze(0))
            cache = out["st_cache"]
            src = out["action_source"].item()
            d = out["action_direction"].item()
            r, c = divmod(src, cfg.grid_w)
            legal = obs[0]["legal_mask"]
            dirm = obs[0]["dir_mask"]
            has_moves = bool(legal.any())
            if not bool(out["action_pass"].item()) and has_moves:
                assert legal[0, r, c], f"illegal source sampled at step {t}"
                assert dirm[d, r, c], f"illegal direction sampled at step {t}"
                action = {"source_idx": src, "direction": d,
                          "split": out["action_split"].item()}
            else:
                # No legal source (all-masked fallback) => collect sends pass.
                action = None
            obs, _, done, info = env.step(action, None)
            assert not info["illegal"][0], "masked action was illegal in env"
            if done:
                break


class TestGreedyEval:
    def test_greedy_rollout_deterministic(self, model, cfg):
        """decoder.greedy=True + same reset seed => identical action streams
        (the paired seat-swapped eval depends on this)."""
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        model.action_decoder.greedy = True
        try:
            streams = []
            for _ in range(2):
                obs = env.reset(seed=99)
                cache = None
                acts = []
                for t in range(8):
                    with torch.no_grad():
                        out = model.step(
                            obs[0]["grid"].unsqueeze(0),
                            obs[0]["globals"].unsqueeze(0),
                            legal_mask=obs[0]["legal_mask"].unsqueeze(0),
                            st_cache=cache, step_idx=t, act_steps=1,
                            direction_mask=obs[0]["dir_mask"].unsqueeze(0))
                    cache = out["st_cache"]
                    acts.append((out["action_source"].item(),
                                 out["action_direction"].item(),
                                 out["action_split"].item(),
                                 bool(out["action_pass"].item())))
                    a = None
                    if not acts[-1][3] and bool(obs[0]["legal_mask"].any()):
                        a = {"source_idx": acts[-1][0],
                             "direction": acts[-1][1], "split": acts[-1][2]}
                    obs, _, done, _ = env.step(a, None)
                    if done:
                        break
                streams.append(acts)
        finally:
            model.action_decoder.greedy = False
        assert streams[0] == streams[1]


class TestAllMaskedFallback:
    """Shared-helper guards for degenerate rows on [B, T, N] inputs —
    the replay-path shape (rollout passes [BT, N] with a 1-D row mask)."""

    @pytest.mark.parametrize("fn", [
        safe_masked_logits,
        _masked_policy_logits,
        _cr_mask_logits,
        _hs_mask_logits,
        _mj_mask_logits,
        PokemonDecoder._mask_logits,
    ], ids=["base", "board", "clash_royale", "hearthstone", "mahjong",
            "pokemon"])
    def test_mask_helper_3d_all_masked_row(self, fn):
        logits = torch.randn(2, 3, 5)
        mask = torch.ones(2, 3, 5, dtype=torch.bool)
        mask[0, 1] = False
        safe = fn(logits, mask)
        assert safe[0, 1, 0] == 0.0
        assert torch.isinf(safe[0, 1, 1:]).all()
        assert torch.allclose(safe[1], logits[1].float())

    def test_masked_log_prob_3d_all_masked_row(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        mask = torch.ones(2, 3, 5, dtype=torch.bool)
        mask[0, 1] = False
        idx = torch.zeros(2, 3, dtype=torch.long)
        lp = masked_log_prob(logits, idx, mask)
        assert lp[0, 1] == 0.0  # one-hot fallback: log p(0) == 0
        lp.sum().backward()
        assert torch.isfinite(logits.grad).all()


class TestDirMask:
    def test_env_dir_mask_matches_get_valid_directions(self):
        env = GeneralsEnv(n_mountains=25, n_cities=3)
        env.reset(seed=5)
        dirm = env.dir_valid
        for idx in list(env.player_land[0]) + list(env.player_land[1]):
            valid = env.get_valid_directions(idx)
            r, c = divmod(idx, env.grid_w)
            for d in range(4):
                assert bool(dirm[d, r, c]) == valid[d]

    def test_masked_direction_logits_suppressed(self, model, cfg):
        """Direction logits at the chosen source are -1e9 on invalid dirs."""
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        obs = env.reset(seed=3)
        # find an owned movable tile on the map edge (has an invalid dir)
        with torch.no_grad():
            out = model.step(
                obs[0]["grid"].unsqueeze(0), obs[0]["globals"].unsqueeze(0),
                legal_mask=obs[0]["legal_mask"].unsqueeze(0),
                act_steps=1,
                direction_mask=obs[0]["dir_mask"].unsqueeze(0))
        src = out["action_source"].item()
        r, c = divmod(src, cfg.grid_w)
        dl = out["direction_logits"][0]
        for d in range(4):
            if not obs[0]["dir_mask"][d, r, c]:
                assert dl[d] <= -1e8, f"invalid dir {d} not masked: {dl[d]}"

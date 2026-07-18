"""Policy heads (autoregressive: Source → Direction / Split) and Value head."""

import torch
import torch.nn as nn


class SourceHead(nn.Module):
    """Predict which tile to select as movement source.  [B, H*W] logits."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_tokens: [B, N, d]
        Returns:
            logits: [B, N]  (raw, no masking — mask applied externally during inference)
        """
        return self.proj(spatial_tokens).squeeze(-1)  # [B, N]


class DirectionHead(nn.Module):
    """Score each direction by pairing the source embedding with the tile the
    move would ENTER.

    Grid adjacency is part of the action-space definition, so the head is
    handed each neighbor's embedding directly instead of having to invent
    "adjacency" through content attention over all 625 tokens — the failure
    mode of the previous cross-attention design, which stayed at uniform
    (ln 4) through three epochs while the source head learned fine.

    A single scorer is shared across the four directions (permutation-
    consistent); off-board neighbors get a learned edge embedding. The
    direction order matches the dir_id labels: 0=down 1=right 2=up 3=left.
    """

    # (drow, dcol) per direction — MUST match dir_id label semantics
    _OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, d_model: int, grid_h: int = 25, grid_w: int = 25):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.edge = nn.Parameter(torch.zeros(d_model))
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        source_embed: torch.Tensor,
        spatial_out: torch.Tensor,
        decision_token: torch.Tensor,
        src_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_embed:   [B, d]
            spatial_out:    [B, N, d]   — all spatial token embeddings
            decision_token: [B, d]
            src_idx:        [B]         — flat index of the source tile
        Returns:
            logits: [B, 4]
        """
        B = spatial_out.shape[0]
        H, W = self.grid_h, self.grid_w
        batch = torch.arange(B, device=spatial_out.device)
        row, col = src_idx // W, src_idx % W

        logits = []
        for drow, dcol in self._OFFSETS:
            r, c = row + drow, col + dcol
            on_board = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            idx = r.clamp(0, H - 1) * W + c.clamp(0, W - 1)
            neighbor = spatial_out[batch, idx]
            neighbor = torch.where(
                on_board.unsqueeze(-1),
                neighbor,
                self.edge.to(neighbor.dtype).expand_as(neighbor),
            )
            logits.append(self.scorer(
                torch.cat([source_embed, neighbor, decision_token], dim=-1)
            ))
        return torch.cat(logits, dim=-1)


class SplitHead(nn.Module):
    """Predict split ratio (0=all-but-1, 1=50%) conditioned on source embedding."""

    def __init__(self, d_model: int, n_split: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_split),
        )

    def forward(
        self, source_embed: torch.Tensor, decision_token: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([source_embed, decision_token], dim=-1)
        return self.net(combined)


class PassHead(nn.Module):
    """Decide whether to pass (no move) given action context, proposed move, and timing.

    Inputs: ll_token, source_embed, direction_logits, split_logits, time_pe.
    Output: single logit (>0 → pass, ≤0 → execute move).
    """

    def __init__(self, d_model: int, n_directions: int = 4, n_split: int = 2,
                 time_pe_dim: int = 32):
        super().__init__()
        self.time_pe_dim = time_pe_dim
        # ll(d) + source_embed(d) + direction(n_dir) + split(n_split) + time_pe
        in_dim = d_model * 2 + n_directions + n_split + time_pe_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        ll_token: torch.Tensor,
        source_embed: torch.Tensor,
        direction_logits: torch.Tensor,
        split_logits: torch.Tensor,
        time_pe: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([ll_token, source_embed, direction_logits,
                              split_logits, time_pe], dim=-1)
        return self.net(combined).squeeze(-1)


class ValueHead(nn.Module):
    """Predict game value from the high-level decision token.

    Output is unbounded: IL targets live in [-1, 1] but RL returns
    (shaped rewards discounted at gamma≈0.999) routinely exceed that
    range, and a tanh-bounded critic saturates on them.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hl_token: torch.Tensor) -> torch.Tensor:
        return self.net(hl_token).squeeze(-1)  # [B]


class HLGaussValueHead(nn.Module):
    """Distributional critic: categorical over fixed support (HL-Gauss).

    Predicts logits over ``n_bins`` support points evenly spaced in
    [-vmax, vmax]; trained with cross-entropy against a Gaussian-smeared
    target distribution (see ``model.losses.hl_gauss_target_probs``).
    ``forward`` returns the distribution's EXPECTATION so the head is a
    drop-in scalar replacement for ``ValueHead`` everywhere (GAE
    bootstrap, eval, play scripts); the PPO update reads the logits via
    ``logits()`` / the network's ``out["value_logits"]``.

    The expectation of a categorical over a bounded support is inherently
    bounded — the unbounded-critic rationale of ``ValueHead`` does not
    apply because this head is only used in the sparse-reward regime
    where true returns live inside the support.
    """

    def __init__(self, d_model: int, n_bins: int = 128, vmax: float = 1.5):
        super().__init__()
        self.n_bins = n_bins
        self.vmax = vmax
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_bins),
        )
        # Not persisted: derived from (n_bins, vmax), and keeping it out of
        # the state dict spares every checkpoint-compat path from a
        # constant buffer.
        self.register_buffer(
            "centers", torch.linspace(-vmax, vmax, n_bins),
            persistent=False,
        )

    def logits(self, hl_token: torch.Tensor) -> torch.Tensor:
        return self.net(hl_token)  # [..., n_bins]

    def expectation_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        return (probs * self.centers).sum(-1)

    def forward(self, hl_token: torch.Tensor) -> torch.Tensor:
        return self.expectation_from_logits(self.logits(hl_token))  # [B]

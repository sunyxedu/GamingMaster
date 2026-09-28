"""Game-agnostic core network — two-stage architecture:

  Stage 1 — History Encoder
            SpatiotemporalBlock × L (separate weights)

  Stage 2 — TRM (Tiny Recursive Model)
            Two-state recursive pondering with shared weights.
            z_L (reasoning scratchpad) and z_H (answer state) are
            iteratively refined by a single shared reasoning module.

The core never changes across games.  Per-game modules plug in at the
boundary defined in ``model.games.base``:

    obs dict → ObservationEncoder → game tokens [B, T, S, d]
             → (+ HL/LL decision tokens, + PE)
             → Stage 1 (spatiotemporal) → Stage 2 (TRM pondering)
             → ActionDecoder → action logits / samples / log-probs
             → ValueHead (from the HL decision token, shared)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.components.heads import HLGaussValueHead, ValueHead
from model.components.transformer import SpatiotemporalBlock, _sinusoidal_pe
from model.components.core import DeepCore, RecurrentDepthCore, record_moe_routes, stack_routes
from model.components.trm import TRM
from model.config import ModelConfig
from model.games.base import ActionDecoder, ObservationEncoder


class GameNetwork(nn.Module):
    """Generic two-stage network parameterised by a game adapter.

        ObservationEncoder → (+decision tokens, +PE) →
        Stage 1: History Encoding (spatiotemporal) →
        Stage 2: TRM (two-state recursive pondering) →
        ActionDecoder + ValueHead
    """

    def __init__(
        self,
        cfg: ModelConfig,
        obs_encoder: ObservationEncoder,
        action_decoder: ActionDecoder,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.temporal_type != "spatiotemporal":
            raise ValueError(
                f"unsupported temporal_type {cfg.temporal_type!r}; the GRU "
                "path was removed — only 'spatiotemporal' exists"
            )

        # --- Game adapter modules ---
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder

        # --- Stage 1: History Encoder ---
        self.st_layers = nn.ModuleList(
            [SpatiotemporalBlock(cfg) for _ in range(cfg.n_st_layers)]
        )

        # --- Stage 2: core (model/components/core.py; "trm" = legacy) ---
        self.core_type = getattr(cfg, "core_type", "trm")
        if (getattr(cfg, "core_gated_memory", "") or getattr(cfg, "core_delta_memory", "")) \
                and self.core_type != "transformer":
            raise ValueError("core memory sub-layers (core_gated_memory / core_delta_memory) "
                             f"need core_type='transformer', not {self.core_type!r}")
        if self.core_type == "trm":
            if getattr(cfg, "moe_experts", 0):
                raise ValueError("MoE is implemented for the transformer / "
                                 "recurrent_depth cores, not the legacy TRM")
            self.trm = TRM(cfg)
            # q_halt is read from the HL decision token, which sits directly
            # after the game tokens (re-set per call: a shared core serves
            # games with different token counts).
            self.trm.q_token_idx = obs_encoder.n_tokens
        elif self.core_type == "recurrent_depth":
            self.core = RecurrentDepthCore(cfg)
        elif self.core_type == "transformer":
            self.core = DeepCore(cfg)
        else:
            raise ValueError(f"unknown core_type {self.core_type!r}")

        # --- Learnable decision tokens ---
        self.hl_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.hl_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- Norms ---
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.mid_norm = nn.LayerNorm(cfg.d_model)
        self.post_norm = nn.LayerNorm(cfg.d_model)

        # --- Value head (reads the HL decision token; game-agnostic class, but
        # not a core module: one per game in a MultiGameNetwork) ---
        if getattr(cfg, "hl_gauss_value", False):
            self.value_head = HLGaussValueHead(
                cfg.d_model, cfg.hl_gauss_bins, cfg.hl_gauss_vmax)
        else:
            self.value_head = ValueHead(cfg.d_model)

        if cfg.predict_next_obs and not obs_encoder.supports_next_obs:
            raise ValueError(
                f"predict_next_obs=True but {type(obs_encoder).__name__} "
                "does not implement next-observation prediction"
            )

        self._init_weights()
        # the core's memory sub-layers start as identities (zero outputs), which the
        # network-wide Linear re-init above would undo
        mem = self._core_memory()
        if mem is not None:
            mem.reset_parameters()
        # the games' add-on modules (model.multi_game.ADDON_PREFIX) start at zero output too
        for m in self.modules():
            if m is not self and hasattr(m, "reset_addon"):
                m.reset_addon()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(self, "trm") and m is self.trm.q_head:
                    continue  # TRM q_head has custom init
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Shared building blocks
    # ------------------------------------------------------------------

    @property
    def supervision_steps(self) -> int:
        """Deep-supervision steps per example (training.deep_supervision):
        N_sup for the legacy TRM, 1 (plain training) for the other cores."""
        return max(1, self.cfg.halt_max_steps) if self.core_type == "trm" else 1

    @property
    def default_depth(self) -> int:
        """Inference depth when act_steps is None: TRM N_sup, recurrent-depth
        r (rd_eval_iters or rd_mean_iters), 1 for the plain transformer."""
        if self.core_type == "trm":
            return max(1, self.cfg.halt_max_steps)
        return self.core.default_iters()

    def _core_pass(self, x: torch.Tensor, steps: Optional[int],
                   valid: Optional[torch.Tensor] = None,
                   routes: Optional[torch.Tensor] = None,
                   bt: Optional[Tuple[int, int]] = None,
                   mem_states: Optional[List[Optional[torch.Tensor]]] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Non-TRM cores: (z [BT, S, d], q_halt zeros [BT], depth).  ``routes``
        [BT, C, S, k]: replayed MoE routing (model.components.core).  ``bt`` /
        ``mem_states``: the core's episode-memory sub-layers (sequence layout of a
        full forward, or one step's states, updated in place)."""
        if self._core_memory() is not None:
            z, depth = self.core(x, steps, valid, routes=routes, bt=bt, mem_states=mem_states)
        else:
            z, depth = self.core(x, steps, valid, routes=routes)
        return z, x.new_zeros(x.shape[0]), depth

    def _core_memory(self):
        """The core's episode-memory sub-layers (model.components.memory), or None."""
        mem = getattr(getattr(self, "core", None), "memory", None)
        return mem if mem else None

    def moe_route_calls(self, act_steps: Optional[int] = None) -> int:
        """MoE layer calls of one core forward at depth ``act_steps`` — the C
        axis of ``step(record_routes=True)["moe_routes"]`` (0: no MoE)."""
        if self.core_type == "trm":
            return 0
        return self.core.moe_calls(act_steps)

    # ------------------------------------------------------------------
    # Static inference state (CUDA-graph / compiled step support)
    # ------------------------------------------------------------------

    def prepare_inference(self, device: Optional[torch.device] = None,
                          dtype: Optional[torch.dtype] = None) -> None:
        """Precompute every lazily-built inference constant so that
        :meth:`step` performs no host→device copy.  Required before
        capturing the step in a CUDA graph (a warm-up step also does it).

        Today that is the recurrent-depth core's fixed initial state s_0
        for this game's token count.  ``dtype`` is the core's residual-stream
        dtype: the parameter dtype by default (float32 also under bf16
        autocast)."""
        core = getattr(self, "core", None)
        if not isinstance(core, RecurrentDepthCore):
            return
        p = next(core.parameters())
        core.prepare_inference(self.obs_encoder.n_tokens + 2,
                               p.device if device is None else torch.device(device),
                               p.dtype if dtype is None else dtype)

    @torch.no_grad()
    def initial_st_cache(self, batch: int) -> List[torch.Tensor]:
        """Materialised Stage-1 state at t=0 for ``batch`` streams: one
        [batch * S, d] float32 tensor per spatiotemporal layer, then the core's
        episode-memory states (model.components.memory; [batch * S, d] per gated
        sub-layer, [batch, heads, dv, dk] per delta sub-layer).

        It holds the same state as ``st_cache=None`` (the recurrence starts
        from its learned h_init either way), but the tensors have a fixed
        shape and address — what a CUDA graph needs (copy each step's new
        state back into them in place).  Step from it with a TENSOR
        ``step_idx`` (the graph-capturable form): with a Python int, a
        present cache at step_idx 0 trips step()'s forgotten-step_idx guard,
        so the int path keeps passing ``st_cache=None`` at t=0.  The values
        are a snapshot of h_init: rebuild after loading new weights.  Only
        the recurrent temporal mixer has a fixed-size state; the attention
        mixer's KV cache grows by one frame per step.
        """
        if self.cfg.temporal_mixer != "recurrent":
            raise ValueError(
                "a static Stage-1 cache needs temporal_mixer='recurrent' "
                f"(got {self.cfg.temporal_mixer!r}: the attention mixer's KV "
                "cache grows every step, so its shapes cannot be captured)")
        S = self.obs_encoder.n_tokens + 2
        states = [layer.temporal_rnn.initial_state(batch * S)
                  .clone(memory_format=torch.contiguous_format)
                  for layer in self.st_layers]
        mem = self._core_memory()
        if mem is not None:
            states.extend(mem.initial_states(batch, S))
        return states

    def _decode_ctx(self, obs: Dict[str, torch.Tensor],
                    step: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """``decode()`` kwargs carrying the encoder's decode context
        (:meth:`ObservationEncoder.decode_context`): ``{"ctx": ctx}``, or
        ``{}`` when the encoder has none, so decoders that take no ``ctx``
        are called exactly as before.  ``obs`` is [B, T, ...], or one frame
        [B, ...] with ``step`` (viewed as T = 1, like the decode itself)."""
        enc = self.obs_encoder
        if type(enc).decode_context is ObservationEncoder.decode_context:
            return {}                                  # the default: no context
        if step:
            obs = {k: v.unsqueeze(1) if torch.is_tensor(v) else v for k, v in obs.items()}
        ctx = enc.decode_context(obs)
        return {} if ctx is None else {"ctx": ctx}

    def _build_pe(self) -> torch.Tensor:
        """Game-token PE (from the encoder) + decision-token PE: [1, S+2, d]."""
        return torch.cat(
            [self.obs_encoder.positional_embedding(), self.hl_pos, self.ll_pos],
            dim=1,
        )

    def _trm_ponder(
        self,
        x: torch.Tensor,
        act_steps: Optional[int],
        carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rows: Optional[torch.Tensor] = None,
        reset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run ``act_steps`` TRM supervision steps starting from ``carry``.

        The first ``act_steps - 1`` steps run without gradient; only the
        last is differentiated (the reference's per-step gradient: every
        supervision step backpropagates through one full recursion, the
        carry entering it is detached).  ``act_steps=None`` → 1 step.

        ``rows`` (LongTensor of flat frame indices) restricts the TRM to
        the frames still being supervised; the other rows keep their
        carried state (their answer is frozen, as when the reference
        swaps a halted example out of the batch).

        ``reset`` [BT] bool (reference ``reset_carry``): rows whose carry is
        replaced by (H_init, L_init) — examples newly swapped into the batch.

        Returns (z_H [BT, S, d], q_halt [BT], new_carry (z_H, z_L) detached).
        """
        steps = max(1, act_steps or 1)
        if carry is not None and reset is not None:
            BT, S, _ = x.shape
            init = self.trm.initial_carry(BT, S, x)
            r = reset.reshape(BT, 1, 1)
            carry = (torch.where(r, init[0].to(carry[0].dtype), carry[0]),
                     torch.where(r, init[1].to(carry[1].dtype), carry[1]))
        full_carry = None
        if rows is not None:
            BT, S, _ = x.shape
            full_carry = carry if carry is not None else self.trm.initial_carry(BT, S, x)
            x = x[rows]
            carry = (full_carry[0][rows], full_carry[1][rows])
        if steps > 1:
            with torch.no_grad():
                for _ in range(steps - 1):
                    _, _, carry = self.trm(x, carry)
            # bf16 autocast caches each weight's half-precision cast on
            # first use — here that first use happens under no_grad, so the
            # cached copy carries no graph edge back to the fp32 weight.
            # torch 2.5 reuses it in the grad-enabled step below, silently
            # severing q_head's weight gradient (q_halt still requires grad
            # via its input; checkpointed modules re-cast during backward
            # recompute and are immune). Drop the poisoned entries so the
            # final step re-casts with autograd tracking.
            torch.clear_autocast_cache()
        z_H, q_halt, new_carry = self.trm(x, carry)
        if full_carry is None:
            return z_H, q_halt, new_carry
        BT = full_carry[0].shape[0]
        z_full = full_carry[0].to(z_H.dtype).index_copy(0, rows, z_H)
        q_full = q_halt.new_zeros(BT).index_copy(0, rows, q_halt)
        carry_full = (
            full_carry[0].to(new_carry[0].dtype).index_copy(0, rows, new_carry[0]),
            full_carry[1].to(new_carry[1].dtype).index_copy(0, rows, new_carry[1]),
        )
        return z_full, q_full, carry_full

    def _trm_act_loop(
        self,
        x: torch.Tensor,
        act_steps: Optional[int],
        mem_states: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: a FIXED number of TRM supervision steps.

        ``act_steps=None`` → ``cfg.halt_max_steps``: the reference runs all
        N_sup steps at test time (q_halt only shortens training).  A fixed
        depth is also what makes RL rollouts exactly replayable.

        Returns (z_H, q_halt, act_depth[B]).
        """
        if self.core_type != "trm":
            z, q, r = self._core_pass(x, act_steps, mem_states=mem_states)
            return z, q, torch.full((x.shape[0],), r, dtype=torch.long, device=x.device)
        self.trm.q_token_idx = self.obs_encoder.n_tokens
        steps = max(1, act_steps if act_steps is not None
                    else self.cfg.halt_max_steps)
        z_H, q_halt, carry = self.trm(x)
        for _ in range(steps - 1):
            z_H, q_halt, carry = self.trm(x, carry)
        depth = torch.full((x.shape[0],), steps, dtype=torch.long,
                           device=x.device)
        return z_H, q_halt, depth

    # ------------------------------------------------------------------
    # Full-sequence forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        act_steps: Optional[int] = None,
        frame_mask: Optional[torch.Tensor] = None,
        carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        active: Optional[torch.Tensor] = None,
        return_carry: bool = False,
        reset: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor] = None,
        init_cache: Optional[List[Optional[torch.Tensor]]] = None,
        t0: int = 0,
        return_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full-sequence forward.

        ``routes`` [B, T, C, S, k] replays a recorded MoE routing (what
        ``step(record_routes=True)`` returned for each frame, at the same
        depth): the RL learner's forward then takes exactly the actor's
        experts (model.components.core, routing replay).

        A sequence in chunks (recurrent temporal mixer, non-TRM cores):
        ``init_cache`` is the memory state the frames continue from — the
        ``step()`` cache layout (Stage-1 states, then the core memory's), e.g. a
        previous chunk's ``out["st_cache"]`` — and ``t0`` the absolute index of the
        first frame (the decoders' time features); ``return_cache`` returns the
        state after the last frame as ``out["st_cache"]`` (rows padded at the end
        carry their padding's updates).  Chunks chained this way compute what one
        forward over the whole sequence does.

        ``carry`` / ``active`` / ``return_carry`` drive deep supervision
        (training.deep_supervision): ``carry`` is the detached (z_H, z_L)
        from the previous supervision step of the SAME batch, ``active``
        [B, T] bool selects the frames that have not halted yet, ``reset``
        [B, T] bool marks frames whose carry restarts from (H_init, L_init)
        (reference ``reset_carry``), and ``out["carry"]`` returns the new
        detached latents.
        """
        cfg = self.cfg
        d = cfg.d_model

        # 1. Game-specific encoding → [B, T, S, d]
        tokens = self.obs_encoder(obs, frame_mask)
        B, T, S_game, _ = tokens.shape
        BT = B * T
        if S_game != self.obs_encoder.n_tokens:
            raise RuntimeError(
                f"{type(self.obs_encoder).__name__} produced {S_game} tokens "
                f"but declares n_tokens={self.obs_encoder.n_tokens}; the "
                f"positional embedding and the TRM q_halt index depend on it"
            )

        # 2. Decision tokens
        hl = self.hl_token.expand(B, T, -1, -1)
        ll = self.ll_token.expand(B, T, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=2)  # [B, T, S, d]

        # 3. PE — absolute time PE only for the attention mixer; the
        # recurrence carries temporal order itself and absolute positions
        # would only reintroduce step-index dependence.
        tokens = tokens + self._build_pe().unsqueeze(0)
        if cfg.temporal_mixer != "recurrent":
            tokens = tokens + _sinusoidal_pe(T, d, tokens.device).unsqueeze(0).unsqueeze(2)

        # ── Stage 1: Spatiotemporal encoding (checkpointed) ──
        tokens = self.pre_norm(tokens)
        chunked = init_cache is not None or return_cache
        if chunked and (cfg.temporal_mixer != "recurrent" or self.core_type == "trm"):
            raise ValueError("a carried memory state (init_cache / return_cache) needs the "
                             "recurrent temporal mixer and a non-TRM core")
        st_states: List[torch.Tensor] = []
        for i, layer in enumerate(self.st_layers):
            if chunked:
                h0 = init_cache[i] if init_cache is not None else None
                tokens, h = (checkpoint(layer, tokens, h0, True, use_reentrant=True)
                             if torch.is_grad_enabled() else layer(tokens, h0, True))
                st_states.append(h)
            else:
                tokens = (checkpoint(layer, tokens, use_reentrant=True)
                          if torch.is_grad_enabled() else layer(tokens))
        tokens = self.mid_norm(tokens)

        # ── Stage 2: TRM pondering ──
        S = tokens.shape[2]
        x = tokens.reshape(BT, S, d)                  # input embeddings for TRM
        rows = (active.reshape(BT).nonzero().squeeze(1)
                if active is not None else None)
        if routes is not None and self.core_type == "trm":
            raise ValueError("the TRM core has no MoE routing to replay")
        if self.core_type == "trm":
            self.trm.q_token_idx = self.obs_encoder.n_tokens
            z_H, q_halt, new_carry = self._trm_ponder(
                x, act_steps, carry, rows,
                reset.reshape(BT) if reset is not None else None)
        else:
            # one pass per example (no carried latents); padded frames are
            # excluded from the MoE load statistics
            valid = frame_mask.reshape(BT) if frame_mask is not None else None
            if routes is not None:
                routes = routes.reshape((BT,) + tuple(routes.shape[2:]))
            mem = self._core_memory()
            mem_states = None
            if chunked and mem is not None:
                n_st = len(self.st_layers)
                mem_states = (list(init_cache[n_st:]) if init_cache is not None
                              else [None] * len(mem.ordered()))
            z_H, q_halt, _ = self._core_pass(x, act_steps, valid, routes, bt=(B, T),
                                             mem_states=mem_states)
            new_carry = None
        z_H = self.post_norm(z_H)

        # ── Decode ──
        game_out = z_H[:, :S_game]
        hl_out = z_H[:, S_game]
        ll_out = z_H[:, S_game + 1]

        out = self.action_decoder.decode(
            game_out, hl_out, ll_out, B=B, T=T, t0=t0, legal=legal, labels=labels,
            **self._decode_ctx(obs),
        )
        if isinstance(self.value_head, HLGaussValueHead):
            # One MLP pass: expose the logits for the HL-Gauss CE loss and
            # derive the scalar from them for every other consumer.
            v_logits = self.value_head.logits(hl_out)
            out["value_logits"] = v_logits.reshape(B, T, -1)
            out["value"] = (self.value_head
                            .expectation_from_logits(v_logits).reshape(B, T))
        else:
            out["value"] = self.value_head(hl_out).reshape(B, T)
        out["q_halt"] = q_halt.reshape(B, T)
        if return_carry:
            out["carry"] = new_carry
        if return_cache:
            out["st_cache"] = st_states + (mem_states or [])

        # Self-supervised next-observation prediction from belief tokens
        if cfg.predict_next_obs:
            out.update(self.obs_encoder.next_obs_outputs(tokens[:, :, :S_game]))
        # Stage-1 belief tokens for external auxiliary heads (e.g. the RL
        # capture-feasibility head) — a view, so returning it is free.
        out["belief_tokens"] = tokens[:, :, :S_game]
        return out

    # ------------------------------------------------------------------
    # Single-step inference (rollout, with Stage-1 cache)
    # ------------------------------------------------------------------

    def step(
        self,
        obs: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
        st_cache: Optional[List[torch.Tensor]] = None,
        step_idx: Union[int, torch.Tensor] = 0,
        act_steps: Optional[int] = None,
        record_routes: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Single-frame inference with Stage-1 temporal cache.

        ``record_routes``: also return ``moe_routes`` [B, C, S, k] — every MoE
        call's selected experts for every token (compact integer dtype), what
        ``forward(routes=...)`` replays; absent for cores without MoE.

        ``legal`` restricts sampling (IL evaluation); pass ``None`` for RL
        mode where the model may select anything.  Returns the decoder's
        logits (T squeezed away), the sampled action dict, value, q_halt
        and the updated ``st_cache``.

        Sync-free in eval mode under ``torch.no_grad()`` with the recurrent
        temporal mixer and a non-TRM core: no ``.item()`` / ``nonzero`` /
        data-dependent Python branch, so the call can be captured in a CUDA
        graph (see ``model.cuda_graph.GraphedStep``).  Condition for MoE
        cores: the layers must take the dense expert path — automatic on an
        accelerator while batch × tokens × experts is at most
        ``MoEFFN.DENSE_MAX_TOKEN_EXPERTS`` (generals 25×25, 8 experts:
        batch ≤ 13), otherwise force it with
        ``MoEFFN.inference_path("dense")`` (GraphedStep does).  For capture,
        pass ``step_idx`` as a single-element integer tensor on the model's
        device (the decoder's time feature then reads it at run time instead
        of baking the capture-time value in), start from
        :meth:`initial_st_cache`, and call :meth:`prepare_inference` (or run
        a warm-up step) first.
        """
        cfg = self.cfg
        d = cfg.d_model

        tokens = self.obs_encoder.encode_step(obs)    # [B, S_game, d]
        B, S_game, _ = tokens.shape
        if S_game != self.obs_encoder.n_tokens:
            raise RuntimeError(
                f"{type(self.obs_encoder).__name__} produced {S_game} tokens "
                f"but declares n_tokens={self.obs_encoder.n_tokens}; the "
                f"positional embedding and the TRM q_halt index depend on it"
            )

        tokens = torch.cat([tokens, self.hl_token.expand(B, -1, -1),
                            self.ll_token.expand(B, -1, -1)], dim=1)
        tokens = tokens + self._build_pe()

        if cfg.temporal_mixer == "recurrent":
            # No absolute time PE on tokens (matches forward); the decoder's
            # time feature below still uses the caller-supplied step index.
            # The fixed-size cache carries no step count, so a caller that
            # threads the cache but forgets step_idx would silently freeze
            # the time feature at 0 — catch it.  (A tensor step_idx is the
            # graph-capturable form; its value is never read on the host.)
            if (st_cache is not None and not isinstance(step_idx, torch.Tensor)
                    and step_idx == 0):
                raise ValueError(
                    "recurrent mode: pass step_idx to step()/step_rl() "
                    "(cache present but step_idx is still 0)"
                )
            t_cur = step_idx
        else:
            t_cur = 0 if st_cache is None else st_cache[0].shape[1]
            t_pe = _sinusoidal_pe(t_cur + 1, d, tokens.device)[t_cur]
            tokens = tokens + t_pe.unsqueeze(0).unsqueeze(0)

        # Stage 1: spatiotemporal with temporal cache
        tokens = self.pre_norm(tokens).unsqueeze(1)
        new_caches: List[torch.Tensor] = []
        for i, layer in enumerate(self.st_layers):
            prev = st_cache[i] if st_cache is not None else None
            tokens, cache_i = layer.forward_step(tokens, prev)
            new_caches.append(cache_i)
        tokens = self.mid_norm(tokens.squeeze(1))
        # the core's episode-memory states follow the Stage-1 ones in the cache
        mem = self._core_memory()
        mem_states: Optional[List[Optional[torch.Tensor]]] = None
        if mem is not None:
            n_st = len(self.st_layers)
            mem_states = (list(st_cache[n_st:]) if st_cache is not None
                          else [None] * len(mem.ordered()))

        # Stage 2: TRM with ACT
        recorded: Dict[str, torch.Tensor] = {}
        if record_routes and self.core_type != "trm":
            with record_moe_routes(self.core) as calls:
                z_H, q_halt, act_depth = self._trm_act_loop(tokens, act_steps, mem_states)
            if calls:
                recorded["moe_routes"] = stack_routes(calls, B, tokens.shape[1],
                                                      self.cfg.moe_experts)
        else:
            z_H, q_halt, act_depth = self._trm_act_loop(tokens, act_steps, mem_states)
        if mem_states is not None:
            new_caches.extend(mem_states)
        z_H = self.post_norm(z_H)

        game_out = z_H[:, :S_game]
        hl_out = z_H[:, S_game]
        ll_out = z_H[:, S_game + 1]

        # Decode as a T=1 sequence at absolute time t_cur, then squeeze
        legal_seq = (
            {k: v.unsqueeze(1) for k, v in legal.items()} if legal else None
        )
        pred = self.action_decoder.decode(
            game_out, hl_out, ll_out, B=B, T=1, t0=t_cur,
            legal=legal_seq, labels=None,
            **self._decode_ctx(obs, step=True),
        )
        pred = {k: v.squeeze(1) for k, v in pred.items()}

        # Sample the remaining action components + log-prob for RL rollout
        rl_out = self.action_decoder.sample(pred, legal)

        return {
            **pred,
            "value": self.value_head(hl_out),
            "q_halt": q_halt,
            "act_depth": act_depth,
            "st_cache": new_caches,
            **rl_out,
            **recorded,
        }

    # ------------------------------------------------------------------
    # PPO: re-evaluate stored actions under the current policy
    # ------------------------------------------------------------------

    def compute_log_prob(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Delegate to the decoder: (log_prob [B,T], entropy [B,T], head_ent)."""
        return self.action_decoder.log_prob_entropy(pred, actions, legal)

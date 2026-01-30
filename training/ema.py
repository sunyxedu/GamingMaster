"""
Exponential Moving Average (EMA) for model weights.

From the TRM paper "Less is More: Recursive Reasoning with Tiny Networks":
EMA is used to maintain a stable version of the model for evaluation,
which helps with training stability.

Usage:
    model = GeneralsModel(config)
    ema = EMA(model, decay=0.999)

    # During training
    for batch in dataloader:
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()
        ema.update()  # Update EMA weights

    # For evaluation
    with ema.average_parameters():
        evaluate(model)  # Uses EMA weights
    # After context manager, original weights are restored
"""

import torch
import torch.nn as nn
from typing import Optional, Iterator, Tuple
from contextlib import contextmanager
import copy


class EMA(nn.Module):
    """
    Exponential Moving Average for model parameters.

    Maintains a shadow copy of model parameters that is updated as:
        shadow = decay * shadow + (1 - decay) * current

    Args:
        model: The model to track
        decay: EMA decay rate (default 0.999)
        device: Device for shadow parameters (default: same as model)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.model = model
        self.decay = decay
        self.device = device

        # Create shadow parameters
        self.shadow = {}
        self.backup = {}

        # Register shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow_param = param.data.clone()
                if device is not None:
                    shadow_param = shadow_param.to(device)
                self.shadow[name] = shadow_param

    @torch.no_grad()
    def update(self):
        """
        Update shadow parameters with current model parameters.

        Should be called after each optimizer step.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    @torch.no_grad()
    def copy_to(self, model: Optional[nn.Module] = None):
        """
        Copy shadow parameters to a model.

        Args:
            model: Target model (default: self.model)
        """
        if model is None:
            model = self.model

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def store(self):
        """Store current model parameters for later restoration."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    @torch.no_grad()
    def restore(self):
        """Restore model parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    @contextmanager
    def average_parameters(self):
        """
        Context manager to temporarily use EMA parameters.

        Usage:
            with ema.average_parameters():
                # Model uses EMA weights here
                output = model(input)
            # Original weights restored after context
        """
        self.store()
        self.copy_to()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)


class EMATrainer:
    """
    Trainer wrapper that handles EMA updates automatically.

    Provides methods for training with EMA and evaluation using EMA weights.

    Args:
        model: The model to train
        optimizer: The optimizer
        ema_decay: EMA decay rate (default 0.999)
        device: Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema_decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # Initialize EMA
        self.ema = EMA(model, decay=ema_decay, device=device)

        # Training state
        self.global_step = 0

    def train_step(
        self,
        loss: torch.Tensor,
        max_grad_norm: float = 1.0,
        update_ema: bool = True
    ) -> float:
        """
        Perform a single training step with EMA update.

        Args:
            loss: Computed loss tensor
            max_grad_norm: Maximum gradient norm for clipping
            update_ema: Whether to update EMA after this step

        Returns:
            Loss value as float
        """
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=max_grad_norm
            )

        # Optimizer step
        self.optimizer.step()

        # Update EMA
        if update_ema:
            self.ema.update()

        self.global_step += 1

        return loss.item()

    @contextmanager
    def eval_mode(self, use_ema: bool = True):
        """
        Context manager for evaluation.

        Args:
            use_ema: If True, use EMA weights for evaluation

        Usage:
            with trainer.eval_mode():
                output = model(input)
        """
        was_training = self.model.training
        self.model.eval()

        if use_ema:
            with self.ema.average_parameters():
                try:
                    yield
                finally:
                    self.model.train(was_training)
        else:
            try:
                yield
            finally:
                self.model.train(was_training)

    def save_checkpoint(self, path: str, extra_state: Optional[dict] = None):
        """
        Save training checkpoint including EMA state.

        Args:
            path: Path to save checkpoint
            extra_state: Additional state to save
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'global_step': self.global_step
        }

        if extra_state:
            state.update(extra_state)

        torch.save(state, path)

    def load_checkpoint(self, path: str, map_location: Optional[torch.device] = None):
        """
        Load training checkpoint including EMA state.

        Args:
            path: Path to checkpoint
            map_location: Device mapping for loading

        Returns:
            Extra state from checkpoint (if any)
        """
        checkpoint = torch.load(path, map_location=map_location)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)

        # Return any extra state
        extra_keys = set(checkpoint.keys()) - {
            'model_state_dict', 'optimizer_state_dict',
            'ema_state_dict', 'global_step'
        }
        return {k: checkpoint[k] for k in extra_keys}


def create_ema_model(model: nn.Module, decay: float = 0.999) -> Tuple[nn.Module, EMA]:
    """
    Convenience function to create a model with EMA tracking.

    Args:
        model: The model to track
        decay: EMA decay rate

    Returns:
        Tuple of (model, ema)
    """
    ema = EMA(model, decay=decay)
    return model, ema

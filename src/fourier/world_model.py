"""Minimal LeWM adapter; each action block contains primitive controls."""
import torch
from torch import nn
from stable_worldmodel.wm.lewm.lewm import LeWM


class FourierWorldModel(LeWM):
    def __init__(self, encoder, predictor, action_block=5, projector=None,
                 action_mean=None, action_scale=None):
        super().__init__(encoder=encoder, predictor=predictor,
                         action_encoder=nn.Identity(), projector=projector,
                         pred_proj=nn.Identity())
        self.action_block = int(action_block)
        if self.action_block < 1:
            raise ValueError("action_block must be positive")
        m = predictor.action_dim
        self.register_buffer("action_mean", torch.tensor(
            [0.] * m if action_mean is None else action_mean, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(
            [1.] * m if action_scale is None else action_scale, dtype=torch.float32))
        if self.action_mean.shape != (m,) or self.action_scale.shape != (m,):
            raise ValueError("Normalization must have primitive action width")
        if not torch.isfinite(self.action_scale).all() or (self.action_scale <= 0).any():
            raise ValueError("Invalid normalization scale")

    def predict(self, emb, act_emb):
        expected = self.action_block * self.predictor.action_dim
        if act_emb.shape[-1] != expected:
            raise ValueError(f"Expected action block width {expected}")
        controls = act_emb.unflatten(-1, (self.action_block, self.predictor.action_dim))
        return self.predictor.rollout(emb, controls)[..., -1, :]


from .predictor import FourierDynamics


class FourierPredictor(nn.Module):
    """Predict each context frame's successor from its raw action block."""

    def __init__(self, input_dim=192, action_dim=25, action_block=5,
                 num_frames=3, num_frequencies=768,
                 frequency_scales=(.25, .5, 1., 2.), init_scale=1e-3,
                 learnable_frequencies=True):
        super().__init__()
        if action_block < 1 or action_dim < 1 or action_dim % action_block:
            raise ValueError("Action-block width must be a multiple of action_block")
        self.num_frames = int(num_frames)
        self.action_block = int(action_block)
        self.macro_action_dim = int(action_dim)
        self.dynamics = FourierDynamics(
            latent_dim=input_dim, action_dim=action_dim // action_block,
            num_frequencies=num_frequencies, frequency_scales=frequency_scales,
            init_scale=init_scale, learnable_frequencies=learnable_frequencies)

    def forward(self, emb, act_emb):
        if act_emb.shape[-1] != self.macro_action_dim:
            raise ValueError("Expected flattened raw action blocks")
        controls = act_emb.unflatten(
            -1, (self.action_block, self.dynamics.action_dim))
        return self.dynamics.rollout(emb, controls)[..., -1, :]

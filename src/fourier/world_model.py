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

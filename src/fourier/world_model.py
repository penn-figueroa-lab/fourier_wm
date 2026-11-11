"""Minimal LeWM adapter; each action block contains primitive controls."""

import torch
from torch import nn
from stable_worldmodel.wm.lewm.lewm import LeWM

LEWM_TARGETS = {
    "stable_worldmodel.wm.lewm.LeWM",
    "fourier.world_model.ProprioLeWM",
    "fourier.world_model.AdjacentHistoryWorldModel",
}


class ProprioLeWM(LeWM):
    """LeWM with a linear observation adapter for released proprioceptive tasks.

    The vision encoder, projector, predictor interface and training loss remain
    LeWM's. Native position/velocity is fused before the existing projector, so
    force-controlled tasks have causal motion information at the first frame.
    Normalization belongs to the checkpoint, not to the evaluation dataset.
    """

    observation_keys = ("pixels", "proprio")

    def __init__(
        self, *args, proprio_dim=4, embed_dim=192, proprio_mean=None, proprio_scale=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.proprio_projection = nn.Linear(proprio_dim, embed_dim, bias=False)
        nn.init.normal_(self.proprio_projection.weight, std=0.01)
        mean = (
            torch.zeros(proprio_dim)
            if proprio_mean is None
            else torch.tensor(proprio_mean, dtype=torch.float32)
        )
        scale = (
            torch.ones(proprio_dim)
            if proprio_scale is None
            else torch.tensor(proprio_scale, dtype=torch.float32)
        )
        if (
            mean.shape != (proprio_dim,)
            or scale.shape != mean.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Invalid training proprioception normalization")
        self.register_buffer("proprio_mean", mean)
        self.register_buffer("proprio_scale", scale)

    def encode(self, info):
        pixels = info["pixels"].to(next(self.encoder.parameters()).dtype)
        b, t = pixels.shape[:2]
        proprio = info["proprio"].to(device=pixels.device, dtype=pixels.dtype)
        if proprio.shape != (b, t, self.proprio_mean.numel()):
            raise ValueError("Proprioception must align with each actual observation")
        vision = self.encoder(
            pixels.flatten(0, 1), interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        normalized = (proprio - self.proprio_mean) / self.proprio_scale
        fused = vision + self.proprio_projection(normalized.flatten(0, 1))
        info["emb"] = self.projector(fused).reshape(b, t, -1)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info


class FourierWorldModel(LeWM):
    def __init__(
        self,
        encoder,
        predictor,
        action_block=5,
        projector=None,
        action_mean=None,
        action_scale=None,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=nn.Identity(),
            projector=projector,
            pred_proj=nn.Identity(),
        )
        self.action_block = int(action_block)
        if self.action_block < 1:
            raise ValueError("action_block must be positive")
        m = predictor.action_dim
        self.register_buffer(
            "action_mean",
            torch.tensor([0.0] * m if action_mean is None else action_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "action_scale",
            torch.tensor([1.0] * m if action_scale is None else action_scale, dtype=torch.float32),
        )
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


from .predictor import FourierDynamics, LinearDynamics, BilinearDynamics


class FourierPredictor(nn.Module):
    """Predict each context frame's successor from its raw action block."""

    def __init__(
        self,
        input_dim=192,
        action_dim=25,
        action_block=5,
        num_frames=3,
        num_frequencies=768,
        frequency_scales=(0.25, 0.5, 1.0, 2.0),
        init_scale=1e-3,
        learnable_frequencies=True,
        dynamics_type="fourier",
    ):
        super().__init__()
        if action_block < 1 or action_dim < 1 or action_dim % action_block:
            raise ValueError("Action-block width must be a multiple of action_block")
        self.num_frames = int(num_frames)
        self.action_block = int(action_block)
        self.macro_action_dim = int(action_dim)
        if dynamics_type not in {"fourier", "linear", "bilinear"}:
            raise ValueError("Choose fourier, linear or bilinear dynamics")
        self.dynamics_type = dynamics_type
        reference = FourierDynamics(
            latent_dim=input_dim,
            action_dim=action_dim // action_block,
            num_frequencies=num_frequencies,
            frequency_scales=frequency_scales,
            init_scale=init_scale,
            learnable_frequencies=learnable_frequencies,
        )
        if dynamics_type == "fourier":
            self.dynamics = reference
        else:
            # Preserve the original Fourier RNG stream for the encoder/projector
            # and copy common A/B initialization. The reference is not retained.
            with torch.random.fork_rng(devices=[]):
                cls = LinearDynamics if dynamics_type == "linear" else BilinearDynamics
                self.dynamics = cls(input_dim, action_dim // action_block, init_scale)
            with torch.no_grad():
                self.dynamics.autonomous_matrix.copy_(reference.autonomous_matrix)
                self.dynamics.control_raw.copy_(reference.control_raw)

    def forward(self, emb, act_emb):
        if act_emb.shape[-1] != self.macro_action_dim:
            raise ValueError("Expected flattened raw action blocks")
        controls = act_emb.unflatten(-1, (self.action_block, self.dynamics.action_dim))
        return self.dynamics.rollout(emb, controls)[..., -1, :]


class HistoryFourierPredictor(FourierPredictor):
    """Image-history dynamics with the unchanged LeWM encoder/loss interface.

    The Fourier state concatenates current and past image embeddings over the
    same block-spaced context seen by LeWM: [z_t, z_(t-5), z_(t-10)]. Missing
    past frames repeat the first available observation. No measured physical
    state or latent differences are added. Each
    forecast recursively applies the native primitive forces, then returns only
    the predicted current-image embedding for LeWM's original loss.
    """

    def __init__(self, input_dim=192, num_frames=3, history_encoding="differences", **kwargs):
        if num_frames < 2:
            raise ValueError("Image-history force dynamics require at least two context frames")
        if history_encoding not in {"concatenate", "differences"}:
            raise ValueError("Choose concatenate, or differences for legacy checkpoints")
        super().__init__(input_dim=input_dim * num_frames, num_frames=num_frames, **kwargs)
        self.observation_dim = input_dim
        # Missing field in older checkpoint configs means the previous encoding.
        self.history_encoding = history_encoding

    def context_state(self, emb):
        if emb.ndim != 3 or emb.shape[-1] != self.observation_dim or emb.shape[1] < 1:
            raise ValueError("Expected image embeddings [B,T>=1,D]")
        time = torch.arange(emb.shape[1], device=emb.device)
        history = [emb[:, (time - lag).clamp_min(0)] for lag in range(self.num_frames)]
        if self.history_encoding == "differences":
            return torch.cat([history[0], *[a - b for a, b in zip(history, history[1:])]], -1)
        return torch.cat(history, -1)

    def forward(self, emb, act_emb):
        full_state = self.context_state(emb)
        return super().forward(full_state, act_emb)[..., : self.observation_dim]


class AdjacentHistoryPredictor(FourierPredictor):
    """Primitive state [z_t,z_(t-1),z_(t-2)], newest to oldest."""

    history_encoding = "concatenate"
    history_stride_physical_steps = 1

    def __init__(self, input_dim=192, num_frames=3, **kwargs):
        if num_frames < 2:
            raise ValueError("Adjacent force history requires at least two frames")
        super().__init__(input_dim=input_dim * num_frames, num_frames=num_frames, **kwargs)
        from .predictor import AdjacentHistoryDynamics

        self.observation_dim = input_dim
        self.dynamics = AdjacentHistoryDynamics(self.dynamics, input_dim, self.dynamics_type)

    def context_state(self, emb):
        if emb.ndim != 3 or emb.shape[-1] != self.observation_dim or emb.shape[1] < 1:
            raise ValueError("Expected chronological adjacent image embeddings [B,H>=1,D]")
        return torch.cat(
            [emb[:, max(0, emb.shape[1] - 1 - lag)] for lag in range(self.num_frames)], -1
        )

    def forward(self, emb, act_emb):
        raise ValueError(
            "Adjacent history requires primitive states; use rollout or sparse training with history_indices"
        )


class AdjacentHistoryWorldModel(LeWM):
    """Unchanged LeWM image encoder/projector with primitive history rollout.

    Candidate actions remain five-step blocks for SWM CEM. Internal state never
    resets at block boundaries; goal scoring sees only the current image latent.
    """

    def __init__(self, *args, encoder_checkpointing=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Adjacent source histories require more encoded images per clip. Keep
        # the full source batch size while recomputing encoder activations.
        if encoder_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def rollout(self, info, action_sequence, history_size=None):
        b, samples, blocks, width = action_sequence.shape
        p = self.predictor
        if width != p.macro_action_dim:
            raise ValueError("Wrong primitive action-block width")
        if "emb" not in info:
            encoded = self.encode({"pixels": info["pixels"][:, 0]})["emb"].detach()
            info["emb"] = encoded[:, None].expand(b, samples, -1, -1)
        emb = info["emb"].reshape(b * samples, -1, p.observation_dim)
        state = p.context_state(emb)
        controls = action_sequence.reshape(
            b * samples, blocks * p.action_block, p.dynamics.action_dim
        )
        states = p.dynamics.rollout(state, controls)
        current = torch.cat(
            (
                state[:, None, : p.observation_dim],
                states[:, p.action_block - 1 :: p.action_block, : p.observation_dim],
            ),
            1,
        )
        info["predicted_emb"] = current.reshape(b, samples, blocks + 1, p.observation_dim)
        return info

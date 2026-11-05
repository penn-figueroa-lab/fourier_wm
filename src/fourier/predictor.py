"""Residual control-affine Fourier dynamics, with exact action Jacobians."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class FourierDynamics(nn.Module):
    """z+ = z + Az + Wf phi_f(z) + [B + reshape(Wg phi_g(z))] u.

    One forward call is one primitive transition. ``rollout`` recursively
    applies any number of controls; no macro-step flattening is implicit.
    All six parameter names match the former single-head implementation.
    """

    def __init__(self, latent_dim=192, action_dim=2, num_frequencies=768,
                 frequency_scales=(.25, .5, 1., 2.), init_scale=1e-3,
                 learnable_frequencies=True):
        super().__init__()
        if min(latent_dim, action_dim, num_frequencies) < 1:
            raise ValueError('Dimensions must be positive')
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError('init_scale must be finite and positive')
        if not frequency_scales or any(not math.isfinite(s) or s <= 0 for s in frequency_scales):
            raise ValueError('frequency_scales must be finite and positive')
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.num_frequencies = num_frequencies
        self.frequency_scales = tuple(frequency_scales)
        self.init_scale = init_scale
        self.autonomous_matrix = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.control_raw = nn.Parameter(torch.empty(latent_dim, action_dim))
        self.omega_f = nn.Parameter(torch.empty(num_frequencies, latent_dim), requires_grad=learnable_frequencies)
        self.omega_g = nn.Parameter(torch.empty(num_frequencies, latent_dim), requires_grad=learnable_frequencies)
        self.fourier_f = nn.Parameter(torch.empty(latent_dim, 2 * num_frequencies))
        self.fourier_g = nn.Parameter(torch.empty(latent_dim * action_dim, 2 * num_frequencies))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # Same Gaussian distributions/fan-in scaling as the near-zero runs.
        # Removing obsolete initialization draws changes seeded RNG ordering.
        row_scales = self.omega_f.new_tensor(self.frequency_scales)[
            torch.arange(self.num_frequencies, device=self.omega_f.device) % len(self.frequency_scales)]
        for omega in (self.omega_f, self.omega_g):
            nn.init.normal_(omega, std=1 / math.sqrt(self.latent_dim))
            omega.mul_(row_scales[:, None])
        nn.init.normal_(self.autonomous_matrix, std=self.init_scale / math.sqrt(self.latent_dim))
        nn.init.normal_(self.control_raw, std=self.init_scale / math.sqrt(self.latent_dim))
        for weight in (self.fourier_f, self.fourier_g):
            nn.init.normal_(weight, std=self.init_scale / math.sqrt(2 * self.num_frequencies))

    @staticmethod
    def features(z, omega):
        phase = F.linear(z, omega)
        return torch.cat((phase.sin(), phase.cos()), -1)

    def forward(self, z, u):
        if z.shape[-1] != self.latent_dim or u.shape[-1] != self.action_dim:
            raise ValueError('Wrong state or action width')
        if z.shape[:-1] != u.shape[:-1]:
            raise ValueError('State and action batch dimensions must match')
        control = self.control_raw + F.linear(self.features(z, self.omega_g), self.fourier_g).unflatten(
            -1, (self.latent_dim, self.action_dim))
        return z + F.linear(z, self.autonomous_matrix) + F.linear(
            self.features(z, self.omega_f), self.fourier_f) + (control @ u.unsqueeze(-1)).squeeze(-1)

    def rollout(self, z, controls):
        """[..., D], [..., H, m] -> [..., H, D], excluding initial state."""
        self._check_sequence(z, controls)
        states = []
        for u in controls.unbind(-2):
            z = self(z, u)
            states.append(z)
        return torch.stack(states, -2)

    def _check_sequence(self, z, controls):
        if (controls.ndim < 2 or controls.shape[-2] < 1
                or controls.shape[:-2] != z.shape[:-1]
                or controls.shape[-1] != self.action_dim or z.shape[-1] != self.latent_dim):
            raise ValueError('Expected matching [..., D] and [..., H>=1, m]')

    def terminal_with_action_jacobian(self, z, controls):
        """Exact terminal state and d z_H/d vec(U), differentiable in weights."""
        self._check_sequence(z, controls)
        horizon = controls.shape[-2]
        tangent = z.new_zeros(*z.shape[:-1], self.latent_dim, horizon * self.action_dim)
        for index, u in enumerate(controls.unbind(-2)):
            features, derivatives = [], []
            for omega in (self.omega_f, self.omega_g):
                phase = F.linear(z, omega)
                phase_tangent = torch.einsum('rd,...dk->...rk', omega, tangent)
                features.append(torch.cat((phase.sin(), phase.cos()), -1))
                derivatives.append(torch.cat((phase.cos()[..., None] * phase_tangent,
                                              -phase.sin()[..., None] * phase_tangent), -2))
            control = self.control_raw + F.linear(features[1], self.fourier_g).unflatten(
                -1, (self.latent_dim, self.action_dim))
            next_z = z + F.linear(z, self.autonomous_matrix) + F.linear(features[0], self.fourier_f)
            next_z = next_z + (control @ u.unsqueeze(-1)).squeeze(-1)
            # Contract output weights before actions: avoid a huge [...,D,2F] intermediate.
            bilinear_tangent = (F.linear(derivatives[1].transpose(-1, -2), self.fourier_g)
                .unflatten(-1, (self.latent_dim, self.action_dim)) * u[..., None, None, :]).sum(-1).transpose(-1, -2)
            tangent = tangent + torch.einsum('di,...ik->...dk', self.autonomous_matrix, tangent)
            tangent = tangent + torch.einsum('dr,...rk->...dk', self.fourier_f, derivatives[0]) + bilinear_tangent
            tangent = tangent + F.pad(control, (index * self.action_dim,
                (horizon - index - 1) * self.action_dim))
            z = next_z
        return z, tangent

    def numpy_parameters(self):
        """FP32 CPU arrays for the optional JAX solver; call after training."""
        values = {key: value.detach().float().cpu().numpy().copy()
                  for key, value in self.state_dict().items()}
        values['B'] = values.pop('control_raw')
        return values

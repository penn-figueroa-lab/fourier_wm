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

    def __init__(
        self,
        latent_dim=192,
        action_dim=2,
        num_frequencies=768,
        frequency_scales=(0.25, 0.5, 1.0, 2.0),
        init_scale=1e-3,
        learnable_frequencies=True,
    ):
        super().__init__()
        if min(latent_dim, action_dim, num_frequencies) < 1:
            raise ValueError("Dimensions must be positive")
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("init_scale must be finite and positive")
        if not frequency_scales or any(not math.isfinite(s) or s <= 0 for s in frequency_scales):
            raise ValueError("frequency_scales must be finite and positive")
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.num_frequencies = num_frequencies
        self.frequency_scales = tuple(frequency_scales)
        self.init_scale = init_scale
        self.autonomous_matrix = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.control_raw = nn.Parameter(torch.empty(latent_dim, action_dim))
        self.omega_f = nn.Parameter(
            torch.empty(num_frequencies, latent_dim), requires_grad=learnable_frequencies
        )
        self.omega_g = nn.Parameter(
            torch.empty(num_frequencies, latent_dim), requires_grad=learnable_frequencies
        )
        self.fourier_f = nn.Parameter(torch.empty(latent_dim, 2 * num_frequencies))
        self.fourier_g = nn.Parameter(torch.empty(latent_dim * action_dim, 2 * num_frequencies))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # Same Gaussian distributions/fan-in scaling as the near-zero runs.
        # Removing obsolete initialization draws changes seeded RNG ordering.
        row_scales = self.omega_f.new_tensor(self.frequency_scales)[
            torch.arange(self.num_frequencies, device=self.omega_f.device)
            % len(self.frequency_scales)
        ]
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
            raise ValueError("Wrong state or action width")
        if z.shape[:-1] != u.shape[:-1]:
            raise ValueError("State and action batch dimensions must match")
        control = self.control_raw + F.linear(
            self.features(z, self.omega_g), self.fourier_g
        ).unflatten(-1, (self.latent_dim, self.action_dim))
        return (
            z
            + F.linear(z, self.autonomous_matrix)
            + F.linear(self.features(z, self.omega_f), self.fourier_f)
            + (control @ u.unsqueeze(-1)).squeeze(-1)
        )

    def rollout(self, z, controls):
        """[..., D], [..., H, m] -> [..., H, D], excluding initial state."""
        self._check_sequence(z, controls)
        states = []
        for u in controls.unbind(-2):
            z = self(z, u)
            states.append(z)
        return torch.stack(states, -2)

    def _check_sequence(self, z, controls):
        if (
            controls.ndim < 2
            or controls.shape[-2] < 1
            or controls.shape[:-2] != z.shape[:-1]
            or controls.shape[-1] != self.action_dim
            or z.shape[-1] != self.latent_dim
        ):
            raise ValueError("Expected matching [..., D] and [..., H>=1, m]")

    def terminal_with_action_jacobian(self, z, controls):
        """Exact terminal state and d z_H/d vec(U), differentiable in weights."""
        self._check_sequence(z, controls)
        horizon = controls.shape[-2]
        tangent = z.new_zeros(*z.shape[:-1], self.latent_dim, horizon * self.action_dim)
        for index, u in enumerate(controls.unbind(-2)):
            features, derivatives = [], []
            for omega in (self.omega_f, self.omega_g):
                phase = F.linear(z, omega)
                phase_tangent = torch.einsum("rd,...dk->...rk", omega, tangent)
                features.append(torch.cat((phase.sin(), phase.cos()), -1))
                derivatives.append(
                    torch.cat(
                        (
                            phase.cos()[..., None] * phase_tangent,
                            -phase.sin()[..., None] * phase_tangent,
                        ),
                        -2,
                    )
                )
            control = self.control_raw + F.linear(features[1], self.fourier_g).unflatten(
                -1, (self.latent_dim, self.action_dim)
            )
            next_z = z + F.linear(z, self.autonomous_matrix) + F.linear(features[0], self.fourier_f)
            next_z = next_z + (control @ u.unsqueeze(-1)).squeeze(-1)
            # Contract output weights before actions: avoid a huge [...,D,2F] intermediate.
            bilinear_tangent = (
                (
                    F.linear(derivatives[1].transpose(-1, -2), self.fourier_g).unflatten(
                        -1, (self.latent_dim, self.action_dim)
                    )
                    * u[..., None, None, :]
                )
                .sum(-1)
                .transpose(-1, -2)
            )
            tangent = tangent + torch.einsum("di,...ik->...dk", self.autonomous_matrix, tangent)
            tangent = (
                tangent
                + torch.einsum("dr,...rk->...dk", self.fourier_f, derivatives[0])
                + bilinear_tangent
            )
            tangent = tangent + F.pad(
                control, (index * self.action_dim, (horizon - index - 1) * self.action_dim)
            )
            z = next_z
        return z, tangent

    def numpy_parameters(self):
        """FP32 CPU arrays for the optional JAX solver; call after training."""
        values = {
            key: value.detach().float().cpu().numpy().copy()
            for key, value in self.state_dict().items()
        }
        values["B"] = values.pop("control_raw")
        return values


class LinearDynamics(FourierDynamics):
    """Nested ablation: z+ = z + Az + Bu, with no Fourier features or bias."""

    def __init__(self, latent_dim=192, action_dim=2, init_scale=1e-3):
        nn.Module.__init__(self)
        if min(latent_dim, action_dim) < 1 or not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("Require positive dimensions and finite positive initialization")
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.init_scale = init_scale
        self.autonomous_matrix = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.control_raw = nn.Parameter(torch.empty(latent_dim, action_dim))
        LinearDynamics.reset_parameters(self)

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.autonomous_matrix, std=self.init_scale / math.sqrt(self.latent_dim))
        nn.init.normal_(self.control_raw, std=self.init_scale / math.sqrt(self.latent_dim))

    def _control(self, z):
        return self.control_raw

    def _state_matrix(self, u):
        return self.autonomous_matrix

    def forward(self, z, u):
        if (
            z.shape[-1] != self.latent_dim
            or u.shape[-1] != self.action_dim
            or z.shape[:-1] != u.shape[:-1]
        ):
            raise ValueError("State and action dimensions must match")
        return (
            z
            + F.linear(z, self.autonomous_matrix)
            + (self._control(z) @ u.unsqueeze(-1)).squeeze(-1)
        )

    def terminal_with_action_jacobian(self, z, controls):
        self._check_sequence(z, controls)
        horizon = controls.shape[-2]
        tangent = z.new_zeros(*z.shape[:-1], self.latent_dim, horizon * self.action_dim)
        for index, u in enumerate(controls.unbind(-2)):
            control = self._control(z)
            tangent = tangent + self._state_matrix(u) @ tangent
            tangent = tangent + F.pad(
                control, (index * self.action_dim, (horizon - index - 1) * self.action_dim)
            )
            z = self(z, u)
        return z, tangent


class BilinearDynamics(LinearDynamics):
    """Nested control-affine ablation: z+ = z + Az + Bu + sum_j u_j N_j z."""

    def __init__(self, latent_dim=192, action_dim=2, init_scale=1e-3):
        super().__init__(latent_dim, action_dim, init_scale)
        self.bilinear_control = nn.Parameter(torch.empty(latent_dim, action_dim, latent_dim))
        nn.init.normal_(self.bilinear_control, std=init_scale / math.sqrt(latent_dim * action_dim))

    def _control(self, z):
        return self.control_raw + torch.einsum("dmj,...j->...dm", self.bilinear_control, z)

    def _state_matrix(self, u):
        return self.autonomous_matrix + torch.einsum("dmj,...m->...dj", self.bilinear_control, u)


class AdjacentHistoryDynamics(FourierDynamics):
    """Learn the newest latent; shift older primitive-frame latents exactly.

    The supplied transition initializes the same Fourier/linear/bilinear family.
    Only output rows for the newest frame are retained as trainable parameters.
    """

    def __init__(self, transition, observation_dim, dynamics_type):
        nn.Module.__init__(self)
        self.latent_dim, self.action_dim = transition.latent_dim, transition.action_dim
        self.observation_dim, self.dynamics_type = observation_dim, dynamics_type
        for name, parameter in transition.named_parameters():
            value = parameter
            if name in {"autonomous_matrix", "control_raw", "fourier_f", "bilinear_control"}:
                value = parameter[:observation_dim]
            elif name == "fourier_g":
                value = parameter[: observation_dim * self.action_dim]
            self.register_parameter(
                name, nn.Parameter(value.detach().clone(), requires_grad=parameter.requires_grad)
            )

    def forward(self, z, u):
        if (
            z.shape[-1] != self.latent_dim
            or u.shape[-1] != self.action_dim
            or z.shape[:-1] != u.shape[:-1]
        ):
            raise ValueError("State and action dimensions must match")
        d = self.observation_dim
        newest = z[..., :d] + F.linear(z, self.autonomous_matrix)
        control = self.control_raw
        if self.dynamics_type == "fourier":
            newest = newest + F.linear(self.features(z, self.omega_f), self.fourier_f)
            control = control + F.linear(self.features(z, self.omega_g), self.fourier_g).unflatten(
                -1, (d, self.action_dim)
            )
        elif self.dynamics_type == "bilinear":
            control = control + torch.einsum("dmj,...j->...dm", self.bilinear_control, z)
        newest = newest + (control @ u.unsqueeze(-1)).squeeze(-1)
        return torch.cat((newest, z[..., :-d]), -1)

    def terminal_with_action_jacobian(self, z, controls):
        """Exact autodiff derivative; history SQP/TV-LQR remains explicitly gated."""
        self._check_sequence(z, controls)

        def terminal(state, actions):
            return self.rollout(state, actions)[-1]

        jacobian = torch.func.vmap(torch.func.jacrev(terminal, argnums=1))(
            z.reshape(-1, self.latent_dim), controls.reshape(-1, *controls.shape[-2:])
        )
        return self.rollout(z, controls)[..., -1, :], jacobian.reshape(
            *z.shape[:-1], self.latent_dim, controls.shape[-2] * self.action_dim
        )

    def numpy_parameters(self):
        raise ValueError(
            "History-state JAX/SQP is not implemented; use the validated Torch CEM path"
        )

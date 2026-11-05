"""JIT batched projected damped Gauss-Newton (the former SQP planner).

Only imports NumPy/JAX: use this module in a separate process/environment when
Torch and JAX CUDA libraries conflict. Export parameters using numpy_parameters.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_default_matmul_precision', 'highest')


def stack_parameters(parameters):
    """One dict or a list of independently trained heads -> leading head axis."""
    heads = [parameters] if isinstance(parameters, dict) else list(parameters)
    if not heads:
        raise ValueError('At least one parameter dictionary is required')
    keys = {'autonomous_matrix', 'B', 'omega_f', 'omega_g', 'fourier_f', 'fourier_g'}
    if any(set(head) != keys for head in heads):
        raise ValueError('Unexpected parameter keys')
    return {key: jnp.asarray(np.stack([p[key] for p in heads]), dtype=jnp.float32) for key in keys}


def make_sqp(horizon=25, action_dim=2, iterations=10, smoothness=.1,
             magnitude=0., damping=.01, bound=3., max_step=1.):
    """Return JIT solve(params, z0[B,D], goal[B,D], initial[B,H,m]).

    params has a leading ensemble axis; each member rolls out independently.
    Members share encoder coordinates. Cost is mean terminal MSE + mean input
    slew penalty (+ optional magnitude). No latent/path/learned cost.
    Uses a damped GN Hessian, not the exact nonlinear Hessian, with a five-point
    projected line search. This is NOT an exact active-set box-QP solver or a
    convergence/safety guarantee. The zero-step candidate ensures no increase
    in the learned-model objective for finite arithmetic and feasible starts.
    """
    if min(horizon, action_dim) < 1 or iterations < 0:
        raise ValueError('Require positive dimensions and nonnegative iterations')
    if any(not math.isfinite(x) for x in (smoothness, magnitude, damping, bound, max_step)):
        raise ValueError('Options must be finite')
    if min(smoothness, magnitude) < 0 or min(damping, bound, max_step) <= 0:
        raise ValueError('Invalid regularization or action limits')
    n = horizon * action_dim
    difference = jnp.eye(n)[action_dim:] - jnp.eye(n)[:-action_dim]
    slew_denominator = max((horizon - 1) * action_dim, 1)

    def one(p, state, controls):
        def step(z, u):
            pf, pg = p['omega_f'] @ z, p['omega_g'] @ z
            features_f = jnp.concatenate((jnp.sin(pf), jnp.cos(pf)))
            features_g = jnp.concatenate((jnp.sin(pg), jnp.cos(pg)))
            control = p['B'] + (p['fourier_g'] @ features_g).reshape(z.shape[-1], action_dim)
            return z + p['autonomous_matrix'] @ z + p['fourier_f'] @ features_f + control @ u, None
        return jax.lax.scan(step, state, controls)[0]

    heads = jax.vmap(jax.vmap(one, in_axes=(None, 0, 0)), in_axes=(0, None, None))
    derivatives = jax.vmap(jax.vmap(jax.jacfwd(one, argnums=2), in_axes=(None, 0, 0)),
                           in_axes=(0, None, None))

    def endpoint(p, z, u):
        value = heads(p, z, u)
        return value.transpose(1, 0, 2).reshape(u.shape[0], -1)

    def cost(p, z, goal, u):
        target = jnp.tile(goal, (1, p['B'].shape[0]))
        terminal = jnp.mean((endpoint(p, z, u) - target) ** 2, -1)
        slew = jnp.sum(jnp.diff(u, axis=1) ** 2, (1, 2)) / slew_denominator
        return terminal + smoothness * slew + magnitude * jnp.mean(u ** 2, (1, 2))

    def solve(p, z, goal, initial):
        if z.ndim != 2 or goal.shape != z.shape or initial.shape != (z.shape[0], horizon, action_dim):
            raise ValueError('Expected z/goal[B,D] and initial[B,H,m]')
        batch, width = z.shape
        d = width * p['B'].shape[0]
        target = jnp.tile(goal, (1, p['B'].shape[0]))
        reg = (2 * smoothness * d / slew_denominator) * (difference.T @ difference)
        reg = reg + 2 * magnitude * d / n * jnp.eye(n)

        def iteration(_, u):
            flat = u.reshape(batch, n)
            residual = endpoint(p, z, u) - target
            jacobian = derivatives(p, z, u).transpose(1, 0, 2, 3, 4).reshape(batch, d, n)
            gradient = 2 * jnp.einsum('bdi,bd->bi', jacobian, residual) + flat @ reg
            normal = 2 * jacobian.transpose(0, 2, 1) @ jacobian + reg + damping * jnp.eye(n)
            direction = jnp.linalg.solve(normal, gradient[..., None])[..., 0]
            direction *= jnp.minimum(1., max_step / jnp.maximum(jnp.max(jnp.abs(direction), -1, keepdims=True), 1e-8))
            alphas = jnp.array([1., .5, .25, .125, 0.])
            candidates = jnp.clip(flat[:, None] - alphas[None, :, None] * direction[:, None], -bound, bound)
            candidates = candidates.reshape(-1, horizon, action_dim)
            losses = cost(p, jnp.repeat(z, 5, 0), jnp.repeat(goal, 5, 0), candidates).reshape(batch, 5)
            return candidates.reshape(batch, 5, horizon, action_dim)[jnp.arange(batch), jnp.argmin(losses, -1)]

        return jax.lax.fori_loop(0, iterations, iteration, jnp.clip(initial, -bound, bound))

    return jax.jit(solve), jax.jit(cost)


def shift_warm_start(controls, executed_steps):
    """Shift [B,H,m], repeat final input to fill tail; use recent observation."""
    if not 1 <= executed_steps <= controls.shape[1]:
        raise ValueError('executed_steps must lie in [1,H]')
    return jnp.concatenate((controls[:, executed_steps:],
                            jnp.repeat(controls[:, -1:], executed_steps, axis=1)), axis=1)


# SWM protocol adapter; the supplied JAX kernels above are unchanged.
class SWMSQPSolver:
    def __init__(self, cost, iterations=10, smoothness=.1, magnitude=0.,
                 damping=.01, bound=3., max_step=1., device="cuda",
                 stats_dataset=None):
        self.cost = cost
        self.device = device
        self.options = dict(iterations=iterations, smoothness=smoothness,
                            magnitude=magnitude, damping=damping,
                            bound=bound, max_step=max_step)
        self.stats_dataset = stats_dataset
        self._configured = False

    def configure(self, *, action_space, n_envs, config):
        from sklearn.preprocessing import StandardScaler
        from stable_worldmodel.data import load_dataset
        self._config, self._n_envs = config, n_envs
        self._primitive_dim = int(np.prod(action_space.shape[1:]))
        model = self.cost.model
        if self._primitive_dim != model.predictor.action_dim:
            raise ValueError("Environment/model primitive action dimensions differ")
        if config.action_block != model.action_block:
            raise ValueError("Training/planning action blocks differ")
        self._params = stack_parameters(model.predictor.numpy_parameters())
        self._solve, self._cost = make_sqp(
            horizon=config.horizon * config.action_block,
            action_dim=self._primitive_dim, **self.options)
        self._mean = model.action_mean.detach().float().cpu().numpy()
        self._scale = model.action_scale.detach().float().cpu().numpy()
        if self.stats_dataset:
            ds = load_dataset(self.stats_dataset, keys_to_load=["action"],
                              keys_to_cache=["action"])
            values = ds.get_col_data("action")
            values = values[~np.isnan(values).any(axis=1)]
            scaler = StandardScaler().fit(values)
            self._eval_mean, self._eval_scale = scaler.mean_, scaler.scale_
        else:
            self._eval_mean, self._eval_scale = self._mean, self._scale
        self._configured = True

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._primitive_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        import torch
        if not self._configured:
            raise RuntimeError("Call configure first")
        model = self.cost.model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        local = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None)
                 if torch.is_tensor(v) else v for k,v in info_dict.items()}
        with torch.no_grad():
            current = model.encode(dict(local))["emb"][:, -1]
            expanded = {k: v.unsqueeze(1) if torch.is_tensor(v) else v
                        for k,v in local.items()}
            goal = self.cost.encode_goal(model, expanded)[:, -1]
        z = jnp.asarray(current.float().cpu().numpy())
        target = jnp.asarray(goal.float().cpu().numpy())
        batch = z.shape[0]
        primitive_horizon = self.horizon * self._config.action_block
        if init_action is None or init_action.shape[1] == 0:
            initial = np.zeros((batch, primitive_horizon, self._primitive_dim),
                               dtype=np.float32)
        else:
            prefix = init_action.detach().float().cpu().numpy().reshape(
                batch, -1, self._primitive_dim)
            prefix = (prefix * self._eval_scale + self._eval_mean - self._mean) / self._scale
            remaining = primitive_horizon - prefix.shape[1]
            if remaining < 0:
                raise ValueError("Warm start exceeds horizon")
            initial = np.concatenate(
                [prefix, np.repeat(prefix[:, -1:], remaining, axis=1)], axis=1)
        initial = jnp.asarray(initial, dtype=jnp.float32)
        optimized = self._solve(self._params, z, target, initial)
        final_cost = np.asarray(self._cost(self._params, z, target, optimized))
        controls = np.asarray(optimized)
        if not np.isfinite(controls).all() or not np.isfinite(final_cost).all():
            raise ValueError("Non-finite SQP output")
        # Convert model-normalized controls back to stock policy's scaler space.
        result = (controls * self._scale + self._mean - self._eval_mean) / self._eval_scale
        actions = torch.from_numpy(np.array(result, dtype=np.float32)).reshape(
            batch, self.horizon, self.action_dim)
        print("SQP_COST", final_cost.tolist(), flush=True)
        return {"actions": actions, "costs": final_cost.tolist()}

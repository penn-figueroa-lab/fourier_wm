"""Invertible command coordinates; these are not measured velocities.

Conversion is applied to dense, current-step records before normalization or
temporal subsampling. Simulator saturation remains part of the environment.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    kinds: tuple[str, ...]
    scale: tuple[float, ...]
    dt: float
    units: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    position_key: str = "proprio"
    position_indices: tuple[int, ...] = ()

    def __post_init__(self):
        n = len(self.kinds)
        if not n or any(len(v) != n for v in (self.scale, self.units, self.low, self.high)):
            raise ValueError("Action channel descriptions must have equal, nonzero widths")
        if any(
            k not in {"velocity", "delta", "position", "force", "torque", "push"}
            for k in self.kinds
        ):
            raise ValueError("Unknown action semantics")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("A positive control interval is required")
        if not np.isfinite([*self.scale, *self.low, *self.high]).all():
            raise ValueError("Action scales and bounds must be finite")
        if any(s <= 0 for s in self.scale) or any(a >= b for a, b in zip(self.low, self.high)):
            raise ValueError("Scales must be positive and lower bounds below upper bounds")
        if "position" in self.kinds and len(self.position_indices) != n:
            raise ValueError("Absolute targets require one current-position index per channel")
        if any(not isinstance(i, int) or i < 0 for i in self.position_indices):
            raise ValueError("Current-position channel indices must be nonnegative integers")

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in value.items()})

    def to_dict(self):
        return asdict(self)


class ActionAdapter:
    def __init__(self, spec):
        self.spec = spec if isinstance(spec, ActionSpec) else ActionSpec.from_dict(spec)

    @property
    def state_dependent(self):
        return "position" in self.spec.kinds

    def _constant(self, values, reference):
        if hasattr(reference, "new_tensor"):
            return reference.new_tensor(values)
        return np.asarray(values, dtype=np.result_type(reference, np.float32))

    def _position(self, action, proprio):
        if not self.state_dependent:
            return action * 0
        if proprio is None:
            raise ValueError("Absolute targets require dense current-step proprioception")
        q = proprio[..., list(self.spec.position_indices)]
        if q.shape != action.shape:
            raise ValueError(
                "Position/action shapes must match; sparse or future positions are invalid"
            )
        return q * self._constant([k == "position" for k in self.spec.kinds], action)

    def to_model(self, action, proprio=None):
        """Native command -> model control (NumPy or differentiable Torch).

        Position/displacement channels become commanded rates. Native force and
        torque channels retain their declared scaling and are never divided by dt.
        These coordinates alone do not supply an inertial model's hidden velocity.
        """
        if "push" in self.spec.kinds:
            raise ValueError(
                "A complete push is not a primitive rate; replay dense commanded motion before Fourier training"
            )
        if action.shape[-1] != len(self.spec.kinds):
            raise ValueError("Action width does not match its declared semantics")
        denominator = self._constant(
            [self.spec.dt if k in {"delta", "position"} else 1 for k in self.spec.kinds], action
        )
        return (
            action * self._constant(self.spec.scale, action) - self._position(action, proprio)
        ) / denominator

    def to_native(self, command, proprio=None, *, clip=False):
        if "push" in self.spec.kinds:
            raise ValueError("A complete push has no primitive-rate inverse")
        if command.shape[-1] != len(self.spec.kinds):
            raise ValueError("Command width does not match its declared semantics")
        denominator = self._constant(
            [self.spec.dt if k in {"delta", "position"} else 1 for k in self.spec.kinds], command
        )
        action = (command * denominator + self._position(command, proprio)) / self._constant(
            self.spec.scale, command
        )
        if clip:
            low, high = (
                self._constant(self.spec.low, action),
                self._constant(self.spec.high, action),
            )
            action = (
                action.clamp(low, high) if hasattr(action, "clamp") else np.clip(action, low, high)
            )
        return action

    def command_bounds(self, proprio=None):
        low, high = np.asarray(self.spec.low), np.asarray(self.spec.high)
        return self.to_model(low, proprio), self.to_model(high, proprio)


@dataclass
class ActionNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        self.mean, self.scale = np.asarray(self.mean), np.asarray(self.scale)
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("Normalization must have primitive action width")
        if not np.isfinite([self.mean, self.scale]).all() or (self.scale <= 0).any():
            raise ValueError("Invalid action normalization")

    @classmethod
    def fit(cls, actions):
        actions = np.asarray(actions)
        finite = np.isfinite(actions).all(-1)
        if not finite.any():
            raise ValueError("No finite training commands")
        return cls(actions[finite].mean(0), np.maximum(actions[finite].std(0), 1e-8))

    def transform(self, x):
        return ((x - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, x):
        return (x * self.scale + self.mean).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

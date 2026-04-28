"""Virtual haptic handpiece — 2nd-order mass-damper + actuator lag + clamp.

Used by the offline bench to close the force/position loop without a real
device. Models the user's hand as a lightly-stiff grip holding the stylus:
the commanded force perturbs the stylus away from a reference trajectory
and the perturbation decays through hand damping / grip stiffness.

The model is:
    f_applied = first-order-filter(saturate(f_cmd), tau=actuator_tau)
    m * a + b * v + k * d = f_applied
    position = reference_position + d
    velocity = reference_velocity + v

Defaults approximate a 3D Systems Touch stylus held in a moderately firm
grip. Users can override via `characterize` (see bench CLI) to fit to a
specific device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VirtualHandpieceParams:
    mass: float = 0.1  # kg
    damping: float = 2.0  # N·s/m
    hand_stiffness: float = 50.0  # N/m (grip impedance)
    actuator_tau: float = 0.002  # s (first-order force actuator lag)
    max_force: float = 3.3  # N (saturation — Touch-class peak)


class VirtualHandpiece:
    """Closed-loop virtual device model."""

    def __init__(self, params: VirtualHandpieceParams | None = None):
        self.params = params if params is not None else VirtualHandpieceParams()
        self._perturbation = np.zeros(3, dtype=np.float32)
        self._perturbation_velocity = np.zeros(3, dtype=np.float32)
        self._applied_force = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self._perturbation.fill(0.0)
        self._perturbation_velocity.fill(0.0)
        self._applied_force.fill(0.0)

    def step(
        self,
        commanded_force: np.ndarray,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance one frame. Returns (position, velocity, applied_force)."""
        p = self.params
        dt = max(float(dt), 1.0e-6)

        f_cmd = np.asarray(commanded_force, dtype=np.float32)
        f_mag = float(np.linalg.norm(f_cmd))
        if p.max_force > 0.0 and f_mag > p.max_force:
            f_cmd = f_cmd * (p.max_force / max(f_mag, 1.0e-8))

        alpha = 1.0 - float(np.exp(-dt / max(p.actuator_tau, 1.0e-6)))
        self._applied_force = self._applied_force + alpha * (f_cmd - self._applied_force)

        mass = max(p.mass, 1.0e-6)
        accel = (
            self._applied_force
            - p.damping * self._perturbation_velocity
            - p.hand_stiffness * self._perturbation
        ) / mass
        self._perturbation_velocity = self._perturbation_velocity + accel * dt
        self._perturbation = self._perturbation + self._perturbation_velocity * dt

        position = np.asarray(reference_position, dtype=np.float32) + self._perturbation
        velocity = (
            np.asarray(reference_velocity, dtype=np.float32) + self._perturbation_velocity
        )
        return (
            position.astype(np.float32, copy=True),
            velocity.astype(np.float32, copy=True),
            self._applied_force.astype(np.float32, copy=True),
        )

    @property
    def applied_force(self) -> np.ndarray:
        return self._applied_force.copy()

    @property
    def perturbation(self) -> np.ndarray:
        return self._perturbation.copy()

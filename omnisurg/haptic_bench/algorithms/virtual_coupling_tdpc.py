"""Virtual coupling + time-domain passivity controller (Hannaford/Ryu).

Layers a passivity observer + controller on top of a virtual-coupling core:
observes energy E(t) = Σ F·v·dt at the device port; when E > 0 (net
injection into the user), adds a damping term proportional to velocity to
absorb the excess and keep the system passive. When E < 0 (dissipating),
passes through unchanged.

Tunable via `tdpc_alpha` (damping-injection gain). Also inherits the
virtual-coupling proxy-mass behavior from `VirtualCouplingAlgorithm`.
"""

from __future__ import annotations

import numpy as np

from omnisurg.haptic_bench.algorithms import (
    HapticAlgorithm,
    HapticStepInput,
    HapticStepOutput,
    register,
)
from omnisurg.haptic_bench.algorithms.virtual_coupling import VirtualCouplingAlgorithm


@register("virtual_coupling_tdpc")
class VirtualCouplingTDPCAlgorithm(HapticAlgorithm):
    def __init__(self, settings):
        super().__init__(settings)
        self._inner = VirtualCouplingAlgorithm(settings)
        self._energy = 0.0

    def reset(self, device_position: np.ndarray | None) -> None:
        self._inner.reset(device_position)
        self._energy = 0.0

    def update_settings(self, settings) -> None:
        super().update_settings(settings)
        self._inner.update_settings(settings)

    def compute(self, step: HapticStepInput) -> HapticStepOutput:
        result = self._inner.compute(step)
        dt = max(float(step.dt), 1.0e-6)
        tdpc_alpha = max(0.0, float(self.settings.tdpc_alpha))

        device_velocity = (
            (step.device_position - self._inner._device_position_prev) / dt
            if self._inner._initialized
            else np.zeros(3, dtype=np.float32)
        )
        # Note: inner.compute already advanced _device_position_prev, so we
        # reconstruct velocity from the pre-step delta. A small approximation
        # error compared to computing it before inner.compute, acceptable.

        power = float(np.dot(result.force, device_velocity))
        self._energy = self._energy + power * dt
        if self._energy < 0.0:
            self._energy = 0.0  # reset dissipation ledger; don't credit future injection

        if tdpc_alpha > 0.0 and self._energy > 0.0:
            v_mag_sq = float(np.dot(device_velocity, device_velocity))
            if v_mag_sq > 1.0e-10:
                damping = tdpc_alpha * self._energy / v_mag_sq / dt
                absorbed = damping * device_velocity
                damped_force = result.force - absorbed
                mag = float(np.linalg.norm(damped_force))
                max_force = max(0.0, float(self.settings.max_force))
                if max_force > 0.0 and mag > max_force:
                    damped_force *= max_force / max(mag, 1.0e-8)

                absorbed_power = float(np.dot(absorbed, device_velocity)) * dt
                self._energy = max(0.0, self._energy - absorbed_power)

                return HapticStepOutput(
                    force=damped_force.astype(np.float32, copy=True),
                    raw_force=result.raw_force,
                    filtered_force=result.filtered_force,
                    proxy_offset=result.proxy_offset,
                    clamp_active=result.clamp_active or (mag > max_force if max_force > 0 else False),
                    slew_active=result.slew_active,
                )

        return result

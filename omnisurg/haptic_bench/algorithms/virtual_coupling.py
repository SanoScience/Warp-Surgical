"""Classic virtual coupling — proxy with inertia.

Same topology as `spring_damper`, but the proxy is a dynamic body with mass
`vc_proxy_mass` driven by a spring-damper towards the reaction target,
instead of the kinematic first-order follower. The output force to the
device is still the spring-damper between proxy and device.

Why it matters: the kinematic follower in `spring_damper` has no inertia,
so proxy velocity can be unnaturally fast on contact transitions, which
is a known oscillation driver. Giving the proxy mass band-limits its
velocity response.
"""

from __future__ import annotations

import numpy as np

from omnisurg.haptic_bench.algorithms import (
    HapticAlgorithm,
    HapticStepInput,
    HapticStepOutput,
    register,
)


@register("virtual_coupling")
class VirtualCouplingAlgorithm(HapticAlgorithm):
    def __init__(self, settings):
        super().__init__(settings)
        self._initialized = False
        self._proxy_position = np.zeros(3, dtype=np.float32)
        self._proxy_velocity = np.zeros(3, dtype=np.float32)
        self._device_position_prev = np.zeros(3, dtype=np.float32)
        self._lowpass_state = np.zeros(3, dtype=np.float32)
        self._force_command_prev = np.zeros(3, dtype=np.float32)

    def reset(self, device_position: np.ndarray | None) -> None:
        base = (
            np.asarray(device_position, dtype=np.float32).copy()
            if device_position is not None
            else np.zeros(3, dtype=np.float32)
        )
        self._initialized = device_position is not None
        self._proxy_position = base.copy()
        self._proxy_velocity = np.zeros(3, dtype=np.float32)
        self._device_position_prev = base.copy()
        self._lowpass_state = np.zeros(3, dtype=np.float32)
        self._force_command_prev = np.zeros(3, dtype=np.float32)

    def compute(self, step: HapticStepInput) -> HapticStepOutput:
        s = self.settings
        dt = max(float(step.dt), 1.0e-6)
        device_position = np.asarray(step.device_position, dtype=np.float32).copy()
        reaction = np.asarray(step.avg_reaction_offset, dtype=np.float32).copy()

        if not self._initialized:
            self.reset(device_position)

        reaction_scale = max(0.0, float(s.reaction_scale))
        max_proxy_offset = max(0.0, float(s.max_proxy_offset))
        spring_k = max(0.0, float(s.spring_k))
        damper_b = max(0.0, float(s.damper_b))
        deadband = max(0.0, float(s.deadband))
        lowpass_alpha = float(np.clip(s.lowpass_alpha, 0.0, 1.0))
        max_force = max(0.0, float(s.max_force))
        slew_rate_limit = max(0.0, float(s.slew_rate_limit))
        proxy_mass = max(float(s.vc_proxy_mass), 1.0e-4)

        target = device_position + reaction_scale * reaction
        target_offset = target - device_position
        offset_mag = float(np.linalg.norm(target_offset))
        if max_proxy_offset > 0.0 and offset_mag > max_proxy_offset:
            target_offset *= max_proxy_offset / max(offset_mag, 1.0e-8)
        target = device_position + target_offset

        proxy_force = spring_k * (target - self._proxy_position) - damper_b * self._proxy_velocity
        proxy_accel = proxy_force / proxy_mass
        self._proxy_velocity = self._proxy_velocity + proxy_accel * dt
        self._proxy_position = self._proxy_position + self._proxy_velocity * dt

        device_velocity = (device_position - self._device_position_prev) / dt
        raw_force = spring_k * (
            self._proxy_position - device_position
        ) + damper_b * (self._proxy_velocity - device_velocity)

        deadband_force = raw_force.copy()
        if float(np.linalg.norm(deadband_force)) < deadband:
            deadband_force.fill(0.0)

        filtered_force = (
            lowpass_alpha * deadband_force + (1.0 - lowpass_alpha) * self._lowpass_state
        )
        self._lowpass_state = filtered_force.copy()

        final_force = filtered_force.copy()
        clamp_active = False
        mag = float(np.linalg.norm(final_force))
        if max_force <= 0.0:
            clamp_active = mag > 0.0
            final_force.fill(0.0)
        elif mag > max_force:
            final_force *= max_force / max(mag, 1.0e-8)
            clamp_active = True

        slew_active = False
        if slew_rate_limit > 0.0:
            max_delta = slew_rate_limit * dt
            delta = final_force - self._force_command_prev
            delta_mag = float(np.linalg.norm(delta))
            if delta_mag > max_delta and delta_mag > 1.0e-8:
                final_force = self._force_command_prev + delta * (max_delta / delta_mag)
                slew_active = True

        self._force_command_prev = final_force.astype(np.float32, copy=True)
        self._device_position_prev = device_position.copy()

        return HapticStepOutput(
            force=self._force_command_prev.copy(),
            raw_force=raw_force.astype(np.float32, copy=True),
            filtered_force=filtered_force.astype(np.float32, copy=True),
            proxy_offset=(self._proxy_position - device_position).astype(
                np.float32, copy=True
            ),
            clamp_active=clamp_active,
            slew_active=slew_active,
        )

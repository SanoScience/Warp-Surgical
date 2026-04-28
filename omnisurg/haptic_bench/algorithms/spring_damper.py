"""Baseline spring-damper force-feedback algorithm.

Verbatim port of the original `Runtime._update_haptic_force_feedback` logic.
Pipeline: proxy-follow towards target offset, spring+damper between proxy
and device, deadband, first-order lowpass, magnitude clamp, slew-rate limit.
"""

from __future__ import annotations

import numpy as np

from omnisurg.haptic_bench.algorithms import (
    HapticAlgorithm,
    HapticStepInput,
    HapticStepOutput,
    register,
)


@register("spring_damper")
class SpringDamperAlgorithm(HapticAlgorithm):
    def __init__(self, settings):
        super().__init__(settings)
        self._initialized = False
        self._proxy_position = np.zeros(3, dtype=np.float32)
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
        self._device_position_prev = base.copy()
        self._proxy_position = base.copy()
        self._lowpass_state = np.zeros(3, dtype=np.float32)
        self._force_command_prev = np.zeros(3, dtype=np.float32)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def compute(self, step: HapticStepInput) -> HapticStepOutput:
        settings = self.settings
        device_position = np.asarray(step.device_position, dtype=np.float32).copy()
        reaction = np.asarray(step.avg_reaction_offset, dtype=np.float32).copy()
        dt = max(float(step.dt), 1.0e-6)

        if not self._initialized:
            self.reset(device_position)

        reaction_scale = max(0.0, float(settings.reaction_scale))
        proxy_follow = float(np.clip(settings.proxy_follow, 0.0, 1.0))
        max_proxy_offset = max(0.0, float(settings.max_proxy_offset))
        spring_k = max(0.0, float(settings.spring_k))
        damper_b = max(0.0, float(settings.damper_b))
        deadband = max(0.0, float(settings.deadband))
        lowpass_alpha = float(np.clip(settings.lowpass_alpha, 0.0, 1.0))
        max_force = max(0.0, float(settings.max_force))
        slew_rate_limit = max(0.0, float(settings.slew_rate_limit))

        proxy_target = device_position + reaction_scale * reaction
        proxy_offset = proxy_target - device_position
        proxy_offset_mag = float(np.linalg.norm(proxy_offset))
        if max_proxy_offset > 0.0 and proxy_offset_mag > max_proxy_offset:
            proxy_offset *= max_proxy_offset / max(proxy_offset_mag, 1.0e-8)
        proxy_target = device_position + proxy_offset

        proxy_prev = self._proxy_position.copy()
        device_prev = self._device_position_prev.copy()
        self._proxy_position = proxy_prev + proxy_follow * (proxy_target - proxy_prev)

        device_velocity = (device_position - device_prev) / dt
        proxy_velocity = (self._proxy_position - proxy_prev) / dt
        raw_force = spring_k * (self._proxy_position - device_position) + damper_b * (
            proxy_velocity - device_velocity
        )

        deadband_force = raw_force.copy()
        if float(np.linalg.norm(deadband_force)) < deadband:
            deadband_force.fill(0.0)

        filtered_force = (
            lowpass_alpha * deadband_force
            + (1.0 - lowpass_alpha) * self._lowpass_state
        )
        self._lowpass_state = filtered_force.copy()

        final_force = filtered_force.copy()
        clamp_active = False
        filtered_force_mag = float(np.linalg.norm(final_force))
        if max_force <= 0.0:
            clamp_active = filtered_force_mag > 0.0
            final_force.fill(0.0)
        elif filtered_force_mag > max_force:
            final_force *= max_force / max(filtered_force_mag, 1.0e-8)
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

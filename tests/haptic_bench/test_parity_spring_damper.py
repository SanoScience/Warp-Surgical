"""Parity test: new SpringDamperAlgorithm must match the old inlined formula.

This is the gate for the algorithm extraction refactor. Byte-identical
(within FP order-of-ops) is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from omnisurg.haptic_bench.algorithms import HapticStepInput
from omnisurg.haptic_bench.algorithms.spring_damper import SpringDamperAlgorithm
from omnisurg.haptic_feedback import HapticFeedbackSettings


def _inlined_reference(settings, frames, dt):
    """Replica of the pre-refactor Runtime._update_haptic_force_feedback logic."""
    proxy_position = np.zeros(3, dtype=np.float32)
    device_position_prev = np.zeros(3, dtype=np.float32)
    lowpass_state = np.zeros(3, dtype=np.float32)
    force_command = np.zeros(3, dtype=np.float32)
    initialized = False

    out_raw, out_filt, out_final = [], [], []
    out_proxy, out_clamp, out_slew = [], [], []

    for device_position, reaction in frames:
        device_position = np.asarray(device_position, dtype=np.float32).copy()
        reaction = np.asarray(reaction, dtype=np.float32).copy()

        if not initialized:
            device_position_prev = device_position.copy()
            proxy_position = device_position.copy()
            lowpass_state = np.zeros(3, dtype=np.float32)
            force_command = np.zeros(3, dtype=np.float32)
            initialized = True

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

        proxy_prev = proxy_position.copy()
        device_prev = device_position_prev.copy()
        proxy_position = proxy_prev + proxy_follow * (proxy_target - proxy_prev)

        device_velocity = (device_position - device_prev) / dt
        proxy_velocity = (proxy_position - proxy_prev) / dt
        raw_force = spring_k * (proxy_position - device_position) + damper_b * (
            proxy_velocity - device_velocity
        )

        deadband_force = raw_force.copy()
        if float(np.linalg.norm(deadband_force)) < deadband:
            deadband_force.fill(0.0)

        filtered_force = lowpass_alpha * deadband_force + (1.0 - lowpass_alpha) * lowpass_state
        lowpass_state = filtered_force.copy()

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
            delta = final_force - force_command
            delta_mag = float(np.linalg.norm(delta))
            if delta_mag > max_delta and delta_mag > 1.0e-8:
                final_force = force_command + delta * (max_delta / delta_mag)
                slew_active = True

        force_command = final_force.astype(np.float32, copy=True)
        device_position_prev = device_position.copy()

        out_raw.append(raw_force.copy())
        out_filt.append(filtered_force.copy())
        out_final.append(force_command.copy())
        out_proxy.append((proxy_position - device_position).astype(np.float32, copy=True))
        out_clamp.append(clamp_active)
        out_slew.append(slew_active)

    return (
        np.asarray(out_raw),
        np.asarray(out_filt),
        np.asarray(out_final),
        np.asarray(out_proxy),
        np.asarray(out_clamp),
        np.asarray(out_slew),
    )


def _run_algorithm(settings, frames, dt):
    algo = SpringDamperAlgorithm(settings)
    out_raw, out_filt, out_final, out_proxy, out_clamp, out_slew = [], [], [], [], [], []
    for device_position, reaction in frames:
        result = algo.compute(
            HapticStepInput(
                device_position=np.asarray(device_position, dtype=np.float32),
                avg_reaction_offset=np.asarray(reaction, dtype=np.float32),
                contact_count=1,
                dt=dt,
            )
        )
        out_raw.append(result.raw_force)
        out_filt.append(result.filtered_force)
        out_final.append(result.force)
        out_proxy.append(result.proxy_offset)
        out_clamp.append(result.clamp_active)
        out_slew.append(result.slew_active)
    return (
        np.asarray(out_raw),
        np.asarray(out_filt),
        np.asarray(out_final),
        np.asarray(out_proxy),
        np.asarray(out_clamp),
        np.asarray(out_slew),
    )


def _synthesize_frames(n=400, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 120.0
    device = np.stack(
        [
            0.01 * np.sin(2 * np.pi * 1.5 * t),
            0.005 * np.cos(2 * np.pi * 2.0 * t),
            0.02 * np.sin(2 * np.pi * 0.8 * t),
        ],
        axis=1,
    ).astype(np.float32)
    reaction = 0.004 * rng.standard_normal(size=(n, 3)).astype(np.float32)
    reaction[:50] = 0.0
    reaction[200:260] += np.array([0.0, 0.0, -0.03], dtype=np.float32)
    return list(zip(device, reaction))


def run_parity():
    settings = HapticFeedbackSettings()
    dt = 1.0 / 120.0
    frames = _synthesize_frames()
    ref = _inlined_reference(settings, frames, dt)
    new = _run_algorithm(settings, frames, dt)

    for name, r, n in zip(
        ["raw", "filtered", "final", "proxy_offset"], ref[:4], new[:4]
    ):
        diff = np.max(np.abs(r - n))
        print(f"[{name}] max abs diff = {diff:.3e}")
        assert diff < 1.0e-6, f"{name} parity broken: max diff {diff}"

    clamp_match = np.array_equal(ref[4], new[4])
    slew_match = np.array_equal(ref[5], new[5])
    print(f"[clamp_active] match = {clamp_match}")
    print(f"[slew_active]  match = {slew_match}")
    assert clamp_match and slew_match, "flag parity broken"

    print("PARITY OK")


if __name__ == "__main__":
    run_parity()

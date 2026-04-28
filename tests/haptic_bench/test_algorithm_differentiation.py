"""Sanity check: each algorithm produces measurably different metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from omnisurg.haptic_bench.runner import run_single
from omnisurg.haptic_feedback import HapticFeedbackSettings


def _synthesize_trace(n: int = 360, fs: float = 120.0):
    t = (np.arange(n) / fs).astype(np.float32)
    dt = np.full(n, 1.0 / fs, dtype=np.float32)
    device = np.zeros((n, 3), dtype=np.float32)
    device[:, 2] = 0.02 * np.sin(2 * np.pi * 0.5 * t)
    velocity = np.zeros_like(device)
    velocity[1:] = (device[1:] - device[:-1]) / dt[1:, None]

    reaction = np.zeros_like(device)
    contact = np.zeros(n, dtype=np.int32)
    contact[120:240] = 1
    reaction[120:240, 2] = -0.008 + 0.002 * np.sin(2 * np.pi * 15.0 * t[120:240])

    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "t": t,
        "dt": dt,
        "device_position": device,
        "device_velocity": velocity,
        "avg_reaction_offset": reaction,
        "contact_count": contact,
    }


def main():
    trace = _synthesize_trace()
    settings = HapticFeedbackSettings(spring_k=50.0, damper_b=2.0, max_force=1.0)
    results: dict[str, dict[str, float]] = {}
    for alg in ("spring_damper", "virtual_coupling", "virtual_coupling_tdpc"):
        _, metrics = run_single(alg, settings, trace)
        results[alg] = metrics
        print(
            f"[{alg:26s}] osc_peak={metrics['oscillation_peak']:.4f}  "
            f"osc_rms={metrics['oscillation_rms']:.4f}  "
            f"E_inj={metrics['energy_injection']:+.4f}  "
            f"rise_ms={metrics['contact_rise_time_ms']:.1f}"
        )

    sd = results["spring_damper"]["oscillation_rms"]
    vc = results["virtual_coupling"]["oscillation_rms"]
    assert abs(sd - vc) > 1.0e-6, "virtual_coupling output matches spring_damper"
    print("DIFFERENTIATION OK")


if __name__ == "__main__":
    main()

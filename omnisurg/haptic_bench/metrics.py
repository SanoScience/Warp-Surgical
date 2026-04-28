"""Metrics over a haptic bench run.

Pure functions over numpy arrays. Inputs are per-frame traces produced by
the bench runner; outputs are scalars composed into a `metric_vector` dict.
No weighting or scalar reduction happens here — the study layer exposes
the full vector to Optuna for multi-objective optimization.

Metric reference:
- `oscillation_peak`: peak magnitude of the commanded-force FFT in the
  20–300 Hz band. Proxy for audible/felt buzz. Lower is better.
- `oscillation_rms`: RMS energy in the same band. Less susceptible to
  single-spike outliers than peak.
- `energy_injection`: running sum of F·v over contact frames. Positive
  means the force loop is injecting energy into the hand — the classic
  haptic-instability signature. Near zero or negative is good.
- `jerk_rms`: RMS of the 3rd derivative of perturbation position.
  Perceptual smoothness proxy.
- `contact_rise_time_ms`: time from first contact frame to 90% of the
  eventual steady-state force. Lower = crisper contact.
- `free_space_force_mean`: mean |F| during frames with no contact.
  Should be ~0; positive values mean parasitic drag.
- `penetration_max_mm`: magnitude of the largest reaction offset seen
  during contact. Rough proxy for how "soft" the wall feels.
- `latency_frames`: indicative actuator delay in frames — derivable from
  virtual device params, written through for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BenchTrace:
    """Per-frame trace emitted by the bench runner."""

    t: np.ndarray            # (N,)
    dt: np.ndarray           # (N,)
    device_position: np.ndarray    # (N, 3) post-perturbation
    reference_position: np.ndarray  # (N, 3) unperturbed
    device_velocity: np.ndarray    # (N, 3) post-perturbation
    commanded_force: np.ndarray    # (N, 3) algorithm output (pre-device clamp)
    applied_force: np.ndarray      # (N, 3) post actuator lag + clamp
    contact_count: np.ndarray      # (N,) int
    reaction_offset: np.ndarray    # (N, 3) contact trace input


def _fs_from_dt(dt: np.ndarray) -> float:
    mean_dt = float(np.mean(dt[dt > 0])) if np.any(dt > 0) else 0.0
    return 1.0 / mean_dt if mean_dt > 0 else 0.0


def oscillation_band(
    force: np.ndarray, fs: float, band: tuple[float, float] = (20.0, 300.0)
) -> tuple[float, float]:
    """Return (peak, rms) of |F| spectrum within the given Hz band."""
    if force.shape[0] < 8 or fs <= 0:
        return 0.0, 0.0
    magnitude = np.linalg.norm(force, axis=1).astype(np.float64)
    magnitude = magnitude - np.mean(magnitude)
    n = magnitude.shape[0]
    window = np.hanning(n)
    spectrum = np.fft.rfft(magnitude * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(spectrum) * (2.0 / np.sum(window))
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0, 0.0
    band_amp = amp[mask]
    peak = float(np.max(band_amp))
    rms = float(np.sqrt(np.mean(band_amp**2)))
    return peak, rms


def energy_injection(
    force: np.ndarray, velocity: np.ndarray, dt: np.ndarray, contact_count: np.ndarray
) -> float:
    mask = contact_count > 0
    if not np.any(mask):
        return 0.0
    power = np.sum(force * velocity, axis=1)
    return float(np.sum(power[mask] * dt[mask]))


def jerk_rms(position: np.ndarray, dt: np.ndarray) -> float:
    if position.shape[0] < 4:
        return 0.0
    mean_dt = float(np.mean(dt[dt > 0])) if np.any(dt > 0) else 1.0
    deriv3 = np.diff(position, n=3, axis=0) / (mean_dt**3)
    return float(np.sqrt(np.mean(np.sum(deriv3**2, axis=1))))


def contact_rise_time_ms(
    force: np.ndarray, contact_count: np.ndarray, dt: np.ndarray
) -> float:
    mask = contact_count > 0
    if not np.any(mask):
        return 0.0
    idx = np.flatnonzero(mask)
    segments: list[np.ndarray] = []
    cur_start = idx[0]
    for i in range(1, idx.shape[0]):
        if idx[i] != idx[i - 1] + 1:
            segments.append(np.arange(cur_start, idx[i - 1] + 1))
            cur_start = idx[i]
    segments.append(np.arange(cur_start, idx[-1] + 1))

    rise_times_s: list[float] = []
    for seg in segments:
        if seg.shape[0] < 4:
            continue
        seg_force = np.linalg.norm(force[seg], axis=1)
        peak = float(np.max(seg_force))
        if peak <= 1.0e-6:
            continue
        target = 0.9 * peak
        above = np.flatnonzero(seg_force >= target)
        if above.size == 0:
            continue
        first_idx = above[0]
        elapsed = float(np.sum(dt[seg[:first_idx + 1]]))
        rise_times_s.append(elapsed)

    if not rise_times_s:
        return 0.0
    return float(np.mean(rise_times_s) * 1000.0)


def free_space_force_mean(force: np.ndarray, contact_count: np.ndarray) -> float:
    mask = contact_count == 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.linalg.norm(force[mask], axis=1)))


def penetration_max_mm(
    reaction_offset: np.ndarray, contact_count: np.ndarray
) -> float:
    mask = contact_count > 0
    if not np.any(mask):
        return 0.0
    mags = np.linalg.norm(reaction_offset[mask], axis=1)
    return float(np.max(mags) * 1000.0)


def compute_metrics(trace: BenchTrace) -> dict[str, float]:
    fs = _fs_from_dt(trace.dt)
    osc_peak, osc_rms = oscillation_band(trace.applied_force, fs)
    return {
        "fs_hz": float(fs),
        "n_frames": int(trace.t.shape[0]),
        "contact_frames": int(np.sum(trace.contact_count > 0)),
        "oscillation_peak": osc_peak,
        "oscillation_rms": osc_rms,
        "energy_injection": energy_injection(
            trace.applied_force, trace.device_velocity, trace.dt, trace.contact_count
        ),
        "jerk_rms": jerk_rms(
            trace.device_position - trace.reference_position, trace.dt
        ),
        "contact_rise_time_ms": contact_rise_time_ms(
            trace.applied_force, trace.contact_count, trace.dt
        ),
        "free_space_force_mean": free_space_force_mean(
            trace.applied_force, trace.contact_count
        ),
        "penetration_max_mm": penetration_max_mm(
            trace.reaction_offset, trace.contact_count
        ),
    }

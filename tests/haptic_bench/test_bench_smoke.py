"""Smoke test: synthesize a contact trace, run the bench, verify metrics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


SCHEMA_VERSION = 1


def _synthesize_trace(path: Path, n: int = 360, fs: float = 120.0):
    """Make a cloth-contact-like trace: device moves into a 'wall' then out."""
    rng = np.random.default_rng(3)
    t = (np.arange(n) / fs).astype(np.float32)
    dt = (np.full(n, 1.0 / fs)).astype(np.float32)

    device = np.zeros((n, 3), dtype=np.float32)
    device[:, 2] = 0.02 * np.sin(2 * np.pi * 0.5 * t)
    device[:, 0] = -0.01 + 0.01 * np.cos(2 * np.pi * 0.3 * t)

    velocity = np.zeros((n, 3), dtype=np.float32)
    velocity[1:] = (device[1:] - device[:-1]) / dt[1:, None]

    reaction = np.zeros((n, 3), dtype=np.float32)
    contact = np.zeros(n, dtype=np.int32)
    contact_window = slice(120, 240)
    contact[contact_window] = 1
    reaction[contact_window, 2] = -0.005 - 0.001 * np.sin(
        2 * np.pi * 2.0 * t[contact_window]
    )

    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        t=t,
        dt=dt,
        device_position=device,
        device_velocity=velocity,
        avg_reaction_offset=reaction,
        contact_count=contact,
    )


def main():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        trace_path = tmp / "trace.npz"
        out_path = tmp / "run.json"
        _synthesize_trace(trace_path)

        cmd = [
            sys.executable,
            "-m",
            "omnisurg.haptic_bench",
            "run",
            "--algorithm",
            "spring_damper",
            "--contact-trace",
            str(trace_path),
            "--out",
            str(out_path),
            "--json",
        ]
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bench run failed: {result.stderr}"

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["schema"] == "haptic-bench/v1"
        metrics = payload["data"]["metrics"]
        assert metrics["n_frames"] == 360
        assert metrics["contact_frames"] == 120
        for key in (
            "oscillation_peak",
            "oscillation_rms",
            "energy_injection",
            "jerk_rms",
            "contact_rise_time_ms",
            "free_space_force_mean",
            "penetration_max_mm",
        ):
            assert key in metrics, f"missing metric: {key}"
        assert metrics["free_space_force_mean"] < 1.0e-4, (
            f"non-zero free-space force: {metrics['free_space_force_mean']}"
        )
        print(json.dumps(metrics, indent=2))
        print("SMOKE OK")


if __name__ == "__main__":
    main()

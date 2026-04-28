"""End-to-end contract: ContactTraceRecorder output loads cleanly into runner."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from omnisurg.haptic_bench.runner import load_contact_trace, run_single
from omnisurg.haptic_feedback import HapticFeedbackDiagnostics, HapticFeedbackSettings
from omnisurg.telemetry import ContactTraceRecorder


@dataclass
class _FakeControllerState:
    scaled_position: np.ndarray
    sample_present: bool = True


@dataclass
class _FakeSimConfig:
    frame_dt: float = 1.0 / 120.0


@dataclass
class _FakeRuntime:
    haptic_feedback_diagnostics: HapticFeedbackDiagnostics
    controller_states: dict
    sim_config: _FakeSimConfig
    sim_time: float = 0.0

    def set_force_feedback_compute_always(self, _enabled: bool) -> None:
        pass


def main():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        out = tmp / "trace.npz"

        diagnostics = HapticFeedbackDiagnostics()
        state = _FakeControllerState(scaled_position=np.zeros(3, dtype=np.float32))
        rt = _FakeRuntime(
            haptic_feedback_diagnostics=diagnostics,
            controller_states={ContactTraceRecorder.CONTROLLER_ID: state},
            sim_config=_FakeSimConfig(),
        )
        recorder = ContactTraceRecorder(rt, out)

        n = 60
        for i in range(n):
            t = i * rt.sim_config.frame_dt
            rt.sim_time = t
            state.scaled_position = np.array(
                [0.0, 0.0, 0.01 * np.sin(2 * np.pi * 0.5 * t)], dtype=np.float32
            )
            contact_on = 20 <= i < 40
            diagnostics.contact_count = 1 if contact_on else 0
            diagnostics.avg_reaction_offset = (
                np.array([0.0, 0.0, -0.005], dtype=np.float32)
                if contact_on
                else np.zeros(3, dtype=np.float32)
            )
            recorder.record()
        recorder.close()

        assert out.exists()
        trace = load_contact_trace(out)
        for key in (
            "schema_version",
            "t",
            "dt",
            "device_position",
            "device_velocity",
            "avg_reaction_offset",
            "contact_count",
        ):
            assert key in trace, f"missing key: {key}"
        assert int(trace["schema_version"]) == 1
        assert trace["t"].shape == (n,)
        assert trace["device_position"].shape == (n, 3)

        settings = HapticFeedbackSettings()
        _, metrics = run_single("spring_damper", settings, trace)
        assert metrics["n_frames"] == n
        assert metrics["contact_frames"] == 20
        print("CONTACT-TRACE ROUNDTRIP OK")


if __name__ == "__main__":
    main()

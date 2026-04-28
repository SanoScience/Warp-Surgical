"""Unit tests for Stage3ValidationRig — PD tracking, ramp, abort."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from omnisurg.input.position_tracking import Stage3ValidationRig
from omnisurg.input.sources import ControllerSample, InputRig


class _ScriptedReplay(InputRig):
    def __init__(self, positions: np.ndarray):
        self._pos = positions
        self._frame = 0

    def poll(self):
        idx = min(self._frame, self._pos.shape[0] - 1)
        self._frame += 1
        return {"right": ControllerSample(position=self._pos[idx].astype(np.float32))}

    def set_force_commands(self, _):
        pass

    def close(self):
        pass

    def supports_force_feedback(self, _id=None):
        return False

    def reset(self):
        self._frame = 0


class _MockDevice(InputRig):
    """Device that perfectly matches whatever target was commanded last frame
    (infinite tracking bandwidth). Lets us unit-test the rig's PD arithmetic."""

    def __init__(self, start: np.ndarray, *, stuck: bool = False):
        self._pos = start.astype(np.float32).copy()
        self._last_force = np.zeros(3, dtype=np.float32)
        self._stuck = stuck

    def poll(self):
        return {"right": ControllerSample(position=self._pos.copy())}

    def set_force_commands(self, forces):
        self._last_force = forces.get("right", np.zeros(3, dtype=np.float32)).copy()

    def supports_force_feedback(self, _id=None):
        return True

    def reset(self):
        pass

    def close(self):
        pass

    def advance(self, new_pos):
        if not self._stuck:
            self._pos = new_pos.astype(np.float32).copy()


def test_ramp_in_stays_small():
    """During ramp-in, tracking error stays bounded and PD force is gentle."""
    target_positions = np.tile(np.array([100.0, 0.0, 0.0], dtype=np.float32), (240, 1))
    replay = _ScriptedReplay(target_positions)
    device = _MockDevice(np.zeros(3, dtype=np.float32))

    rig = Stage3ValidationRig(
        replay, device,
        sim_frame_dt=1.0 / 120.0,
        kp=0.02, kd=0.005,
        force_cap=1.5,
        abort_error=50.0,
        abort_error_frames=10,
        ramp_in_frames=180,
    )

    pd_mags: list[float] = []
    for _ in range(60):
        rig.poll()
        rig.set_force_commands({"right": np.zeros(3, dtype=np.float32)})
        snap = rig.last_dispatched
        pd_mags.append(float(np.linalg.norm(snap["pd_force"])))
        device.advance(snap["total_force"] * 0 + snap["effective_target"])

    assert max(pd_mags) < 1.5, f"PD force peaked at {max(pd_mags):.3f} N (above cap)"
    assert rig._frame == 60 and not rig.aborted
    print(f"[ramp_in] peak PD force during first 60 frames: {max(pd_mags):.4f} N  OK")


def test_abort_on_stuck_device():
    """If the device can't track the target, abort fires after the threshold window."""
    target_positions = np.tile(np.array([100.0, 0.0, 0.0], dtype=np.float32), (240, 1))
    replay = _ScriptedReplay(target_positions)
    device = _MockDevice(np.zeros(3, dtype=np.float32), stuck=True)

    rig = Stage3ValidationRig(
        replay, device,
        sim_frame_dt=1.0 / 120.0,
        kp=0.02, kd=0.005,
        force_cap=1.5,
        abort_error=10.0,  # mm
        abort_error_frames=6,
        ramp_in_frames=30,
    )

    fired_at = None
    for i in range(200):
        rig.poll()
        rig.set_force_commands({"right": np.zeros(3, dtype=np.float32)})
        if rig.aborted and fired_at is None:
            fired_at = i
            break
    assert fired_at is not None, "Stage3 rig never aborted despite stuck device"
    assert fired_at > 30, f"Abort fired during ramp-in (frame {fired_at})"
    print(f"[abort] fired at frame {fired_at} after stuck-device sustained error  OK")


def test_contact_force_sums_correctly():
    """Contact force from sim gets added to PD force before dispatch."""
    target_positions = np.zeros((10, 3), dtype=np.float32)
    replay = _ScriptedReplay(target_positions)
    device = _MockDevice(np.zeros(3, dtype=np.float32))

    rig = Stage3ValidationRig(
        replay, device,
        sim_frame_dt=1.0 / 120.0,
        kp=0.0, kd=0.0,  # PD disabled — total = contact only
        force_cap=1.5,
        abort_error=50.0, abort_error_frames=100,
        ramp_in_frames=0,
    )
    contact = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    for _ in range(5):
        rig.poll()
        rig.set_force_commands({"right": contact.copy()})
    snap = rig.last_dispatched
    diff = float(np.linalg.norm(snap["total_force"] - contact))
    assert diff < 1.0e-6, f"total_force should equal contact when PD off, got diff={diff}"
    print(f"[contact_sum] total==contact when Kp=Kd=0  OK")


def main():
    test_ramp_in_stays_small()
    test_abort_on_stuck_device()
    test_contact_force_sums_correctly()
    print("STAGE3 RIG OK")


if __name__ == "__main__":
    main()

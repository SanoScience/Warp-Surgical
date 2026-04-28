"""Stage-3 real-device validation rig.

Drives a real haptic device to physically follow a recorded position trace
while the running sim computes contact forces in parallel. Both the
position-tracking PD output and the contact force are summed at the device,
then saturated to a common cap. The whole point is: a tuned force-feedback
algorithm gets evaluated against the real device's dynamics (latency,
friction, encoder noise, actuator lag) with a fully deterministic trajectory
— same trace, same contact events, run after run.

THIS IS ASSUMED UNATTENDED. Do not hold the stylus while stage-3 runs.
The abort logic is specifically designed to catch a human grabbing the
stylus as a sustained tracking error and ramp forces to zero.

Loop-rate note: the PD here runs at sim rate (~120 Hz). The device callback
still runs at ~1 kHz and holds the last commanded force between sim ticks.
This means tracking bandwidth is limited by sim rate, not device rate, which
is acceptable for v1 — the goal is to compare algorithms against a
predictable device, not to win tracking bandwidth competitions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from omnisurg.input.sources import ControllerSample, InputRig


class Stage3ValidationRig(InputRig):
    """Wrap (replay_rig, live_rig) to drive the real device to replay positions
    while still dispatching sim-computed contact forces on top.

    Sim sees the replay positions exactly as it would under `--replay`. The
    device sees `PD(target, measured) + contact_force`, saturated to
    `force_cap`. Safety layers:
      * Ramp-in over `ramp_in_frames` blends smoothly from the device's
        measured pose at t=0 toward the first replay target.
      * After ramp-in, tracking error magnitude is monitored; if it exceeds
        `abort_error` for `abort_error_frames` consecutive frames (a human
        grabbing the stylus, or impossible kinematics), all forces drop to
        zero and the rig latches into aborted state.
      * `close()` always zeroes forces before tearing the device down.
    """

    ABORT_ACTIONS = ("zero", "hold", "continue")

    def __init__(
        self,
        replay_rig: InputRig,
        force_rig: InputRig,
        *,
        sim_frame_dt: float,
        controller_id: str = "right",
        kp: float = 30.0,
        kd: float = 1.0,
        force_cap: float = 1.5,
        abort_error: float = 0.010,
        abort_error_frames: int = 6,
        ramp_in_frames: int = 180,
        abort_action: str = "zero",
        on_abort: Callable[[], None] | None = None,
    ):
        if abort_action not in self.ABORT_ACTIONS:
            raise ValueError(
                f"abort_action must be one of {self.ABORT_ACTIONS}, got {abort_action!r}"
            )

        self._replay_rig = replay_rig
        self._force_rig = force_rig
        self._controller_id = controller_id
        self._sim_frame_dt = max(float(sim_frame_dt), 1.0e-6)
        self._kp = max(0.0, float(kp))
        self._kd = max(0.0, float(kd))
        self._force_cap = max(0.0, float(force_cap))
        self._abort_error = max(0.0, float(abort_error))
        self._abort_error_frames = max(1, int(abort_error_frames))
        self._ramp_in_frames = max(0, int(ramp_in_frames))
        self._abort_action = abort_action
        self._on_abort = on_abort

        self._frame = 0
        self._consecutive_error_frames = 0
        self._aborted = False
        self._abort_event_logged = False
        self._playback_done = False
        self._playback_done_announced = False
        self._last_target: np.ndarray | None = None
        self._last_measured: np.ndarray | None = None
        self._ramp_start: np.ndarray | None = None
        self._last_dispatched: dict = self._empty_dispatched()

    @staticmethod
    def _empty_dispatched() -> dict:
        zero = np.zeros(3, dtype=np.float32)
        return {
            "target": zero.copy(),
            "measured": zero.copy(),
            "effective_target": zero.copy(),
            "pd_force": zero.copy(),
            "contact_force": zero.copy(),
            "total_force": zero.copy(),
            "tracking_error": 0.0,
            "in_ramp": True,
            "saturated": False,
        }

    def poll(self) -> dict[str, ControllerSample]:
        frame = self._replay_rig.poll()
        sample = frame.get(self._controller_id) if frame else None
        if sample is not None and sample.position is not None:
            self._last_target = np.asarray(sample.position, dtype=np.float32).copy()

        checker = getattr(self._replay_rig, "replay_exhausted", None)
        if callable(checker) and checker(self._controller_id):
            if not self._playback_done:
                self._playback_done = True
                if not self._playback_done_announced:
                    print("[stage3] playback complete — device idle (zero force). Press P to replay.")
                    self._playback_done_announced = True
        return frame

    def set_force_commands(self, force_commands: dict[str, np.ndarray]):
        zero = np.zeros(3, dtype=np.float32)
        if self._aborted:
            if self._abort_action == "zero":
                self._force_rig.set_force_commands(
                    {cid: zero.copy() for cid in force_commands}
                    or {self._controller_id: zero.copy()}
                )
            # "hold": don't dispatch — device holds whatever was last commanded.
            return

        if self._playback_done:
            # Trace finished: send zero force so the device falls limp. User
            # can press P to reset(); ramp_start will re-seed from wherever
            # the stylus ended up (e.g. resting on the desk).
            self._force_rig.set_force_commands(
                {cid: zero.copy() for cid in force_commands}
                or {self._controller_id: zero.copy()}
            )
            self._last_dispatched = self._empty_dispatched()
            return

        live_frame = self._force_rig.poll() or {}
        measured_sample = live_frame.get(self._controller_id)
        if measured_sample is None or measured_sample.position is None:
            # Device not reporting pose — safest to zero and wait.
            self._force_rig.set_force_commands(
                {cid: zero.copy() for cid in force_commands}
                or {self._controller_id: zero.copy()}
            )
            self._last_dispatched = self._empty_dispatched()
            return

        measured = np.asarray(measured_sample.position, dtype=np.float32).copy()
        target = (
            self._last_target.copy()
            if self._last_target is not None
            else measured.copy()
        )

        if self._ramp_start is None:
            self._ramp_start = measured.copy()
        in_ramp = self._frame < self._ramp_in_frames
        if in_ramp and self._ramp_in_frames > 0:
            blend = float(self._frame + 1) / float(self._ramp_in_frames)
            effective_target = (1.0 - blend) * self._ramp_start + blend * target
        else:
            effective_target = target

        if self._last_measured is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            velocity = (measured - self._last_measured) / self._sim_frame_dt
        self._last_measured = measured.copy()

        error = effective_target - measured
        pd_force = self._kp * error - self._kd * velocity

        contact_force = np.asarray(
            force_commands.get(self._controller_id, zero),
            dtype=np.float32,
        ).copy()

        # During ramp-in the device hasn't reached the trajectory yet, so any
        # contact force the sim generates is for a position the real device
        # does not occupy — it's phantom and would crowd out the PD budget.
        # Only start dispatching contact force once ramp-in finishes.
        if in_ramp:
            effective_contact_force = zero.copy()
        else:
            effective_contact_force = contact_force

        total_force = pd_force + effective_contact_force
        total_mag = float(np.linalg.norm(total_force))
        saturated = False
        if self._force_cap > 0.0 and total_mag > self._force_cap:
            total_force = total_force * (self._force_cap / max(total_mag, 1.0e-8))
            saturated = True

        error_mag = float(np.linalg.norm(error))
        if not in_ramp and self._abort_error > 0.0:
            if error_mag > self._abort_error:
                self._consecutive_error_frames += 1
                if self._consecutive_error_frames >= self._abort_error_frames:
                    self._fire_abort(error_mag)
                    if self._abort_action != "continue":
                        return
                    # "continue": log once per excursion, reset counter, keep going.
                    self._consecutive_error_frames = 0
            else:
                self._consecutive_error_frames = 0
                self._abort_event_logged = False

        dispatch = {cid: np.asarray(f, dtype=np.float32).copy() for cid, f in force_commands.items()}
        dispatch[self._controller_id] = total_force.astype(np.float32, copy=True)
        self._force_rig.set_force_commands(dispatch)

        self._last_dispatched = {
            "target": target.copy(),
            "measured": measured.copy(),
            "effective_target": effective_target.astype(np.float32, copy=True),
            "pd_force": pd_force.astype(np.float32, copy=True),
            "contact_force": effective_contact_force.copy(),
            "total_force": total_force.astype(np.float32, copy=True),
            "tracking_error": error_mag,
            "in_ramp": in_ramp,
            "saturated": saturated,
        }
        self._frame += 1

    def _fire_abort(self, error_mag: float) -> None:
        if self._abort_action == "continue":
            if not self._abort_event_logged:
                print(
                    f"[stage3] WARN: tracking error {error_mag:.2f} exceeded "
                    f"{self._abort_error:.2f} (native trace units) for "
                    f"{self._abort_error_frames} frames. Continuing per --stage3-abort-action=continue."
                )
                self._abort_event_logged = True
            return
        self._aborted = True
        suffix = (
            "Zeroing forces." if self._abort_action == "zero"
            else "Holding last force (no further dispatch)."
        )
        print(
            f"[stage3] ABORT: tracking error {error_mag:.2f} held above "
            f"{self._abort_error:.2f} (native trace units) for "
            f"{self._abort_error_frames} frames. {suffix}"
        )
        if self._abort_action == "zero":
            zero = np.zeros(3, dtype=np.float32)
            try:
                self._force_rig.set_force_commands({self._controller_id: zero.copy()})
            except Exception:
                pass
        if self._on_abort is not None:
            try:
                self._on_abort()
            except Exception:
                pass

    def supports_force_feedback(self, controller_id: str | None = None) -> bool:
        return self._force_rig.supports_force_feedback(controller_id)

    def reset(self):
        self._replay_rig.reset()
        self._frame = 0
        self._consecutive_error_frames = 0
        self._aborted = False
        self._abort_event_logged = False
        self._playback_done = False
        self._playback_done_announced = False
        self._ramp_start = None
        self._last_measured = None
        self._last_dispatched = self._empty_dispatched()

    def close(self):
        if self._abort_action != "hold":
            zero = np.zeros(3, dtype=np.float32)
            try:
                self._force_rig.set_force_commands({self._controller_id: zero.copy()})
            except Exception:
                pass
        try:
            self._replay_rig.close()
        finally:
            self._force_rig.close()

    @property
    def last_dispatched(self) -> dict:
        return dict(self._last_dispatched)

    @property
    def aborted(self) -> bool:
        return self._aborted


def analyse_trace_for_stage3(
    trace_path: str,
    *,
    sim_frame_dt: float,
    kp: float,
    kd: float,
    force_cap: float,
    abort_error: float,
) -> dict:
    """Pre-flight analysis: scan a replay trace, estimate worst-case PD demands.

    No device contact. Returns a summary dict the CLI prints before handing
    control to the device. If the estimated worst-case exceeds `force_cap`,
    the dry-run flags it so the user can back off Kp/Kd before a live run.
    """
    data = np.load(trace_path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Replay trace '{trace_path}' has unexpected shape {data.shape}")

    positions = data[:, :3].astype(np.float32)
    if positions.shape[0] < 2:
        return {"n_frames": int(positions.shape[0]), "warning": "trace too short"}

    diffs = np.diff(positions, axis=0)
    velocities = diffs / max(sim_frame_dt, 1.0e-6)
    speed = np.linalg.norm(velocities, axis=1)
    max_speed = float(np.max(speed))
    mean_speed = float(np.mean(speed))

    excursion = positions - np.mean(positions, axis=0, keepdims=True)
    max_excursion = float(np.max(np.linalg.norm(excursion, axis=1)))

    first = positions[0]
    worst_ramp_error = float(np.linalg.norm(first))

    worst_pd_force_during_track = kp * abort_error + kd * max_speed
    worst_pd_force_ramp = kp * worst_ramp_error + kd * max_speed

    return {
        "trace_path": trace_path,
        "n_frames": int(positions.shape[0]),
        "duration_s": float(positions.shape[0] * sim_frame_dt),
        "max_speed_per_s": max_speed,
        "mean_speed_per_s": mean_speed,
        "max_excursion": max_excursion,
        "first_sample_magnitude": worst_ramp_error,
        "worst_pd_force_during_track_N": worst_pd_force_during_track,
        "worst_pd_force_ramp_N": worst_pd_force_ramp,
        "force_cap_N": force_cap,
        "exceeds_force_cap": (
            worst_pd_force_during_track > force_cap
            or worst_pd_force_ramp > force_cap
        ),
        "units_note": "speed/excursion/magnitude are in native trace units (mm for Touch).",
    }

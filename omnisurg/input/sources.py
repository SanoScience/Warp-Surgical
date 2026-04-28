from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ControllerSample:
    """Normalized controller sample used by multi-controller rigs."""

    position: np.ndarray | None = None
    rotation: np.ndarray | None = None
    button: bool = False
    grip: float = 0.0

    @property
    def active(self) -> bool:
        return self.position is not None or self.rotation is not None

    @classmethod
    def from_sample_dict(cls, sample: dict | None):
        if not sample:
            return cls()

        position = _as_array(sample.get("position"), expected_size=3)
        rotation = _as_array(sample.get("rotation"), expected_size=4)
        button = bool(sample.get("button", False))
        grip = _as_unit_interval(sample.get("grip"))
        if grip is None:
            grip = 1.0 if button else 0.0
        return cls(
            position=position,
            rotation=rotation,
            button=button,
            grip=grip,
        )


class InputSource(ABC):
    """Single-controller input device or replay feed."""

    @abstractmethod
    def poll(self) -> dict:
        """Return the latest controller sample or an empty dict when unavailable."""

    def close(self):
        pass

    def set_force(self, force_xyz):
        pass

    def supports_force_feedback(self) -> bool:
        return False

    def reset(self):
        pass


class InputRig(ABC):
    """Multi-controller input rig composed from one or more input sources."""

    @abstractmethod
    def poll(self) -> dict[str, ControllerSample]:
        """Return the latest controller samples keyed by controller role."""

    def close(self):
        pass

    def set_force_commands(self, force_commands: dict[str, np.ndarray]):
        pass

    def supports_force_feedback(self, controller_id: str | None = None) -> bool:
        return False

    def reset(self):
        pass


class MultiSourceRig(InputRig):
    """Compose multiple single-controller sources into a multi-controller rig."""

    def __init__(self, sources: dict[str, InputSource]):
        self._sources = dict(sources)

    def poll(self) -> dict[str, ControllerSample]:
        frame: dict[str, ControllerSample] = {}
        for controller_id, source in self._sources.items():
            sample = ControllerSample.from_sample_dict(source.poll())
            if sample.active or sample.button or sample.grip > 0.0:
                frame[controller_id] = sample
        return frame

    def replay_exhausted(self, controller_id: str) -> bool:
        source = self._sources.get(controller_id)
        return bool(getattr(source, "exhausted", False))

    def close(self):
        for source in self._sources.values():
            source.close()

    def set_force_commands(self, force_commands: dict[str, np.ndarray]):
        zero = np.zeros(3, dtype=np.float32)
        for controller_id, source in self._sources.items():
            force = force_commands.get(controller_id, zero)
            source.set_force(force)

    def supports_force_feedback(self, controller_id: str | None = None) -> bool:
        if controller_id is not None:
            source = self._sources.get(controller_id)
            return bool(source is not None and source.supports_force_feedback())
        return any(source.supports_force_feedback() for source in self._sources.values())

    def reset(self):
        for source in self._sources.values():
            source.reset()


class LiveHapticSource(InputSource):
    def __init__(self, scale: float = 1.0, device_name: str = "Default Device"):
        from omnisurg.input.device import HapticController

        self._device_name = device_name
        self._ctrl = HapticController(device_name=device_name, scale=scale)
        self._reported_failure = False

    def poll(self) -> dict:
        if self._ctrl is None:
            return {}

        try:
            if hasattr(self._ctrl, "poll_state"):
                sample = self._ctrl.poll_state()
                return {
                    "position": np.array(sample["position"], dtype=np.float32),
                    "rotation": np.array(sample["rotation"], dtype=np.float32),
                    "button": bool(sample["button"]),
                }

            return {
                "position": np.array(self._ctrl.get_scaled_position(), dtype=np.float32),
                "rotation": np.array(self._ctrl.get_rotation(), dtype=np.float32),
                "button": bool(self._ctrl.is_button_pressed()),
            }
        except RuntimeError as exc:
            if not self._reported_failure:
                print(f'Live haptic source "{self._device_name}" stopped updating: {exc}')
                self._reported_failure = True
            self.close()
            return {}

    def close(self):
        if self._ctrl is not None:
            self._ctrl.close()
            self._ctrl = None

    def set_force(self, force_xyz):
        if self._ctrl is None:
            return
        self._ctrl.set_force(force_xyz)

    def supports_force_feedback(self) -> bool:
        return self._ctrl is not None


class LiveMiniMouSource(InputSource):
    def __init__(self, *, scale: float = 1.0, root: str | Path | None = None, device_index: int = 0):
        from omnisurg.input.follou import MiniMouController

        self._description = f"MiniMou[{device_index}]"
        self._ctrl = MiniMouController(root=root, device_index=device_index, scale=scale)
        self._reported_failure = False

    def poll(self) -> dict:
        if self._ctrl is None:
            return {}

        try:
            sample = self._ctrl.poll()
            sample["position"] = np.asarray(sample.get("position"), dtype=np.float32)
            sample["rotation"] = np.asarray(sample.get("rotation"), dtype=np.float32)
            return sample
        except Exception as exc:
            if not self._reported_failure:
                print(f'Live input source "{self._description}" stopped updating: {exc}')
                self._reported_failure = True
            self.close()
            return {}

    def close(self):
        if self._ctrl is not None:
            self._ctrl.close()
            self._ctrl = None


class ReplayInputSource(InputSource):
    """Replay a deterministic haptic trace saved as an `(N, 7)` or `(N, 8)` NumPy array.

    Once the trace is exhausted, the last recorded sample is held indefinitely
    so downstream logic (force-feedback diagnostics, telemetry) keeps seeing a
    valid device pose instead of dropping to zero. Call :meth:`reset` (bound to
    `P` in the viewer) to restart from the beginning.
    """

    def __init__(self, path: str):
        self._path = path
        self._data = np.load(path)
        self._frame = 0
        self._end_announced = False

    @property
    def exhausted(self) -> bool:
        return self._frame >= len(self._data)

    def reset(self):
        self._frame = 0
        self._end_announced = False

    def poll(self) -> dict:
        if len(self._data) == 0:
            return {}

        if self._frame >= len(self._data):
            idx = len(self._data) - 1
            if not self._end_announced:
                print(f"[replay] trace exhausted, holding last pose ({self._path})")
                self._end_announced = True
        else:
            idx = self._frame
            self._frame += 1

        sample = self._data[idx]
        result = {
            "position": sample[:3].astype(np.float32),
            "rotation": sample[3:7].astype(np.float32),
        }
        if sample.shape[0] >= 8:
            result["button"] = bool(sample[7] > 0.5)
        if sample.shape[0] >= 9:
            result["grip"] = float(np.clip(sample[8], 0.0, 1.0))
        return result


class RecordingRig(InputRig):
    """Wrap an `InputRig` to record per-controller samples to `.npy` traces.

    Each saved row has 9 float32 columns: `[px, py, pz, qx, qy, qz, qw, button, grip]`,
    matching the format accepted by `ReplayInputSource`.
    """

    def __init__(self, rig: InputRig, output_paths: dict[str, str | Path]):
        self._rig = rig
        self._output_paths: dict[str, Path] = {
            controller_id: Path(path)
            for controller_id, path in output_paths.items()
            if path is not None
        }
        if not self._output_paths:
            raise ValueError("RecordingRig requires at least one output path")
        self._buffers: dict[str, list[list[float]]] = {cid: [] for cid in self._output_paths}
        self._active = False
        self._take_counter = 0

    @property
    def is_recording(self) -> bool:
        return self._active

    @property
    def output_paths(self) -> dict[str, Path]:
        return dict(self._output_paths)

    def poll(self) -> dict[str, ControllerSample]:
        frame = self._rig.poll()
        if self._active:
            for controller_id, buffer in self._buffers.items():
                sample = frame.get(controller_id)
                if sample is None or sample.position is None or sample.rotation is None:
                    continue
                buffer.append([
                    float(sample.position[0]),
                    float(sample.position[1]),
                    float(sample.position[2]),
                    float(sample.rotation[0]),
                    float(sample.rotation[1]),
                    float(sample.rotation[2]),
                    float(sample.rotation[3]),
                    1.0 if sample.button else 0.0,
                    float(sample.grip),
                ])
        return frame

    def start_recording(self):
        if self._active:
            return
        for buffer in self._buffers.values():
            buffer.clear()
        self._active = True
        print(f"[record] take {self._take_counter + 1} started")

    def stop_recording(self, *, save: bool = True) -> dict[str, Path]:
        if not self._active:
            return {}
        self._active = False
        written: dict[str, Path] = {}
        if save:
            for controller_id, buffer in self._buffers.items():
                if not buffer:
                    print(f"[record] {controller_id}: no frames captured, skipping")
                    continue
                path = self._take_path(controller_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.asarray(buffer, dtype=np.float32))
                written[controller_id] = path
                print(f"[record] {controller_id}: wrote {len(buffer)} frames to {path}")
        self._take_counter += 1
        for buffer in self._buffers.values():
            buffer.clear()
        return written

    def toggle_recording(self) -> bool:
        if self._active:
            self.stop_recording()
        else:
            self.start_recording()
        return self._active

    def _take_path(self, controller_id: str) -> Path:
        base = self._output_paths[controller_id]
        if self._take_counter == 0:
            return base
        return base.with_name(f"{base.stem}.{self._take_counter:03d}{base.suffix}")

    def close(self):
        if self._active:
            self.stop_recording(save=True)
        self._rig.close()

    def set_force_commands(self, force_commands: dict[str, np.ndarray]):
        self._rig.set_force_commands(force_commands)

    def supports_force_feedback(self, controller_id: str | None = None) -> bool:
        return self._rig.supports_force_feedback(controller_id)

    def reset(self):
        self._rig.reset()


class ReplayForceFeedbackRig(InputRig):
    """Drive poll data from a replay rig while dispatching forces to a live rig.

    Positions, rotations, and buttons come from the replay trace so the simulation
    remains deterministic, but force-feedback commands generated by the live sim
    are forwarded to a real haptic rig so the operator can feel the playback.
    """

    def __init__(self, replay_rig: InputRig, force_rig: InputRig):
        self._replay_rig = replay_rig
        self._force_rig = force_rig

    def poll(self) -> dict[str, ControllerSample]:
        return self._replay_rig.poll()

    def set_force_commands(self, force_commands: dict[str, np.ndarray]):
        self._force_rig.set_force_commands(force_commands)

    def supports_force_feedback(self, controller_id: str | None = None) -> bool:
        return self._force_rig.supports_force_feedback(controller_id)

    def reset(self):
        self._replay_rig.reset()

    def close(self):
        try:
            self._replay_rig.close()
        finally:
            self._force_rig.close()


class BimanualOpenHapticsRig(MultiSourceRig):
    """Two-controller rig backed by two OpenHaptics devices."""

    def __init__(
        self,
        *,
        scale: float = 1.0,
        right_device_name: str = "Default Device",
        left_device_name: str = "Left Device",
        include_left: bool = True,
    ):
        sources: dict[str, InputSource] = {
            "right": LiveHapticSource(scale=scale, device_name=right_device_name),
        }
        if include_left:
            sources["left"] = LiveHapticSource(scale=scale, device_name=left_device_name)
        super().__init__(sources)


class BimanualReplayRig(MultiSourceRig):
    """Two-controller rig composed from replay traces."""

    def __init__(self, *, right_path: str | None = None, left_path: str | None = None):
        sources: dict[str, InputSource] = {}
        if right_path is not None:
            sources["right"] = ReplayInputSource(right_path)
        if left_path is not None:
            sources["left"] = ReplayInputSource(left_path)
        if not sources:
            raise ValueError("At least one replay trace must be provided")
        super().__init__(sources)


def _as_array(value, *, expected_size: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size < expected_size:
        return None
    return array[:expected_size].copy()


def _as_unit_interval(value) -> float | None:
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric):
        return None

    return float(np.clip(numeric, 0.0, 1.0))

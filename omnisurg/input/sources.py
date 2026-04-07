from __future__ import annotations

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

    @property
    def active(self) -> bool:
        return self.position is not None or self.rotation is not None

    @classmethod
    def from_sample_dict(cls, sample: dict | None):
        if not sample:
            return cls()

        position = _as_array(sample.get("position"), expected_size=3)
        rotation = _as_array(sample.get("rotation"), expected_size=4)
        return cls(
            position=position,
            rotation=rotation,
            button=bool(sample.get("button", False)),
        )


class InputSource(ABC):
    """Single-controller input device or replay feed."""

    @abstractmethod
    def poll(self) -> dict:
        """Return the latest controller sample or an empty dict when unavailable."""

    def close(self):
        pass


class InputRig(ABC):
    """Multi-controller input rig composed from one or more input sources."""

    @abstractmethod
    def poll(self) -> dict[str, ControllerSample]:
        """Return the latest controller samples keyed by controller role."""

    def close(self):
        pass


class MultiSourceRig(InputRig):
    """Compose multiple single-controller sources into a multi-controller rig."""

    def __init__(self, sources: dict[str, InputSource]):
        self._sources = dict(sources)

    def poll(self) -> dict[str, ControllerSample]:
        frame: dict[str, ControllerSample] = {}
        for controller_id, source in self._sources.items():
            sample = ControllerSample.from_sample_dict(source.poll())
            if sample.active or sample.button:
                frame[controller_id] = sample
        return frame

    def close(self):
        for source in self._sources.values():
            source.close()


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
    """Replay a deterministic haptic trace saved as an `(N, 7)` or `(N, 8)` NumPy array."""

    def __init__(self, path: str):
        self._data = np.load(path)
        self._frame = 0

    def poll(self) -> dict:
        if self._frame >= len(self._data):
            return {}

        sample = self._data[self._frame]
        self._frame += 1
        result = {
            "position": sample[:3].astype(np.float32),
            "rotation": sample[3:7].astype(np.float32),
        }
        if sample.shape[0] >= 8:
            result["button"] = bool(sample[7] > 0.5)
        return result


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

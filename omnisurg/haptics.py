from abc import ABC, abstractmethod

import numpy as np


class InputSource(ABC):
    """Abstraction over haptic input so the runtime is device-agnostic."""

    @abstractmethod
    def poll(self) -> dict:
        """Return the latest sample.

        Expected keys (all optional):
            "position": np.ndarray shape (3,) float32
            "rotation": np.ndarray shape (4,) float32  (quaternion xyzw)
        Returns an empty dict when no data is available.
        """
        ...

    def close(self):
        pass


class LiveHapticSource(InputSource):
    """Wraps the real OpenHaptics device via haptic_device.HapticController."""

    def __init__(self, scale: float = 1.0):
        from haptic_device import HapticController

        self._ctrl = HapticController(scale=scale)

    def poll(self) -> dict:
        return {
            "position": np.array(self._ctrl.get_scaled_position(), dtype=np.float32),
            "rotation": np.array(self._ctrl.get_rotation(), dtype=np.float32),
        }

    def close(self):
        self._ctrl = None


class ReplayInputSource(InputSource):
    """Plays back a recorded haptic trace for deterministic testing.

    The trace file is a NumPy .npy with shape (N, 7): [px, py, pz, qx, qy, qz, qw].
    """

    def __init__(self, path: str):
        self._data = np.load(path)
        self._frame = 0

    def poll(self) -> dict:
        if self._frame >= len(self._data):
            return {}
        sample = self._data[self._frame]
        self._frame += 1
        return {
            "position": sample[:3].astype(np.float32),
            "rotation": sample[3:7].astype(np.float32),
        }

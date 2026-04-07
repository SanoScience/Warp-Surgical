from abc import ABC, abstractmethod

import numpy as np


class InputSource(ABC):
    """Abstraction over live and replayed haptic samples."""

    @abstractmethod
    def poll(self) -> dict:
        """Return the latest sample or an empty dict when no input is available."""

    def close(self):
        pass


class LiveHapticSource(InputSource):
    def __init__(self, scale: float = 1.0):
        from omnisurg.input.device import HapticController

        self._ctrl = HapticController(scale=scale)

    def poll(self) -> dict:
        return {
            "position": np.array(self._ctrl.get_scaled_position(), dtype=np.float32),
            "rotation": np.array(self._ctrl.get_rotation(), dtype=np.float32),
        }

    def close(self):
        self._ctrl = None


class ReplayInputSource(InputSource):
    """Replay a deterministic haptic trace saved as an `(N, 7)` NumPy array."""

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

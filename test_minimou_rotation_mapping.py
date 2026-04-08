from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

import omnisurg.main as omnisurg_main


def _live_args(**overrides):
    values = {
        "right_input_backend": "minimou",
        "left_input_backend": "none",
        "right_device_name": "Default Device",
        "left_device_name": "Left Device",
        "right_device_index": 0,
        "left_device_index": 1,
        "follou_root": r"G:\warp\python_device_manager",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeSource:
    def __init__(self, root: str, device_index: int, sample=None):
        self.root = root
        self.device_index = device_index
        self.sample = sample if sample is not None else {}

    def poll(self):
        return dict(self.sample)

    def close(self):
        pass


class TestMiniMouRotationMapping(TestCase):
    def test_live_rig_reflects_minimou_quaternion_x(self):
        sample = {
            "position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "rotation": np.array([0.25, 0.5, -0.75, 1.0], dtype=np.float32),
            "button": False,
            "grip": 0.0,
        }

        def build_source(*, root: str, device_index: int, scale: float = 1.0):
            return _FakeSource(root, device_index, sample=sample)

        rig = None
        with mock.patch("omnisurg.haptics.LiveMiniMouSource", side_effect=build_source):
            rig = omnisurg_main._build_live_input_rig(_live_args())

        try:
            frame = rig.poll()
            self.assertIn("right", frame)
            np.testing.assert_allclose(
                frame["right"].rotation,
                np.array([-0.25, 0.5, -0.75, 1.0], dtype=np.float32),
            )
            np.testing.assert_allclose(
                frame["right"].position,
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
        finally:
            if rig is not None:
                rig.close()

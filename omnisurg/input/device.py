import time
from dataclasses import dataclass, field

import numpy as np
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice


@dataclass
class DeviceState:
    button: bool = False
    position: list = field(default_factory=list)
    rotation: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    gimbals: list = field(default_factory=list)
    force: list = field(default_factory=list)


class HapticController:
    def __init__(self, device_name="Default Device", scale=2.5):
        self.device_state = DeviceState()
        self.device_state.force = [0.0, 0.0, 0.0]
        self.device_state.rotation = [0.0, 0.0, 0.0, 1.0]
        self.scale = scale

        self.invert_x = False
        self.invert_y = False
        self.invert_z = False
        self.invert_w = True

        self._setup_callback()
        self.device = HapticDevice(device_name=device_name, callback=self._state_callback)
        time.sleep(0.2)

    def _setup_callback(self):
        @hd_callback
        def state_callback():
            transform = hd.get_transform()
            joints = hd.get_joints()
            gimbals = hd.get_gimbals()

            self.device_state.position = [transform[3][0], transform[3][1], transform[3][2]]

            rotation_matrix = np.array([
                [transform[0][0], transform[0][1], transform[0][2]],
                [transform[1][0], transform[1][1], transform[1][2]],
                [transform[2][0], transform[2][1], transform[2][2]],
            ])
            self.device_state.rotation = self._matrix_to_quaternion(rotation_matrix)

            self.device_state.joints = [joints[0], joints[1], joints[2]]
            self.device_state.gimbals = [gimbals[0], gimbals[1], gimbals[2]]
            hd.set_force(self.device_state.force)
            button = hd.get_buttons()
            self.device_state.button = button == 1

        self._state_callback = state_callback

    def _matrix_to_quaternion(self, matrix):
        trace = np.trace(matrix)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s

        if self.invert_x:
            x = -x
        if self.invert_y:
            y = -y
        if self.invert_z:
            z = -z
        if self.invert_w:
            w = -w

        return [x, y, z, w]

    def get_scaled_position(self):
        return [pos * self.scale for pos in self.device_state.position]

    def get_rotation(self):
        return self.device_state.rotation.copy()

    def set_force(self, force):
        self.device_state.force = force

    def is_button_pressed(self):
        return self.device_state.button

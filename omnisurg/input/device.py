import threading
import time
from ctypes import CFUNCTYPE, POINTER, c_char_p, c_uint, c_void_p, c_ushort
from dataclasses import dataclass, field

import numpy as np
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_define import (
    HD_CALLBACK_CONTINUE,
    HD_CALLBACK_DONE,
    HD_DEVICE_BUTTON_1,
    HD_DEVICE_MODEL_TYPE,
    HD_DEVICE_VENDOR,
    HD_FORCE_OUTPUT,
    HD_INVALID_HANDLE,
    HD_MAX_SCHEDULER_PRIORITY,
    HD_SUCCESS,
    HDCallbackCode,
    HDErrorInfo,
    HDenum,
    HHD,
    HDstring,
)


@dataclass
class DeviceState:
    button: bool = False
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    joints: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gimbals: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


_lib_hd = hd._lib_hd
_scheduler_lock = threading.Lock()
_scheduler_refcount = 0
_HD_CALLBACK = CFUNCTYPE(HDCallbackCode, POINTER(c_void_p))


def _get_last_error_code() -> int:
    _lib_hd.hdGetError.restype = HDErrorInfo
    return int(_lib_hd.hdGetError().errorCode)


def _raise_on_error(context: str):
    error_code = _get_last_error_code()
    if error_code != HD_SUCCESS:
        raise RuntimeError(f"{context} failed with HD error 0x{error_code:04X}")


def _init_device(device_name: str) -> int:
    _lib_hd.hdInitDevice.argtypes = [c_char_p]
    _lib_hd.hdInitDevice.restype = HHD
    device_id = int(_lib_hd.hdInitDevice(device_name.encode("utf-8")))
    error_code = _get_last_error_code()
    if device_id == int(HD_INVALID_HANDLE) or error_code != HD_SUCCESS:
        raise RuntimeError(
            f'Unable to initialize haptic device "{device_name}" (HD error 0x{error_code:04X})'
        )
    return device_id


def _make_current_device(device_id: int):
    _lib_hd.hdMakeCurrentDevice.argtypes = [HHD]
    _lib_hd.hdMakeCurrentDevice.restype = None
    _lib_hd.hdMakeCurrentDevice(device_id)
    _raise_on_error(f"hdMakeCurrentDevice({device_id})")


def _begin_frame(device_id: int):
    _lib_hd.hdBeginFrame.argtypes = [HHD]
    _lib_hd.hdBeginFrame.restype = None
    _lib_hd.hdBeginFrame(device_id)


def _end_frame(device_id: int):
    _lib_hd.hdEndFrame.argtypes = [HHD]
    _lib_hd.hdEndFrame.restype = None
    _lib_hd.hdEndFrame(device_id)


def _enable_force_output():
    _lib_hd.hdEnable.argtypes = [HDenum]
    _lib_hd.hdEnable.restype = None
    _lib_hd.hdEnable(HD_FORCE_OUTPUT)
    _raise_on_error("hdEnable(HD_FORCE_OUTPUT)")


def _safe_get_string(code: int) -> str | None:
    _lib_hd.hdGetString.argtypes = [HDenum]
    _lib_hd.hdGetString.restype = HDstring
    value = _lib_hd.hdGetString(code)
    if not value:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _register_scheduler_callback(callback) -> int:
    global _scheduler_refcount

    _lib_hd.hdScheduleAsynchronous.restype = c_uint
    _lib_hd.hdUnschedule.argtypes = [c_uint]
    _lib_hd.hdUnschedule.restype = None
    _lib_hd.hdStartScheduler.restype = None
    _lib_hd.hdStopScheduler.restype = None

    with _scheduler_lock:
        handle = int(_lib_hd.hdScheduleAsynchronous(callback, None, c_ushort(HD_MAX_SCHEDULER_PRIORITY)))
        _raise_on_error("hdScheduleAsynchronous")
        try:
            if _scheduler_refcount == 0:
                _lib_hd.hdStartScheduler()
                _raise_on_error("hdStartScheduler")
        except Exception:
            _lib_hd.hdUnschedule(handle)
            _get_last_error_code()
            raise
        _scheduler_refcount += 1
        return handle


def _release_scheduler_callback(handle: int | None):
    global _scheduler_refcount

    if handle is None:
        return

    _lib_hd.hdUnschedule.argtypes = [c_uint]
    _lib_hd.hdUnschedule.restype = None
    _lib_hd.hdStopScheduler.restype = None

    with _scheduler_lock:
        if _scheduler_refcount == 1:
            _lib_hd.hdStopScheduler()
            _get_last_error_code()
        _lib_hd.hdUnschedule(handle)
        _get_last_error_code()
        if _scheduler_refcount > 0:
            _scheduler_refcount -= 1


class HapticController:
    def __init__(self, device_name: str = "Default Device", scale: float = 2.5):
        self.device_name = device_name
        self.scale = scale
        self.device_state = DeviceState()
        self.invert_x = False
        self.invert_y = False
        self.invert_z = False
        self.invert_w = True

        self._device_id: int | None = None
        self._scheduler_handle: int | None = None
        self._state_lock = threading.Lock()
        self._callback_error: Exception | None = None
        self._closed = False

        print(f"Initializing haptic device with name {device_name}")
        try:
            self._device_id = _init_device(device_name)
            _make_current_device(self._device_id)
            vendor = _safe_get_string(HD_DEVICE_VENDOR) or "unknown-vendor"
            model = _safe_get_string(HD_DEVICE_MODEL_TYPE) or "unknown-model"
            _enable_force_output()
            self._state_callback = self._create_state_callback()
            self._scheduler_handle = _register_scheduler_callback(self._state_callback)
            time.sleep(0.05)
            print(f"Initialized device! {vendor}/{model}")
        except Exception:
            self.close()
            raise

    @property
    def device_id(self) -> int:
        if self._device_id is None:
            raise RuntimeError(f'Haptic device "{self.device_name}" is not initialized')
        return self._device_id

    def _create_state_callback(self):
        @_HD_CALLBACK
        def state_callback(_user_data):
            try:
                _make_current_device(self.device_id)
                _begin_frame(self.device_id)
                transform = hd.get_transform()
                joints = hd.get_joints()
                gimbals = hd.get_gimbals()
                button_mask = hd.get_buttons()

                rotation_matrix = np.array(
                    [
                        [transform[0][0], transform[0][1], transform[0][2]],
                        [transform[1][0], transform[1][1], transform[1][2]],
                        [transform[2][0], transform[2][1], transform[2][2]],
                    ],
                    dtype=np.float64,
                )
                position = [float(transform[3][0]), float(transform[3][1]), float(transform[3][2])]
                rotation = self._matrix_to_quaternion(rotation_matrix)
                joints_value = [float(joints[0]), float(joints[1]), float(joints[2])]
                gimbals_value = [float(gimbals[0]), float(gimbals[1]), float(gimbals[2])]

                with self._state_lock:
                    self.device_state.position = position
                    self.device_state.rotation = rotation
                    self.device_state.joints = joints_value
                    self.device_state.gimbals = gimbals_value
                    self.device_state.button = bool(button_mask & HD_DEVICE_BUTTON_1) or bool(button_mask)
                    force = list(self.device_state.force)

                hd.set_force(force)
                _end_frame(self.device_id)
                if _get_last_error_code() != HD_SUCCESS:
                    self._callback_error = RuntimeError(
                        f'Haptic callback for "{self.device_name}" reported an HD error'
                    )
                    return HD_CALLBACK_DONE
                return HD_CALLBACK_CONTINUE
            except Exception as exc:
                self._callback_error = exc
                try:
                    if self._device_id is not None:
                        _end_frame(self.device_id)
                except Exception:
                    pass
                return HD_CALLBACK_DONE

        return state_callback

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

    def _raise_if_callback_failed(self):
        if self._callback_error is not None:
            raise RuntimeError(
                f'Haptic device "{self.device_name}" stopped updating: {self._callback_error}'
            )

    def get_scaled_position(self):
        self._raise_if_callback_failed()
        with self._state_lock:
            return [float(pos) * self.scale for pos in self.device_state.position]

    def get_rotation(self):
        self._raise_if_callback_failed()
        with self._state_lock:
            return self.device_state.rotation.copy()

    def set_force(self, force):
        with self._state_lock:
            self.device_state.force = [float(value) for value in force[:3]]

    def is_button_pressed(self):
        self._raise_if_callback_failed()
        with self._state_lock:
            return bool(self.device_state.button)

    def close(self):
        if self._closed:
            return
        self._closed = True

        try:
            _release_scheduler_callback(self._scheduler_handle)
        finally:
            self._scheduler_handle = None

        if self._device_id is not None:
            try:
                time.sleep(0.02)
                hd.close_device(self._device_id)
                _get_last_error_code()
            finally:
                self._device_id = None

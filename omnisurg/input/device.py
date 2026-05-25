import math
import threading
import time
from ctypes import CFUNCTYPE, POINTER, c_char_p, c_uint, c_void_p, c_ushort
from dataclasses import dataclass, field

import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_define import (
    HD_CALLBACK_CONTINUE,
    HD_CALLBACK_DONE,
    HD_DEFAULT_SCHEDULER_PRIORITY,
    HD_DEVICE_BUTTON_1,
    HD_DEVICE_BUTTON_2,
    HD_DEVICE_MODEL_TYPE,
    HD_DEVICE_VENDOR,
    HD_FORCE_OUTPUT,
    HD_FORCE_RAMPING,
    HD_INVALID_HANDLE,
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
    button2: bool = False
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    joints: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gimbals: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


_lib_hd = hd._lib_hd
_scheduler_lock = threading.Lock()
_scheduler_handle: int | None = None
_registered_controllers: dict[int, "HapticController"] = {}
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


def _enable_force_ramping():
    _lib_hd.hdEnable.argtypes = [HDenum]
    _lib_hd.hdEnable.restype = None
    _lib_hd.hdEnable(HD_FORCE_RAMPING)
    _raise_on_error("hdEnable(HD_FORCE_RAMPING)")


def _safe_get_string(code: int) -> str | None:
    _lib_hd.hdGetString.argtypes = [HDenum]
    _lib_hd.hdGetString.restype = HDstring
    value = _lib_hd.hdGetString(code)
    if not value:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _register_scheduler_callback(callback, priority: float = HD_DEFAULT_SCHEDULER_PRIORITY) -> int:
    _lib_hd.hdScheduleAsynchronous.restype = c_uint
    _lib_hd.hdUnschedule.argtypes = [c_uint]
    _lib_hd.hdUnschedule.restype = None
    _lib_hd.hdStartScheduler.restype = None
    _lib_hd.hdStopScheduler.restype = None

    requested_priority = int(max(0, min(float(priority), 65535.0)))
    handle = int(_lib_hd.hdScheduleAsynchronous(callback, None, c_ushort(requested_priority)))
    _raise_on_error("hdScheduleAsynchronous")
    try:
        _lib_hd.hdStartScheduler()
        _raise_on_error("hdStartScheduler")
    except Exception:
        _lib_hd.hdUnschedule(handle)
        _get_last_error_code()
        raise
    return handle


def _release_scheduler_callback(handle: int | None):
    if handle is None:
        return

    _lib_hd.hdUnschedule.argtypes = [c_uint]
    _lib_hd.hdUnschedule.restype = None
    _lib_hd.hdStopScheduler.restype = None
    _lib_hd.hdStopScheduler()
    _get_last_error_code()
    _lib_hd.hdUnschedule(handle)
    _get_last_error_code()


def _mark_scheduler_failed(message: str):
    for controller in _registered_controllers.values():
        if controller._callback_error is None:
            controller._callback_error = RuntimeError(message)


@_HD_CALLBACK
def _global_state_callback(_user_data):
    with _scheduler_lock:
        controllers = tuple(_registered_controllers.values())
        if not controllers:
            return HD_CALLBACK_CONTINUE

        for controller in controllers:
            if controller._closed or controller._device_id is None:
                continue

            try:
                _make_current_device(controller.device_id)
                _begin_frame(controller.device_id)
                transform = hd.get_transform()
                button_mask = hd.get_buttons()
                position = [float(transform[3][0]), float(transform[3][1]), float(transform[3][2])]
                rotation = controller._matrix_to_quaternion(
                    float(transform[0][0]),
                    float(transform[0][1]),
                    float(transform[0][2]),
                    float(transform[1][0]),
                    float(transform[1][1]),
                    float(transform[1][2]),
                    float(transform[2][0]),
                    float(transform[2][1]),
                    float(transform[2][2]),
                )

                with controller._state_lock:
                    controller.device_state.position = position
                    controller.device_state.rotation = rotation
                    controller.device_state.button = bool(button_mask & HD_DEVICE_BUTTON_1)
                    controller.device_state.button2 = bool(button_mask & HD_DEVICE_BUTTON_2)
                    force = list(controller.device_state.force)

                if controller.force_feedback:
                    hd.set_force(force)
                _end_frame(controller.device_id)
                if _get_last_error_code() != HD_SUCCESS:
                    _mark_scheduler_failed(
                        f'Global haptic scheduler reported an HD error while updating "{controller.device_name}"'
                    )
                    return HD_CALLBACK_DONE
            except Exception as exc:
                try:
                    if controller._device_id is not None:
                        _end_frame(controller.device_id)
                except Exception:
                    pass
                _mark_scheduler_failed(f"Global haptic scheduler stopped: {exc}")
                return HD_CALLBACK_DONE

    return HD_CALLBACK_CONTINUE


def _register_controller(controller: "HapticController"):
    global _scheduler_handle

    with _scheduler_lock:
        _registered_controllers[controller.device_id] = controller
        if _scheduler_handle is None:
            try:
                _scheduler_handle = _register_scheduler_callback(
                    _global_state_callback,
                    priority=controller.scheduler_priority,
                )
            except Exception:
                _registered_controllers.pop(controller.device_id, None)
                raise


def _unregister_controller(controller: "HapticController"):
    global _scheduler_handle

    handle_to_release = None
    with _scheduler_lock:
        if controller._device_id is not None:
            _registered_controllers.pop(controller._device_id, None)
        if not _registered_controllers and _scheduler_handle is not None:
            handle_to_release = _scheduler_handle
            _scheduler_handle = None

    if handle_to_release is not None:
        _release_scheduler_callback(handle_to_release)


class HapticController:
    def __init__(
        self,
        device_name: str = "Default Device",
        scale: float = 2.5,
        scheduler_priority: float = HD_DEFAULT_SCHEDULER_PRIORITY,
        force_feedback: bool = True,
    ):
        self.device_name = device_name
        self.scale = scale
        self.scheduler_priority = float(scheduler_priority)
        self.force_feedback = bool(force_feedback)
        self.device_state = DeviceState()
        self.invert_x = False
        self.invert_y = False
        self.invert_z = False
        self.invert_w = True

        self._device_id: int | None = None
        self._state_lock = threading.Lock()
        self._callback_error: Exception | None = None
        self._closed = False

        print(f"Initializing haptic device with name {device_name}")
        try:
            self._device_id = _init_device(device_name)
            _make_current_device(self._device_id)
            vendor = _safe_get_string(HD_DEVICE_VENDOR) or "unknown-vendor"
            model = _safe_get_string(HD_DEVICE_MODEL_TYPE) or "unknown-model"
            if self.force_feedback:
                _enable_force_output()
                _enable_force_ramping()
            _register_controller(self)
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

    def _matrix_to_quaternion(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        trace = m00 + m11 + m22

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
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

    def poll_state(self) -> dict[str, list | bool]:
        self._raise_if_callback_failed()
        with self._state_lock:
            return {
                "position": [float(pos) * self.scale for pos in self.device_state.position],
                "rotation": self.device_state.rotation.copy(),
                "button": bool(self.device_state.button),
                "button2": bool(self.device_state.button2),
                "valid": True,
            }

    def set_force(self, force):
        if not self.force_feedback:
            return
        with self._state_lock:
            self.device_state.force = [float(value) for value in force[:3]]

    def is_button_pressed(self):
        self._raise_if_callback_failed()
        with self._state_lock:
            return bool(self.device_state.button)

    def is_button2_pressed(self):
        self._raise_if_callback_failed()
        with self._state_lock:
            return bool(self.device_state.button2)

    def close(self):
        if self._closed:
            return
        self._closed = True

        _unregister_controller(self)
        if self._device_id is not None:
            try:
                time.sleep(0.02)
                hd.close_device(self._device_id)
                _get_last_error_code()
            finally:
                self._device_id = None

import functools
from ctypes import CDLL, CFUNCTYPE, POINTER, byref, c_void_p
from sys import platform

from .hd import begin_frame, end_frame, get_current_device, get_error, make_current_device
from .hd_define import (
    HD_CALLBACK_CONTINUE,
    HD_CALLBACK_DONE,
    HD_MAX_SCHEDULER_PRIORITY,
    HDCallbackCode,
    HDSchedulerHandle,
)

if platform == "linux" or platform == "linux2":
    _lib_hd = CDLL("libHD.so")
elif platform == "win32":
    _lib_hd = CDLL("HD.dll")

def hd_callback(input_function=None, *, device_id: int | None = None):
    def _decorate(func):
        @functools.wraps(func)
        @CFUNCTYPE(HDCallbackCode, POINTER(c_void_p))
        def _callback(pUserData):
            """Callback function for one haptic device.

            Dual-device setups must bind the callback to a specific HHD before
            querying state. This mirrors the OpenHaptics dual-device samples,
            which call ``hdMakeCurrentDevice`` for the device whose state is
            being read.
            """
            current_device = get_current_device() if device_id is None else device_id
            make_current_device(current_device)
            begin_frame(current_device)
            try:
                func()
            finally:
                end_frame(current_device)
            if get_error():
                return HD_CALLBACK_DONE
            return HD_CALLBACK_CONTINUE

        return _callback

    if input_function is None:
        return _decorate
    return _decorate(input_function)


def hdAsyncSheduler(callback):
    pUserData = c_void_p()
    _lib_hd.hdScheduleAsynchronous.restype = HDSchedulerHandle
    return _lib_hd.hdScheduleAsynchronous(callback, byref(pUserData), HD_MAX_SCHEDULER_PRIORITY)


def hdSyncSheduler(callback):
    pUserData = c_void_p()
    _lib_hd.hdScheduleSynchronous.restype = HDSchedulerHandle
    return _lib_hd.hdScheduleSynchronous(callback, byref(pUserData), HD_MAX_SCHEDULER_PRIORITY)

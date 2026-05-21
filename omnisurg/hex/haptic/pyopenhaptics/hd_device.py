from __future__ import annotations

from collections.abc import Callable

from .hd import (
    close_device,
    enable_force,
    enable_force_ramping,
    get_error,
    get_model,
    get_vendor,
    init_device,
    make_current_device,
    unschedule,
)
from .hd import (
    start_scheduler as hd_start_scheduler,
)
from .hd import (
    stop_scheduler as hd_stop_scheduler,
)
from .hd_callback import hdAsyncSheduler, hdSyncSheduler
from .hd_define import HD_BAD_HANDLE, HD_INVALID_HANDLE


class HapticDevice:
    _active_device_count = 0
    _scheduler_started = False

    def __init__(
        self,
        callback: Callable | None = None,
        device_name: str = "Default Device",
        scheduler_type: str = "async",
        *,
        auto_start_scheduler: bool = True,
        enable_force_output: bool = False,
    ):

        print(f"Initializing haptic device with name {device_name}")
        self.scheduler_handle = HD_INVALID_HANDLE
        self._closed = False

        self.id = init_device(device_name)
        if self.id in (None, HD_BAD_HANDLE):
            self.id = HD_BAD_HANDLE
            print("Unable to initialize the device. Check the connection!")
            return

        make_current_device(self.id)

        print(f"Intialized device! {self.__vendor__()}/{self.__model__()}")
        if enable_force_output:
            enable_force()
            enable_force_ramping()

        HapticDevice._active_device_count += 1
        if callback is not None:
            self.scheduler(callback, scheduler_type)
        if auto_start_scheduler:
            self.start_scheduler()

    @classmethod
    def start_scheduler(cls):
        if cls._scheduler_started:
            return
        hd_start_scheduler()
        if get_error():
            raise SystemError("failed to start OpenHaptics scheduler")
        cls._scheduler_started = True

    @classmethod
    def stop_scheduler_if_idle(cls):
        if cls._active_device_count > 0 or not cls._scheduler_started:
            return
        hd_stop_scheduler()
        cls._scheduler_started = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.scheduler_handle != HD_INVALID_HANDLE:
            unschedule(self.scheduler_handle)
            self.scheduler_handle = HD_INVALID_HANDLE
        if self.id != HD_BAD_HANDLE:
            close_device(self.id)
            HapticDevice._active_device_count = max(0, HapticDevice._active_device_count - 1)
        self.stop_scheduler_if_idle()

    def scheduler(self, callback, scheduler_type):
        if scheduler_type == "async":
            self.scheduler_handle = hdAsyncSheduler(callback)
        else:
            self.scheduler_handle = hdSyncSheduler(callback)


    @staticmethod
    def __vendor__() -> str:
        return get_vendor()

    @staticmethod
    def __model__() -> str:
        return get_model()

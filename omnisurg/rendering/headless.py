class HeadlessRenderer:
    """No-op renderer used for automated smoke tests and deterministic stepping."""

    def __init__(self):
        self._running = True
        self.frame_count = 0

    def begin_frame(self, time: float):
        self.frame_count += 1

    def end_frame(self):
        pass

    def is_running(self) -> bool:
        return self._running

    def close(self):
        self._running = False

    def log_mesh(self, *args, **kwargs):
        pass

    def log_instances(self, *args, **kwargs):
        pass

    def log_points(self, *args, **kwargs):
        pass

    def set_input_callbacks(self, on_key_press=None, on_key_release=None):
        pass

class HeadlessRenderer:
    """No-op renderer used for automated smoke tests and deterministic stepping."""

    def __init__(self):
        self._running = True
        self._paused = False
        self.frame_count = 0
        self.show_particles = False
        self.show_ui = False

    def begin_frame(self, time: float):
        self.frame_count += 1

    def end_frame(self):
        pass

    def is_running(self) -> bool:
        return self._running

    def close(self):
        self._running = False

    def set_model(self, *_args, **_kwargs):
        pass

    def set_camera(self, *_args, **_kwargs):
        pass

    def log_state(self, *_args, **_kwargs):
        pass

    def log_mesh(self, *args, **kwargs):
        pass

    def log_instances(self, *args, **kwargs):
        pass

    def log_points(self, *args, **kwargs):
        pass

    def log_lines(self, *args, **kwargs):
        pass

    def draw_cryo_surface(self, *args, **kwargs):
        return None

    def screen_to_world_ray(self, *_args, **_kwargs):
        raise RuntimeError("headless renderer does not expose screen_to_world_ray")

    def is_paused(self) -> bool:
        return bool(self._paused)

    def is_ui_capturing(self) -> bool:
        return False

    def set_input_callbacks(self, on_key_press=None, on_key_release=None):
        pass

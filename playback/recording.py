import os

import numpy as np


class RecordingManager:
    def __init__(self, viewer, output_path: str, frames_dir: str, fps: int):
        self.viewer = viewer
        self.output_path = output_path
        self.frames_dir = frames_dir
        self.fps = fps

        self.record_ui = False
        self._imageio = None
        self._video_writer = None
        self._recording = False
        self._frame_idx = 0

        self._init_backend()

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _init_backend(self):
        try:
            import imageio.v2 as iio  # noqa: PLC0415
        except Exception:
            try:
                import imageio as iio  # noqa: PLC0415
            except Exception:
                iio = None
        self._imageio = iio

    def start(self):
        if self._recording:
            return
        if not hasattr(self.viewer, "get_frame"):
            print("Viewer does not support get_frame(); cannot record.")
            return
        self._frame_idx = 0
        out_dir = os.path.dirname(self.output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        self._video_writer = None
        if self._imageio is not None:
            try:
                self._video_writer = self._imageio.get_writer(self.output_path, fps=self.fps)
            except Exception as exc:
                print(f"Failed to open video writer ({exc}); writing frames instead.")
                self._video_writer = None
        self._recording = True
        print(f"Recording started -> {self.output_path}")

    def stop(self):
        if not self._recording and self._video_writer is None:
            return
        if self._video_writer is not None:
            try:
                self._video_writer.close()
                print(f"Video saved -> {self.output_path}")
            except Exception as exc:
                print(f"Error closing video writer: {exc}")
        self._video_writer = None
        self._recording = False

    def capture(self):
        if not self._recording or not hasattr(self.viewer, "get_frame"):
            return
        frame_wp = self.viewer.get_frame(render_ui=self.record_ui)
        frame_np = frame_wp.numpy()
        if self._video_writer is not None:
            self._video_writer.append_data(frame_np)
        elif self._imageio is not None:
            frame_path = os.path.join(self.frames_dir, f"frame_{self._frame_idx:06d}.png")
            self._imageio.imwrite(frame_path, frame_np)
        else:
            frame_path = os.path.join(self.frames_dir, f"frame_{self._frame_idx:06d}.npy")
            np.save(frame_path, frame_np)
        self._frame_idx += 1

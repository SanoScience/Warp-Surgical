from __future__ import annotations

import collections
import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omnisurg.runtime import Runtime


_CSV_COLUMNS = (
    "frame",
    "t",
    "dt",
    "spring_k",
    "damper_b",
    "max_force",
    "lowpass_alpha",
    "slew_rate_limit",
    "contact_count",
    "proxy_offset_x",
    "proxy_offset_y",
    "proxy_offset_z",
    "proxy_offset_mag",
    "raw_force_x",
    "raw_force_y",
    "raw_force_z",
    "raw_force_mag",
    "filtered_force_x",
    "filtered_force_y",
    "filtered_force_z",
    "filtered_force_mag",
    "cmd_force_x",
    "cmd_force_y",
    "cmd_force_z",
    "cmd_force_mag",
    "clamp_active",
    "slew_active",
)


class ForceTelemetry:
    """Live force-feedback telemetry for analyzing oscillations/instability.

    Streams rolling scalar plots to the Newton GL viewer's built-in "Plots"
    window (one line per signal, auto-scrolling history) and optionally logs
    per-frame samples to a CSV file for offline FFT / stability analysis.
    """

    def __init__(
        self,
        runtime: "Runtime",
        *,
        viewer_plots: bool = True,
        csv_path: str | Path | None = None,
        history_size: int = 250,
    ):
        self._runtime = runtime
        self._viewer_plots = viewer_plots and self._viewer_supports_plots()
        self._csv_path: Path | None = Path(csv_path) if csv_path else None
        self._csv_file = None
        self._csv_writer = None
        if self._csv_path is not None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = self._csv_path.open("w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(_CSV_COLUMNS)
        self._frame = 0
        self._history_size = int(history_size)
        self._force_history: dict[str, collections.deque] = {
            "raw": collections.deque(maxlen=self._history_size),
            "filtered": collections.deque(maxlen=self._history_size),
            "cmd": collections.deque(maxlen=self._history_size),
        }
        self._implot_available = False
        if self._viewer_plots:
            self._register_combined_force_plot()
        if self.enabled:
            setter = getattr(runtime, "set_force_feedback_compute_always", None)
            if callable(setter):
                setter(True)

    def _register_combined_force_plot(self) -> None:
        renderer = getattr(self._runtime.renderer, "_renderer", None)
        if renderer is None or not hasattr(renderer, "register_ui_callback"):
            return
        try:
            from imgui_bundle import implot
        except ImportError:
            print("[telemetry] imgui_bundle.implot not available; combined force chart disabled")
            return
        if implot.get_current_context() is None:
            implot.create_context()
        self._implot_available = True
        renderer.register_ui_callback(self._render_combined_force_plot, position="free")

    def _render_combined_force_plot(self, imgui) -> None:
        if not self._implot_available:
            return
        from imgui_bundle import implot

        imgui.set_next_window_size(imgui.ImVec2(520, 260), imgui.Cond_.appearing)
        if imgui.begin("Haptic Forces"):
            plot_size = imgui.ImVec2(-1, -1)
            if implot.begin_plot("##forces", plot_size):
                implot.setup_axes("frame", "|F| (N)")
                n = self._history_size
                for label, buf in self._force_history.items():
                    if not buf:
                        continue
                    arr = np.full(n, np.nan, dtype=np.float32)
                    arr[n - len(buf):] = np.asarray(buf, dtype=np.float32)
                    implot.plot_line(label, arr)
                implot.end_plot()
        imgui.end()

    @property
    def enabled(self) -> bool:
        return self._viewer_plots or self._csv_writer is not None

    def _viewer_supports_plots(self) -> bool:
        renderer = getattr(self._runtime, "renderer", None)
        if renderer is None:
            return False
        inner = getattr(renderer, "_renderer", None)
        return inner is not None and hasattr(inner, "log_scalar")

    def record(self) -> None:
        if not self.enabled:
            return

        rt = self._runtime
        diagnostics = rt.haptic_feedback_diagnostics
        settings = rt.haptic_feedback_settings

        proxy_offset = np.asarray(diagnostics.proxy_offset, dtype=np.float32)
        raw_force = np.asarray(diagnostics.raw_force, dtype=np.float32)
        filtered_force = np.asarray(diagnostics.filtered_force, dtype=np.float32)
        cmd_force = np.asarray(diagnostics.final_force, dtype=np.float32)

        proxy_offset_mag = float(np.linalg.norm(proxy_offset))
        raw_force_mag = float(np.linalg.norm(raw_force))
        filtered_force_mag = float(np.linalg.norm(filtered_force))
        cmd_force_mag = float(np.linalg.norm(cmd_force))

        if self._viewer_plots:
            self._force_history["raw"].append(raw_force_mag)
            self._force_history["filtered"].append(filtered_force_mag)
            self._force_history["cmd"].append(cmd_force_mag)
            log = rt.renderer.log_scalar
            log("proxy/offset_mag", proxy_offset_mag)
            log("contacts", float(diagnostics.contact_count))

        if self._csv_writer is not None:
            dt = float(rt.sim_config.frame_dt)
            t = float(getattr(rt, "sim_time", self._frame * dt))
            self._csv_writer.writerow(
                [
                    self._frame,
                    t,
                    dt,
                    float(settings.spring_k),
                    float(settings.damper_b),
                    float(settings.max_force),
                    float(settings.lowpass_alpha),
                    float(settings.slew_rate_limit),
                    int(diagnostics.contact_count),
                    float(proxy_offset[0]),
                    float(proxy_offset[1]),
                    float(proxy_offset[2]),
                    proxy_offset_mag,
                    float(raw_force[0]),
                    float(raw_force[1]),
                    float(raw_force[2]),
                    raw_force_mag,
                    float(filtered_force[0]),
                    float(filtered_force[1]),
                    float(filtered_force[2]),
                    filtered_force_mag,
                    float(cmd_force[0]),
                    float(cmd_force[1]),
                    float(cmd_force[2]),
                    cmd_force_mag,
                    int(bool(diagnostics.clamp_active)),
                    int(bool(diagnostics.slew_active)),
                ]
            )

        self._frame += 1

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            print(f"[telemetry] wrote {self._frame} frames to {self._csv_path}")

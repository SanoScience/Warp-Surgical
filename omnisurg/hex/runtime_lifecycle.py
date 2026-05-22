# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle bookkeeping for internal hex runtime drivers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HexFrameLoopConfig:
    """Resolved frame-loop settings shared by hex runtime drivers."""

    frame_dt: float
    max_frames: int | None
    exit_after_init: bool


def build_hex_frame_loop_config(args: Any) -> HexFrameLoopConfig:
    """Resolve frame timing and stop-count semantics from parsed runtime args."""

    fps = max(1, int(getattr(args, "fps", 60)))
    requested_frames = getattr(args, "frames", None)
    if requested_frames is None:
        max_frames = 1 if str(getattr(args, "viewer", "")) == "headless" else None
    else:
        max_frames = max(0, int(requested_frames))
    return HexFrameLoopConfig(
        frame_dt=1.0 / float(fps),
        max_frames=max_frames,
        exit_after_init=bool(getattr(args, "exit_after_init", False)),
    )


@dataclass
class HexFrameLoopState:
    """Frame counters and stop state for a hex runtime frame loop."""

    config: HexFrameLoopConfig
    frame: int = 0
    completed_frames: int = 0
    _start_time: float = field(default_factory=time.perf_counter)
    _stopped: bool = False

    @property
    def frame_dt(self) -> float:
        return float(self.config.frame_dt)

    @property
    def frame_time(self) -> float:
        return float(self.frame) * self.frame_dt

    @property
    def elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._start_time)

    @property
    def average_fps(self) -> float:
        elapsed = self.elapsed
        return float(self.completed_frames) / elapsed if elapsed > 0.0 else 0.0

    def reset(self) -> None:
        self.frame = 0
        self.completed_frames = 0
        self._start_time = time.perf_counter()
        self._stopped = False

    def should_run_frame(self) -> bool:
        if self.config.exit_after_init or self._stopped:
            return False
        if self.config.max_frames is None:
            return True
        return self.frame < int(self.config.max_frames)

    def complete_frame(self, *, render_running: bool = True) -> bool:
        self.completed_frames += 1
        self.frame += 1
        if not render_running:
            self._stopped = True
        return self.should_run_frame()

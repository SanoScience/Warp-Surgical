from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnisurg.hex.runtime_lifecycle import HexFrameLoopState, build_hex_frame_loop_config


def _args(**overrides):
    values = {
        "viewer": "gl",
        "frames": None,
        "fps": 60,
        "exit_after_init": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_headless_without_explicit_frames_resolves_to_one_frame():
    config = build_hex_frame_loop_config(_args(viewer="headless", frames=None))

    assert config.max_frames == 1


def test_gl_without_explicit_frames_resolves_to_unlimited():
    config = build_hex_frame_loop_config(_args(viewer="gl", frames=None))

    assert config.max_frames is None


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (0, 0),
        (1, 1),
        (-3, 0),
    ],
)
def test_explicit_frames_are_clamped_non_negative(frames, expected):
    config = build_hex_frame_loop_config(_args(frames=frames))

    assert config.max_frames == expected


@pytest.mark.parametrize("fps", [0, -30])
def test_non_positive_fps_uses_finite_default_dt(fps):
    config = build_hex_frame_loop_config(_args(fps=fps))

    assert config.frame_dt == pytest.approx(1.0)


def test_render_stopped_viewer_stops_unlimited_loop():
    loop = HexFrameLoopState(build_hex_frame_loop_config(_args(frames=None)))

    assert loop.should_run_frame()
    assert not loop.complete_frame(render_running=False)
    assert loop.frame == 1
    assert loop.completed_frames == 1
    assert not loop.should_run_frame()


def test_reset_clears_frame_and_completed_frame_counts():
    loop = HexFrameLoopState(build_hex_frame_loop_config(_args(frames=5)))
    loop.complete_frame()
    loop.complete_frame()

    loop.reset()

    assert loop.frame == 0
    assert loop.completed_frames == 0
    assert loop.should_run_frame()


def test_exit_after_init_never_runs_a_frame():
    loop = HexFrameLoopState(build_hex_frame_loop_config(_args(exit_after_init=True)))

    assert not loop.should_run_frame()

from __future__ import annotations

import argparse

import numpy as np
import pytest

from omnisurg.hex import app_runtime
from omnisurg.hex import runtime as hex_runtime
from omnisurg.hex.app import OmniSurgHexApp
from omnisurg.hex.data.types import PreparedVolume
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable
from omnisurg.hex.runtime_config import normalize_hex_runtime_args


APP_RUNTIME_ONLY_DESTS = {
    "crop_visible_classes",
    "crop_visible_margin_voxels",
    "digimouse_cache",
    "digimouse_rebuild_cache",
    "digimouse_use_cache",
    "downsample",
    "dynamic_bones",
}


def _make_volume() -> PreparedVolume:
    return PreparedVolume(
        labels=np.ones((2, 2, 2), dtype=np.uint8),
        voxel_size_m=0.005,
        materials=MaterialTable(DEFAULT_MATERIALS),
        class_map={0: "background", 1: "synthetic_block"},
    )


def _runtime_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {action.dest for action in parser._actions if action.option_strings and action.dest != "help"}


def _parse_and_normalize(argv: list[str]) -> argparse.Namespace:
    parser = hex_runtime._build_lifecycle_parser()
    args = parser.parse_args(argv)
    normalize_hex_runtime_args(args, parser=parser)
    return args


def test_app_and_session_parsers_share_common_runtime_dests():
    app_dests = _runtime_dests(app_runtime._build_app_runtime_parser())
    session_dests = _runtime_dests(hex_runtime._build_lifecycle_parser())

    assert APP_RUNTIME_ONLY_DESTS <= app_dests
    assert not (APP_RUNTIME_ONLY_DESTS & session_dests)
    assert len(session_dests) == 103
    assert app_dests - APP_RUNTIME_ONLY_DESTS == session_dests


def test_default_instrument_tool_modes_normalize_by_input_backend():
    assert _parse_and_normalize([]).instrument_tool_modes == ("diathermy", "diathermy")
    assert _parse_and_normalize(["--input-backend", "fallback"]).instrument_tool_modes == (
        "diathermy",
        "diathermy",
    )
    assert _parse_and_normalize(["--input-backend", "minimou"]).instrument_tool_modes == ("grasper", "grasper")


def test_cutting_instrument_tool_mode_aliases_to_diathermy():
    args = _parse_and_normalize(["--instrument-tool-modes", "cutting", "scissors"])

    assert args.instrument_tool_modes == ("diathermy", "scissors")


def test_slang_procedural_material_promotes_gl_viewer_to_slang():
    parser = hex_runtime._build_lifecycle_parser()
    args = parser.parse_args(["--viewer", "gl", "--slang-procedural-material", "material.slang"])

    slang_viewer_requested = normalize_hex_runtime_args(args, parser=parser)

    assert slang_viewer_requested is True
    assert args.viewer == "slang"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--usd", "out.usd"], "--frames is required when using --usd"),
        (["--usd", "out.usd", "--frames", "1", "--viewer", "slang"], "--viewer slang cannot be combined with --usd"),
        (["--instrument-radius-scale", "-1"], "--instrument-radius-scale must be >= 0"),
        (["--instrument-contact-iterations", "0"], "--instrument-contact-iterations must be >= 1"),
        (["--instrument-max-correction-scale", "-1"], "--instrument-max-correction-scale must be >= 0"),
    ],
)
def test_invalid_session_launcher_runtime_config_exits_with_argparse_error(capsys, argv, message):
    with pytest.raises(SystemExit) as exc_info:
        hex_runtime.HexAppLauncher(_make_volume(), argv).run()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_invalid_app_runtime_only_config_exits_with_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        OmniSurgHexApp(_make_volume()).run(("--crop-visible-margin-voxels", "-1"))

    assert exc_info.value.code == 2
    assert "--crop-visible-margin-voxels must be >= 0" in capsys.readouterr().err

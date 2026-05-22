# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from omnisurg.hex import app_runtime
from omnisurg.hex import runtime as hex_runtime
from omnisurg.hex import setup as hex_setup
from omnisurg.hex.data.types import PreparedVolume
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable
from omnisurg.hex.shape_matching_solver import (
    HIERARCHICAL_SHAPE_MATCHING_FULL27,
    HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    SHAPE_MATCHING_GS_WEIGHT_FULL,
    SHAPE_MATCHING_GS_WEIGHT_SQRT,
    SHAPE_MATCHING_SOLVE_COLORED_GS,
    SHAPE_MATCHING_SOLVE_GATHER,
    SHAPE_MATCHING_SOLVE_SCATTER,
)


def _args(**overrides):
    values = {
        "atlas": "Digimouse/atlas/atlas",
        "downsample": 16,
        "atlas_pad": 1,
        "digimouse_cache": None,
        "digimouse_use_cache": True,
        "digimouse_rebuild_cache": False,
        "size": 0,
        "voxel": 0.005,
        "drop_height": 0.04,
        "global_scale": 1.0,
        "crop_visible_classes": None,
        "crop_visible_margin_voxels": 4,
        "shape_matching_mode": None,
        "shape_matching_gather": False,
        "shape_matching_gs_weighting": "averaged",
        "shape_matching_gs_support_alpha": -1.0,
        "hierarchical_shape_matching": "outer8",
        "l2_hierarchical_shape_matching": "outer8",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _volume(**overrides) -> PreparedVolume:
    values = {
        "labels": np.ones((2, 2, 2), dtype=np.uint8),
        "voxel_size_m": 0.005,
        "materials": MaterialTable(DEFAULT_MATERIALS),
        "class_map": {0: "background", 1: "muscle"},
    }
    values.update(overrides)
    return PreparedVolume(**values)


def test_prepared_atlas_setup_preserves_volume_metadata_texture_origin_and_skips_visible_crop(tmp_path):
    settings = tmp_path / "panel.json"
    settings.write_text(json.dumps({"materials": [{"id": 1, "name": "muscle", "visible": True}]}), encoding="utf-8")
    texture = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    volume = _volume(
        texture_rgb=texture,
        origin=(1.0, 2.0, 3.0),
        voxel_size_m=0.25,
        metadata={"scene_label": "prepared case"},
    )

    setup = hex_setup.resolve_hex_atlas_setup(
        _args(drop_height=0.5, global_scale=2.0, crop_visible_classes=settings),
        prepared_volume=volume,
    )

    assert setup.atlas.voxel_size == pytest.approx(0.5)
    assert setup.atlas.origin == (1.0, 2.0, 3.0)
    assert setup.origin == pytest.approx((2.0, 4.0, 7.0))
    assert setup.scene_label == "prepared case  (global_scale=2)"
    assert setup.visible_class_crop is None
    assert setup.prepared_texture_rgb is volume.texture_rgb
    assert "visible_class_crop" not in setup.atlas.metadata


def test_synthetic_setup_visible_crop_updates_atlas_origin_metadata_and_scene_label(tmp_path):
    settings = tmp_path / "panel.json"
    settings.write_text(json.dumps({"materials": [{"id": 1, "name": "muscle", "visible": True}]}), encoding="utf-8")

    setup = hex_setup.resolve_hex_atlas_setup(
        _args(size=3, voxel=1.0, drop_height=10.0, crop_visible_classes=settings, crop_visible_margin_voxels=0)
    )

    assert setup.atlas.labels.shape == (3, 3, 3)
    assert np.all(setup.atlas.labels == 1)
    assert setup.atlas.origin == pytest.approx((1.0, 1.0, 1.0))
    assert setup.origin == pytest.approx((-1.5, -1.5, 10.0))
    assert setup.visible_class_crop is not None
    assert setup.atlas.metadata["visible_class_crop"]["crop_min"] == [1, 1, 1]
    assert setup.scene_label == "block 3^3  (visible crop)"


def test_solver_mode_resolution_matches_parser_choices_and_constants():
    modes = hex_setup.resolve_hex_solver_modes(_args())
    assert modes.shape_matching_mode == SHAPE_MATCHING_SOLVE_SCATTER
    assert modes.hierarchical_shape_matching_mode == HIERARCHICAL_SHAPE_MATCHING_OUTER8
    assert modes.l2_hierarchical_shape_matching_mode == HIERARCHICAL_SHAPE_MATCHING_OUTER8

    modes = hex_setup.resolve_hex_solver_modes(_args(shape_matching_gather=True))
    assert modes.shape_matching_mode == SHAPE_MATCHING_SOLVE_GATHER

    modes = hex_setup.resolve_hex_solver_modes(
        _args(
            shape_matching_mode="gs",
            shape_matching_gs_weighting="sqrt",
            shape_matching_gs_support_alpha=0.25,
            hierarchical_shape_matching="full27",
            l2_hierarchical_shape_matching="full125",
        )
    )
    assert modes.shape_matching_mode == SHAPE_MATCHING_SOLVE_COLORED_GS
    assert modes.shape_matching_gs_weighting == SHAPE_MATCHING_GS_WEIGHT_SQRT
    assert modes.shape_matching_gs_support_alpha == pytest.approx(0.25)
    assert modes.hierarchical_shape_matching_mode == HIERARCHICAL_SHAPE_MATCHING_FULL27
    assert modes.l2_hierarchical_shape_matching_mode == HIERARCHICAL_SHAPE_MATCHING_FULL27

    parser = hex_runtime._build_lifecycle_parser()
    parsed = parser.parse_args(
        [
            "--shape-matching-mode",
            "gs",
            "--shape-matching-gs-weighting",
            "full",
            "--hierarchical-shape-matching",
            "full27",
            "--l2-hierarchical-shape-matching",
            "full125",
        ]
    )
    modes = hex_setup.resolve_hex_solver_modes(parsed)
    assert modes.shape_matching_mode == SHAPE_MATCHING_SOLVE_COLORED_GS
    assert modes.shape_matching_gs_weighting == SHAPE_MATCHING_GS_WEIGHT_FULL
    assert modes.hierarchical_shape_matching_mode == HIERARCHICAL_SHAPE_MATCHING_FULL27
    assert modes.l2_hierarchical_shape_matching_mode == HIERARCHICAL_SHAPE_MATCHING_FULL27


def test_app_runtime_and_session_call_shared_core_setup(monkeypatch):
    volume = _volume()
    calls = []

    class StopAfterSharedSetup(RuntimeError):
        pass

    def fake_build_hex_core_setup(args, *, prepared_volume=None, startup_phases=None, **kwargs):
        del args, kwargs
        calls.append((prepared_volume, startup_phases))
        raise StopAfterSharedSetup

    monkeypatch.setattr(app_runtime, "build_hex_core_setup", fake_build_hex_core_setup)
    with pytest.raises(StopAfterSharedSetup):
        app_runtime.run_prepared_volume(
            volume,
            (
                "--viewer",
                "headless",
                "--input-backend",
                "off",
                "--exit-after-init",
            ),
        )
    assert calls[-1][0] is volume
    assert isinstance(calls[-1][1], list)

    monkeypatch.setattr(hex_runtime, "build_hex_core_setup", fake_build_hex_core_setup)
    parser = hex_runtime._build_lifecycle_parser()
    session = hex_runtime.HexRuntimeSession(
        volume,
        parser.parse_args(
            [
                "--viewer",
                "headless",
                "--input-backend",
                "off",
                "--exit-after-init",
            ]
        ),
    )
    with pytest.raises(StopAfterSharedSetup):
        session.init()
    assert calls[-1][0] is volume
    assert isinstance(calls[-1][1], list)

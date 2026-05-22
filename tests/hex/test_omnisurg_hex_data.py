# SPDX-License-Identifier: Apache-2.0
"""Tests for the OmniSurg Hex data contract, cache, and loaders."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable, Phase
from omnisurg.hex import app_runtime
from omnisurg.hex import runtime as hex_runtime
from omnisurg.hex.cli import _default_root
from omnisurg.hex.data.cache import build_manifest, make_cache_key, try_read_valid_cache, write_npz_atomic
from omnisurg.hex.data.crop import crop_labels_to_visible_classes, select_visible_class_ids
from omnisurg.hex.data.loaders.abdomen_atlas import AbdomenAtlasLoader
from omnisurg.hex.data.loaders.preprocessed_npz import PreprocessedNpzLoader
from omnisurg.hex.data.materials import generate_material_table, remap_labels_to_internal
from omnisurg.hex.data.segmentation import load_builtin_class_map, load_builtin_material_defaults
from omnisurg.hex.data.types import PreparedVolume, PreprocessConfig


class _StopAfterBuild(RuntimeError):
    pass


def test_cli_default_dataset_roots_are_project_relative():
    digimouse_root = _default_root("digimouse")
    vhp_root = _default_root("vhp")
    abdomen_root = _default_root("abdomen-atlas")
    kits23_root = _default_root("kits23")

    assert digimouse_root is not None and digimouse_root.is_absolute()
    assert vhp_root is not None and vhp_root.is_absolute()
    assert abdomen_root is not None and abdomen_root.is_absolute()
    assert kits23_root is not None and kits23_root.is_absolute()
    assert digimouse_root.parts[-3:] == ("Digimouse", "atlas", "atlas")
    assert vhp_root.name == "VHP"
    assert abdomen_root.parts[-2:] == ("AbdomenAtlas", "combined_labels.nii.gz")
    assert kits23_root.name == "Kits2023"


def test_cli_default_dataset_roots_honor_data_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OMNISURG_DATA_ROOT", str(tmp_path))

    assert _default_root("digimouse") == tmp_path / "Digimouse" / "atlas" / "atlas"
    assert _default_root("vhp") == tmp_path / "VHP"
    assert _default_root("abdomen-atlas") == tmp_path / "AbdomenAtlas" / "combined_labels.nii.gz"
    assert _default_root("kits23") == tmp_path / "Kits2023"
    assert _default_root("synthetic") is None


def test_omnisurg_runtime_never_pins_bone_particles(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_build_hex_particle_grid(*args, **kwargs):
        captured.update(kwargs)
        raise _StopAfterBuild

    volume = PreparedVolume(
        labels=np.ones((1, 1, 1), dtype=np.uint8),
        voxel_size_m=0.005,
        materials=MaterialTable(DEFAULT_MATERIALS),
        class_map={0: "background", 1: "muscle"},
    )
    monkeypatch.setattr(hex_runtime, "build_hex_particle_grid", fake_build_hex_particle_grid)

    with pytest.raises(_StopAfterBuild):
        hex_runtime.HexRuntime.from_volume(
            volume,
            (
                "--exit-after-init",
                "--viewer",
                "headless",
                "--input-backend",
                "off",
                "--cryo-renderer",
                "off",
                "--no-gl-interop",
            ),
        ).init()

    assert captured["kinematic_bones"] is False


def test_runtime_locked_node_enforcement_pins_position_and_velocity():
    device = wp.get_device("cpu")
    q = wp.array(
        np.asarray([[0.0, 0.0, 0.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    qd = wp.array(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    state = SimpleNamespace(particle_q=q, particle_qd=qd)
    locked_indices = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device=device)
    locked_positions = wp.array(np.asarray([[2.5, 3.5, 4.5]], dtype=np.float32), dtype=wp.vec3, device=device)

    app_runtime._enforce_locked_nodes(state, locked_indices, locked_positions, 1, device)
    wp.synchronize_device(device)

    assert np.allclose(state.particle_q.numpy()[1], [2.5, 3.5, 4.5])
    assert np.allclose(state.particle_qd.numpy()[1], [0.0, 0.0, 0.0])


def test_runtime_locked_node_merge_updates_existing_position():
    locked_indices = np.full(4, -1, dtype=np.int32)
    locked_positions = np.zeros((4, 3), dtype=np.float32)
    locked_slot_by_node: dict[int, int] = {}

    count = app_runtime._merge_locked_node_positions(
        locked_indices,
        locked_positions,
        locked_slot_by_node,
        0,
        np.asarray([2, 3], dtype=np.int32),
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )
    count = app_runtime._merge_locked_node_positions(
        locked_indices,
        locked_positions,
        locked_slot_by_node,
        count,
        np.asarray([2], dtype=np.int32),
        np.asarray([[9.0, 8.0, 7.0]], dtype=np.float32),
    )

    assert count == 2
    assert locked_indices[:2].tolist() == [2, 3]
    assert np.allclose(locked_positions[0], [9.0, 8.0, 7.0])


def test_cache_manifest_roundtrip_and_invalidation(tmp_path: Path):
    source = tmp_path / "source.raw"
    source.write_bytes(b"one")
    key = make_cache_key(dataset="unit", sources=[source], options={"pad": 1}, material_hash="abc")
    manifest = build_manifest(
        dataset="unit",
        cache_key=key,
        class_map={0: "background", 1: "liver"},
        raw_to_internal={0: 0, 7: 1},
        source_shape=(2, 2, 2),
        source_spacing_mm=(1.0, 2.0, 2.0),
        target_voxel_mm=2.0,
        pad=1,
        preprocess_options={"pad": 1},
        source_files=[source],
        material_defaults_hash="abc",
    )
    path = tmp_path / "cache.npz"
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    write_npz_atomic(path, labels=labels, manifest=manifest)

    cached = try_read_valid_cache(path, key)
    assert cached is not None
    cached_labels, texture, cached_manifest = cached
    assert texture is None
    assert np.array_equal(cached_labels, labels)
    assert cached_manifest["raw_to_internal"] == {"0": 0, "7": 1}
    assert try_read_valid_cache(path, "wrong-key") is None


def test_cache_atomic_write_preserves_existing_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source.raw"
    source.write_bytes(b"one")
    key = make_cache_key(dataset="unit", sources=[source], options={}, material_hash="abc")
    manifest = build_manifest(
        dataset="unit",
        cache_key=key,
        class_map={0: "background"},
        raw_to_internal={0: 0},
        source_shape=(1, 1, 1),
        source_spacing_mm=(1.0, 1.0, 1.0),
        target_voxel_mm=1.0,
        pad=0,
        preprocess_options={},
        source_files=[source],
        material_defaults_hash="abc",
    )
    path = tmp_path / "cache.npz"
    write_npz_atomic(path, labels=np.zeros((1, 1, 1), dtype=np.uint8), manifest=manifest)
    before = path.read_bytes()

    def fail_savez(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(np, "savez", fail_savez)
    with pytest.raises(RuntimeError):
        write_npz_atomic(path, labels=np.ones((1, 1, 1), dtype=np.uint8), manifest=manifest)
    assert path.read_bytes() == before


def test_remap_labels_and_material_overrides(tmp_path: Path):
    labels = np.asarray([[[0, 7], [9, 7]]], dtype=np.int16)
    mapped, class_map, raw_to_internal = remap_labels_to_internal(
        labels,
        {0: "background", 7: "liver", 8: "absent", 9: "bone"},
    )
    assert mapped.tolist() == [[[0, 1], [2, 1]]]
    assert class_map == {0: "background", 1: "liver", 2: "bone"}
    assert raw_to_internal == {0: 0, 7: 1, 9: 2}

    table, metadata = generate_material_table(
        class_map,
        overrides={"bone": {"phase": "rigid", "color": "#332211", "cuttable": False, "stiffness_scale": 2.0}},
    )
    assert table[2].phase is Phase.RIGID
    assert table[2].color == pytest.approx((0x33 / 255, 0x22 / 255, 0x11 / 255))
    assert metadata["material_cuttable"] == [False, True, False]
    assert metadata["material_stiffness_scale"][2] == 2.0


def test_visible_crop_selection_matches_by_id_and_name_excluding_background(tmp_path: Path):
    settings = tmp_path / "panel.json"
    settings.write_text(
        json.dumps(
            {
                "materials": [
                    {"id": 0, "name": "background", "visible": True},
                    {"id": 99, "name": "liver", "visible": True},
                    {"id": 3, "visible": True},
                    {"id": 1, "name": "procedural", "visible": True},
                    {"id": 4, "name": "bone", "visible": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    selection = select_visible_class_ids(settings, {0: "background", 1: "liver", 3: "kidney"})

    assert selection.ids == (1, 3)
    assert selection.names == ("liver", "kidney")


def test_visible_crop_selection_prefers_names_when_saved_ids_shift(tmp_path: Path):
    settings = tmp_path / "panel-shifted.json"
    settings.write_text(
        json.dumps(
            {
                "materials": [
                    {"id": 1, "name": "skin", "visible": True},
                    {"id": 2, "name": "bone", "visible": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    selection = select_visible_class_ids(
        settings,
        {0: "background", 1: "procedural", 2: "skin", 3: "bone"},
    )

    assert selection.ids == (2, 3)
    assert selection.names == ("skin", "bone")


def test_visible_crop_keeps_margin_neighborhood_and_zeroes_distant_tissue(tmp_path: Path):
    settings = tmp_path / "panel.json"
    settings.write_text(json.dumps({"materials": [{"id": 1, "name": "organ", "visible": True}]}), encoding="utf-8")
    labels = np.zeros((6, 6, 9), dtype=np.uint8)
    labels[2, 2, 2] = 1
    labels[2, 2, 6] = 1
    labels[3, 2, 2] = 2
    labels[2, 2, 4] = 2

    crop = crop_labels_to_visible_classes(labels, {0: "background", 1: "organ", 2: "other"}, settings, margin_voxels=1)

    assert crop.labels.shape == (3, 3, 7)
    assert crop.crop_min == (1, 1, 1)
    assert crop.crop_max == (4, 4, 8)
    assert crop.labels[1, 1, 1] == 1
    assert crop.labels[1, 1, 5] == 1
    assert crop.labels[2, 1, 1] == 2
    assert crop.labels[1, 1, 3] == 0
    assert crop.metadata["zeroed_voxel_count"] == 1


def test_visible_crop_margin_zero_is_visible_only(tmp_path: Path):
    settings = tmp_path / "panel.json"
    settings.write_text(json.dumps({"materials": [{"name": "organ", "visible": True}]}), encoding="utf-8")
    labels = np.zeros((3, 3, 5), dtype=np.uint8)
    labels[1, 1, 1] = 1
    labels[1, 1, 2] = 2
    labels[1, 1, 3] = 1

    crop = crop_labels_to_visible_classes(labels, {0: "background", 1: "organ", 2: "other"}, settings, margin_voxels=0)

    assert crop.labels.shape == (1, 1, 3)
    assert crop.labels[:, :, 0].item() == 1
    assert crop.labels[:, :, 1].item() == 0
    assert crop.labels[:, :, 2].item() == 1


def test_visible_crop_rejects_empty_or_absent_visible_classes(tmp_path: Path):
    no_visible = tmp_path / "no_visible.json"
    no_visible.write_text(json.dumps({"materials": [{"id": 1, "name": "organ", "visible": False}]}), encoding="utf-8")
    labels = np.zeros((2, 2, 2), dtype=np.uint8)
    labels[0, 0, 0] = 1

    with pytest.raises(ValueError, match="no visible non-background"):
        crop_labels_to_visible_classes(labels, {0: "background", 1: "organ"}, no_visible)

    absent = tmp_path / "absent.json"
    absent.write_text(json.dumps({"materials": [{"id": 2, "name": "absent", "visible": True}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not present"):
        crop_labels_to_visible_classes(labels, {0: "background", 1: "organ", 2: "absent"}, absent)


def test_builtin_segmentation_specs_are_dataset_specific():
    digimouse = load_builtin_class_map("digimouse")
    vhp = load_builtin_class_map("vhp")
    abdomen = load_builtin_class_map("abdomen-atlas")
    kits23 = load_builtin_class_map("kits23")

    assert digimouse[2] == "skeleton"
    assert "skeleton" not in set(vhp.values())
    assert abdomen[5] == "liver"
    assert kits23[1] == "kidney"
    assert kits23[2] == "tumor"
    assert "external_cerebrum" not in set(abdomen.values())
    assert load_builtin_material_defaults("digimouse")["skeleton"]["phase"] == "rigid"
    assert load_builtin_material_defaults("vhp") == {}
    assert load_builtin_material_defaults("kits23")["kidney"]["color"] == [0.56, 0.25, 0.22]


def test_preprocessed_npz_loading_with_and_without_texture(tmp_path: Path):
    path = tmp_path / "prepared.npz"
    labels = np.asarray([[[0, 5], [5, 0]]], dtype=np.uint8)
    texture = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    texture[..., 1] = 200
    manifest = {"class_map": {"0": "background", "5": "organ"}, "target_voxel_mm": 3.0}
    np.savez(path, labels=labels, texture_rgb=texture, manifest=np.asarray(json.dumps(manifest)))

    loader = PreprocessedNpzLoader()
    vol = loader.load(
        PreprocessConfig(
            dataset="preprocessed",
            root=path,
            cache_dir=tmp_path / "cache",
            pad=0,
            use_cache=False,
        )
    )
    assert vol.labels.tolist() == [[[0, 1], [1, 0]]]
    assert vol.voxel_size_m == pytest.approx(0.003)
    assert vol.texture_rgb is not None
    assert vol.texture_rgb[..., 1].max() == 200

    vol_no_texture = loader.load(
        PreprocessConfig(
            dataset="preprocessed",
            root=path,
            cache_dir=tmp_path / "cache",
            pad=0,
            use_cache=False,
            load_texture=False,
        )
    )
    assert vol_no_texture.texture_rgb is None


def test_preprocessed_npz_applies_visible_crop_before_padding_and_crops_texture(tmp_path: Path):
    path = tmp_path / "prepared.npz"
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[2, 2, 2] = 5
    labels[3, 2, 2] = 9
    labels[0, 0, 0] = 9
    texture = np.zeros((5, 5, 5, 3), dtype=np.uint8)
    for x in range(5):
        for y in range(5):
            for z in range(5):
                texture[x, y, z, 0] = (x * 100 + y * 10 + z) % 256
    manifest = {"class_map": {"0": "background", "5": "organ", "9": "other"}, "target_voxel_mm": 2.0}
    np.savez(path, labels=labels, texture_rgb=texture, manifest=np.asarray(json.dumps(manifest)))
    settings = tmp_path / "panel.json"
    settings.write_text(json.dumps({"materials": [{"name": "organ", "visible": True}]}), encoding="utf-8")

    vol = PreprocessedNpzLoader().load(
        PreprocessConfig(
            dataset="preprocessed",
            root=path,
            cache_dir=tmp_path / "cache",
            pad=1,
            use_cache=False,
            visible_class_crop_settings_path=settings,
            visible_class_crop_margin_voxels=1,
        )
    )

    assert vol.labels.shape == (5, 5, 5)
    assert vol.texture_rgb is not None
    assert vol.texture_rgb.shape == (5, 5, 5, 3)
    assert vol.labels[2, 2, 2] == 1
    assert vol.labels[3, 2, 2] == 2
    assert int(np.count_nonzero(vol.labels)) == 2
    assert vol.texture_rgb[2, 2, 2, 0] == texture[2, 2, 2, 0]
    assert vol.origin == pytest.approx((0.002, 0.002, 0.002))
    assert vol.metadata["visible_class_crop"]["crop_min"] == [1, 1, 1]


def test_abdomen_atlas_nifti_resamples_to_isotropic(tmp_path: Path):
    nib = pytest.importorskip("nibabel")
    data = np.zeros((2, 2, 4), dtype=np.uint8)
    data[:, :, 1:3] = 5
    img = nib.Nifti1Image(data, affine=np.diag([1.0, 1.0, 2.5, 1.0]))
    path = tmp_path / "combined_labels.nii.gz"
    nib.save(img, path)

    loader = AbdomenAtlasLoader()
    vol = loader.load(
        PreprocessConfig(
            dataset="abdomen-atlas",
            root=path,
            cache_dir=tmp_path / "cache",
            pad=1,
            use_cache=True,
        )
    )
    assert vol.voxel_size_m == pytest.approx(0.0025)
    assert vol.labels.shape == (3, 3, 6)
    assert vol.class_map == {0: "background", 1: "liver"}
    assert vol.labels.max() == 1
    assert vol.metadata["material_names"] == ["background", "liver"]
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1

    vol_cached = loader.load(
        PreprocessConfig(
            dataset="abdomen-atlas",
            root=path,
            cache_dir=tmp_path / "cache",
            pad=1,
            use_cache=True,
        )
    )
    assert np.array_equal(vol_cached.labels, vol.labels)

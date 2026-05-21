# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the KiTS23 NIfTI loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omnisurg.hex.cli import DATASET_CHOICES, _default_root, build_parser
from omnisurg.hex.data.loaders.kits23 import Kits23Loader
from omnisurg.hex.data.registry import choose_loader, get_loader, list_loaders
from omnisurg.hex.data.types import PreprocessConfig


def _write_nifti(path: Path, labels: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    nib = pytest.importorskip("nibabel")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.ascontiguousarray(labels), affine=np.diag([*spacing, 1.0]))
    nib.save(img, path)


def _config(root: Path, tmp_path: Path, **kwargs) -> PreprocessConfig:
    options = {
        "dataset": "kits23",
        "root": root,
        "cache_dir": tmp_path / "cache",
        "pad": 0,
        "use_cache": False,
    }
    options.update(kwargs)
    return PreprocessConfig(**options)


def test_kits23_loads_flat_case_and_maps_tumor_classes(tmp_path: Path):
    root = tmp_path / "Kits2023"
    labels = np.asarray([[[0, 1], [2, 3]]], dtype=np.uint8)
    case = root / "case_00002.nii.gz"
    _write_nifti(case, labels)

    volume = Kits23Loader().load(_config(root, tmp_path, options={"case": "2"}))

    assert volume.metadata["dataset"] == "kits23"
    assert volume.metadata["case_id"] == "case_00002"
    assert volume.class_map == {0: "background", 1: "kidney", 2: "tumor", 3: "tumor_2"}
    assert volume.labels.tolist() == labels.tolist()
    assert volume.metadata["material_names"] == ["background", "kidney", "tumor", "tumor_2"]


def test_kits23_loads_case_directory_segmentation_and_resamples(tmp_path: Path):
    case_dir = tmp_path / "KiTS23" / "case_00001"
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    data[:, :, 0] = 1
    _write_nifti(case_dir / "segmentation.nii.gz", data, spacing=(1.0, 1.0, 2.5))

    volume = Kits23Loader().load(_config(case_dir.parent, tmp_path, pad=1))

    assert volume.voxel_size_m == pytest.approx(0.0025)
    assert volume.labels.shape == (3, 3, 4)
    assert volume.class_map == {0: "background", 1: "kidney"}
    assert volume.metadata["scene_label"] == "KiTS23 case_00001"


def test_kits23_cache_roundtrip(tmp_path: Path):
    root = tmp_path / "Kits2023"
    _write_nifti(root / "case_00000.nii.gz", np.asarray([[[1, 2]]], dtype=np.uint8))

    loader = Kits23Loader()
    config = _config(root, tmp_path, use_cache=True)
    volume = loader.load(config)
    cached = loader.load(config)

    assert np.array_equal(cached.labels, volume.labels)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    assert cached.metadata["cache_manifest"]["dataset"] == "kits23"


def test_kits23_registry_aliases_and_cli_exposure(tmp_path: Path):
    root = tmp_path / "Kits2023"
    _write_nifti(root / "case_00000.nii.gz", np.asarray([[[1]]], dtype=np.uint8))

    assert Kits23Loader().can_load(root)
    assert Kits23Loader().can_load(root / "case_00000.nii.gz")
    assert choose_loader(root).name == "kits23"
    assert "kits23" in list_loaders()
    assert "kits23" in DATASET_CHOICES
    assert "kits2023" in DATASET_CHOICES
    assert _default_root("kits23") is not None
    assert _default_root("kits23").name == "Kits2023"
    for alias in ("kits23", "kits-23", "kits2023", "kits-2023"):
        assert get_loader(alias).name == "kits23"
        args = build_parser().parse_args(["--dataset", alias, "--data", str(root), "--case", "0"])
        assert args.dataset == alias
        assert args.case == "0"

# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the 3Dircadb DICOM loader."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from omnisurg.hex.cli import DATASET_CHOICES, _default_root, build_parser
from omnisurg.hex.data.loaders.ircad_3d import ThreeDircadbLoader
from omnisurg.hex.data.registry import choose_loader, get_loader, list_loaders
from omnisurg.hex.data.types import PreprocessConfig


def _write_dicom_slice(
    path: Path,
    pixels: np.ndarray,
    *,
    instance: int,
    series_uid: str,
    position: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float, float, float] | None = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 1.0,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
) -> None:
    arr = np.ascontiguousarray(pixels)
    if arr.ndim != 2:
        raise ValueError("test DICOM helper expects one 2D slice")

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Unit^Test"
    ds.PatientID = "unit"
    ds.Modality = "CT"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = int(instance)
    ds.Rows = int(arr.shape[0])
    ds.Columns = int(arr.shape[1])
    ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    ds.SliceThickness = float(slice_thickness)
    if position is not None:
        ds.ImagePositionPatient = [float(v) for v in position]
    if orientation is not None:
        ds.ImageOrientationPatient = [float(v) for v in orientation]
    if rescale_slope is not None:
        ds.RescaleSlope = float(rescale_slope)
    if rescale_intercept is not None:
        ds.RescaleIntercept = float(rescale_intercept)

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = int(arr.dtype.itemsize * 8)
    ds.BitsStored = int(arr.dtype.itemsize * 8)
    ds.HighBit = int(ds.BitsStored - 1)
    ds.PixelRepresentation = 1 if np.issubdtype(arr.dtype, np.signedinteger) else 0
    ds.PixelData = arr.tobytes()
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:
        ds.save_as(path, write_like_original=False)


def _write_zipped_dicom_slice(zip_path: Path, member: str, pixels: np.ndarray, **kwargs) -> None:
    tmp = zip_path.parent / f".{Path(member).name}.dcm"
    _write_dicom_slice(tmp, pixels, **kwargs)
    try:
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.write(tmp, member)
    finally:
        tmp.unlink(missing_ok=True)


def _root(tmp_path: Path, cohort: int = 1) -> Path:
    root = tmp_path / f"3Dircadb{cohort}" / f"3Dircadb{cohort}.1"
    (root / "LABELLED_DICOM").mkdir(parents=True)
    return root


def _zipped_root(tmp_path: Path, cohort: int = 2) -> Path:
    root = tmp_path / f"3Dircadb{cohort}" / f"3Dircadb{cohort}.1"
    root.mkdir(parents=True)
    return root


def _config(root: Path, tmp_path: Path, **kwargs) -> PreprocessConfig:
    options = {
        "dataset": "3dircadb",
        "root": root,
        "cache_dir": tmp_path / "cache",
        "pad": 0,
        "use_cache": False,
        "load_texture": False,
    }
    options.update(kwargs)
    return PreprocessConfig(**options)


def test_3dircadb_loader_detects_nested_patient_folders(tmp_path: Path):
    loader = ThreeDircadbLoader()
    root_v1 = _root(tmp_path, cohort=1)
    root_v2 = _zipped_root(tmp_path, cohort=2)
    _write_zipped_dicom_slice(
        root_v2 / "LABELLED_DICOM.zip",
        "LABELLED_DICOM/image_0",
        np.asarray([[8]], dtype=np.uint16),
        instance=1,
        series_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
    )

    assert loader.can_load(root_v1)
    assert loader.can_load(tmp_path / "3Dircadb1")
    assert loader.can_load(root_v2)
    assert loader.can_load(tmp_path / "3Dircadb2")
    assert choose_loader(root_v1).name == "3dircadb"
    assert choose_loader(root_v2).name == "3dircadb"


def test_3dircadb_loads_3dircadb2_root_shape(tmp_path: Path):
    root = _zipped_root(tmp_path, cohort=2)
    _write_zipped_dicom_slice(
        root / "LABELLED_DICOM.zip",
        "LABELLED_DICOM/image_0",
        np.asarray([[65]], dtype=np.uint16),
        instance=1,
        series_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
    )
    _write_zipped_dicom_slice(
        root / "MASKS_DICOM.zip",
        "MASKS_DICOM/rightlung/image_0",
        np.asarray([[1]], dtype=np.uint16),
        instance=1,
        series_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
    )

    volume = ThreeDircadbLoader().load(_config(root, tmp_path))

    assert volume.metadata["dataset"] == "3dircadb"
    assert volume.metadata["scene_label"] == "3Dircadb 3Dircadb2.1"
    assert volume.class_map == {0: "background", 1: "lungs"}


def test_3dircadb1_sorts_dicom_slices_by_position(tmp_path: Path):
    root = _root(tmp_path)
    label_dir = root / "LABELLED_DICOM"
    series_uid = generate_uid()
    _write_dicom_slice(label_dir / "z2.dcm", np.asarray([[3]], dtype=np.uint16), instance=30, series_uid=series_uid, position=(0.0, 0.0, 2.0))
    _write_dicom_slice(label_dir / "z0.dcm", np.asarray([[1]], dtype=np.uint16), instance=10, series_uid=series_uid, position=(0.0, 0.0, 0.0))
    _write_dicom_slice(label_dir / "z1.dcm", np.asarray([[2]], dtype=np.uint16), instance=20, series_uid=series_uid, position=(0.0, 0.0, 1.0))

    volume = ThreeDircadbLoader().load(_config(root, tmp_path))

    assert volume.labels.shape == (1, 1, 3)
    assert volume.labels[0, 0, :].tolist() == [1, 2, 3]


def test_3dircadb1_sorts_dicom_slices_by_instance_when_position_missing(tmp_path: Path):
    root = _root(tmp_path)
    label_dir = root / "LABELLED_DICOM"
    series_uid = generate_uid()
    _write_dicom_slice(label_dir / "third.dcm", np.asarray([[3]], dtype=np.uint16), instance=3, series_uid=series_uid, position=None, orientation=None)
    _write_dicom_slice(label_dir / "first.dcm", np.asarray([[1]], dtype=np.uint16), instance=1, series_uid=series_uid, position=None, orientation=None)
    _write_dicom_slice(label_dir / "second.dcm", np.asarray([[2]], dtype=np.uint16), instance=2, series_uid=series_uid, position=None, orientation=None)

    volume = ThreeDircadbLoader().load(_config(root, tmp_path))

    assert volume.labels[0, 0, :].tolist() == [1, 2, 3]


def test_3dircadb1_remaps_bundled_raw_labels_to_internal_ids(tmp_path: Path):
    root = _root(tmp_path)
    label_dir = root / "LABELLED_DICOM"
    _write_dicom_slice(
        label_dir / "labels.dcm",
        np.asarray([[0, 8], [9, 8]], dtype=np.uint16),
        instance=1,
        series_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
    )

    volume = ThreeDircadbLoader().load(_config(root, tmp_path))

    assert volume.class_map == {0: "background", 1: "liver", 2: "liver_tumor"}
    assert volume.labels[:, :, 0].tolist() == [[0, 1], [2, 1]]


def test_3dircadb1_resamples_anisotropic_labels_to_default_isotropic_spacing(tmp_path: Path):
    root = _root(tmp_path)
    label_dir = root / "LABELLED_DICOM"
    series_uid = generate_uid()
    for idx, z in enumerate((0.0, 2.5, 5.0, 7.5), start=1):
        _write_dicom_slice(
            label_dir / f"{idx}.dcm",
            np.full((2, 2), 8, dtype=np.uint16),
            instance=idx,
            series_uid=series_uid,
            position=(0.0, 0.0, z),
            pixel_spacing=(1.0, 1.0),
            slice_thickness=2.5,
        )

    volume = ThreeDircadbLoader().load(_config(root, tmp_path, pad=1))

    assert volume.voxel_size_m == pytest.approx(0.0025)
    assert volume.labels.shape == (3, 3, 6)
    assert volume.class_map == {0: "background", 1: "liver"}


def test_3dircadb1_cache_roundtrip(tmp_path: Path):
    root = _root(tmp_path)
    label_dir = root / "LABELLED_DICOM"
    _write_dicom_slice(
        label_dir / "labels.dcm",
        np.asarray([[8]], dtype=np.uint16),
        instance=1,
        series_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
    )

    loader = ThreeDircadbLoader()
    config = _config(root, tmp_path, use_cache=True)
    volume = loader.load(config)
    cached = loader.load(config)

    assert np.array_equal(cached.labels, volume.labels)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    assert cached.metadata["cache_manifest"]["dataset"] == "3dircadb"


def test_3dircadb1_loads_optional_patient_dicom_texture(tmp_path: Path):
    root = _zipped_root(tmp_path)
    _write_zipped_dicom_slice(
        root / "LABELLED_DICOM.zip",
        "LABELLED_DICOM/image_0",
        np.asarray([[8]], dtype=np.uint16),
        instance=1,
        series_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
    )
    _write_zipped_dicom_slice(
        root / "PATIENT_DICOM.zip",
        "PATIENT_DICOM/image_0",
        np.asarray([[42]], dtype=np.int16),
        instance=1,
        series_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
        rescale_slope=1.0,
        rescale_intercept=-1024.0,
    )

    volume = ThreeDircadbLoader().load(_config(root, tmp_path, load_texture=True))

    assert volume.texture_rgb is not None
    assert volume.texture_rgb.shape == (1, 1, 1, 3)
    assert volume.texture_rgb.dtype == np.uint8


def test_3dircadb_registry_aliases_and_cli_exposure(tmp_path: Path):
    assert "3dircadb" in list_loaders()
    assert "3dircadb" in DATASET_CHOICES
    assert "3dircadb1" in DATASET_CHOICES
    assert "3dircadb2" in DATASET_CHOICES
    assert _default_root("3dircadb") is None
    for alias in ("3dircadb", "3dircadb1", "3dircadb2", "ircad-3d", "3d-ircadb"):
        assert get_loader(alias).name == "3dircadb"
        args = build_parser().parse_args(["--dataset", alias, "--data", str(tmp_path)])
        assert args.dataset == alias

    crop_args = build_parser().parse_args(
        [
            "--dataset",
            "3dircadb",
            "--data",
            str(tmp_path),
            "--crop-visible-classes",
            str(tmp_path / "panel.json"),
            "--crop-visible-margin-voxels",
            "7",
        ]
    )
    config = PreprocessConfig(
        dataset=crop_args.dataset,
        root=crop_args.data,
        visible_class_crop_settings_path=crop_args.crop_visible_classes,
        visible_class_crop_margin_voxels=crop_args.crop_visible_margin_voxels,
    )
    assert config.visible_class_crop_settings_path == tmp_path / "panel.json"
    assert config.visible_class_crop_margin_voxels == 7

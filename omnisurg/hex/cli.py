# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for OmniSurg Hex."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .app import OmniSurgHexApp
from .data.registry import get_loader
from .data.types import PreprocessConfig

DATASET_CHOICES = (
    "digimouse",
    "vhp",
    "abdomen-atlas",
    "3dircadb",
    "3dircadb1",
    "3dircadb2",
    "ircad-3d",
    "3d-ircadb",
    "kits23",
    "kits-23",
    "kits2023",
    "kits-2023",
    "preprocessed",
    "synthetic",
)
OMNISURG_DATA_ROOT_ENV = "OMNISURG_DATA_ROOT"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    configured = os.environ.get(OMNISURG_DATA_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return _project_root()


def _dataset_key(dataset: str) -> str:
    return dataset.strip().lower().replace("_", "-")


def _default_root(dataset: str) -> Path | None:
    root = _data_root()
    key = _dataset_key(dataset)
    if key == "digimouse":
        return root / "Digimouse" / "atlas" / "atlas"
    if key == "vhp":
        return root / "VHP"
    if key in {"abdomen-atlas", "abdomenatlas"}:
        return root / "AbdomenAtlas" / "combined_labels.nii.gz"
    if key in {"kits23", "kits-23", "kits2023", "kits-2023"}:
        return root / "Kits2023"
    if key in {"synthetic", "synthetic-block"}:
        return None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnisurg hex",
        description="Run the OmniSurg Hex corner-grid surgical cutting app.",
        add_help=True,
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="digimouse",
        help="Dataset loader to use.",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset root or preprocessed .npz path.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory for prepared .npz cache bundles.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild the prepared cache even if it is valid.")
    parser.add_argument("--no-cache", action="store_true", help="Disable reading and writing prepared cache bundles.")
    parser.add_argument("--pad", type=int, default=1, help="Background voxel padding applied before simulation.")
    parser.add_argument("--downsample", type=int, default=None, help="Integer categorical downsample factor.")
    parser.add_argument("--target-voxel-mm", type=float, default=None, help="Target isotropic simulation voxel size.")
    parser.add_argument(
        "--texture",
        type=str,
        default="auto",
        help="Texture mode: auto, off, or a .npy/.npz RGB volume path.",
    )
    parser.add_argument("--class-map", type=Path, default=None, help="JSON class-map override.")
    parser.add_argument("--case", default=None, help="Case id for multi-case datasets such as KiTS23.")
    parser.add_argument("--material-overrides", type=Path, default=None, help="JSON material override file.")
    parser.add_argument(
        "--crop-visible-classes",
        type=Path,
        default=None,
        help="Segmentation panel settings JSON used to crop around visible classes.",
    )
    parser.add_argument(
        "--crop-visible-margin-voxels",
        type=int,
        default=4,
        help="Voxel margin around visible crop seed classes when --crop-visible-classes is set.",
    )
    parser.add_argument("--size", type=int, default=8, help="Synthetic block edge length in voxels.")
    parser.add_argument("--voxel", type=float, default=0.005, help="Synthetic voxel size in metres.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, runtime_args = parser.parse_known_args(argv)

    texture_text = str(args.texture)
    if texture_text.lower() == "off":
        load_texture = False
        texture_path = None
    elif texture_text.lower() == "auto":
        load_texture = True
        texture_path = None
    else:
        load_texture = True
        texture_path = Path(texture_text)

    root = args.data if args.data is not None else _default_root(args.dataset)
    dataset_key = _dataset_key(args.dataset)
    if dataset_key == "preprocessed" and root is None:
        parser.error("--data is required with --dataset preprocessed")
    if dataset_key in {"3dircadb", "3dircadb1", "3dircadb2", "ircad-3d", "3d-ircadb"} and root is None:
        parser.error("--data is required with --dataset 3dircadb")
    if int(args.crop_visible_margin_voxels) < 0:
        parser.error("--crop-visible-margin-voxels must be >= 0")

    loader = get_loader(args.dataset)
    config = PreprocessConfig(
        dataset=args.dataset,
        root=root,
        cache_dir=args.cache_dir,
        downsample=args.downsample,
        target_voxel_mm=args.target_voxel_mm,
        pad=args.pad,
        use_cache=not bool(args.no_cache),
        rebuild_cache=bool(args.rebuild_cache),
        load_texture=load_texture,
        texture_path=texture_path,
        class_map_path=args.class_map,
        material_overrides_path=args.material_overrides,
        visible_class_crop_settings_path=args.crop_visible_classes,
        visible_class_crop_margin_voxels=args.crop_visible_margin_voxels,
        options={
            "size": int(args.size),
            "voxel_m": float(args.voxel),
            "case": args.case,
        },
    )
    volume = loader.load(config)

    app_args = list(runtime_args)
    app_args.extend(["--atlas", "__omnisurg_prepared__", "--atlas-pad", "0"])
    app_args.extend(["--cryo-texture", "__omnisurg_prepared__" if volume.texture_rgb is not None else ""])
    return OmniSurgHexApp(volume).run(app_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

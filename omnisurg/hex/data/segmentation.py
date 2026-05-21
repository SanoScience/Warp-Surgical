# SPDX-License-Identifier: Apache-2.0
"""Dataset-specific segmentation class-map resources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

DATASET_SPEC_FILES: dict[str, str] = {
    "digimouse": "digimouse.json",
    "vhp": "vhp.json",
    "abdomen-atlas": "abdomen_atlas.json",
    "abdomen_atlas": "abdomen_atlas.json",
    "3dircadb": "3dircadb.json",
    "3dircadb1": "3dircadb.json",
    "3dircadb2": "3dircadb.json",
    "ircad-3d": "3dircadb.json",
    "3d-ircadb": "3dircadb.json",
    "kits23": "kits23.json",
    "kits-23": "kits23.json",
    "kits2023": "kits23.json",
    "kits-2023": "kits23.json",
}


def load_segmentation_spec(dataset: str) -> dict[str, Any]:
    """Load the bundled segmentation spec for one dataset."""

    key = str(dataset).strip().lower().replace("_", "-")
    filename = DATASET_SPEC_FILES.get(key)
    if filename is None:
        raise KeyError(f"no bundled segmentation spec for dataset {dataset!r}")
    text = files("omnisurg.hex.data.class_maps").joinpath(filename).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"segmentation spec {filename} must be a JSON object")
    return dict(raw)


def load_builtin_class_map(dataset: str) -> dict[int, str]:
    """Return the bundled raw-label class map for one dataset."""

    spec = load_segmentation_spec(dataset)
    classes = spec.get("classes", [])
    out: dict[int, str] = {}
    for item in classes:
        if not isinstance(item, Mapping) or "id" not in item or "name" not in item:
            raise ValueError(f"invalid class-map entry in {dataset!r}: {item!r}")
        out[int(item["id"])] = str(item["name"])
    out.setdefault(0, "background")
    return dict(sorted(out.items()))


def load_builtin_material_defaults(dataset: str) -> dict[str, dict[str, Any]]:
    """Return dataset-specific material defaults keyed by class name."""

    spec = load_segmentation_spec(dataset)
    defaults = spec.get("material_defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError(f"material_defaults in {dataset!r} must be a JSON object")
    return {str(k): dict(v) for k, v in defaults.items()}

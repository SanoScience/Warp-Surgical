# SPDX-License-Identifier: Apache-2.0
"""Class-map normalization and generated material tables."""

from __future__ import annotations

import colorsys
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from omnisurg.hex.materials import BACKGROUND, Material, MaterialTable, Phase

from .segmentation import load_builtin_class_map, load_builtin_material_defaults

DIGIMOUSE_CLASS_MAP: dict[int, str] = load_builtin_class_map("digimouse")
VHP_CLASS_MAP: dict[int, str] = load_builtin_class_map("vhp")
ABDOMEN_ATLAS_CLASS_MAP: dict[int, str] = load_builtin_class_map("abdomen-atlas")
THREE_DIRCADB_CLASS_MAP: dict[int, str] = load_builtin_class_map("3dircadb")
KITS23_CLASS_MAP: dict[int, str] = load_builtin_class_map("kits23")
DIGIMOUSE_MATERIAL_HINTS: dict[str, dict[str, Any]] = load_builtin_material_defaults("digimouse")
THREE_DIRCADB_MATERIAL_HINTS: dict[str, dict[str, Any]] = load_builtin_material_defaults("3dircadb")
KITS23_MATERIAL_HINTS: dict[str, dict[str, Any]] = load_builtin_material_defaults("kits23")
THREE_DIRCADB1_CLASS_MAP = THREE_DIRCADB_CLASS_MAP
THREE_DIRCADB1_MATERIAL_HINTS = THREE_DIRCADB_MATERIAL_HINTS


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def stable_hash(value: object) -> str:
    """Return a short stable hash for manifest/cache fields."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_class_map(path: str | Path | None) -> dict[int, str]:
    """Load a class map from JSON.

    Accepted shapes are ``{"1": "liver"}``, ``{"classes": [{"id": 1, "name": "liver"}]}``,
    or ``[{"id": 1, "name": "liver"}]``.
    """

    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, Mapping) and "classes" in raw:
        raw = raw["classes"]
    if isinstance(raw, list):
        out: dict[int, str] = {}
        for item in raw:
            if not isinstance(item, Mapping) or "id" not in item or "name" not in item:
                raise ValueError(f"invalid class-map entry: {item!r}")
            out[int(item["id"])] = str(item["name"])
        return out
    if isinstance(raw, Mapping):
        return {int(k): str(v) for k, v in raw.items()}
    raise ValueError(f"unsupported class-map JSON shape in {path}")


def load_material_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load material override JSON keyed by internal id or material name."""

    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, Mapping) and "materials" in raw:
        raw = raw["materials"]
    if not isinstance(raw, Mapping):
        raise ValueError("material overrides must be a JSON object")
    return {str(k): dict(v) for k, v in raw.items()}


def normalize_class_map(class_map: Mapping[int, str] | None, labels: np.ndarray | None = None) -> dict[int, str]:
    """Return a map containing background and every observed label."""

    out = {int(k): str(v) for k, v in (class_map or {}).items()}
    out.setdefault(0, "background")
    if labels is not None and labels.size:
        for raw_id in np.unique(labels).astype(np.int64).tolist():
            out.setdefault(int(raw_id), f"class_{int(raw_id)}")
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def remap_labels_to_internal(
    raw_labels: np.ndarray,
    class_map: Mapping[int, str] | None = None,
) -> tuple[np.ndarray, dict[int, str], dict[int, int]]:
    """Map arbitrary raw labels to contiguous internal ids with background at 0."""

    raw = np.asarray(raw_labels)
    cmap = normalize_class_map(class_map, raw)
    if raw.size:
        raw_ids = sorted(int(raw_id) for raw_id in np.unique(raw).astype(np.int64).tolist() if int(raw_id) != 0)
    else:
        raw_ids = sorted(int(k) for k in cmap if int(k) != 0)

    raw_to_internal = {0: 0}
    internal_class_map = {0: cmap.get(0, "background")}
    for internal_id, raw_id in enumerate(raw_ids, start=1):
        raw_to_internal[int(raw_id)] = int(internal_id)
        internal_class_map[int(internal_id)] = cmap.get(int(raw_id), f"class_{int(raw_id)}")

    if raw.size == 0:
        return np.ascontiguousarray(raw.astype(np.uint8, copy=False)), internal_class_map, raw_to_internal

    max_raw = int(raw.max())
    if max_raw >= 0 and max_raw < 65536 and min(raw_to_internal) >= 0:
        lut = np.zeros(max(max_raw + 1, 1), dtype=np.int32)
        for raw_id, internal_id in raw_to_internal.items():
            if 0 <= raw_id < lut.size:
                lut[raw_id] = internal_id
        mapped = lut[raw.astype(np.int64)]
    else:
        mapped = np.zeros(raw.shape, dtype=np.int32)
        for raw_id, internal_id in raw_to_internal.items():
            mapped[raw == raw_id] = internal_id
    return _smallest_label_dtype(mapped), internal_class_map, raw_to_internal


def _smallest_label_dtype(labels: np.ndarray) -> np.ndarray:
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif max_label <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    else:
        dtype = np.int32
    return np.ascontiguousarray(labels.astype(dtype, copy=False))


def pad_labels(labels: np.ndarray, pad: int) -> np.ndarray:
    """Apply background padding to a label volume."""

    if int(pad) <= 0:
        return np.ascontiguousarray(labels)
    return np.ascontiguousarray(np.pad(labels, int(pad), mode="constant", constant_values=0))


def pad_texture_rgb(texture: np.ndarray | None, pad: int, scale: int = 1) -> np.ndarray | None:
    """Pad a texture volume by the corresponding number of texture voxels."""

    if texture is None or int(pad) <= 0:
        return None if texture is None else np.ascontiguousarray(texture)
    p = int(pad) * max(1, int(scale))
    return np.ascontiguousarray(
        np.pad(texture, ((p, p), (p, p), (p, p), (0, 0)), mode="constant", constant_values=0)
    )


def _default_color(idx: int) -> tuple[float, float, float]:
    hue = (0.08 + idx * 0.61803398875) % 1.0
    sat = 0.48 + 0.18 * ((idx * 37) % 3) / 2.0
    val = 0.78 + 0.12 * ((idx * 53) % 2)
    return tuple(float(c) for c in colorsys.hsv_to_rgb(hue, sat, val))


def _parse_phase(value: Any) -> Phase:
    if isinstance(value, Phase):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "soft":
            return Phase.SOFT
        if text == "rigid":
            return Phase.RIGID
        if text == "fluid":
            return Phase.FLUID
    return Phase(int(value))


def _parse_color(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            return tuple(int(text[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        raise ValueError(f"unsupported color string {value!r}; use '#RRGGBB'")
    vals = tuple(float(v) for v in value)
    if len(vals) != 3:
        raise ValueError(f"color must have 3 components, got {value!r}")
    if max(vals) > 1.0:
        vals = tuple(v / 255.0 for v in vals)
    return tuple(min(1.0, max(0.0, v)) for v in vals)


def generate_material_table(
    class_map: Mapping[int, str],
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    defaults_by_name: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[MaterialTable, dict[str, list[Any]]]:
    """Generate a contiguous material table plus UI metadata arrays."""

    cmap = normalize_class_map(class_map)
    max_id = max(cmap) if cmap else 0
    overrides = overrides or {}
    defaults_by_name = defaults_by_name or {}
    materials: list[Material] = []
    visible: list[bool] = []
    cuttable: list[bool] = []
    stiffness_scale: list[float] = []

    for idx in range(max_id + 1):
        name = cmap.get(idx, f"class_{idx}")
        if idx == 0:
            base = replace(BACKGROUND, id=0, name=name)
            defaults: dict[str, Any] = {"visible": False, "cuttable": False, "stiffness_scale": 0.0}
        else:
            base = Material(
                id=idx,
                name=name,
                phase=Phase.SOFT,
                mass=1.0,
                resistance=80.0,
                conductivity=0.9,
                color=_default_color(idx),
            )
            defaults = {"visible": True, "cuttable": True, "stiffness_scale": 1.0}
            defaults.update(defaults_by_name.get(name, {}))

        override = dict(overrides.get(str(idx), {}))
        override.update(overrides.get(name, {}))
        material = base
        if "phase" in defaults:
            material = replace(material, phase=_parse_phase(defaults["phase"]))
        if "mass" in defaults:
            material = replace(material, mass=float(defaults["mass"]))
        if "resistance" in defaults:
            material = replace(material, resistance=float(defaults["resistance"]))
        if "conductivity" in defaults:
            material = replace(material, conductivity=float(defaults["conductivity"]))
        if "color" in defaults:
            material = replace(material, color=_parse_color(defaults["color"]))

        if "phase" in override:
            material = replace(material, phase=_parse_phase(override["phase"]))
        if "mass" in override:
            material = replace(material, mass=float(override["mass"]))
        if "resistance" in override:
            material = replace(material, resistance=float(override["resistance"]))
        if "conductivity" in override:
            material = replace(material, conductivity=float(override["conductivity"]))
        if "color" in override:
            material = replace(material, color=_parse_color(override["color"]))

        materials.append(material)
        visible.append(bool(override.get("visible", defaults.get("visible", True))))
        cuttable.append(bool(override.get("cuttable", defaults.get("cuttable", idx != 0))))
        stiffness_scale.append(float(override.get("stiffness_scale", defaults.get("stiffness_scale", 1.0))))

    return MaterialTable(materials), {
        "material_visible": visible,
        "material_cuttable": cuttable,
        "material_stiffness_scale": stiffness_scale,
        "material_names": [m.name for m in materials],
    }


def material_defaults_hash(
    class_map: Mapping[int, str],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    defaults_by_name: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Hash material-generation inputs that affect cache manifests."""

    return stable_hash(
        {
            "class_map": {str(k): v for k, v in sorted(class_map.items())},
            "overrides": overrides or {},
            "defaults_by_name": defaults_by_name or {},
        }
    )

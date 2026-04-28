from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np


HAPTIC_PRESET_SCHEMA_VERSION = 1
DEFAULT_HAPTIC_PRESET_FILENAME = "default.json"


def _zero_vec3() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


@dataclass
class HapticFeedbackSettings:
    enabled: bool = True
    reaction_scale: float = 1.0
    proxy_follow: float = 0.35
    max_proxy_offset: float = 0.02
    spring_k: float = 3.0
    damper_b: float = 0.08
    deadband: float = 0.0005
    lowpass_alpha: float = 0.35
    max_force: float = 0.05
    slew_rate_limit: float = 0.75
    vc_proxy_mass: float = 0.05
    tdpc_alpha: float = 1.0
    algorithm: str = "spring_damper"


@dataclass
class HapticFeedbackDiagnostics:
    current_preset_name: str = "Default"
    dirty: bool = False
    contact_count: int = 0
    avg_reaction_offset: np.ndarray = field(default_factory=_zero_vec3)
    proxy_offset: np.ndarray = field(default_factory=_zero_vec3)
    raw_force: np.ndarray = field(default_factory=_zero_vec3)
    filtered_force: np.ndarray = field(default_factory=_zero_vec3)
    final_force: np.ndarray = field(default_factory=_zero_vec3)
    clamp_active: bool = False
    slew_active: bool = False
    force_feedback_available: bool = False
    status_message: str = ""
    status_is_error: bool = False


@dataclass(frozen=True)
class HapticPresetRecord:
    path: Path
    name: str
    settings: HapticFeedbackSettings


def copy_feedback_settings(settings: HapticFeedbackSettings) -> HapticFeedbackSettings:
    return replace(settings)


def settings_to_dict(settings: HapticFeedbackSettings) -> dict[str, bool | float]:
    result: dict[str, bool | float] = {}
    for spec in fields(HapticFeedbackSettings):
        result[spec.name] = getattr(settings, spec.name)
    return result


def settings_almost_equal(a: HapticFeedbackSettings, b: HapticFeedbackSettings, atol: float = 1.0e-6) -> bool:
    for spec in fields(HapticFeedbackSettings):
        left = getattr(a, spec.name)
        right = getattr(b, spec.name)
        if isinstance(left, bool):
            if bool(left) != bool(right):
                return False
            continue
        if isinstance(left, str):
            if str(left) != str(right):
                return False
            continue
        if abs(float(left) - float(right)) > atol:
            return False
    return True


def sanitize_preset_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "preset"
    return f"{cleaned}.json"


def _coerce_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f'Preset field "{field_name}" must be a boolean')


def _coerce_float(value, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Preset field "{field_name}" must be numeric') from exc
    if not math.isfinite(numeric):
        raise ValueError(f'Preset field "{field_name}" must be finite')
    return numeric


def _coerce_str(value, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f'Preset field "{field_name}" must be a non-empty string')
    return value


def settings_from_dict(payload: dict) -> HapticFeedbackSettings:
    defaults = HapticFeedbackSettings()
    kwargs = {}
    for spec in fields(HapticFeedbackSettings):
        default_value = getattr(defaults, spec.name)
        raw_value = payload.get(spec.name, default_value)
        if isinstance(default_value, bool):
            kwargs[spec.name] = _coerce_bool(raw_value, spec.name)
        elif isinstance(default_value, str):
            kwargs[spec.name] = _coerce_str(raw_value, spec.name)
        else:
            kwargs[spec.name] = _coerce_float(raw_value, spec.name)
    return HapticFeedbackSettings(**kwargs)


def load_haptic_preset(path: Path) -> HapticPresetRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Preset "{path.name}" is not valid JSON') from exc

    version = payload.get("schema_version", HAPTIC_PRESET_SCHEMA_VERSION)
    if int(version) != HAPTIC_PRESET_SCHEMA_VERSION:
        raise ValueError(
            f'Preset "{path.name}" uses schema_version={version}, expected {HAPTIC_PRESET_SCHEMA_VERSION}'
        )

    name = str(payload.get("name") or path.stem)
    settings = settings_from_dict(payload)
    return HapticPresetRecord(path=path, name=name, settings=settings)


def scan_haptic_presets(preset_dir: Path) -> tuple[list[HapticPresetRecord], list[str]]:
    presets: list[HapticPresetRecord] = []
    errors: list[str] = []
    if not preset_dir.exists():
        return presets, errors

    for path in sorted(preset_dir.glob("*.json")):
        try:
            presets.append(load_haptic_preset(path))
        except ValueError as exc:
            errors.append(str(exc))
    presets.sort(key=lambda item: (item.name.lower(), item.path.name.lower()))
    return presets, errors


def save_haptic_preset(path: Path, name: str, settings: HapticFeedbackSettings) -> HapticPresetRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": HAPTIC_PRESET_SCHEMA_VERSION,
        "name": name,
        **settings_to_dict(settings),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return HapticPresetRecord(path=path, name=name, settings=copy_feedback_settings(settings))

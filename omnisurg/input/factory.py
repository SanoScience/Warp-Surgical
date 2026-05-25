from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from omnisurg.input.sources import FallbackInputSource, InputSource, MultiSourceRig, ReplayInputSource

CANONICAL_INPUT_ROLES = ("right", "left")
INPUT_BACKENDS = ("off", "fallback", "openhaptics", "minimou", "replay")
LEGACY_INPUT_BACKEND_ALIASES = {"none": "off"}


class InputOpenError(RuntimeError):
    """Raised when a required input source cannot be opened."""


@dataclass(frozen=True)
class RoleInputConfig:
    role: str
    backend: str = "off"
    device_name: str | None = None
    device_index: int | None = None
    replay_path: str | Path | None = None
    scale: float = 1.0
    force_feedback: bool = False


@dataclass
class InputOpenResult:
    sources: dict[str, InputSource] = field(default_factory=dict)
    descriptors: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def normalize_input_backend(value: str | None) -> str:
    backend = "off" if value is None else str(value).strip().lower()
    backend = LEGACY_INPUT_BACKEND_ALIASES.get(backend, backend)
    if backend not in INPUT_BACKENDS:
        choices = ", ".join(INPUT_BACKENDS)
        raise ValueError(f"unknown input backend {value!r}; expected one of: {choices}")
    return backend


def open_input_rig(
    configs: Sequence[RoleInputConfig] | Mapping[str, RoleInputConfig],
    *,
    require_all: bool = False,
) -> MultiSourceRig | None:
    result = open_input_sources(configs, require_all=require_all)
    if not result.sources:
        return None
    return MultiSourceRig(result.sources)


def open_input_sources(
    configs: Sequence[RoleInputConfig] | Mapping[str, RoleInputConfig],
    *,
    require_all: bool = False,
) -> InputOpenResult:
    ordered_configs = _ordered_configs(configs)
    result = InputOpenResult()
    opened: list[InputSource] = []

    try:
        for config in ordered_configs:
            backend = normalize_input_backend(config.backend)
            if backend == "off":
                continue

            try:
                source, descriptor = _open_one_source(config, backend=backend)
            except Exception as exc:  # noqa: BLE001
                message = _format_failure(config, backend, exc)
                result.failures.append(message)
                if require_all:
                    raise InputOpenError("; ".join(result.failures)) from exc
                continue

            result.sources[config.role] = source
            result.descriptors[config.role] = descriptor
            opened.append(source)
    except Exception:
        _close_sources_reverse(opened)
        raise

    return result


def _ordered_configs(
    configs: Sequence[RoleInputConfig] | Mapping[str, RoleInputConfig],
) -> list[RoleInputConfig]:
    if isinstance(configs, Mapping):
        by_role = dict(configs)
        ordered: list[RoleInputConfig] = []
        for role in CANONICAL_INPUT_ROLES:
            config = by_role.get(role)
            if config is not None:
                ordered.append(config)
        for role, config in by_role.items():
            if role not in CANONICAL_INPUT_ROLES:
                ordered.append(config)
        return ordered
    return list(configs)


def _open_one_source(config: RoleInputConfig, *, backend: str) -> tuple[InputSource, str]:
    if backend == "fallback":
        return FallbackInputSource(), "fallback"

    if backend == "replay":
        if config.replay_path is None:
            raise ValueError(f"{config.role} replay backend requires a replay path")
        path = str(config.replay_path)
        return ReplayInputSource(path), f"replay:{path}"

    if backend == "minimou":
        from omnisurg.haptics import LiveMiniMouSource

        index = 0 if config.device_index is None else int(config.device_index)
        return _construct_minimou_source(LiveMiniMouSource, index=index, scale=config.scale), f"minimou:{index}"

    if backend == "openhaptics":
        from omnisurg.haptics import LiveHapticSource

        requested_name = config.device_name or _default_device_name(config.role)
        last_exc: Exception | None = None
        for candidate_name in _device_name_candidates(requested_name):
            try:
                source = _construct_haptic_source(
                    LiveHapticSource,
                    device_name=candidate_name,
                    scale=config.scale,
                    force_feedback=config.force_feedback,
                )
                descriptor = f"openhaptics:{candidate_name}"
                if candidate_name != requested_name:
                    descriptor += f' (alias for "{requested_name}")'
                return source, descriptor
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    raise ValueError(f"unsupported input backend {backend!r}")


def _construct_haptic_source(source_cls, *, device_name: str, scale: float, force_feedback: bool):
    try:
        return source_cls(device_name=device_name, scale=scale, force_feedback=force_feedback)
    except TypeError as exc:
        if "force_feedback" not in str(exc):
            raise
        return source_cls(device_name=device_name, scale=scale)


def _construct_minimou_source(source_cls, *, index: int, scale: float):
    return source_cls(device_index=index, scale=scale)


def _device_name_candidates(device_name: str) -> tuple[str, ...]:
    aliases = {
        "Device Left": "Left Device",
        "Left Device": "Device Left",
    }
    candidates = [device_name]
    alias = aliases.get(device_name)
    if alias is not None and alias not in candidates:
        candidates.append(alias)
    return tuple(candidates)


def _default_device_name(role: str) -> str:
    return "Left Device" if role == "left" else "Default Device"


def _format_failure(config: RoleInputConfig, backend: str, exc: Exception) -> str:
    if backend == "openhaptics":
        detail = config.device_name or _default_device_name(config.role)
        return f"{config.role} ({backend}:{detail}): {exc}"
    if backend == "minimou":
        index = 0 if config.device_index is None else int(config.device_index)
        return f"{config.role} ({backend}#{index}): {exc}"
    if backend == "replay":
        return f"{config.role} ({backend}:{config.replay_path}): {exc}"
    return f"{config.role} ({backend}): {exc}"


def _close_sources_reverse(sources: Sequence[InputSource]) -> None:
    first_error: BaseException | None = None
    first_traceback = None
    for source in reversed(tuple(sources)):
        close = getattr(source, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc
                first_traceback = exc.__traceback__
    if first_error is not None:
        raise first_error.with_traceback(first_traceback)

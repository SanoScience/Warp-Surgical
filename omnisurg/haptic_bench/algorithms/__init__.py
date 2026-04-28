"""Haptic force-feedback algorithm registry.

Each algorithm implements `HapticAlgorithm` and is registered by name so the
live runtime and the offline bench can select it by string. Algorithms own
their own persistent per-step state (proxy position, filter memory, last
force command). Runtime calls `reset(device_position)` whenever it needs to
re-seed that state (session start, loss of tracking, zero-contact frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_REGISTRY: dict[str, type["HapticAlgorithm"]] = {}


def register(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f'Haptic algorithm "{name}" already registered')
        _REGISTRY[name] = cls
        cls.ALGORITHM_NAME = name
        return cls

    return decorator


def create(name: str, settings) -> "HapticAlgorithm":
    if name not in _REGISTRY:
        raise KeyError(
            f'Unknown haptic algorithm "{name}". Registered: {sorted(_REGISTRY)}'
        )
    return _REGISTRY[name](settings)


def list_algorithms() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class HapticStepInput:
    device_position: np.ndarray
    avg_reaction_offset: np.ndarray
    contact_count: int
    dt: float


@dataclass
class HapticStepOutput:
    force: np.ndarray
    raw_force: np.ndarray
    filtered_force: np.ndarray
    proxy_offset: np.ndarray
    clamp_active: bool
    slew_active: bool


class HapticAlgorithm:
    """Base class for haptic force-feedback algorithms."""

    ALGORITHM_NAME: str = "unknown"

    def __init__(self, settings):
        self.settings = settings

    def reset(self, device_position: np.ndarray | None) -> None:
        raise NotImplementedError

    def compute(self, step: HapticStepInput) -> HapticStepOutput:
        raise NotImplementedError

    def update_settings(self, settings) -> None:
        self.settings = settings


# Register built-in algorithms on import so `create(name)` works immediately.
from omnisurg.haptic_bench.algorithms import spring_damper  # noqa: E402,F401
from omnisurg.haptic_bench.algorithms import virtual_coupling  # noqa: E402,F401
from omnisurg.haptic_bench.algorithms import virtual_coupling_tdpc  # noqa: E402,F401

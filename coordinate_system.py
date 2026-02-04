"""
Coordinate system transformations.

This module defines a consistent coordinate system for the surgical simulation,
providing clear transformations between different spaces:

- Haptic space: Raw coordinates from the haptic device
- Simulation space: Physics simulation coordinates (meters)
- Render space: OpenGL rendering coordinates
- USD space: Universal Scene Description export coordinates

All positions in simulation space are in meters.
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class CoordinateSystem:
    """Defines coordinate system transformations.

    All scale factors are multiplied with positions to convert from the source
    space to the target space.
    """

    # Haptic device -> simulation space
    # Haptic devices typically output in mm, convert to simulation meters
    HAPTIC_TO_SIM_SCALE: float = 0.01

    # USD asset -> simulation space
    # USD models are typically in cm, convert to simulation meters
    USD_TO_SIM_SCALE: float = 0.02

    # Simulation -> OpenGL render space
    # Rendering at 1:1 scale
    SIM_TO_RENDER_SCALE: float = 1.0

    # Simulation -> USD export space
    # Export at a larger scale for visualization
    SIM_TO_USD_SCALE: float = 20.0

    # Haptic position offset (applied after scaling, in simulation space)
    HAPTIC_POSITION_OFFSET: Tuple[float, float, float] = (0.0, 1.0, -4.0)


def haptic_to_sim(position: List[float], coords: CoordinateSystem = None) -> List[float]:
    """Convert haptic device position to simulation space.

    Args:
        position: Position in haptic device coordinates [x, y, z].
        coords: Coordinate system to use (default if None).

    Returns:
        Position in simulation space [x, y, z].
    """
    if coords is None:
        coords = CoordinateSystem()

    scale = coords.HAPTIC_TO_SIM_SCALE
    offset = coords.HAPTIC_POSITION_OFFSET

    return [
        position[0] * scale + offset[0],
        position[1] * scale + offset[1],
        position[2] * scale + offset[2]
    ]


def sim_to_haptic(position: List[float], coords: CoordinateSystem = None) -> List[float]:
    """Convert simulation space position to haptic device coordinates.

    Args:
        position: Position in simulation space [x, y, z].
        coords: Coordinate system to use (default if None).

    Returns:
        Position in haptic device coordinates [x, y, z].
    """
    if coords is None:
        coords = CoordinateSystem()

    scale = coords.HAPTIC_TO_SIM_SCALE
    offset = coords.HAPTIC_POSITION_OFFSET

    return [
        (position[0] - offset[0]) / scale,
        (position[1] - offset[1]) / scale,
        (position[2] - offset[2]) / scale
    ]


def usd_to_sim(position: List[float], coords: CoordinateSystem = None) -> List[float]:
    """Convert USD asset position to simulation space.

    Args:
        position: Position in USD asset coordinates [x, y, z].
        coords: Coordinate system to use (default if None).

    Returns:
        Position in simulation space [x, y, z].
    """
    if coords is None:
        coords = CoordinateSystem()

    scale = coords.USD_TO_SIM_SCALE
    return [p * scale for p in position]


def sim_to_usd(position: List[float], coords: CoordinateSystem = None) -> List[float]:
    """Convert simulation space position to USD export coordinates.

    Args:
        position: Position in simulation space [x, y, z].
        coords: Coordinate system to use (default if None).

    Returns:
        Position in USD export coordinates [x, y, z].
    """
    if coords is None:
        coords = CoordinateSystem()

    scale = coords.SIM_TO_USD_SCALE
    return [p * scale for p in position]


def sim_to_render(position: List[float], coords: CoordinateSystem = None) -> List[float]:
    """Convert simulation space position to render coordinates.

    Args:
        position: Position in simulation space [x, y, z].
        coords: Coordinate system to use (default if None).

    Returns:
        Position in render coordinates [x, y, z].
    """
    if coords is None:
        coords = CoordinateSystem()

    scale = coords.SIM_TO_RENDER_SCALE
    return [p * scale for p in position]


def scale_matrix_for_usd(scale: float, coords: CoordinateSystem = None) -> np.ndarray:
    """Create a 4x4 scale matrix for USD coordinate transformation.

    Args:
        scale: Additional scale factor to apply.
        coords: Coordinate system to use (default if None).

    Returns:
        4x4 transformation matrix.
    """
    if coords is None:
        coords = CoordinateSystem()

    total_scale = coords.USD_TO_SIM_SCALE * scale
    return np.array([
        [total_scale, 0, 0, 0],
        [0, total_scale, 0, 0],
        [0, 0, total_scale, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def apply_haptic_offset_raw(
    position: List[float],
    offset: Tuple[float, float, float] = (0.0, 100.0, -400.0)
) -> List[float]:
    """Apply raw haptic offset (used before scaling).

    This preserves the legacy behavior where the offset is applied in haptic space.

    Args:
        position: Raw haptic device position [x, y, z].
        offset: Offset to apply (default matches legacy behavior).

    Returns:
        Offset position [x, y, z].
    """
    return [
        position[0] + offset[0],
        position[1] + offset[1],
        position[2] + offset[2]
    ]

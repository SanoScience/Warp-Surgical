# SPDX-License-Identifier: Apache-2.0
"""OmniSurg Hex application layer for labelled hexahedral tissue volumes."""

from ._version import __version__
from .deletion import DeviceDeletionResult
from .heat import HexHeatState, make_hex_heat_state
from .hex_grid import build_hex_particle_grid, build_hierarchical_shape_matching_clusters, build_shape_matching_clusters
from .shape_matching_solver import (
    HIERARCHICAL_SHAPE_MATCHING_FULL27,
    HIERARCHICAL_SHAPE_MATCHING_LABELS,
    HIERARCHICAL_SHAPE_MATCHING_OFF,
    HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    L2_HIERARCHICAL_SHAPE_MATCHING_LABELS,
    SHAPE_MATCHING_GS_WEIGHT_ALPHAS,
    SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
    SHAPE_MATCHING_GS_WEIGHT_FULL,
    SHAPE_MATCHING_GS_WEIGHT_LABELS,
    SHAPE_MATCHING_GS_WEIGHT_SQRT,
    SHAPE_MATCHING_SOLVE_COLORED_GS,
    SHAPE_MATCHING_SOLVE_GATHER,
    SHAPE_MATCHING_SOLVE_LABELS,
    SHAPE_MATCHING_SOLVE_SCATTER,
    HexShapeMatchingSolver,
)
from .data.types import PreparedVolume, PreprocessConfig
from .materials import Material, MaterialTable, digimouse_material_table


def __getattr__(name: str):
    if name == "OmniSurgHexApp":
        from .app import OmniSurgHexApp

        return OmniSurgHexApp
    if name in {"HexAppLauncher", "HexRuntime"}:
        from .runtime import HexAppLauncher, HexRuntime

        return {"HexAppLauncher": HexAppLauncher, "HexRuntime": HexRuntime}[name]
    raise AttributeError(name)

__all__ = [
    "HIERARCHICAL_SHAPE_MATCHING_FULL27",
    "HIERARCHICAL_SHAPE_MATCHING_LABELS",
    "HIERARCHICAL_SHAPE_MATCHING_OFF",
    "HIERARCHICAL_SHAPE_MATCHING_OUTER8",
    "L2_HIERARCHICAL_SHAPE_MATCHING_LABELS",
    "SHAPE_MATCHING_GS_WEIGHT_ALPHAS",
    "SHAPE_MATCHING_GS_WEIGHT_AVERAGED",
    "SHAPE_MATCHING_GS_WEIGHT_FULL",
    "SHAPE_MATCHING_GS_WEIGHT_LABELS",
    "SHAPE_MATCHING_GS_WEIGHT_SQRT",
    "SHAPE_MATCHING_SOLVE_COLORED_GS",
    "SHAPE_MATCHING_SOLVE_GATHER",
    "SHAPE_MATCHING_SOLVE_LABELS",
    "SHAPE_MATCHING_SOLVE_SCATTER",
    "HexHeatState",
    "DeviceDeletionResult",
    "HexAppLauncher",
    "HexRuntime",
    "Material",
    "MaterialTable",
    "OmniSurgHexApp",
    "PreparedVolume",
    "PreprocessConfig",
    "HexShapeMatchingSolver",
    "__version__",
    "build_hex_particle_grid",
    "build_hierarchical_shape_matching_clusters",
    "build_shape_matching_clusters",
    "digimouse_material_table",
    "make_hex_heat_state",
]

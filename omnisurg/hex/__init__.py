# SPDX-License-Identifier: Apache-2.0
"""OmniSurg Hex application layer for labelled hexahedral tissue volumes."""

from ._version import __version__
from .app import OmniSurgHexApp
from .corner_delete import DeviceDeletionResult
from .corner_heat import CornerHeatState, make_corner_heat_state
from .corner_grid import build_corner_l1_shape_matching_clusters, build_corner_shape_matching_clusters
from .corner_solver import (
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
    SolverCornerShapeMatching,
)
from .data.types import PreparedVolume, PreprocessConfig
from .materials import Material, MaterialTable, digimouse_material_table
from .runtime import HexAppLauncher

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
    "CornerHeatState",
    "DeviceDeletionResult",
    "HexAppLauncher",
    "Material",
    "MaterialTable",
    "OmniSurgHexApp",
    "PreparedVolume",
    "PreprocessConfig",
    "SolverCornerShapeMatching",
    "__version__",
    "build_corner_l1_shape_matching_clusters",
    "build_corner_shape_matching_clusters",
    "digimouse_material_table",
    "make_corner_heat_state",
]

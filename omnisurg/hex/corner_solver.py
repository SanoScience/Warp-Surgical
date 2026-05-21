# SPDX-License-Identifier: Apache-2.0
"""Particle-only corner-grid solver with optional shape matching."""

from __future__ import annotations

import numpy as np
import warp as wp
from newton._src.core.types import override
from newton._src.geometry.flags import ParticleFlags
from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers.solver import SolverBase
from newton._src.solvers.xpbd.kernels import (
    apply_particle_deltas,
    solve_particle_particle_contacts,
    solve_springs,
)

from .corner_grid import HierarchicalShapeMatchingClusters, ShapeMatchingClusters
from .kernels.grab import project_grab_distance_constraints
from .kernels.ground_plane import solve_particle_ground_plane_contacts
from .kernels.instrument import (
    project_kinematic_sphere_mc_triangle_node_positions,
    project_kinematic_sphere_particle_positions,
)
from .kernels.shape_matching import (
    accumulate_runtime_cluster_counts_kernel,
    apply_shape_matching_particle_gather_uniform8,
    compute_coarse_sleep_projection_kernel,
    compute_l0_runtime_active_kernel,
    compute_shape_matching_cluster_poses_uniform8,
    finalize_position_update_from_q,
    finalize_runtime_cluster_inv_weights_kernel,
    mark_wake_blocks_from_deleted_cells_kernel,
    mask_l1_projection_covered_by_l2_kernel,
    project_shape_matching_children_uniform8,
    project_shape_matching_children_uniform8_table,
    prolongate_shape_matching_corrections_uniform8,
    prolongate_shape_matching_corrections_uniform8_table,
    solve_shape_matching_clusters_uniform8,
    solve_shape_matching_clusters_uniform8_colored_gs,
    solve_shape_matching_clusters_uniform8_colored_gs_template_active,
    solve_shape_matching_clusters_uniform8_template_active,
    solve_shape_matching_clusters_uniform27,
    solve_shape_matching_clusters_uniform27_colored_gs,
    solve_shape_matching_clusters_uniform125,
    solve_shape_matching_clusters_uniform125_colored_gs,
    solve_volume_constraints_uniform8,
)

HIERARCHICAL_SHAPE_MATCHING_OFF = 0
HIERARCHICAL_SHAPE_MATCHING_OUTER8 = 1
HIERARCHICAL_SHAPE_MATCHING_FULL27 = 2
HIERARCHICAL_SHAPE_MATCHING_LABELS = ("Off", "Outer8", "Full27")
L2_HIERARCHICAL_SHAPE_MATCHING_LABELS = ("Off", "Outer8", "Full125")
SHAPE_MATCHING_SOLVE_SCATTER = 0
SHAPE_MATCHING_SOLVE_GATHER = 1
SHAPE_MATCHING_SOLVE_COLORED_GS = 2
SHAPE_MATCHING_SOLVE_LABELS = ("Scatter Jacobi", "Gather Jacobi", "Colored GS")
SHAPE_MATCHING_GS_WEIGHT_AVERAGED = 0
SHAPE_MATCHING_GS_WEIGHT_SQRT = 1
SHAPE_MATCHING_GS_WEIGHT_FULL = 2
SHAPE_MATCHING_GS_WEIGHT_LABELS = ("Averaged", "Sqrt", "Full")
SHAPE_MATCHING_GS_WEIGHT_ALPHAS = (1.0, 0.5, 0.0)
KINEMATIC_SPHERE_CONTACT_PARTICLES = 0
KINEMATIC_SPHERE_CONTACT_MC_TRIANGLES = 1


class SolverCornerShapeMatching(SolverBase):
    """Local particle solver for corner grids with springs and shape matching."""

    def __init__(
        self,
        model: Model,
        clusters: ShapeMatchingClusters,
        *,
        iterations: int = 4,
        enable_springs: bool = True,
        enable_shape_matching: bool = True,
        enable_self_collisions: bool = False,
        enable_ground_plane: bool = False,
        shape_matching_stiffness: float = 0.5,
        shape_matching_passes: int = 1,
        shape_matching_use_gather: bool = False,
        shape_matching_mode: int | None = None,
        shape_matching_relaxation: float = 1.0,
        shape_matching_gs_weighting: int = SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
        shape_matching_gs_support_alpha: float = -1.0,
        shape_matching_use_computed_prolongation: bool = True,
        enable_volume_preservation: bool = False,
        volume_preservation_stiffness: float = 0.0,
        volume_preservation_passes: int = 1,
        hierarchy: HierarchicalShapeMatchingClusters | None = None,
        hierarchical_shape_matching_mode: int = HIERARCHICAL_SHAPE_MATCHING_OFF,
        hierarchical_shape_matching_stiffness: float = 0.35,
        hierarchical_shape_matching_relaxation: float = 1.0,
        hierarchical_shape_matching_passes: int = 1,
        hierarchical_shape_matching_use_gs: bool = False,
        hierarchical_shape_matching_outer8_prolongation: bool = True,
        hierarchical_shape_matching_outer8_absolute_projection: bool = False,
        l2_hierarchical_shape_matching_mode: int = HIERARCHICAL_SHAPE_MATCHING_OFF,
        l2_hierarchical_shape_matching_stiffness: float = 0.25,
        l2_hierarchical_shape_matching_relaxation: float = 1.0,
        l2_hierarchical_shape_matching_passes: int = 1,
        l2_hierarchical_shape_matching_use_gs: bool = False,
        l2_hierarchical_shape_matching_outer8_prolongation: bool = True,
        l2_hierarchical_shape_matching_outer8_absolute_projection: bool = False,
        sleep_l0_shape_matching: bool = False,
        sleep_l0_wake_halo_blocks: int = 1,
        rotation_iterations: int = 8,
        ground_height: float = 0.0,
        ground_contact_relaxation: float = 0.9,
    ) -> None:
        if model.body_count != 0:
            raise ValueError(f"SolverCornerShapeMatching only supports particle-only models; got body_count={model.body_count}")
        if model.shape_count != 0:
            raise ValueError(
                f"SolverCornerShapeMatching does not support rigid/shape contacts in v1; got shape_count={model.shape_count}"
            )
        if model.edge_count != 0 or model.tet_count != 0 or model.tri_count != 0 or model.joint_count != 0:
            raise ValueError("SolverCornerShapeMatching only supports particles, springs, self-collision, and shape-matching clusters")
        if clusters.uniform_size != 8:
            raise NotImplementedError(f"only uniform_size=8 clusters are supported in v1, got {clusters.uniform_size}")
        if clusters.num_clusters == 0:
            raise ValueError("shape-matching cluster sidecar is empty")
        cluster_arrays = (
            clusters.offsets,
            clusters.indices,
            clusters.indices_by_slot,
            clusters.rest_centers,
            clusters.rest_local_positions,
            clusters.rest_local_positions_by_slot,
            clusters.rest_local_template,
            clusters.coefficients,
            clusters.active,
            clusters.colors,
            clusters.color_offsets,
            clusters.color_cluster_indices,
            clusters.source_cell,
            clusters.cell_to_cluster,
            clusters.particle_cluster_counts,
            clusters.particle_cluster_inv_weights,
            clusters.particle_cluster_offsets,
            clusters.particle_cluster_indices,
            clusters.particle_cluster_member_offsets,
        )
        for cluster_array in cluster_arrays:
            if cluster_array.device != model.device:
                raise ValueError(
                    "shape-matching cluster buffers must live on the same device as the model; "
                    f"got {cluster_array.device} and {model.device}"
                )

        super().__init__(model=model)

        self.clusters = clusters
        self.iterations = int(iterations)
        self.enable_springs = bool(enable_springs)
        self.enable_shape_matching = bool(enable_shape_matching)
        self.enable_self_collisions = bool(enable_self_collisions)
        self.enable_ground_plane = bool(enable_ground_plane)
        self.shape_matching_stiffness = float(shape_matching_stiffness)
        self.shape_matching_passes = int(shape_matching_passes)
        if shape_matching_mode is None:
            shape_matching_mode = SHAPE_MATCHING_SOLVE_GATHER if shape_matching_use_gather else SHAPE_MATCHING_SOLVE_SCATTER
        self.shape_matching_mode = int(shape_matching_mode)
        self.shape_matching_relaxation = float(shape_matching_relaxation)
        self.shape_matching_gs_weighting = int(shape_matching_gs_weighting)
        self.shape_matching_gs_support_alpha = float(shape_matching_gs_support_alpha)
        self.shape_matching_use_computed_prolongation = bool(shape_matching_use_computed_prolongation)
        self.enable_volume_preservation = bool(enable_volume_preservation)
        self.volume_preservation_stiffness = float(volume_preservation_stiffness)
        self.volume_preservation_passes = int(volume_preservation_passes)
        self.hierarchy = hierarchy
        self.hierarchical_shape_matching_mode = int(hierarchical_shape_matching_mode)
        self.hierarchical_shape_matching_stiffness = float(hierarchical_shape_matching_stiffness)
        self.hierarchical_shape_matching_relaxation = float(hierarchical_shape_matching_relaxation)
        self.hierarchical_shape_matching_passes = int(hierarchical_shape_matching_passes)
        self.hierarchical_shape_matching_use_gs = bool(hierarchical_shape_matching_use_gs)
        self.hierarchical_shape_matching_outer8_prolongation = bool(hierarchical_shape_matching_outer8_prolongation)
        self.hierarchical_shape_matching_outer8_absolute_projection = bool(
            hierarchical_shape_matching_outer8_absolute_projection
        )
        self.l2_hierarchical_shape_matching_mode = int(l2_hierarchical_shape_matching_mode)
        self.l2_hierarchical_shape_matching_stiffness = float(l2_hierarchical_shape_matching_stiffness)
        self.l2_hierarchical_shape_matching_relaxation = float(l2_hierarchical_shape_matching_relaxation)
        self.l2_hierarchical_shape_matching_passes = int(l2_hierarchical_shape_matching_passes)
        self.l2_hierarchical_shape_matching_use_gs = bool(l2_hierarchical_shape_matching_use_gs)
        self.l2_hierarchical_shape_matching_outer8_prolongation = bool(l2_hierarchical_shape_matching_outer8_prolongation)
        self.l2_hierarchical_shape_matching_outer8_absolute_projection = bool(
            l2_hierarchical_shape_matching_outer8_absolute_projection
        )
        self.sleep_l0_shape_matching = bool(sleep_l0_shape_matching)
        self.sleep_l0_wake_halo_blocks = max(0, int(sleep_l0_wake_halo_blocks))
        self.rotation_iterations = int(rotation_iterations)
        self.ground_height = float(ground_height)
        self.ground_contact_relaxation = float(ground_contact_relaxation)
        self.kinematic_sphere_q_prev = None
        self.kinematic_sphere_q = None
        self.kinematic_sphere_count = 0
        self.kinematic_sphere_radius = 0.0
        self.kinematic_sphere_interpolation_alpha = 1.0
        self.kinematic_sphere_contact_relaxation = 0.9
        self.kinematic_sphere_contact_iterations = 1
        self.kinematic_sphere_max_correction = 0.0
        self.kinematic_sphere_contact_mode = KINEMATIC_SPHERE_CONTACT_PARTICLES
        self.kinematic_sphere_mc_cell_nodes = None
        self.kinematic_sphere_mc_cell_active = None
        self.kinematic_sphere_mc_vertex_pos = None
        self.kinematic_sphere_mc_tri_indices = None
        self.kinematic_sphere_mc_triangle_count = 0
        self.grab_distance_constraints = ()

        self.particle_q_init = wp.empty_like(model.particle_q)
        self.particle_deltas = wp.empty_like(model.particle_q)
        self.kinematic_sphere_delta_counts = wp.zeros(model.particle_count, dtype=wp.int32, device=model.device)
        self._particle_q_scratch = wp.empty_like(model.particle_q)
        self._particle_qd_scratch = wp.empty_like(model.particle_qd)
        self._hierarchy_q_snapshot = wp.empty_like(model.particle_q)
        self.spring_constraint_lambdas = (
            wp.empty(model.spring_count, dtype=wp.float32, device=model.device) if model.spring_count else None
        )

        identity_quats = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (clusters.num_clusters, 1))
        self.cluster_rotations = wp.array(identity_quats, dtype=wp.quat, device=model.device)
        self.cluster_translations = wp.array(
            clusters.rest_centers_host.astype(np.float32, copy=False),
            dtype=wp.vec3,
            device=model.device,
        )
        self.hierarchy_outer8_rotations = None
        self.hierarchy_outer8_translations = None
        self.hierarchy_full27_rotations = None
        self.hierarchy_full27_translations = None
        self.hierarchy_l2_outer8_rotations = None
        self.hierarchy_l2_outer8_translations = None
        self.hierarchy_l2_full125_rotations = None
        self.hierarchy_l2_full125_translations = None

        if hierarchy is not None:
            self._validate_hierarchy(hierarchy)
            if hierarchy.outer8 is not None:
                identity_quats = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (hierarchy.outer8.num_clusters, 1))
                self.hierarchy_outer8_rotations = wp.array(identity_quats, dtype=wp.quat, device=model.device)
                self.hierarchy_outer8_translations = wp.array(
                    hierarchy.outer8.rest_centers_host.astype(np.float32, copy=False),
                    dtype=wp.vec3,
                    device=model.device,
                )
            if hierarchy.full27 is not None:
                identity_quats = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (hierarchy.full27.num_clusters, 1))
                self.hierarchy_full27_rotations = wp.array(identity_quats, dtype=wp.quat, device=model.device)
                self.hierarchy_full27_translations = wp.array(
                    hierarchy.full27.rest_centers_host.astype(np.float32, copy=False),
                    dtype=wp.vec3,
                    device=model.device,
                )
            if hierarchy.l2_outer8 is not None:
                identity_quats = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (hierarchy.l2_outer8.num_clusters, 1))
                self.hierarchy_l2_outer8_rotations = wp.array(identity_quats, dtype=wp.quat, device=model.device)
                self.hierarchy_l2_outer8_translations = wp.array(
                    hierarchy.l2_outer8.rest_centers_host.astype(np.float32, copy=False),
                    dtype=wp.vec3,
                    device=model.device,
                )
            if hierarchy.l2_full125 is not None:
                identity_quats = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (hierarchy.l2_full125.num_clusters, 1))
                self.hierarchy_l2_full125_rotations = wp.array(identity_quats, dtype=wp.quat, device=model.device)
                self.hierarchy_l2_full125_translations = wp.array(
                    hierarchy.l2_full125.rest_centers_host.astype(np.float32, copy=False),
                    dtype=wp.vec3,
                    device=model.device,
                )

        self.l0_runtime_active_host = clusters.active_host.copy()
        self.l0_runtime_particle_cluster_counts_host = clusters.particle_cluster_counts_host.copy()
        self.l0_runtime_particle_cluster_inv_weights_host = clusters.particle_cluster_inv_weights_host.copy()
        self.l0_runtime_active = wp.array(self.l0_runtime_active_host, dtype=wp.int32, device=model.device)
        self.l0_runtime_particle_cluster_counts = wp.array(
            self.l0_runtime_particle_cluster_counts_host,
            dtype=wp.int32,
            device=model.device,
        )
        self.l0_runtime_particle_cluster_inv_weights = wp.array(
            self.l0_runtime_particle_cluster_inv_weights_host,
            dtype=wp.float32,
            device=model.device,
        )
        self._l0_fast_uniform8_active = self._compute_l0_fast_uniform8_possible()

        self.l1_sleep_projection_active_host = None
        self.l1_sleep_projection_active = None
        self.l1_sleep_projection_active_count = 0
        self.l1_sleep_projection_active_count_device = wp.zeros(1, dtype=wp.int32, device=model.device)
        self.l2_sleep_projection_active_host = None
        self.l2_sleep_projection_active = None
        self.l2_sleep_projection_active_count = 0
        self.l2_sleep_projection_active_count_device = wp.zeros(1, dtype=wp.int32, device=model.device)
        self.sleeping_l0_cluster_count_device = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._l1_sleep_wake_mask = None
        self._l2_sleep_wake_mask = None
        if hierarchy is not None and hierarchy.outer8 is not None:
            self.l1_sleep_projection_active_host = np.zeros(hierarchy.outer8.num_clusters, dtype=np.int32)
            self.l1_sleep_projection_active = wp.array(
                self.l1_sleep_projection_active_host,
                dtype=wp.int32,
                device=model.device,
            )
            if hierarchy.outer8_block_lookup_shape is not None:
                self._l1_sleep_wake_mask = wp.zeros(
                    hierarchy.outer8_block_lookup_shape,
                    dtype=wp.int32,
                    device=model.device,
                )
        if hierarchy is not None and hierarchy.l2_outer8 is not None:
            self.l2_sleep_projection_active_host = np.zeros(hierarchy.l2_outer8.num_clusters, dtype=np.int32)
            self.l2_sleep_projection_active = wp.array(
                self.l2_sleep_projection_active_host,
                dtype=wp.int32,
                device=model.device,
            )
            if hierarchy.l2_outer8_block_lookup_shape is not None:
                self._l2_sleep_wake_mask = wp.zeros(
                    hierarchy.l2_outer8_block_lookup_shape,
                    dtype=wp.int32,
                    device=model.device,
                )
        self._l1_sleep_wake_block_keys: set[tuple[int, int, int]] = set()
        self._l2_sleep_wake_block_keys: set[tuple[int, int, int]] = set()
        self._last_sleep_deleted_cells_device = None
        self._last_sleep_deleted_count_device = wp.zeros(1, dtype=wp.int32, device=model.device)
        self.sleeping_l0_cluster_count = 0
        self.refresh_l0_sleep_state()

        if model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

    @property
    def shape_matching_use_gather(self) -> bool:
        """Legacy boolean alias for the L0 gather/scatter Jacobi modes."""
        return int(self.shape_matching_mode) == SHAPE_MATCHING_SOLVE_GATHER

    @shape_matching_use_gather.setter
    def shape_matching_use_gather(self, enabled: bool) -> None:
        self.shape_matching_mode = SHAPE_MATCHING_SOLVE_GATHER if bool(enabled) else SHAPE_MATCHING_SOLVE_SCATTER

    def _l0_effective_shape_matching_stiffness(self) -> float:
        return float(self.shape_matching_stiffness) * float(self.shape_matching_relaxation)

    def _l1_effective_shape_matching_stiffness(self) -> float:
        return float(self.hierarchical_shape_matching_stiffness) * float(self.hierarchical_shape_matching_relaxation)

    def _l2_effective_shape_matching_stiffness(self) -> float:
        return float(self.l2_hierarchical_shape_matching_stiffness) * float(self.l2_hierarchical_shape_matching_relaxation)

    def _l0_shape_matching_mode(self) -> int:
        mode = int(self.shape_matching_mode)
        if mode < SHAPE_MATCHING_SOLVE_SCATTER or mode > SHAPE_MATCHING_SOLVE_COLORED_GS:
            mode = SHAPE_MATCHING_SOLVE_SCATTER
        self.shape_matching_mode = mode
        return mode

    def _shape_matching_gs_weighting(self) -> int:
        weighting = int(self.shape_matching_gs_weighting)
        if weighting < SHAPE_MATCHING_GS_WEIGHT_AVERAGED or weighting > SHAPE_MATCHING_GS_WEIGHT_FULL:
            weighting = SHAPE_MATCHING_GS_WEIGHT_AVERAGED
        self.shape_matching_gs_weighting = weighting
        return weighting

    def _shape_matching_gs_support_alpha(self, weighting: int) -> float:
        alpha = float(self.shape_matching_gs_support_alpha)
        if alpha < 0.0:
            return float(SHAPE_MATCHING_GS_WEIGHT_ALPHAS[int(weighting)])
        if alpha > 1.0:
            alpha = 1.0
        self.shape_matching_gs_support_alpha = alpha
        return alpha

    def _l0_volume_preservation_active(self) -> bool:
        return bool(
            self.enable_volume_preservation
            and self.volume_preservation_stiffness > 0.0
            and self.volume_preservation_passes > 0
        )

    def _compute_l0_fast_uniform8_possible(self) -> bool:
        if not self.clusters.rest_local_template_valid:
            return False
        if self.clusters.indices_host.size != self.clusters.num_clusters * self.clusters.uniform_size:
            return False
        particle_ids = np.unique(self.clusters.indices_host)
        if particle_ids.size == 0:
            return False
        flags = self.model.particle_flags.numpy()[particle_ids]
        inv_mass = self.model.particle_inv_mass.numpy()[particle_ids]
        return bool(
            np.all((flags & int(ParticleFlags.ACTIVE)) != 0)
            and np.all(inv_mass > 0.0)
        )

    @property
    def l0_fast_uniform8_active(self) -> bool:
        """Whether the topology-sensitive L0 uniform8 fast path can shape the next capture."""
        return bool(self._l0_fast_uniform8_active and self.clusters.rest_local_template_valid)

    def disable_l0_fast_uniform8_for_cutting(self) -> bool:
        """Disable the pre-cut L0 fast path and report whether graph shape changed."""
        was_active = self.l0_fast_uniform8_active
        self._l0_fast_uniform8_active = False
        return bool(was_active)

    def _validate_hierarchy(self, hierarchy: HierarchicalShapeMatchingClusters) -> None:
        if hierarchy.outer8 is not None and hierarchy.outer8.uniform_size != 8:
            raise NotImplementedError(f"outer8 hierarchy requires uniform_size=8, got {hierarchy.outer8.uniform_size}")
        if hierarchy.l2_outer8 is not None and hierarchy.l2_outer8.uniform_size != 8:
            raise NotImplementedError(f"L2 outer8 hierarchy requires uniform_size=8, got {hierarchy.l2_outer8.uniform_size}")
        if hierarchy.full27 is not None and hierarchy.full27.uniform_size != 27:
            raise NotImplementedError(f"full27 hierarchy requires uniform_size=27, got {hierarchy.full27.uniform_size}")
        if hierarchy.l2_full125 is not None and hierarchy.l2_full125.uniform_size != 125:
            raise NotImplementedError(f"L2 full125 hierarchy requires uniform_size=125, got {hierarchy.l2_full125.uniform_size}")

        for clusters in (hierarchy.outer8, hierarchy.full27, hierarchy.l2_outer8, hierarchy.l2_full125):
            if clusters is None:
                continue
            cluster_arrays = (
                clusters.offsets,
                clusters.indices,
                clusters.indices_by_slot,
                clusters.rest_centers,
                clusters.rest_local_positions,
                clusters.rest_local_positions_by_slot,
                clusters.rest_local_template,
                clusters.coefficients,
                clusters.active,
                clusters.colors,
                clusters.color_offsets,
                clusters.color_cluster_indices,
                clusters.source_cell,
                clusters.cell_to_cluster,
                clusters.particle_cluster_counts,
                clusters.particle_cluster_inv_weights,
                clusters.particle_cluster_offsets,
                clusters.particle_cluster_indices,
                clusters.particle_cluster_member_offsets,
            )
            for cluster_array in cluster_arrays:
                if cluster_array.device != self.model.device:
                    raise ValueError(
                        "hierarchical shape-matching buffers must live on the same device as the model; "
                        f"got {cluster_array.device} and {self.model.device}"
                    )

        for prolongation in (hierarchy.outer8_prolongation, hierarchy.l2_outer8_prolongation):
            if prolongation is None:
                continue
            for array in (
                prolongation.parent_indices,
                prolongation.parent_weights,
                prolongation.child_cluster,
                prolongation.node_grid_xyz,
                prolongation.grid_to_node,
            ):
                if array.device != self.model.device:
                    raise ValueError(
                        "shape-matching prolongation buffers must live on the same device as the model; "
                        f"got {array.device} and {self.model.device}"
                    )

        for array in (
            hierarchy.outer8_block_keys,
            hierarchy.l2_outer8_block_keys,
            hierarchy.outer8_sleepable,
            hierarchy.l2_outer8_sleepable,
            hierarchy.outer8_block_lookup,
            hierarchy.l2_outer8_block_lookup,
        ):
            if array is not None and array.device != self.model.device:
                raise ValueError(
                    "hierarchical sleep metadata buffers must live on the same device as the model; "
                    f"got {array.device} and {self.model.device}"
                )

    def _sleep_enabled_for_runtime_masks(self) -> bool:
        return bool(
            self.sleep_l0_shape_matching
            and self.enable_shape_matching
            and self._l0_effective_shape_matching_stiffness() > 0.0
            and self.shape_matching_passes > 0
            and self.hierarchy is not None
        )

    def _coarse_projection_mask_host(
        self,
        clusters: ShapeMatchingClusters | None,
        block_keys_host: np.ndarray | None,
        sleepable_host: np.ndarray | None,
        wake_block_keys: set[tuple[int, int, int]],
        level_active: bool,
    ) -> np.ndarray | None:
        if clusters is None:
            return None
        projection_active = np.zeros(clusters.num_clusters, dtype=np.int32)
        if (
            not self._sleep_enabled_for_runtime_masks()
            or not level_active
            or block_keys_host is None
            or block_keys_host.shape != (clusters.num_clusters, 3)
        ):
            return projection_active

        projection_active[:] = clusters.active_host.astype(np.int32, copy=False)
        if sleepable_host is not None and sleepable_host.shape == (clusters.num_clusters,):
            projection_active *= sleepable_host.astype(np.int32, copy=False)
        if wake_block_keys:
            woken = np.zeros(clusters.num_clusters, dtype=bool)
            for idx, key in enumerate(block_keys_host):
                woken[idx] = (int(key[0]), int(key[1]), int(key[2])) in wake_block_keys
            projection_active[woken] = 0
        return projection_active

    def _runtime_inv_weights_from_active(self, cluster_active_host: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        particle_cluster_counts = np.zeros(self.model.particle_count, dtype=np.int32)
        active_cluster_ids = np.nonzero(cluster_active_host != 0)[0]
        if active_cluster_ids.size > 0:
            if self.clusters.indices_host.size == self.clusters.num_clusters * self.clusters.uniform_size:
                member_indices = self.clusters.indices_host.reshape(self.clusters.num_clusters, self.clusters.uniform_size)[
                    active_cluster_ids
                ].reshape(-1)
                np.add.at(particle_cluster_counts, member_indices, 1)
            else:
                for cluster_idx in active_cluster_ids.tolist():
                    start = int(self.clusters.offsets_host[cluster_idx])
                    end = int(self.clusters.offsets_host[cluster_idx + 1])
                    np.add.at(particle_cluster_counts, self.clusters.indices_host[start:end], 1)

        particle_cluster_inv_weights = np.zeros(self.model.particle_count, dtype=np.float32)
        valid = particle_cluster_counts > 0
        particle_cluster_inv_weights[valid] = 1.0 / particle_cluster_counts[valid].astype(np.float32)
        return particle_cluster_counts, particle_cluster_inv_weights

    def _sync_l0_runtime_hosts(self) -> None:
        self.l0_runtime_active_host[:] = self.l0_runtime_active.numpy()
        self.l0_runtime_particle_cluster_counts_host[:] = self.l0_runtime_particle_cluster_counts.numpy()
        self.l0_runtime_particle_cluster_inv_weights_host[:] = self.l0_runtime_particle_cluster_inv_weights.numpy()
        self.sleeping_l0_cluster_count = int(self.sleeping_l0_cluster_count_device.numpy()[0])

        if self.l1_sleep_projection_active is not None and self.l1_sleep_projection_active_host is not None:
            self.l1_sleep_projection_active_host[:] = self.l1_sleep_projection_active.numpy()
            self.l1_sleep_projection_active_count = int(self.l1_sleep_projection_active_count_device.numpy()[0])
        else:
            self.l1_sleep_projection_active_count = 0

        if self.l2_sleep_projection_active is not None and self.l2_sleep_projection_active_host is not None:
            self.l2_sleep_projection_active_host[:] = self.l2_sleep_projection_active.numpy()
            self.l2_sleep_projection_active_count = int(self.l2_sleep_projection_active_count_device.numpy()[0])
        else:
            self.l2_sleep_projection_active_count = 0

    def refresh_l0_sleep_state(self, *, sync_host: bool = True) -> None:
        """Recompute runtime L0 masks after sleep or hierarchy mode changes."""
        self.sleeping_l0_cluster_count_device.zero_()
        self.l1_sleep_projection_active_count_device.zero_()
        self.l2_sleep_projection_active_count_device.zero_()

        if self.l1_sleep_projection_active is not None:
            self.l1_sleep_projection_active.zero_()
        if self.l2_sleep_projection_active is not None:
            self.l2_sleep_projection_active.zero_()

        hierarchy = self.hierarchy
        if not self._sleep_enabled_for_runtime_masks() or hierarchy is None:
            self.l0_runtime_active.assign(self.clusters.active)
            self.l0_runtime_particle_cluster_counts.assign(self.clusters.particle_cluster_counts)
            self.l0_runtime_particle_cluster_inv_weights.assign(self.clusters.particle_cluster_inv_weights)
            if sync_host:
                self._sync_l0_runtime_hosts()
            return

        l2_projection_available = int(
            hierarchy.l2_outer8 is not None
            and self.l2_sleep_projection_active is not None
            and hierarchy.l2_outer8_block_keys is not None
            and hierarchy.l2_outer8_sleepable is not None
            and self._l2_sleep_wake_mask is not None
        )
        l2_level_enabled = int(
            l2_projection_available != 0
            and int(self.l2_hierarchical_shape_matching_mode) == HIERARCHICAL_SHAPE_MATCHING_OUTER8
            and self._l2_effective_shape_matching_stiffness() > 0.0
            and self.l2_hierarchical_shape_matching_passes > 0
        )
        if l2_projection_available != 0:
            assert hierarchy.l2_outer8 is not None
            assert hierarchy.l2_outer8_block_keys is not None
            assert hierarchy.l2_outer8_sleepable is not None
            assert self.l2_sleep_projection_active is not None
            assert self._l2_sleep_wake_mask is not None
            wp.launch(
                compute_coarse_sleep_projection_kernel,
                dim=hierarchy.l2_outer8.num_clusters,
                inputs=[
                    hierarchy.l2_outer8.active,
                    hierarchy.l2_outer8_block_keys,
                    hierarchy.l2_outer8_sleepable,
                    self._l2_sleep_wake_mask,
                    l2_level_enabled,
                    self.l2_sleep_projection_active,
                    self.l2_sleep_projection_active_count_device,
                ],
                device=self.model.device,
            )

        l1_projection_available = int(
            hierarchy.outer8 is not None
            and self.l1_sleep_projection_active is not None
            and hierarchy.outer8_block_keys is not None
            and hierarchy.outer8_sleepable is not None
            and self._l1_sleep_wake_mask is not None
        )
        l1_level_enabled = int(
            l1_projection_available != 0
            and int(self.hierarchical_shape_matching_mode) == HIERARCHICAL_SHAPE_MATCHING_OUTER8
            and self._l1_effective_shape_matching_stiffness() > 0.0
            and self.hierarchical_shape_matching_passes > 0
        )
        if l1_projection_available != 0:
            assert hierarchy.outer8 is not None
            assert hierarchy.outer8_block_keys is not None
            assert hierarchy.outer8_sleepable is not None
            assert self.l1_sleep_projection_active is not None
            assert self._l1_sleep_wake_mask is not None
            wp.launch(
                compute_coarse_sleep_projection_kernel,
                dim=hierarchy.outer8.num_clusters,
                inputs=[
                    hierarchy.outer8.active,
                    hierarchy.outer8_block_keys,
                    hierarchy.outer8_sleepable,
                    self._l1_sleep_wake_mask,
                    l1_level_enabled,
                    self.l1_sleep_projection_active,
                    self.l1_sleep_projection_active_count_device,
                ],
                device=self.model.device,
            )
            if l2_projection_available != 0 and hierarchy.l2_outer8 is not None and self.l2_sleep_projection_active is not None:
                wp.launch(
                    mask_l1_projection_covered_by_l2_kernel,
                    dim=hierarchy.outer8.num_clusters,
                    inputs=[
                        hierarchy.outer8.source_cell,
                        hierarchy.l2_outer8.cell_to_cluster,
                        self.l2_sleep_projection_active,
                        self.l1_sleep_projection_active,
                        self.l1_sleep_projection_active_count_device,
                    ],
                    device=self.model.device,
                )

        self.l0_runtime_active.zero_()
        wp.launch(
            compute_l0_runtime_active_kernel,
            dim=self.clusters.num_clusters,
            inputs=[
                self.clusters.active,
                self.clusters.source_cell,
                hierarchy.outer8.cell_to_cluster if hierarchy.outer8 is not None else self.clusters.cell_to_cluster,
                self.l1_sleep_projection_active if self.l1_sleep_projection_active is not None else self.clusters.active,
                int(l1_projection_available != 0),
                hierarchy.l2_outer8.cell_to_cluster if hierarchy.l2_outer8 is not None else self.clusters.cell_to_cluster,
                self.l2_sleep_projection_active if self.l2_sleep_projection_active is not None else self.clusters.active,
                int(l2_projection_available != 0),
                self.l0_runtime_active,
                self.sleeping_l0_cluster_count_device,
            ],
            device=self.model.device,
        )
        self.l0_runtime_particle_cluster_counts.zero_()
        self.l0_runtime_particle_cluster_inv_weights.zero_()
        wp.launch(
            accumulate_runtime_cluster_counts_kernel,
            dim=self.clusters.num_clusters,
            inputs=[
                self.l0_runtime_active,
                self.clusters.offsets,
                self.clusters.indices,
                self.l0_runtime_particle_cluster_counts,
            ],
            device=self.model.device,
        )
        wp.launch(
            finalize_runtime_cluster_inv_weights_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.l0_runtime_particle_cluster_counts,
                self.l0_runtime_particle_cluster_inv_weights,
            ],
            device=self.model.device,
        )

        if sync_host:
            self._sync_l0_runtime_hosts()

    def _record_level_wake_blocks(
        self,
        clusters: ShapeMatchingClusters | None,
        block_keys_host: np.ndarray | None,
        deleted_cells: np.ndarray,
        wake_block_keys: set[tuple[int, int, int]],
    ) -> None:
        if clusters is None or block_keys_host is None or block_keys_host.shape != (clusters.num_clusters, 3):
            return
        valid_cells = deleted_cells[(deleted_cells >= 0) & (deleted_cells < clusters.cell_to_cluster_host.shape[0])]
        if valid_cells.size == 0:
            return
        cluster_ids = clusters.cell_to_cluster_host[valid_cells]
        cluster_ids = np.unique(cluster_ids[cluster_ids >= 0])
        if cluster_ids.size == 0:
            return

        halo = max(0, int(self.sleep_l0_wake_halo_blocks))
        for key in block_keys_host[cluster_ids]:
            bx, by, bz = (int(key[0]), int(key[1]), int(key[2]))
            for dx in range(-halo, halo + 1):
                for dy in range(-halo, halo + 1):
                    for dz in range(-halo, halo + 1):
                        wake_block_keys.add((bx + dx, by + dy, bz + dz))

    def update_l0_sleep_after_deletion(self, deleted_cells) -> bool:
        """Persistently wake cut coarse blocks plus their Chebyshev halo."""
        deleted_cells_np = np.asarray(deleted_cells, dtype=np.int32).reshape(-1)
        if deleted_cells_np.size == 0:
            self.refresh_l0_sleep_state()
            return False
        self._last_sleep_deleted_cells_device = wp.array(
            deleted_cells_np,
            dtype=wp.int32,
            device=self.model.device,
        )
        self._last_sleep_deleted_count_device.fill_(int(deleted_cells_np.size))
        return self.update_l0_sleep_after_deletion_device(
            self._last_sleep_deleted_cells_device,
            self._last_sleep_deleted_count_device,
            int(deleted_cells_np.size),
            sync_host=True,
        )

    def update_l0_sleep_after_deletion_device(
        self,
        deleted_cells_device: wp.array | None,
        deleted_count_device: wp.array | None,
        candidate_capacity: int,
        *,
        sync_host: bool = False,
    ) -> bool:
        """Persistently wake cut coarse blocks from a compact device deletion result."""
        hierarchy = self.hierarchy
        candidate_capacity = int(candidate_capacity)
        if deleted_cells_device is None or deleted_count_device is None or candidate_capacity <= 0:
            self.refresh_l0_sleep_state(sync_host=sync_host)
            return False

        graph_shape_changed = self.disable_l0_fast_uniform8_for_cutting()
        if hierarchy is not None:
            halo = max(0, int(self.sleep_l0_wake_halo_blocks))
            if (
                hierarchy.outer8 is not None
                and hierarchy.outer8_block_keys is not None
                and self._l1_sleep_wake_mask is not None
            ):
                wp.launch(
                    mark_wake_blocks_from_deleted_cells_kernel,
                    dim=candidate_capacity,
                    inputs=[
                        deleted_cells_device,
                        deleted_count_device,
                        int(hierarchy.outer8.cell_to_cluster.shape[0]),
                        hierarchy.outer8.cell_to_cluster,
                        hierarchy.outer8_block_keys,
                        halo,
                        self._l1_sleep_wake_mask,
                    ],
                    device=self.model.device,
                )
            if (
                hierarchy.l2_outer8 is not None
                and hierarchy.l2_outer8_block_keys is not None
                and self._l2_sleep_wake_mask is not None
            ):
                wp.launch(
                    mark_wake_blocks_from_deleted_cells_kernel,
                    dim=candidate_capacity,
                    inputs=[
                        deleted_cells_device,
                        deleted_count_device,
                        int(hierarchy.l2_outer8.cell_to_cluster.shape[0]),
                        hierarchy.l2_outer8.cell_to_cluster,
                        hierarchy.l2_outer8_block_keys,
                        halo,
                        self._l2_sleep_wake_mask,
                    ],
                    device=self.model.device,
                )
        if sync_host or self._sleep_enabled_for_runtime_masks():
            self.refresh_l0_sleep_state(sync_host=sync_host)
        return bool(graph_shape_changed)

    def _reset_pose_buffer(self, clusters, rotations, translations) -> None:
        if clusters is None or rotations is None or translations is None:
            return
        identity_quats = np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (clusters.num_clusters, 1),
        )
        rotations.assign(identity_quats)
        translations.assign(clusters.rest_centers_host.astype(np.float32, copy=False))

    def reset_runtime_state(self) -> None:
        """Clear deletion/sleep runtime caches after restoring the topology."""
        self._l0_fast_uniform8_active = self._compute_l0_fast_uniform8_possible()
        if self._l1_sleep_wake_mask is not None:
            self._l1_sleep_wake_mask.zero_()
        if self._l2_sleep_wake_mask is not None:
            self._l2_sleep_wake_mask.zero_()
        self._l1_sleep_wake_block_keys.clear()
        self._l2_sleep_wake_block_keys.clear()
        self._last_sleep_deleted_cells_device = None
        self._last_sleep_deleted_count_device.zero_()

        self._reset_pose_buffer(self.clusters, self.cluster_rotations, self.cluster_translations)
        hierarchy = self.hierarchy
        if hierarchy is not None:
            self._reset_pose_buffer(hierarchy.outer8, self.hierarchy_outer8_rotations, self.hierarchy_outer8_translations)
            self._reset_pose_buffer(hierarchy.full27, self.hierarchy_full27_rotations, self.hierarchy_full27_translations)
            self._reset_pose_buffer(
                hierarchy.l2_outer8,
                self.hierarchy_l2_outer8_rotations,
                self.hierarchy_l2_outer8_translations,
            )
            self._reset_pose_buffer(
                hierarchy.l2_full125,
                self.hierarchy_l2_full125_rotations,
                self.hierarchy_l2_full125_translations,
            )

        self.refresh_l0_sleep_state()

    def _l0_active_array(self):
        if self._sleep_enabled_for_runtime_masks():
            return self.l0_runtime_active
        return self.clusters.active

    def _l0_particle_cluster_inv_weights_array(self):
        if self._sleep_enabled_for_runtime_masks():
            return self.l0_runtime_particle_cluster_inv_weights
        return self.clusters.particle_cluster_inv_weights

    def _project_l0_sleeping_children(
        self,
        current_q: wp.array,
        current_qd: wp.array,
        next_q: wp.array,
        next_qd: wp.array,
        dt: float,
    ):
        if not self._sleep_enabled_for_runtime_masks():
            return current_q, current_qd, next_q, next_qd

        hierarchy = self.hierarchy
        assert hierarchy is not None
        l0_stiffness = self._l0_effective_shape_matching_stiffness()

        if (
            hierarchy.l2_outer8_prolongation is not None
            and self.l2_sleep_projection_active is not None
        ):
            if self.shape_matching_use_computed_prolongation:
                assert hierarchy.l2_outer8_block_keys is not None
                wp.launch(
                    kernel=project_shape_matching_children_uniform8,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        self.l0_runtime_particle_cluster_counts,
                        hierarchy.l2_outer8_prolongation.child_cluster,
                        hierarchy.l2_outer8_block_keys,
                        hierarchy.l2_outer8_prolongation.node_grid_xyz,
                        hierarchy.l2_outer8_prolongation.grid_to_node,
                        hierarchy.l2_outer8_prolongation.block_size,
                        hierarchy.l2_outer8_prolongation.max_grid_coord[0],
                        hierarchy.l2_outer8_prolongation.max_grid_coord[1],
                        hierarchy.l2_outer8_prolongation.max_grid_coord[2],
                        self.l2_sleep_projection_active,
                        l0_stiffness,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
            else:
                wp.launch(
                    kernel=project_shape_matching_children_uniform8_table,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        self.l0_runtime_particle_cluster_counts,
                        hierarchy.l2_outer8_prolongation.parent_indices,
                        hierarchy.l2_outer8_prolongation.parent_weights,
                        hierarchy.l2_outer8_prolongation.child_cluster,
                        self.l2_sleep_projection_active,
                        l0_stiffness,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
            current_q, next_q = next_q, current_q
            current_qd, next_qd = next_qd, current_qd

        if (
            hierarchy.outer8_prolongation is not None
            and self.l1_sleep_projection_active is not None
        ):
            if self.shape_matching_use_computed_prolongation:
                assert hierarchy.outer8_block_keys is not None
                wp.launch(
                    kernel=project_shape_matching_children_uniform8,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        self.l0_runtime_particle_cluster_counts,
                        hierarchy.outer8_prolongation.child_cluster,
                        hierarchy.outer8_block_keys,
                        hierarchy.outer8_prolongation.node_grid_xyz,
                        hierarchy.outer8_prolongation.grid_to_node,
                        hierarchy.outer8_prolongation.block_size,
                        hierarchy.outer8_prolongation.max_grid_coord[0],
                        hierarchy.outer8_prolongation.max_grid_coord[1],
                        hierarchy.outer8_prolongation.max_grid_coord[2],
                        self.l1_sleep_projection_active,
                        l0_stiffness,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
            else:
                wp.launch(
                    kernel=project_shape_matching_children_uniform8_table,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        self.l0_runtime_particle_cluster_counts,
                        hierarchy.outer8_prolongation.parent_indices,
                        hierarchy.outer8_prolongation.parent_weights,
                        hierarchy.outer8_prolongation.child_cluster,
                        self.l1_sleep_projection_active,
                        l0_stiffness,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
            current_q, next_q = next_q, current_q
            current_qd, next_qd = next_qd, current_qd

        return current_q, current_qd, next_q, next_qd

    def _run_l0_volume_preservation(
        self,
        current_q: wp.array,
        current_qd: wp.array,
        next_q: wp.array,
        next_qd: wp.array,
        dt: float,
    ):
        if not self._l0_volume_preservation_active():
            return current_q, current_qd, next_q, next_qd

        for _ in range(max(0, int(self.volume_preservation_passes))):
            self.particle_deltas.zero_()
            wp.launch(
                kernel=solve_volume_constraints_uniform8,
                dim=self.clusters.num_clusters,
                inputs=[
                    current_q,
                    self.model.particle_inv_mass,
                    self.model.particle_flags,
                    self.clusters.indices_by_slot,
                    self.clusters.rest_local_positions_by_slot,
                    self.clusters.rest_local_template,
                    self.clusters.coefficients,
                    self._l0_active_array(),
                    self._l0_particle_cluster_inv_weights_array(),
                    self.clusters.num_clusters,
                    int(self.clusters.rest_local_template_valid),
                    float(self.volume_preservation_stiffness),
                ],
                outputs=[self.particle_deltas],
                device=self.model.device,
            )
            wp.launch(
                kernel=apply_particle_deltas,
                dim=self.model.particle_count,
                inputs=[
                    self.particle_q_init,
                    current_q,
                    self.model.particle_flags,
                    self.particle_deltas,
                    dt,
                    self.model.particle_max_velocity,
                ],
                outputs=[next_q, next_qd],
                device=self.model.device,
            )
            current_q, next_q = next_q, current_q
            current_qd, next_qd = next_qd, current_qd
            current_q, current_qd, next_q, next_qd = self._project_l0_sleeping_children(
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
            )

        return current_q, current_qd, next_q, next_qd

    def _run_outer8_shape_matching_level(
        self,
        clusters: ShapeMatchingClusters | None,
        prolongation,
        block_keys,
        cluster_rotations,
        cluster_translations,
        stiffness: float,
        passes: int,
        current_q: wp.array,
        current_qd: wp.array,
        next_q: wp.array,
        next_qd: wp.array,
        dt: float,
        use_gs: bool = False,
        gs_support_alpha: float = 1.0,
        use_prolongation: bool = True,
        absolute_projection: bool = False,
    ):
        if clusters is None or prolongation is None or stiffness <= 0.0 or passes <= 0:
            return current_q, current_qd, next_q, next_qd
        assert cluster_rotations is not None
        assert cluster_translations is not None

        for shape_pass in range(max(0, int(passes))):
            self._hierarchy_q_snapshot.assign(current_q)
            if use_gs:
                if (shape_pass & 1) == 0:
                    color_range = range(8)
                else:
                    color_range = range(7, -1, -1)
                for color in color_range:
                    color_start = int(clusters.color_offsets_host[color])
                    color_end = int(clusters.color_offsets_host[color + 1])
                    color_count = color_end - color_start
                    if color_count <= 0:
                        continue
                    wp.launch(
                        kernel=solve_shape_matching_clusters_uniform8_colored_gs,
                        dim=color_count,
                        inputs=[
                            current_q,
                            self.model.particle_inv_mass,
                            self.model.particle_flags,
                            clusters.color_cluster_indices,
                            clusters.indices_by_slot,
                            clusters.rest_local_positions_by_slot,
                            clusters.rest_local_template,
                            clusters.coefficients,
                            clusters.active,
                            clusters.particle_cluster_inv_weights,
                            clusters.num_clusters,
                            int(clusters.rest_local_template_valid),
                            color_start,
                            stiffness,
                            gs_support_alpha,
                            self.rotation_iterations,
                        ],
                        outputs=[cluster_rotations],
                        device=self.model.device,
                    )
                wp.launch(
                    kernel=finalize_position_update_from_q,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_flags,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    device=self.model.device,
                )
            else:
                self.particle_deltas.zero_()
                wp.launch(
                    kernel=solve_shape_matching_clusters_uniform8,
                    dim=clusters.num_clusters,
                    inputs=[
                        current_q,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        clusters.indices_by_slot,
                        clusters.rest_local_positions_by_slot,
                        clusters.rest_local_template,
                        clusters.coefficients,
                        clusters.active,
                        clusters.particle_cluster_inv_weights,
                        clusters.num_clusters,
                        int(clusters.rest_local_template_valid),
                        stiffness,
                        self.rotation_iterations,
                    ],
                    outputs=[cluster_rotations, cluster_translations, self.particle_deltas],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=apply_particle_deltas,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        self.model.particle_flags,
                        self.particle_deltas,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
                current_q, next_q = next_q, current_q
                current_qd, next_qd = next_qd, current_qd

            if not use_prolongation:
                continue

            if self.shape_matching_use_computed_prolongation:
                assert block_keys is not None
                wp.launch(
                    kernel=prolongate_shape_matching_corrections_uniform8,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        self._hierarchy_q_snapshot,
                        prolongation.child_cluster,
                        block_keys,
                        prolongation.node_grid_xyz,
                        prolongation.grid_to_node,
                        prolongation.block_size,
                        prolongation.max_grid_coord[0],
                        prolongation.max_grid_coord[1],
                        prolongation.max_grid_coord[2],
                        clusters.active,
                        int(bool(absolute_projection)),
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
            else:
                wp.launch(
                    kernel=prolongate_shape_matching_corrections_uniform8_table,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        self._hierarchy_q_snapshot,
                        prolongation.parent_indices,
                        prolongation.parent_weights,
                        prolongation.child_cluster,
                        clusters.active,
                        int(bool(absolute_projection)),
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
            current_q, next_q = next_q, current_q
            current_qd, next_qd = next_qd, current_qd

        return current_q, current_qd, next_q, next_qd

    def _run_full27_shape_matching_level(
        self,
        clusters: ShapeMatchingClusters | None,
        cluster_rotations,
        cluster_translations,
        stiffness: float,
        passes: int,
        current_q: wp.array,
        current_qd: wp.array,
        next_q: wp.array,
        next_qd: wp.array,
        dt: float,
        use_gs: bool = False,
        gs_support_alpha: float = 1.0,
    ):
        if clusters is None or stiffness <= 0.0 or passes <= 0:
            return current_q, current_qd, next_q, next_qd
        assert cluster_rotations is not None
        assert cluster_translations is not None

        for shape_pass in range(max(0, int(passes))):
            if use_gs:
                if (shape_pass & 1) == 0:
                    color_range = range(8)
                else:
                    color_range = range(7, -1, -1)
                for color in color_range:
                    color_start = int(clusters.color_offsets_host[color])
                    color_end = int(clusters.color_offsets_host[color + 1])
                    color_count = color_end - color_start
                    if color_count <= 0:
                        continue
                    wp.launch(
                        kernel=solve_shape_matching_clusters_uniform27_colored_gs,
                        dim=color_count,
                        inputs=[
                            current_q,
                            self.model.particle_inv_mass,
                            self.model.particle_flags,
                            clusters.color_cluster_indices,
                            clusters.indices_by_slot,
                            clusters.rest_local_positions_by_slot,
                            clusters.rest_local_template,
                            clusters.coefficients,
                            clusters.active,
                            clusters.particle_cluster_inv_weights,
                            clusters.num_clusters,
                            int(clusters.rest_local_template_valid),
                            color_start,
                            stiffness,
                            gs_support_alpha,
                            self.rotation_iterations,
                        ],
                        outputs=[cluster_rotations],
                        device=self.model.device,
                    )
                wp.launch(
                    kernel=finalize_position_update_from_q,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_flags,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    device=self.model.device,
                )
            else:
                self.particle_deltas.zero_()
                wp.launch(
                    kernel=solve_shape_matching_clusters_uniform27,
                    dim=clusters.num_clusters,
                    inputs=[
                        current_q,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        clusters.indices_by_slot,
                        clusters.rest_local_positions_by_slot,
                        clusters.rest_local_template,
                        clusters.coefficients,
                        clusters.active,
                        clusters.particle_cluster_inv_weights,
                        clusters.num_clusters,
                        int(clusters.rest_local_template_valid),
                        stiffness,
                        self.rotation_iterations,
                    ],
                    outputs=[cluster_rotations, cluster_translations, self.particle_deltas],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=apply_particle_deltas,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        self.model.particle_flags,
                        self.particle_deltas,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
                current_q, next_q = next_q, current_q
                current_qd, next_qd = next_qd, current_qd

        return current_q, current_qd, next_q, next_qd

    def _run_full125_shape_matching_level(
        self,
        clusters: ShapeMatchingClusters | None,
        cluster_rotations,
        cluster_translations,
        stiffness: float,
        passes: int,
        current_q: wp.array,
        current_qd: wp.array,
        next_q: wp.array,
        next_qd: wp.array,
        dt: float,
        use_gs: bool = False,
        gs_support_alpha: float = 1.0,
    ):
        if clusters is None or stiffness <= 0.0 or passes <= 0:
            return current_q, current_qd, next_q, next_qd
        assert cluster_rotations is not None
        assert cluster_translations is not None

        for shape_pass in range(max(0, int(passes))):
            if use_gs:
                if (shape_pass & 1) == 0:
                    color_range = range(8)
                else:
                    color_range = range(7, -1, -1)
                for color in color_range:
                    color_start = int(clusters.color_offsets_host[color])
                    color_end = int(clusters.color_offsets_host[color + 1])
                    color_count = color_end - color_start
                    if color_count <= 0:
                        continue
                    wp.launch(
                        kernel=solve_shape_matching_clusters_uniform125_colored_gs,
                        dim=color_count,
                        inputs=[
                            current_q,
                            self.model.particle_inv_mass,
                            self.model.particle_flags,
                            clusters.color_cluster_indices,
                            clusters.indices_by_slot,
                            clusters.rest_local_positions_by_slot,
                            clusters.rest_local_template,
                            clusters.coefficients,
                            clusters.active,
                            clusters.particle_cluster_inv_weights,
                            clusters.num_clusters,
                            int(clusters.rest_local_template_valid),
                            color_start,
                            stiffness,
                            gs_support_alpha,
                            self.rotation_iterations,
                        ],
                        outputs=[cluster_rotations],
                        device=self.model.device,
                    )
                wp.launch(
                    kernel=finalize_position_update_from_q,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        current_qd,
                        self.model.particle_flags,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    device=self.model.device,
                )
            else:
                self.particle_deltas.zero_()
                wp.launch(
                    kernel=solve_shape_matching_clusters_uniform125,
                    dim=clusters.num_clusters,
                    inputs=[
                        current_q,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        clusters.indices_by_slot,
                        clusters.rest_local_positions_by_slot,
                        clusters.rest_local_template,
                        clusters.coefficients,
                        clusters.active,
                        clusters.particle_cluster_inv_weights,
                        clusters.num_clusters,
                        int(clusters.rest_local_template_valid),
                        stiffness,
                        self.rotation_iterations,
                    ],
                    outputs=[cluster_rotations, cluster_translations, self.particle_deltas],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=apply_particle_deltas,
                    dim=self.model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        self.model.particle_flags,
                        self.particle_deltas,
                        dt,
                        self.model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=self.model.device,
                )
                current_q, next_q = next_q, current_q
                current_qd, next_qd = next_qd, current_qd

        return current_q, current_qd, next_q, next_qd

    def _run_hierarchical_shape_matching(
        self,
        current_q: wp.array,
        current_qd: wp.array,
        next_q: wp.array,
        next_qd: wp.array,
        dt: float,
    ):
        hierarchy = self.hierarchy
        if hierarchy is None or not self.enable_shape_matching:
            return current_q, current_qd, next_q, next_qd

        gs_weighting = self._shape_matching_gs_weighting()
        gs_support_alpha = self._shape_matching_gs_support_alpha(gs_weighting)
        l1_stiffness = self._l1_effective_shape_matching_stiffness()
        l2_stiffness = self._l2_effective_shape_matching_stiffness()

        if int(self.l2_hierarchical_shape_matching_mode) == HIERARCHICAL_SHAPE_MATCHING_OUTER8:
            current_q, current_qd, next_q, next_qd = self._run_outer8_shape_matching_level(
                hierarchy.l2_outer8,
                hierarchy.l2_outer8_prolongation,
                hierarchy.l2_outer8_block_keys,
                self.hierarchy_l2_outer8_rotations,
                self.hierarchy_l2_outer8_translations,
                l2_stiffness,
                self.l2_hierarchical_shape_matching_passes,
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
                self.l2_hierarchical_shape_matching_use_gs,
                gs_support_alpha,
                self.l2_hierarchical_shape_matching_outer8_prolongation,
                self.l2_hierarchical_shape_matching_outer8_absolute_projection,
            )
        elif int(self.l2_hierarchical_shape_matching_mode) == HIERARCHICAL_SHAPE_MATCHING_FULL27:
            current_q, current_qd, next_q, next_qd = self._run_full125_shape_matching_level(
                hierarchy.l2_full125,
                self.hierarchy_l2_full125_rotations,
                self.hierarchy_l2_full125_translations,
                l2_stiffness,
                self.l2_hierarchical_shape_matching_passes,
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
                self.l2_hierarchical_shape_matching_use_gs,
                gs_support_alpha,
            )

        mode = int(self.hierarchical_shape_matching_mode)
        if mode == HIERARCHICAL_SHAPE_MATCHING_OUTER8:
            current_q, current_qd, next_q, next_qd = self._run_outer8_shape_matching_level(
                hierarchy.outer8,
                hierarchy.outer8_prolongation,
                hierarchy.outer8_block_keys,
                self.hierarchy_outer8_rotations,
                self.hierarchy_outer8_translations,
                l1_stiffness,
                self.hierarchical_shape_matching_passes,
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
                self.hierarchical_shape_matching_use_gs,
                gs_support_alpha,
                self.hierarchical_shape_matching_outer8_prolongation,
                self.hierarchical_shape_matching_outer8_absolute_projection,
            )
        elif mode == HIERARCHICAL_SHAPE_MATCHING_FULL27:
            current_q, current_qd, next_q, next_qd = self._run_full27_shape_matching_level(
                hierarchy.full27,
                self.hierarchy_full27_rotations,
                self.hierarchy_full27_translations,
                l1_stiffness,
                self.hierarchical_shape_matching_passes,
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
                self.hierarchical_shape_matching_use_gs,
                gs_support_alpha,
            )

        return current_q, current_qd, next_q, next_qd

    def clear_kinematic_sphere_contacts(self) -> None:
        self.kinematic_sphere_count = 0
        self.kinematic_sphere_radius = 0.0

    def clear_grab_distance_constraints(self) -> None:
        self.grab_distance_constraints = ()

    def set_grab_distance_constraints(self, constraints) -> None:
        normalized = []
        for grab_indices, grab_offsets, grab_count, pull_target, stiffness in constraints:
            count = int(grab_count)
            if count <= 0:
                continue
            target = np.asarray(pull_target, dtype=np.float32).reshape(3)
            normalized.append(
                (
                    grab_indices,
                    grab_offsets,
                    count,
                    (float(target[0]), float(target[1]), float(target[2])),
                    float(stiffness),
                )
            )
        self.grab_distance_constraints = tuple(normalized)

    def set_kinematic_sphere_contacts(
        self,
        sphere_q_prev: wp.array,
        sphere_q: wp.array,
        *,
        sphere_count: int,
        sphere_radius: float,
        interpolation_alpha: float,
        relaxation: float,
        iterations: int = 1,
        max_correction: float = 0.0,
    ) -> None:
        self.kinematic_sphere_q_prev = sphere_q_prev
        self.kinematic_sphere_q = sphere_q
        self.kinematic_sphere_count = int(sphere_count)
        self.kinematic_sphere_radius = float(sphere_radius)
        self.kinematic_sphere_interpolation_alpha = float(interpolation_alpha)
        self.kinematic_sphere_contact_relaxation = float(relaxation)
        self.kinematic_sphere_contact_iterations = max(1, int(iterations))
        self.kinematic_sphere_max_correction = max(0.0, float(max_correction))
        self.kinematic_sphere_contact_mode = KINEMATIC_SPHERE_CONTACT_PARTICLES

    def set_kinematic_sphere_mc_triangle_contacts(
        self,
        sphere_q_prev: wp.array,
        sphere_q: wp.array,
        *,
        sphere_count: int,
        sphere_radius: float,
        interpolation_alpha: float,
        relaxation: float,
        cell_nodes: wp.array,
        cell_active: wp.array,
        vertex_pos: wp.array,
        tri_indices: wp.array,
        triangle_count: int,
        iterations: int = 1,
        max_correction: float = 0.0,
    ) -> None:
        self.kinematic_sphere_q_prev = sphere_q_prev
        self.kinematic_sphere_q = sphere_q
        self.kinematic_sphere_count = int(sphere_count)
        self.kinematic_sphere_radius = float(sphere_radius)
        self.kinematic_sphere_interpolation_alpha = float(interpolation_alpha)
        self.kinematic_sphere_contact_relaxation = float(relaxation)
        self.kinematic_sphere_contact_iterations = max(1, int(iterations))
        self.kinematic_sphere_max_correction = max(0.0, float(max_correction))
        self.kinematic_sphere_contact_mode = KINEMATIC_SPHERE_CONTACT_MC_TRIANGLES
        self.kinematic_sphere_mc_cell_nodes = cell_nodes
        self.kinematic_sphere_mc_cell_active = cell_active
        self.kinematic_sphere_mc_vertex_pos = vertex_pos
        self.kinematic_sphere_mc_tri_indices = tri_indices
        self.kinematic_sphere_mc_triangle_count = int(triangle_count)

    def _kinematic_sphere_contacts_active(self) -> bool:
        base_active = bool(
            self.kinematic_sphere_q_prev is not None
            and self.kinematic_sphere_q is not None
            and self.kinematic_sphere_count > 0
            and self.kinematic_sphere_radius > 0.0
        )
        if not base_active:
            return False
        if self.kinematic_sphere_contact_mode == KINEMATIC_SPHERE_CONTACT_MC_TRIANGLES:
            return bool(
                self.kinematic_sphere_mc_cell_nodes is not None
                and self.kinematic_sphere_mc_cell_active is not None
                and self.kinematic_sphere_mc_vertex_pos is not None
                and self.kinematic_sphere_mc_tri_indices is not None
                and self.kinematic_sphere_mc_triangle_count > 0
            )
        return True

    def _project_kinematic_sphere_contacts(self, particle_q: wp.array) -> None:
        if not self._kinematic_sphere_contacts_active():
            return
        assert self.kinematic_sphere_q_prev is not None
        assert self.kinematic_sphere_q is not None
        if self.kinematic_sphere_contact_mode == KINEMATIC_SPHERE_CONTACT_MC_TRIANGLES:
            assert self.kinematic_sphere_mc_cell_nodes is not None
            assert self.kinematic_sphere_mc_cell_active is not None
            assert self.kinematic_sphere_mc_vertex_pos is not None
            assert self.kinematic_sphere_mc_tri_indices is not None
            project_kinematic_sphere_mc_triangle_node_positions(
                particle_q,
                self.model.particle_inv_mass,
                self.model.particle_flags,
                self.kinematic_sphere_mc_cell_nodes,
                self.kinematic_sphere_mc_cell_active,
                self.kinematic_sphere_mc_vertex_pos,
                self.kinematic_sphere_mc_tri_indices,
                self.kinematic_sphere_mc_triangle_count,
                self.kinematic_sphere_q_prev,
                self.kinematic_sphere_q,
                self.kinematic_sphere_count,
                self.kinematic_sphere_radius,
                self.kinematic_sphere_interpolation_alpha,
                self.kinematic_sphere_contact_relaxation,
                self.particle_deltas,
                self.kinematic_sphere_delta_counts,
                iterations=self.kinematic_sphere_contact_iterations,
                max_correction=self.kinematic_sphere_max_correction,
                device=self.model.device,
            )
            return

        project_kinematic_sphere_particle_positions(
            particle_q,
            self.model,
            self.kinematic_sphere_q_prev,
            self.kinematic_sphere_q,
            self.kinematic_sphere_count,
            self.kinematic_sphere_radius,
            self.kinematic_sphere_interpolation_alpha,
            self.kinematic_sphere_contact_relaxation,
            iterations=self.kinematic_sphere_contact_iterations,
            max_correction=self.kinematic_sphere_max_correction,
            device=self.model.device,
        )

    def _project_grab_distance_constraints(self, particle_q: wp.array, particle_qd: wp.array) -> None:
        for grab_indices, grab_offsets, grab_count, pull_target, stiffness in self.grab_distance_constraints:
            project_grab_distance_constraints(
                particle_q,
                particle_qd,
                self.model.particle_inv_mass,
                self.model.particle_flags,
                grab_indices,
                grab_offsets,
                grab_count,
                pull_target,
                stiffness,
                device=self.model.device,
            )

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        del control, contacts

        model = self.model
        self.particle_q_init.assign(state_in.particle_q)

        self.integrate_particles(model, state_in, state_out, dt)

        current_q = state_out.particle_q
        current_qd = state_out.particle_qd
        next_q = self._particle_q_scratch
        next_qd = self._particle_qd_scratch

        self_collision_active = self.enable_self_collisions and model.particle_count > 1 and model.particle_grid is not None
        if self_collision_active:
            search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(current_q, radius=search_radius)

        if int(self.iterations) <= 0:
            current_q, current_qd, next_q, next_qd = self._run_hierarchical_shape_matching(
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
            )
            current_q, current_qd, next_q, next_qd = self._project_l0_sleeping_children(
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
            )
            self._project_grab_distance_constraints(current_q, current_qd)
            self._project_kinematic_sphere_contacts(current_q)

        for _ in range(self.iterations):
            springs_active = bool(self.enable_springs and model.spring_count)
            ground_active = bool(self.enable_ground_plane)
            shape_passes = max(0, int(self.shape_matching_passes))
            l0_stiffness = self._l0_effective_shape_matching_stiffness()
            l0_mode = self._l0_shape_matching_mode()
            l0_gs_weighting = self._shape_matching_gs_weighting()
            l0_gs_support_alpha = self._shape_matching_gs_support_alpha(l0_gs_weighting)
            shape_active = bool(self.enable_shape_matching and l0_stiffness > 0.0 and shape_passes > 0)
            shape_gather_active = bool(shape_active and l0_mode == SHAPE_MATCHING_SOLVE_GATHER)
            shape_scatter_active = bool(shape_active and l0_mode == SHAPE_MATCHING_SOLVE_SCATTER)
            shape_gs_active = bool(shape_active and l0_mode == SHAPE_MATCHING_SOLVE_COLORED_GS)
            volume_active = self._l0_volume_preservation_active()
            l0_fast_uniform8_active = bool(self._l0_fast_uniform8_active and self.clusters.rest_local_template_valid)
            base_delta_active = bool(self_collision_active or springs_active or ground_active)

            if base_delta_active or shape_scatter_active:
                self.particle_deltas.zero_()

            if self_collision_active:
                wp.launch(
                    kernel=solve_particle_particle_contacts,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        current_q,
                        current_qd,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.particle_flags,
                        model.particle_mu,
                        model.particle_cohesion,
                        model.particle_max_radius,
                        dt,
                        0.9,
                    ],
                    outputs=[self.particle_deltas],
                    device=model.device,
                )

            if springs_active:
                assert self.spring_constraint_lambdas is not None
                self.spring_constraint_lambdas.zero_()
                wp.launch(
                    kernel=solve_springs,
                    dim=model.spring_count,
                    inputs=[
                        current_q,
                        current_qd,
                        model.particle_inv_mass,
                        model.spring_indices,
                        model.spring_rest_length,
                        model.spring_stiffness,
                        model.spring_damping,
                        dt,
                        self.spring_constraint_lambdas,
                    ],
                    outputs=[self.particle_deltas],
                    device=model.device,
                )

            if ground_active:
                wp.launch(
                    kernel=solve_particle_ground_plane_contacts,
                    dim=model.particle_count,
                    inputs=[
                        current_q,
                        current_qd,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.particle_flags,
                        self.ground_height,
                        model.particle_mu,
                        dt,
                        self.ground_contact_relaxation,
                    ],
                    outputs=[self.particle_deltas],
                    device=model.device,
                )

            if volume_active:
                if base_delta_active:
                    wp.launch(
                        kernel=apply_particle_deltas,
                        dim=model.particle_count,
                        inputs=[
                            self.particle_q_init,
                            current_q,
                            model.particle_flags,
                            self.particle_deltas,
                            dt,
                            model.particle_max_velocity,
                        ],
                        outputs=[next_q, next_qd],
                        device=model.device,
                    )
                    current_q, next_q = next_q, current_q
                    current_qd, next_qd = next_qd, current_qd
                    base_delta_active = False

                current_q, current_qd, next_q, next_qd = self._run_l0_volume_preservation(
                    current_q,
                    current_qd,
                    next_q,
                    next_qd,
                    dt,
                )

            if base_delta_active:
                wp.launch(
                    kernel=apply_particle_deltas,
                    dim=model.particle_count,
                    inputs=[
                        self.particle_q_init,
                        current_q,
                        model.particle_flags,
                        self.particle_deltas,
                        dt,
                        model.particle_max_velocity,
                    ],
                    outputs=[next_q, next_qd],
                    device=model.device,
                )
                current_q, next_q = next_q, current_q
                current_qd, next_qd = next_qd, current_qd
                base_delta_active = False
                if shape_scatter_active:
                    self.particle_deltas.zero_()

            current_q, current_qd, next_q, next_qd = self._run_hierarchical_shape_matching(
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
            )
            current_q, current_qd, next_q, next_qd = self._project_l0_sleeping_children(
                current_q,
                current_qd,
                next_q,
                next_qd,
                dt,
            )

            if shape_gather_active:
                for shape_pass in range(shape_passes):
                    wp.launch(
                        kernel=compute_shape_matching_cluster_poses_uniform8,
                        dim=self.clusters.num_clusters,
                        inputs=[
                            current_q,
                            model.particle_flags,
                            self.clusters.indices_by_slot,
                            self.clusters.rest_local_positions_by_slot,
                            self.clusters.rest_local_template,
                            self.clusters.coefficients,
                            self._l0_active_array(),
                            self.clusters.num_clusters,
                            int(self.clusters.rest_local_template_valid),
                            l0_stiffness,
                            self.rotation_iterations,
                        ],
                        outputs=[self.cluster_rotations, self.cluster_translations],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=apply_shape_matching_particle_gather_uniform8,
                        dim=model.particle_count,
                        inputs=[
                            self.particle_q_init,
                            current_q,
                            model.particle_inv_mass,
                            model.particle_flags,
                            self.particle_deltas,
                            self.clusters.particle_cluster_offsets,
                            self.clusters.particle_cluster_indices,
                            self.clusters.particle_cluster_member_offsets,
                            self.clusters.rest_local_positions_by_slot,
                            self.clusters.rest_local_template,
                            self.clusters.coefficients,
                            self._l0_active_array(),
                            self._l0_particle_cluster_inv_weights_array(),
                            self.clusters.num_clusters,
                            int(self.clusters.rest_local_template_valid),
                            l0_stiffness,
                            self.cluster_rotations,
                            self.cluster_translations,
                            int(base_delta_active and shape_pass == 0),
                            dt,
                            model.particle_max_velocity,
                        ],
                        outputs=[next_q, next_qd],
                        device=model.device,
                    )
                    current_q, next_q = next_q, current_q
                    current_qd, next_qd = next_qd, current_qd
                    current_q, current_qd, next_q, next_qd = self._project_l0_sleeping_children(
                        current_q,
                        current_qd,
                        next_q,
                        next_qd,
                        dt,
                    )
            else:
                if shape_scatter_active:
                    for _shape_pass in range(shape_passes):
                        self.particle_deltas.zero_()
                        if l0_fast_uniform8_active:
                            wp.launch(
                                kernel=solve_shape_matching_clusters_uniform8_template_active,
                                dim=self.clusters.num_clusters,
                                inputs=[
                                    current_q,
                                    self.clusters.indices_by_slot,
                                    self.clusters.rest_local_template,
                                    self.clusters.coefficients,
                                    self._l0_active_array(),
                                    self._l0_particle_cluster_inv_weights_array(),
                                    self.clusters.num_clusters,
                                    l0_stiffness,
                                    self.rotation_iterations,
                                ],
                                outputs=[self.cluster_rotations, self.cluster_translations, self.particle_deltas],
                                device=model.device,
                            )
                        else:
                            wp.launch(
                                kernel=solve_shape_matching_clusters_uniform8,
                                dim=self.clusters.num_clusters,
                                inputs=[
                                    current_q,
                                    model.particle_inv_mass,
                                    model.particle_flags,
                                    self.clusters.indices_by_slot,
                                    self.clusters.rest_local_positions_by_slot,
                                    self.clusters.rest_local_template,
                                    self.clusters.coefficients,
                                    self._l0_active_array(),
                                    self._l0_particle_cluster_inv_weights_array(),
                                    self.clusters.num_clusters,
                                    int(self.clusters.rest_local_template_valid),
                                    l0_stiffness,
                                    self.rotation_iterations,
                                ],
                                outputs=[self.cluster_rotations, self.cluster_translations, self.particle_deltas],
                                device=model.device,
                            )
                        wp.launch(
                            kernel=apply_particle_deltas,
                            dim=model.particle_count,
                            inputs=[
                                self.particle_q_init,
                                current_q,
                                model.particle_flags,
                                self.particle_deltas,
                                dt,
                                model.particle_max_velocity,
                            ],
                            outputs=[next_q, next_qd],
                            device=model.device,
                        )
                        current_q, next_q = next_q, current_q
                        current_qd, next_qd = next_qd, current_qd
                        current_q, current_qd, next_q, next_qd = self._project_l0_sleeping_children(
                            current_q,
                            current_qd,
                            next_q,
                            next_qd,
                            dt,
                        )
                elif shape_gs_active:
                    if base_delta_active:
                        wp.launch(
                            kernel=apply_particle_deltas,
                            dim=model.particle_count,
                            inputs=[
                                self.particle_q_init,
                                current_q,
                                model.particle_flags,
                                self.particle_deltas,
                                dt,
                                model.particle_max_velocity,
                            ],
                            outputs=[next_q, next_qd],
                            device=model.device,
                        )
                        current_q, next_q = next_q, current_q
                        current_qd, next_qd = next_qd, current_qd

                    for shape_pass in range(shape_passes):
                        if (shape_pass & 1) == 0:
                            color_range = range(8)
                        else:
                            color_range = range(7, -1, -1)
                        for color in color_range:
                            color_start = int(self.clusters.color_offsets_host[color])
                            color_end = int(self.clusters.color_offsets_host[color + 1])
                            color_count = color_end - color_start
                            if color_count <= 0:
                                continue
                            if l0_fast_uniform8_active:
                                wp.launch(
                                    kernel=solve_shape_matching_clusters_uniform8_colored_gs_template_active,
                                    dim=color_count,
                                    inputs=[
                                        current_q,
                                        self.clusters.color_cluster_indices,
                                        self.clusters.indices_by_slot,
                                        self.clusters.rest_local_template,
                                        self.clusters.coefficients,
                                        self._l0_active_array(),
                                        self._l0_particle_cluster_inv_weights_array(),
                                        self.clusters.num_clusters,
                                        color_start,
                                        l0_stiffness,
                                        l0_gs_support_alpha,
                                        self.rotation_iterations,
                                    ],
                                    outputs=[self.cluster_rotations],
                                    device=model.device,
                                )
                            else:
                                wp.launch(
                                    kernel=solve_shape_matching_clusters_uniform8_colored_gs,
                                    dim=color_count,
                                    inputs=[
                                        current_q,
                                        model.particle_inv_mass,
                                        model.particle_flags,
                                        self.clusters.color_cluster_indices,
                                        self.clusters.indices_by_slot,
                                        self.clusters.rest_local_positions_by_slot,
                                        self.clusters.rest_local_template,
                                        self.clusters.coefficients,
                                        self._l0_active_array(),
                                        self._l0_particle_cluster_inv_weights_array(),
                                        self.clusters.num_clusters,
                                        int(self.clusters.rest_local_template_valid),
                                        color_start,
                                        l0_stiffness,
                                        l0_gs_support_alpha,
                                        self.rotation_iterations,
                                    ],
                                    outputs=[self.cluster_rotations],
                                    device=model.device,
                                )
                        current_q, current_qd, next_q, next_qd = self._project_l0_sleeping_children(
                            current_q,
                            current_qd,
                            next_q,
                            next_qd,
                            dt,
                        )
                    wp.launch(
                        kernel=finalize_position_update_from_q,
                        dim=model.particle_count,
                        inputs=[
                            self.particle_q_init,
                            current_q,
                            current_qd,
                            model.particle_flags,
                            dt,
                            model.particle_max_velocity,
                        ],
                        device=model.device,
                    )
                else:
                    if base_delta_active:
                        wp.launch(
                            kernel=apply_particle_deltas,
                            dim=model.particle_count,
                            inputs=[
                                self.particle_q_init,
                                current_q,
                                model.particle_flags,
                                self.particle_deltas,
                                dt,
                                model.particle_max_velocity,
                            ],
                            outputs=[next_q, next_qd],
                            device=model.device,
                        )
                        current_q, next_q = next_q, current_q
                        current_qd, next_qd = next_qd, current_qd

            self._project_grab_distance_constraints(current_q, current_qd)
            self._project_kinematic_sphere_contacts(current_q)

        if current_q.ptr != state_out.particle_q.ptr:
            state_out.particle_q.assign(current_q)
            state_out.particle_qd.assign(current_qd)

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        del state
        if contacts.force is not None:
            contacts.force.zero_()


__all__ = [
    "HIERARCHICAL_SHAPE_MATCHING_FULL27",
    "HIERARCHICAL_SHAPE_MATCHING_LABELS",
    "HIERARCHICAL_SHAPE_MATCHING_OFF",
    "HIERARCHICAL_SHAPE_MATCHING_OUTER8",
    "KINEMATIC_SPHERE_CONTACT_MC_TRIANGLES",
    "KINEMATIC_SPHERE_CONTACT_PARTICLES",
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
    "SolverCornerShapeMatching",
]

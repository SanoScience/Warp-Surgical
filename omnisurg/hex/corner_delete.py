# SPDX-License-Identifier: Apache-2.0
"""Host-side voxel deletion for the corner-node cell lattice."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from newton._src.geometry.flags import ParticleFlags

from .corner_grid import (
    CONSTRAINT_BODY_DIAGONAL,
    CONSTRAINT_EDGE,
    CONSTRAINT_FACE_DIAGONAL,
    CornerGridAuxState,
    ShapeMatchingClusters,
)
from .kernels.corner_delete import (
    deactivate_deleted_cell_clusters_kernel,
    deactivate_single_deleted_cell_cluster_kernel,
    delete_cells_sparse_kernel,
    delete_single_cell_complete_kernel,
    finalize_deactivated_cluster_weights_kernel,
    finalize_deleted_cell_nodes_kernel,
    finalize_deleted_cell_springs_kernel,
    select_ray_segment_cells_kernel,
    select_ray_surface_cells_kernel,
    select_sphere_particle_contact_cells_kernel,
    validate_cell_active_kernel,
    validate_cluster_state_kernel,
    validate_node_state_kernel,
    validate_spring_state_kernel,
)
from .tool import Tool


def _normalize_vec3(value, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-8:
        raise ValueError(f"{name} must be non-zero")
    return vec / norm


def _point_segment_distance_sq(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    seg = p1 - p0
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= 1.0e-12:
        delta = points - p0[None, :]
        return np.einsum("ij,ij->i", delta, delta, optimize=True)
    t = np.einsum("ij,j->i", points - p0[None, :], seg, optimize=True) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = p0[None, :] + t[:, None] * seg[None, :]
    delta = points - closest
    return np.einsum("ij,ij->i", delta, delta, optimize=True)


@dataclass
class DeviceDeletionResult:
    """Device-resident result for a launched cell deletion batch."""

    deleted_cells_device: wp.array | None
    deleted_count_device: wp.array
    candidate_capacity: int
    topology_revision: int
    host_synced: bool = False


@dataclass
class CornerDeletionState:
    """Device-authoritative bookkeeping for cell deletion and mass updates.

    NumPy mirror arrays on this object are compatibility/debug snapshots. They
    can be stale after async deletion until ``sync_host_mirrors()`` or
    ``sync_last_deleted()`` is called.
    """

    model: newton.Model
    aux: CornerGridAuxState
    delete_padding: float

    def __post_init__(self) -> None:
        self.cell_active = self.aux.cell_active_host.copy()
        self.node_support_count = self.aux.node_support_count_host.copy()
        self.spring_support_count = self.aux.spring_support_count_host.copy()
        self.node_mass = self.model.particle_mass.numpy().copy()
        self.node_inv_mass = self.model.particle_inv_mass.numpy().copy()
        self.locked_node_mask = np.zeros(self.model.particle_count, dtype=bool)
        self.particle_flags = self.model.particle_flags.numpy().copy()
        self.spring_stiffness = self.model.spring_stiffness.numpy().copy()
        self.spring_damping = self.model.spring_damping.numpy().copy()
        self.spring_indices = self.model.spring_indices.numpy().reshape(-1, 2).copy()
        self.material_stiffness_scale = np.ones(len(self.aux.materials), dtype=np.float32)
        self.material_stiffness_scale_device = wp.array(
            self.material_stiffness_scale,
            dtype=wp.float32,
            device=self.model.device,
        )
        self._dummy_int = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._dummy_float = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self._delete_counter = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._candidate_count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._deleted_total_device = wp.array(
            np.asarray([int((self.cell_active == 0).sum())], dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.node_mass_device = wp.array(self.node_mass, dtype=wp.float32, device=self.model.device)
        self.locked_node_mask_device = wp.zeros(self.model.particle_count, dtype=wp.int32, device=self.model.device)
        self._deleted_cells_capacity = 0
        self._deleted_cells_device: wp.array | None = None
        self._last_candidate_cells_device: wp.array | None = None
        self._cluster_deactivation_buffers: dict[int, tuple[int, wp.array, wp.array]] = {}
        self._plane_candidate_cells = wp.zeros(self.aux.num_cells, dtype=wp.int32, device=self.model.device)
        self._plane_candidate_count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self.last_deleted_cells_host = np.empty(0, dtype=np.int32)
        self.last_deleted_cells_device: wp.array | None = None
        self.last_deleted_count_device: wp.array | None = None
        self.last_deleted_count: int = 0
        self.last_deleted_topology_revision: int = -1
        self.deleted_total = int((self.cell_active == 0).sum())
        self.last_deletion_result: DeviceDeletionResult | None = None
        self._dirty_domains: set[str] = set()
        self._last_deleted_host_synced = True
        self._validation_error_count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._validation_first_error = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        # Monotonic counter bumped whenever cell_active or node-level particle
        # flags change. Renderers (SurfaceRenderer, CryoTextureAtlas) use this
        # to cache topology-dependent work across frames. Any future writer to
        # cell_active or node ACTIVE flags must also bump this.
        self.topology_revision: int = 0
        self._reset_cell_active = self.cell_active.copy()
        self._reset_node_support_count = self.node_support_count.copy()
        self._reset_spring_support_count = self.spring_support_count.copy()
        self._reset_node_mass = self.node_mass.copy()
        self._reset_node_inv_mass = self.node_inv_mass.copy()
        self._reset_particle_flags = self.particle_flags.copy()
        self._reset_deleted_total = int(self.deleted_total)
        self._reset_cluster_snapshots: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @property
    def deleted_total_device(self) -> wp.array:
        """Device-side monotonic count of successfully deleted cells."""
        return self._deleted_total_device

    def _iter_shape_matching_clusters(self):
        if self.aux.shape_matching_clusters is not None:
            yield self.aux.shape_matching_clusters
        hierarchy = self.aux.shape_matching_hierarchy
        if hierarchy is None:
            return
        for clusters in (
            hierarchy.outer8,
            hierarchy.full27,
            hierarchy.l2_outer8,
            hierarchy.l2_full125,
        ):
            if clusters is not None:
                yield clusters

    def capture_reset_baseline(self) -> None:
        """Remember the current undeleted cluster state for later reset.

        ``CornerDeletionState`` is often constructed before shape-matching
        clusters are attached to ``aux``. Capture cluster baselines lazily, just
        before the first deletion or explicit reset, so reset can restore the
        full topology without rebuilding the grid.
        """
        for clusters in self._iter_shape_matching_clusters():
            key = id(clusters)
            if key in self._reset_cluster_snapshots:
                continue
            self._reset_cluster_snapshots[key] = (
                clusters.active_host.copy(),
                clusters.particle_cluster_counts_host.copy(),
                clusters.particle_cluster_inv_weights_host.copy(),
            )

    def reset(self) -> None:
        """Restore deletion-side topology and masses while preserving tunable parameters."""
        self.capture_reset_baseline()

        self.cell_active = self._reset_cell_active.copy()
        self.node_support_count = self._reset_node_support_count.copy()
        self.spring_support_count = self._reset_spring_support_count.copy()
        self.node_mass = self._reset_node_mass.copy()
        self.node_inv_mass = self._reset_node_inv_mass.copy()
        self.particle_flags = self._reset_particle_flags.copy()
        self.locked_node_mask.fill(False)

        self.aux.cell_active_host = self.cell_active
        self.aux.node_support_count_host = self.node_support_count
        self.aux.spring_support_count_host = self.spring_support_count
        self.aux.cell_active.assign(self.cell_active)
        self.aux.node_support_count.assign(self.node_support_count)
        self.model.particle_flags.assign(self.particle_flags)
        self._push_effective_node_masses()
        self._refresh_springs()

        for clusters in self._iter_shape_matching_clusters():
            snapshot = self._reset_cluster_snapshots.get(id(clusters))
            if snapshot is None:
                continue
            active, counts, inv_weights = snapshot
            clusters.active_host[:] = active
            clusters.particle_cluster_counts_host[:] = counts
            clusters.particle_cluster_inv_weights_host[:] = inv_weights
            clusters.active.assign(clusters.active_host)
            clusters.particle_cluster_counts.assign(clusters.particle_cluster_counts_host)
            clusters.particle_cluster_inv_weights.assign(clusters.particle_cluster_inv_weights_host)

        self.deleted_total = int(self._reset_deleted_total)
        self._deleted_total_device.fill_(self.deleted_total)
        self._candidate_count.zero_()
        self._plane_candidate_count.zero_()
        self._last_candidate_cells_device = None
        self._clear_last_deleted()
        self._dirty_domains.clear()
        self.topology_revision += 1

    def _mark_dirty(self, *domains: str) -> None:
        self._dirty_domains.update(domains)

    def _ensure_deleted_cells_buffer(self, capacity: int) -> wp.array:
        capacity = max(1, int(capacity))
        if self._deleted_cells_device is None or capacity > self._deleted_cells_capacity:
            self._deleted_cells_device = wp.zeros(capacity, dtype=wp.int32, device=self.model.device)
            self._deleted_cells_capacity = capacity
        return self._deleted_cells_device

    def _ensure_cluster_deactivation_buffers(self, clusters: ShapeMatchingClusters, capacity: int) -> tuple[wp.array, wp.array]:
        capacity = max(1, int(capacity))
        key = id(clusters)
        current = self._cluster_deactivation_buffers.get(key)
        if current is None or capacity > current[0]:
            ids = wp.zeros(capacity, dtype=wp.int32, device=self.model.device)
            count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
            self._cluster_deactivation_buffers[key] = (capacity, ids, count)
            return ids, count
        _, ids, count = current
        return ids, count

    def _sync_cluster_host(self, clusters: ShapeMatchingClusters | None) -> None:
        if clusters is None:
            return
        clusters.active_host = clusters.active.numpy().astype(np.int32, copy=True)
        clusters.particle_cluster_counts_host = clusters.particle_cluster_counts.numpy().astype(np.int32, copy=True)
        clusters.particle_cluster_inv_weights_host = clusters.particle_cluster_inv_weights.numpy().astype(
            np.float32,
            copy=True,
        )

    def _sync_all_cluster_hosts(self) -> None:
        self._sync_cluster_host(self.aux.shape_matching_clusters)
        hierarchy = self.aux.shape_matching_hierarchy
        if hierarchy is None:
            return
        self._sync_cluster_host(hierarchy.outer8)
        self._sync_cluster_host(hierarchy.full27)
        self._sync_cluster_host(hierarchy.l2_outer8)

    def _normalize_sync_domains(self, domains) -> set[str]:
        if domains == "all" or domains is None:
            return {"cells", "nodes", "mass", "springs", "clusters", "stats", "last-delete"}
        if isinstance(domains, str):
            raw = {domains}
        else:
            raw = {str(domain) for domain in domains}
        normalized: set[str] = set()
        for domain in raw:
            if domain in {"all", "*"}:
                normalized.update({"cells", "nodes", "mass", "springs", "clusters", "stats", "last-delete"})
            elif domain in {"nodes/mass", "node-mass"}:
                normalized.update({"nodes", "mass"})
            elif domain in {"last", "last_deleted", "last-delete"}:
                normalized.add("last-delete")
            else:
                normalized.add(domain)
        return normalized

    def sync_last_deleted(self) -> int:
        """Synchronize only the compact last-deleted list from device to host."""
        if self.last_deleted_count_device is None or self.last_deleted_cells_device is None:
            self.last_deleted_cells_host = np.empty(0, dtype=np.int32)
            self.last_deleted_count = 0
            self._last_deleted_host_synced = True
            return 0

        count = int(self.last_deleted_count_device.numpy()[0])
        count = max(0, min(count, int(self.last_deleted_cells_device.shape[0])))
        if count > 0:
            self.last_deleted_cells_host = self.last_deleted_cells_device.numpy()[:count].astype(np.int32, copy=True)
        else:
            self.last_deleted_cells_host = np.empty(0, dtype=np.int32)
        self.last_deleted_count = count
        self._last_deleted_host_synced = True
        self._dirty_domains.discard("last-delete")
        if self.last_deletion_result is not None:
            self.last_deletion_result.host_synced = True
        return count

    def sync_host_mirrors(self, domains="all") -> None:
        """Refresh requested NumPy mirrors from device-authoritative arrays."""
        requested = self._normalize_sync_domains(domains)
        if "last-delete" in requested:
            self.sync_last_deleted()

        if "cells" in requested:
            self.cell_active = self.aux.cell_active.numpy().astype(np.int32, copy=True)
            self.aux.cell_active_host = self.cell_active
            self._dirty_domains.discard("cells")

        if "nodes" in requested or "mass" in requested:
            self.node_support_count = self.aux.node_support_count.numpy().astype(np.int32, copy=True)
            self.node_mass = self.node_mass_device.numpy().astype(np.float32, copy=True)
            self.node_inv_mass = self.model.particle_inv_mass.numpy().astype(np.float32, copy=True)
            self.locked_node_mask = self.locked_node_mask_device.numpy().astype(np.int32, copy=False) != 0
            self.particle_flags = self.model.particle_flags.numpy().astype(np.int32, copy=True)
            self.aux.node_support_count_host = self.node_support_count
            self._dirty_domains.discard("nodes")
            self._dirty_domains.discard("mass")

        if "springs" in requested:
            self.spring_support_count = self.aux.spring_support_count.numpy().astype(np.int32, copy=True)
            self.spring_stiffness = self.model.spring_stiffness.numpy().astype(np.float32, copy=True)
            self.spring_damping = self.model.spring_damping.numpy().astype(np.float32, copy=True)
            self.aux.spring_support_count_host = self.spring_support_count
            self._dirty_domains.discard("springs")

        if "clusters" in requested:
            self._sync_all_cluster_hosts()
            self._dirty_domains.discard("clusters")

        if "stats" in requested:
            self.deleted_total = int(self._deleted_total_device.numpy()[0])
            self._dirty_domains.discard("stats")

    def _effective_node_mass(self) -> np.ndarray:
        mass = self.node_mass.copy()
        mass[self.locked_node_mask] = 0.0
        return mass

    def _push_effective_node_masses(self) -> None:
        self.node_mass_device.assign(self.node_mass)
        self.locked_node_mask_device.assign(self.locked_node_mask.astype(np.int32, copy=False))
        self.model.particle_mass.assign(self._effective_node_mass())
        self.model.particle_inv_mass.assign(self.node_inv_mass)

    def _clear_last_deleted(self) -> None:
        self._delete_counter.zero_()
        self.last_deleted_cells_host = np.empty(0, dtype=np.int32)
        self.last_deleted_cells_device = None
        self.last_deleted_count_device = None
        self.last_deleted_count = 0
        self.last_deleted_topology_revision = -1
        self.last_deletion_result = None
        self._last_deleted_host_synced = True
        self._dirty_domains.discard("last-delete")

    def _commit_deleted_cells(
        self,
        deleted_cells: np.ndarray,
        deleted_cells_device: wp.array,
    ) -> None:
        self.topology_revision += 1
        self.last_deleted_cells_host = deleted_cells.astype(np.int32, copy=True)
        self.last_deleted_cells_device = deleted_cells_device
        self.last_deleted_count = int(deleted_cells.size)
        self.last_deleted_count_device = wp.array(
            np.asarray([self.last_deleted_count], dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.last_deleted_topology_revision = int(self.topology_revision)
        self.last_deletion_result = DeviceDeletionResult(
            deleted_cells_device=deleted_cells_device,
            deleted_count_device=self.last_deleted_count_device,
            candidate_capacity=int(deleted_cells_device.shape[0]),
            topology_revision=int(self.topology_revision),
            host_synced=True,
        )
        self._last_deleted_host_synced = True

    def _refresh_springs(self) -> None:
        if self._dirty_domains.intersection({"springs"}):
            self.sync_host_mirrors({"springs"})
        if self.spring_indices.size > 0:
            endpoint_materials = self.aux.node_material_host[self.spring_indices]
            material_scale = 0.5 * (
                self.material_stiffness_scale[endpoint_materials[:, 0]]
                + self.material_stiffness_scale[endpoint_materials[:, 1]]
            )
        else:
            material_scale = np.zeros_like(self.spring_support_count, dtype=np.float32)
        self.spring_stiffness = (
            self.aux.spring_unit_stiffness_host
            * self.spring_support_count.astype(np.float32)
            * material_scale
        )
        self.spring_damping = (
            self.aux.spring_unit_damping_host
            * self.spring_support_count.astype(np.float32)
        )
        self.aux.spring_support_count_host = self.spring_support_count
        self.aux.spring_unit_stiffness.assign(self.aux.spring_unit_stiffness_host)
        self.aux.spring_unit_damping.assign(self.aux.spring_unit_damping_host)
        self.aux.spring_support_count.assign(self.spring_support_count)
        self.aux.spring_enabled.assign((self.spring_support_count > 0).astype(np.int32))
        self.model.spring_stiffness.assign(self.spring_stiffness)
        self.model.spring_damping.assign(self.spring_damping)
        self.material_stiffness_scale_device.assign(self.material_stiffness_scale)
        self._dirty_domains.discard("springs")

    def _refresh_touched_springs_host(self, spring_ids: np.ndarray) -> None:
        if spring_ids.size == 0:
            return
        unique_springs = np.unique(spring_ids.astype(np.int32, copy=False))
        endpoint_materials = self.aux.node_material_host[self.spring_indices[unique_springs]]
        material_scale = 0.5 * (
            self.material_stiffness_scale[endpoint_materials[:, 0]]
            + self.material_stiffness_scale[endpoint_materials[:, 1]]
        )
        support = self.spring_support_count[unique_springs].astype(np.float32)
        self.spring_stiffness[unique_springs] = (
            self.aux.spring_unit_stiffness_host[unique_springs]
            * support
            * material_scale
        )
        self.spring_damping[unique_springs] = (
            self.aux.spring_unit_damping_host[unique_springs]
            * support
        )
        self.aux.spring_support_count_host = self.spring_support_count

    def _refresh_touched_shape_matching_host(
        self,
        deleted_cells: np.ndarray,
        node_ids: np.ndarray,
    ) -> None:
        clusters = self.aux.shape_matching_clusters
        if clusters is not None and deleted_cells.size > 0:
            cluster_ids = clusters.cell_to_cluster_host[deleted_cells]
            cluster_ids = cluster_ids[cluster_ids >= 0]
            if cluster_ids.size > 0:
                clusters.active_host[np.unique(cluster_ids)] = 0

        if clusters is not None and node_ids.size > 0:
            unique_nodes = np.unique(node_ids.astype(np.int32, copy=False))
            counts = self.node_support_count[unique_nodes].astype(np.int32, copy=False)
            inv_weights = np.zeros(counts.shape, dtype=np.float32)
            valid = counts > 0
            inv_weights[valid] = 1.0 / counts[valid].astype(np.float32)
            clusters.particle_cluster_counts_host[unique_nodes] = counts
            clusters.particle_cluster_inv_weights_host[unique_nodes] = inv_weights

        if deleted_cells.size > 0:
            self._refresh_touched_l1_shape_matching_host(deleted_cells)

    def _recompute_cluster_membership_weights(self, clusters: ShapeMatchingClusters) -> None:
        particle_cluster_counts = np.zeros(self.model.particle_count, dtype=np.int32)
        active_cluster_ids = np.nonzero(clusters.active_host != 0)[0]

        if active_cluster_ids.size > 0:
            if clusters.indices_host.size == clusters.num_clusters * clusters.uniform_size:
                member_indices = clusters.indices_host.reshape(clusters.num_clusters, clusters.uniform_size)[
                    active_cluster_ids
                ].reshape(-1)
                np.add.at(particle_cluster_counts, member_indices, 1)
            else:
                for cluster_idx in active_cluster_ids.tolist():
                    start = int(clusters.offsets_host[cluster_idx])
                    end = int(clusters.offsets_host[cluster_idx + 1])
                    np.add.at(particle_cluster_counts, clusters.indices_host[start:end], 1)

        particle_cluster_inv_weights = np.zeros(self.model.particle_count, dtype=np.float32)
        valid = particle_cluster_counts > 0
        particle_cluster_inv_weights[valid] = 1.0 / particle_cluster_counts[valid].astype(np.float32)

        clusters.particle_cluster_counts_host[:] = particle_cluster_counts
        clusters.particle_cluster_inv_weights_host[:] = particle_cluster_inv_weights
        clusters.active.assign(clusters.active_host)
        clusters.particle_cluster_counts.assign(clusters.particle_cluster_counts_host)
        clusters.particle_cluster_inv_weights.assign(clusters.particle_cluster_inv_weights_host)

    def _refresh_touched_l1_shape_matching_host(self, deleted_cells: np.ndarray) -> None:
        hierarchy = self.aux.shape_matching_hierarchy
        if hierarchy is None:
            return

        for clusters in (hierarchy.outer8, hierarchy.full27, hierarchy.l2_outer8):
            if clusters is None:
                continue
            cluster_ids = clusters.cell_to_cluster_host[deleted_cells]
            cluster_ids = cluster_ids[cluster_ids >= 0]
            if cluster_ids.size == 0:
                continue
            clusters.active_host[np.unique(cluster_ids)] = 0
            self._recompute_cluster_membership_weights(clusters)

    def _deactivate_deleted_clusters(self, clusters: ShapeMatchingClusters | None, candidate_capacity: int) -> None:
        if clusters is None or candidate_capacity <= 0:
            return
        if self.last_deleted_cells_device is None or self.last_deleted_count_device is None:
            return
        deactivated_cluster_ids, deactivated_count = self._ensure_cluster_deactivation_buffers(
            clusters,
            candidate_capacity,
        )
        deactivated_count.zero_()
        wp.launch(
            deactivate_deleted_cell_clusters_kernel,
            dim=int(candidate_capacity),
            inputs=[
                self.last_deleted_cells_device,
                self.last_deleted_count_device,
                int(self.aux.num_cells),
                clusters.cell_to_cluster,
                clusters.active,
                clusters.offsets,
                clusters.indices,
                clusters.particle_cluster_counts,
                deactivated_cluster_ids,
                deactivated_count,
            ],
            device=self.model.device,
        )
        wp.launch(
            finalize_deactivated_cluster_weights_kernel,
            dim=int(candidate_capacity),
            inputs=[
                deactivated_cluster_ids,
                deactivated_count,
                clusters.offsets,
                clusters.indices,
                clusters.particle_cluster_counts,
                clusters.particle_cluster_inv_weights,
            ],
            device=self.model.device,
        )

    def _deactivate_all_shape_matching_clusters(self, candidate_capacity: int) -> None:
        self._deactivate_deleted_clusters(self.aux.shape_matching_clusters, candidate_capacity)
        hierarchy = self.aux.shape_matching_hierarchy
        if hierarchy is None:
            return
        self._deactivate_deleted_clusters(hierarchy.outer8, candidate_capacity)
        self._deactivate_deleted_clusters(hierarchy.full27, candidate_capacity)
        self._deactivate_deleted_clusters(hierarchy.l2_outer8, candidate_capacity)

    def _deactivate_single_deleted_cluster(self, clusters: ShapeMatchingClusters | None) -> None:
        if clusters is None:
            return
        if self.last_deleted_cells_device is None or self.last_deleted_count_device is None:
            return
        wp.launch(
            deactivate_single_deleted_cell_cluster_kernel,
            dim=1,
            inputs=[
                self.last_deleted_cells_device,
                self.last_deleted_count_device,
                int(self.aux.num_cells),
                clusters.cell_to_cluster,
                clusters.active,
                clusters.offsets,
                clusters.indices,
                clusters.particle_cluster_counts,
                clusters.particle_cluster_inv_weights,
            ],
            device=self.model.device,
        )

    def _deactivate_all_single_deleted_clusters(self) -> None:
        self._deactivate_single_deleted_cluster(self.aux.shape_matching_clusters)
        hierarchy = self.aux.shape_matching_hierarchy
        if hierarchy is None:
            return
        self._deactivate_single_deleted_cluster(hierarchy.outer8)
        self._deactivate_single_deleted_cluster(hierarchy.full27)
        self._deactivate_single_deleted_cluster(hierarchy.l2_outer8)

    def _launch_single_delete_gpu_from_device(self, cell_ids: wp.array) -> DeviceDeletionResult:
        deleted_cells_device = self._ensure_deleted_cells_buffer(1)
        self._delete_counter.zero_()
        wp.launch(
            delete_single_cell_complete_kernel,
            dim=1,
            inputs=[
                cell_ids,
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.cell_mass,
                self.aux.cell_constraint_offsets,
                self.aux.cell_constraint_indices,
                self.aux.cell_active,
                self.aux.node_support_count,
                self.node_mass_device,
                self.locked_node_mask_device,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.model.particle_flags,
                self.aux.spring_support_count,
                self.model.spring_indices,
                self.aux.node_material,
                self.material_stiffness_scale_device,
                self.aux.spring_unit_stiffness,
                self.aux.spring_unit_damping,
                self.aux.spring_enabled,
                self.model.spring_stiffness,
                self.model.spring_damping,
                deleted_cells_device,
                self._delete_counter,
                self._deleted_total_device,
            ],
            device=self.model.device,
        )

        self.last_deleted_cells_device = deleted_cells_device
        self.last_deleted_count_device = self._delete_counter
        self.last_deleted_cells_host = np.empty(0, dtype=np.int32)
        self.last_deleted_count = 0
        self.last_deleted_topology_revision = int(self.topology_revision)
        self._last_deleted_host_synced = False
        self._deactivate_all_single_deleted_clusters()

        result = DeviceDeletionResult(
            deleted_cells_device=deleted_cells_device,
            deleted_count_device=self._delete_counter,
            candidate_capacity=1,
            topology_revision=int(self.topology_revision),
            host_synced=False,
        )
        self.last_deletion_result = result
        self._mark_dirty("cells", "nodes", "mass", "springs", "clusters", "last-delete", "stats")
        return result

    def _launch_sparse_delete_gpu_from_device(
        self,
        cell_ids: wp.array,
        candidate_capacity: int,
        candidate_count_device: wp.array | None = None,
    ) -> DeviceDeletionResult:
        candidate_capacity = int(candidate_capacity)
        if candidate_capacity <= 0:
            self._clear_last_deleted()
            return DeviceDeletionResult(
                deleted_cells_device=None,
                deleted_count_device=self._delete_counter,
                candidate_capacity=0,
                topology_revision=int(self.topology_revision),
                host_synced=True,
            )
        if candidate_capacity > int(cell_ids.shape[0]):
            raise ValueError(
                "candidate_capacity exceeds cell_ids length: "
                f"{candidate_capacity} > {int(cell_ids.shape[0])}"
            )
        if candidate_count_device is None:
            self._candidate_count.fill_(candidate_capacity)
            candidate_count_device = self._candidate_count

        deleted_cells_device = self._ensure_deleted_cells_buffer(candidate_capacity)
        self._delete_counter.zero_()
        wp.launch(
            delete_cells_sparse_kernel,
            dim=candidate_capacity,
            inputs=[
                cell_ids,
                candidate_capacity,
                candidate_count_device,
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.cell_mass,
                self.aux.cell_constraint_offsets,
                self.aux.cell_constraint_indices,
                self.aux.cell_active,
                self.aux.node_support_count,
                self.node_mass_device,
                self.aux.spring_support_count,
                deleted_cells_device,
                self._delete_counter,
                self._deleted_total_device,
            ],
            device=self.model.device,
        )

        self.last_deleted_cells_device = deleted_cells_device
        self.last_deleted_count_device = self._delete_counter
        self.last_deleted_cells_host = np.empty(0, dtype=np.int32)
        self.last_deleted_count = 0
        self.last_deleted_topology_revision = int(self.topology_revision)
        self._last_deleted_host_synced = False

        wp.launch(
            finalize_deleted_cell_nodes_kernel,
            dim=(candidate_capacity, 8),
            inputs=[
                deleted_cells_device,
                self._delete_counter,
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.node_support_count,
                self.node_mass_device,
                self.locked_node_mask_device,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.model.particle_flags,
            ],
            device=self.model.device,
        )
        wp.launch(
            finalize_deleted_cell_springs_kernel,
            dim=(candidate_capacity, 28),
            inputs=[
                deleted_cells_device,
                self._delete_counter,
                int(self.aux.num_cells),
                self.aux.cell_constraint_offsets,
                self.aux.cell_constraint_indices,
                self.aux.spring_support_count,
                self.model.spring_indices,
                self.aux.node_material,
                self.material_stiffness_scale_device,
                self.aux.spring_unit_stiffness,
                self.aux.spring_unit_damping,
                self.aux.spring_enabled,
                self.model.spring_stiffness,
                self.model.spring_damping,
            ],
            device=self.model.device,
        )
        self._deactivate_all_shape_matching_clusters(candidate_capacity)

        result = DeviceDeletionResult(
            deleted_cells_device=deleted_cells_device,
            deleted_count_device=self._delete_counter,
            candidate_capacity=candidate_capacity,
            topology_revision=int(self.topology_revision),
            host_synced=False,
        )
        self.last_deletion_result = result
        self._mark_dirty("cells", "nodes", "mass", "springs", "clusters", "last-delete", "stats")
        return result

    def _refresh_shape_matching_clusters(self) -> None:
        clusters = self.aux.shape_matching_clusters
        if clusters is None:
            return

        active = (self.cell_active[clusters.source_cell_host] != 0).astype(np.int32, copy=False)
        particle_cluster_counts = np.zeros(self.model.particle_count, dtype=np.int32)
        active_cluster_ids = np.nonzero(active != 0)[0]

        if active_cluster_ids.size > 0:
            if clusters.uniform_size == 8 and clusters.indices_host.size == clusters.num_clusters * 8:
                member_indices = clusters.indices_host.reshape(clusters.num_clusters, 8)[active_cluster_ids].reshape(-1)
                np.add.at(particle_cluster_counts, member_indices, 1)
            else:
                for cluster_idx in active_cluster_ids.tolist():
                    start = int(clusters.offsets_host[cluster_idx])
                    end = int(clusters.offsets_host[cluster_idx + 1])
                    np.add.at(particle_cluster_counts, clusters.indices_host[start:end], 1)

        particle_cluster_inv_weights = np.zeros(self.model.particle_count, dtype=np.float32)
        valid = particle_cluster_counts > 0
        particle_cluster_inv_weights[valid] = 1.0 / particle_cluster_counts[valid].astype(np.float32)

        clusters.active_host[:] = active
        clusters.particle_cluster_counts_host[:] = particle_cluster_counts
        clusters.particle_cluster_inv_weights_host[:] = particle_cluster_inv_weights
        clusters.active.assign(clusters.active_host)
        clusters.particle_cluster_counts.assign(clusters.particle_cluster_counts_host)
        clusters.particle_cluster_inv_weights.assign(clusters.particle_cluster_inv_weights_host)

    def set_family_stiffnesses(
        self,
        edge: float | None = None,
        face: float | None = None,
        body: float | None = None,
    ) -> None:
        """Update the per-family unit stiffness and push live spring values."""
        spring_type = self.aux.spring_type_host
        changed = False
        if edge is not None:
            self.aux.spring_unit_stiffness_host[spring_type == CONSTRAINT_EDGE] = float(edge)
            changed = True
        if face is not None:
            self.aux.spring_unit_stiffness_host[spring_type == CONSTRAINT_FACE_DIAGONAL] = float(face)
            changed = True
        if body is not None:
            self.aux.spring_unit_stiffness_host[spring_type == CONSTRAINT_BODY_DIAGONAL] = float(body)
            changed = True
        if changed:
            self._refresh_springs()

    def set_material_stiffness_scale(self, values) -> None:
        """Update per-material spring stiffness multipliers and push live spring values."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != self.material_stiffness_scale.shape:
            raise ValueError(
                "material stiffness scale shape mismatch: "
                f"expected {self.material_stiffness_scale.shape}, got {arr.shape}"
            )
        self.material_stiffness_scale = arr.copy()
        self.material_stiffness_scale_device.assign(self.material_stiffness_scale)
        self._refresh_springs()

    def lock_nodes(self, node_indices: np.ndarray | list[int]) -> int:
        """Set selected, still-supported nodes to kinematic mass/inv-mass zero.

        ``node_mass`` remains the support-derived physical mass for deletion
        bookkeeping; the model receives an effective mass with locked nodes
        zeroed so later cell deletions cannot underflow the host mirror.
        """
        if self._dirty_domains.intersection({"nodes", "mass"}):
            self.sync_host_mirrors({"nodes", "mass"})
        candidates = np.asarray(node_indices, dtype=np.int64).reshape(-1)
        if candidates.size == 0:
            return 0
        valid = (candidates >= 0) & (candidates < self.model.particle_count)
        if not np.any(valid):
            return 0
        nodes = np.unique(candidates[valid].astype(np.int32, copy=False))
        supported = self.node_support_count[nodes] > 0
        nodes = nodes[supported]
        if nodes.size == 0:
            return 0

        newly_locked = nodes[~self.locked_node_mask[nodes]]
        self.locked_node_mask[nodes] = True
        self.node_inv_mass[nodes] = 0.0
        self._push_effective_node_masses()
        return int(newly_locked.size)

    def set_locked_nodes(self, node_indices: np.ndarray | list[int]) -> int:
        """Replace the current locked-node set and restore masses for unlocked nodes."""
        if self._dirty_domains.intersection({"nodes", "mass"}):
            self.sync_host_mirrors({"nodes", "mass"})

        candidates = np.asarray(node_indices, dtype=np.int64).reshape(-1)
        lock_mask = np.zeros(self.model.particle_count, dtype=bool)
        if candidates.size > 0:
            valid = (candidates >= 0) & (candidates < self.model.particle_count)
            if np.any(valid):
                nodes = np.unique(candidates[valid].astype(np.int32, copy=False))
                supported = self.node_support_count[nodes] > 0
                lock_mask[nodes[supported]] = True

        self.locked_node_mask[:] = lock_mask
        self.node_inv_mass.fill(0.0)
        movable = (self.node_support_count > 0) & (self.node_mass > 0.0) & ~self.locked_node_mask
        self.node_inv_mass[movable] = 1.0 / self.node_mass[movable]
        self._push_effective_node_masses()
        return int(np.count_nonzero(self.locked_node_mask))

    def unlock_all_nodes(self) -> int:
        """Restore dynamic mass/inv-mass for every still-supported locked node."""
        if self._dirty_domains.intersection({"nodes", "mass"}):
            self.sync_host_mirrors({"nodes", "mass"})
        unlocked_count = int(np.count_nonzero(self.locked_node_mask))
        if unlocked_count == 0:
            return 0

        self.locked_node_mask.fill(False)
        self.node_inv_mass.fill(0.0)
        movable = (self.node_support_count > 0) & (self.node_mass > 0.0)
        self.node_inv_mass[movable] = 1.0 / self.node_mass[movable]
        self._push_effective_node_masses()
        return unlocked_count

    def _apply_deleted_cells_host(
        self,
        cell_indices: np.ndarray | list[int],
        *,
        ignore_invalid: bool = False,
    ) -> np.ndarray:
        """Apply deletion bookkeeping to host mirrors and return cells changed."""
        candidates = np.asarray(cell_indices, dtype=np.int64).reshape(-1)
        if candidates.size == 0:
            return np.empty(0, dtype=np.int32)
        self.capture_reset_baseline()

        active_bit = int(ParticleFlags.ACTIVE)
        deleted_cells: list[int] = []
        touched_nodes: list[int] = []
        touched_springs: list[int] = []
        for raw_idx in candidates.tolist():
            cell_idx = int(raw_idx)
            if cell_idx < 0 or cell_idx >= self.aux.num_cells:
                if ignore_invalid:
                    continue
                raise IndexError(f"cell index out of range: {cell_idx}")
            if self.cell_active[cell_idx] == 0:
                continue
            self.cell_active[cell_idx] = 0
            deleted_cells.append(cell_idx)

            cell_nodes = self.aux.cell_nodes_host[cell_idx]
            touched_nodes.extend(int(n) for n in cell_nodes.tolist())
            self.node_support_count[cell_nodes] -= 1
            self.node_mass[cell_nodes] -= self.aux.cell_mass_host[cell_idx] * (1.0 / 8.0)

            start = int(self.aux.cell_constraint_offsets_host[cell_idx])
            end = int(self.aux.cell_constraint_offsets_host[cell_idx + 1])
            spring_ids = self.aux.cell_constraint_indices_host[start:end]
            touched_springs.extend(int(s) for s in spring_ids.tolist())
            self.spring_support_count[spring_ids] -= 1

        deleted_now = len(deleted_cells)
        if deleted_now == 0:
            return np.empty(0, dtype=np.int32)

        deleted_cells_np = np.asarray(deleted_cells, dtype=np.int32)

        touched_node_arr = np.asarray(touched_nodes, dtype=np.int32)
        touched_spring_arr = np.asarray(touched_springs, dtype=np.int32)
        unique_nodes = (
            np.unique(touched_node_arr)
            if touched_node_arr.size > 0
            else np.empty(0, dtype=np.int32)
        )
        unique_springs = (
            np.unique(touched_spring_arr)
            if touched_spring_arr.size > 0
            else np.empty(0, dtype=np.int32)
        )

        if unique_nodes.size > 0 and np.any(self.node_support_count[unique_nodes] < 0):
            raise RuntimeError("node support count underflow during cell deletion")
        if unique_springs.size > 0 and np.any(self.spring_support_count[unique_springs] < 0):
            raise RuntimeError("spring support count underflow during cell deletion")

        if unique_nodes.size > 0:
            supported_nodes = self.node_support_count[unique_nodes] > 0
            self.node_mass[unique_nodes[~supported_nodes]] = 0.0
            self.locked_node_mask[unique_nodes[~supported_nodes]] = False
            self.node_mass[unique_nodes] = np.maximum(self.node_mass[unique_nodes], 0.0)
            positive_mass = self.node_mass[unique_nodes] > 0.0
            self.node_inv_mass[unique_nodes] = 0.0
            movable = positive_mass & ~self.locked_node_mask[unique_nodes]
            inv_nodes = unique_nodes[movable]
            self.node_inv_mass[inv_nodes] = 1.0 / self.node_mass[inv_nodes]
            self.particle_flags[unique_nodes[supported_nodes]] |= active_bit
            self.particle_flags[unique_nodes[~supported_nodes]] &= ~active_bit

        self.deleted_total += deleted_now
        self.aux.cell_active_host = self.cell_active
        self.aux.node_support_count_host = self.node_support_count

        self._refresh_touched_springs_host(unique_springs)
        self._refresh_touched_shape_matching_host(
            deleted_cells_np,
            unique_nodes,
        )
        return deleted_cells_np

    def _finish_sync_delete(self, previous_revision: int) -> int:
        deleted_now = self.sync_last_deleted()
        if deleted_now <= 0:
            self.topology_revision = int(previous_revision)
            self._clear_last_deleted()
            self._dirty_domains.clear()
            return 0
        self.last_deleted_cells_device = wp.array(
            self.last_deleted_cells_host,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.last_deleted_count_device = wp.array(
            np.asarray([deleted_now], dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.last_deletion_result = DeviceDeletionResult(
            deleted_cells_device=self.last_deleted_cells_device,
            deleted_count_device=self.last_deleted_count_device,
            candidate_capacity=int(deleted_now),
            topology_revision=int(self.topology_revision),
            host_synced=True,
        )
        self.sync_host_mirrors("all")
        return int(deleted_now)

    def delete_device_cells_async(
        self,
        cell_ids: wp.array,
        delete_count: int | None = None,
        *,
        candidate_count_device: wp.array | None = None,
        optimistic_revision: bool = True,
    ) -> DeviceDeletionResult:
        """Launch deletion from a device id array without synchronizing host mirrors."""
        if delete_count is None:
            delete_count = int(cell_ids.shape[0])
        delete_count = int(delete_count)
        if delete_count <= 0:
            self._clear_last_deleted()
            return DeviceDeletionResult(
                deleted_cells_device=None,
                deleted_count_device=self._delete_counter,
                candidate_capacity=0,
                topology_revision=int(self.topology_revision),
                host_synced=True,
            )
        self.capture_reset_baseline()
        if optimistic_revision:
            self.topology_revision += 1
        if delete_count == 1 and candidate_count_device is None:
            result = self._launch_single_delete_gpu_from_device(cell_ids)
        else:
            result = self._launch_sparse_delete_gpu_from_device(
                cell_ids,
                delete_count,
                candidate_count_device=candidate_count_device,
            )
        result.topology_revision = int(self.topology_revision)
        self.last_deleted_topology_revision = int(self.topology_revision)
        return result

    def delete_device_cells(self, cell_ids: wp.array, delete_count: int | None = None) -> int:
        """Compatibility wrapper: delete device cells and refresh host mirrors."""
        previous_revision = int(self.topology_revision)
        self.delete_device_cells_async(cell_ids, delete_count, optimistic_revision=True)
        return self._finish_sync_delete(previous_revision)

    def delete_cells_async(
        self,
        cell_indices: np.ndarray | list[int],
        *,
        optimistic_revision: bool = True,
    ) -> DeviceDeletionResult:
        """Launch deletion from host-provided cell ids without syncing mirrors."""
        candidates = np.asarray(cell_indices, dtype=np.int64).reshape(-1)
        if candidates.size == 0:
            self._clear_last_deleted()
            return DeviceDeletionResult(
                deleted_cells_device=None,
                deleted_count_device=self._delete_counter,
                candidate_capacity=0,
                topology_revision=int(self.topology_revision),
                host_synced=True,
            )
        invalid = (candidates < 0) | (candidates >= self.aux.num_cells)
        if np.any(invalid):
            raise IndexError(f"cell index out of range: {int(candidates[np.flatnonzero(invalid)[0]])}")
        cell_ids = wp.array(
            candidates.astype(np.int32, copy=False),
            dtype=wp.int32,
            device=self.model.device,
        )
        self._last_candidate_cells_device = cell_ids
        return self.delete_device_cells_async(
            cell_ids,
            int(candidates.size),
            optimistic_revision=optimistic_revision,
        )

    def delete_cells(self, cell_indices: np.ndarray | list[int]) -> int:
        """Deactivate specific cells and update shared node / spring state."""
        previous_revision = int(self.topology_revision)
        self.delete_cells_async(cell_indices, optimistic_revision=True)
        return self._finish_sync_delete(previous_revision)

    def delete_cells_by_tool(self, tool: Tool, particle_q) -> int:
        """Delete active cells whose deformed centroids intersect an active tool."""
        if not tool.active:
            return 0

        self.sync_host_mirrors({"cells"})
        q = particle_q if isinstance(particle_q, np.ndarray) else particle_q.numpy()
        active_cells = self.cell_active != 0
        if not np.any(active_cells):
            return 0

        active_indices = np.nonzero(active_cells)[0]
        centers = q[self.aux.cell_nodes_host[active_indices]].mean(axis=1)

        delete_mask = np.zeros(active_indices.shape[0], dtype=bool)
        seg_p0 = tool.segments.p0.numpy()
        seg_p1 = tool.segments.p1.numpy()
        seg_radius = tool.segments.radius.numpy()
        seg_role = tool.segments.role_electrode.numpy()

        for seg_idx in range(int(tool.segments.num_segments)):
            if int(seg_role[seg_idx]) == 0:
                continue
            radius = float(seg_radius[seg_idx]) + float(self.delete_padding)
            dist_sq = _point_segment_distance_sq(centers, seg_p0[seg_idx], seg_p1[seg_idx])
            delete_mask |= dist_sq <= radius * radius

        if not np.any(delete_mask):
            return 0
        return self.delete_cells(active_indices[delete_mask])

    def delete_cells_by_ray_surface_async(
        self,
        particle_q: wp.array,
        ray0_origin,
        ray0_direction,
        ray1_origin,
        ray1_direction,
        depth: float,
        material_cuttable: wp.array | None = None,
        padding: float | None = None,
    ) -> int:
        """Delete active cells intersecting the finite surface between two rays."""
        depth = float(depth)
        if depth <= 0.0:
            self._clear_last_deleted()
            return DeviceDeletionResult(
                deleted_cells_device=None,
                deleted_count_device=self._delete_counter,
                candidate_capacity=0,
                topology_revision=int(self.topology_revision),
                host_synced=True,
            )

        origin0 = np.asarray(ray0_origin, dtype=np.float32).reshape(3)
        origin1 = np.asarray(ray1_origin, dtype=np.float32).reshape(3)
        direction0 = _normalize_vec3(ray0_direction, "ray0_direction")
        direction1 = _normalize_vec3(ray1_direction, "ray1_direction")
        end0 = origin0 + direction0 * depth
        end1 = origin1 + direction1 * depth

        cut_padding = self.delete_padding if padding is None else float(padding)
        cut_padding = max(0.0, cut_padding)

        self._plane_candidate_count.zero_()
        wp.launch(
            select_ray_surface_cells_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.cell_material,
                self.aux.cell_active,
                material_cuttable if material_cuttable is not None else self._dummy_int,
                int(material_cuttable is not None),
                particle_q,
                wp.vec3(float(origin0[0]), float(origin0[1]), float(origin0[2])),
                wp.vec3(float(end0[0]), float(end0[1]), float(end0[2])),
                wp.vec3(float(origin1[0]), float(origin1[1]), float(origin1[2])),
                wp.vec3(float(end1[0]), float(end1[1]), float(end1[2])),
                cut_padding,
            ],
            outputs=[self._plane_candidate_cells, self._plane_candidate_count],
            device=self.model.device,
        )
        self.topology_revision += 1
        result = self._launch_sparse_delete_gpu_from_device(
            self._plane_candidate_cells,
            int(self.aux.num_cells),
            candidate_count_device=self._plane_candidate_count,
        )
        result.topology_revision = int(self.topology_revision)
        self.last_deleted_topology_revision = int(self.topology_revision)
        return result

    def delete_cells_by_ray_surface(
        self,
        particle_q: wp.array,
        ray0_origin,
        ray0_direction,
        ray1_origin,
        ray1_direction,
        depth: float,
        material_cuttable: wp.array | None = None,
        padding: float | None = None,
    ) -> int:
        """Delete active cells intersecting the finite surface between two rays."""
        previous_revision = int(self.topology_revision)
        self.delete_cells_by_ray_surface_async(
            particle_q,
            ray0_origin,
            ray0_direction,
            ray1_origin,
            ray1_direction,
            depth,
            material_cuttable=material_cuttable,
            padding=padding,
        )
        return self._finish_sync_delete(previous_revision)

    def delete_cells_by_spheres_particle_contacts_async(
        self,
        particle_q: wp.array,
        sphere_q: wp.array,
        sphere_cut_enabled: wp.array,
        sphere_count: int,
        sphere_radius: float,
        material_cuttable: wp.array | None = None,
    ) -> DeviceDeletionResult:
        """Delete cells with active corner particles intersecting enabled kinematic spheres."""
        sphere_count = int(sphere_count)
        sphere_radius = float(sphere_radius)
        if sphere_count <= 0 or sphere_radius <= 0.0:
            self._clear_last_deleted()
            return DeviceDeletionResult(
                deleted_cells_device=None,
                deleted_count_device=self._delete_counter,
                candidate_capacity=0,
                topology_revision=int(self.topology_revision),
                host_synced=True,
            )

        self._plane_candidate_count.zero_()
        wp.launch(
            select_sphere_particle_contact_cells_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.cell_material,
                self.aux.cell_active,
                material_cuttable if material_cuttable is not None else self._dummy_int,
                int(material_cuttable is not None),
                particle_q,
                self.model.particle_radius,
                self.model.particle_flags,
                sphere_q,
                sphere_cut_enabled,
                sphere_count,
                sphere_radius,
            ],
            outputs=[self._plane_candidate_cells, self._plane_candidate_count],
            device=self.model.device,
        )

        self.topology_revision += 1
        result = self._launch_sparse_delete_gpu_from_device(
            self._plane_candidate_cells,
            int(self.aux.num_cells),
            candidate_count_device=self._plane_candidate_count,
        )
        result.topology_revision = int(self.topology_revision)
        self.last_deleted_topology_revision = int(self.topology_revision)
        return result

    def delete_cells_by_ray_segment_from_cell_async(
        self,
        particle_q: wp.array,
        start_cell_ids: wp.array,
        ray_direction,
        depth: float,
        material_cuttable: wp.array | None = None,
        padding: float | None = None,
    ) -> int:
        """Delete active cells along a ray segment starting at a picked cell centre."""
        depth = float(depth)
        if depth <= 0.0:
            self._clear_last_deleted()
            return DeviceDeletionResult(
                deleted_cells_device=None,
                deleted_count_device=self._delete_counter,
                candidate_capacity=0,
                topology_revision=int(self.topology_revision),
                host_synced=True,
            )

        direction = _normalize_vec3(ray_direction, "ray_direction")
        cut_padding = self.delete_padding if padding is None else float(padding)
        cut_padding = max(0.0, cut_padding)

        self._plane_candidate_count.zero_()
        wp.launch(
            select_ray_segment_cells_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.cell_material,
                self.aux.cell_active,
                material_cuttable if material_cuttable is not None else self._dummy_int,
                int(material_cuttable is not None),
                particle_q,
                start_cell_ids,
                wp.vec3(float(direction[0]), float(direction[1]), float(direction[2])),
                depth,
                cut_padding,
            ],
            outputs=[self._plane_candidate_cells, self._plane_candidate_count],
            device=self.model.device,
        )

        self.topology_revision += 1
        result = self._launch_sparse_delete_gpu_from_device(
            self._plane_candidate_cells,
            int(self.aux.num_cells),
            candidate_count_device=self._plane_candidate_count,
        )
        result.topology_revision = int(self.topology_revision)
        self.last_deleted_topology_revision = int(self.topology_revision)
        return result

    def delete_cells_by_ray_segment_from_cell(
        self,
        particle_q: wp.array,
        start_cell_ids: wp.array,
        ray_direction,
        depth: float,
        material_cuttable: wp.array | None = None,
        padding: float | None = None,
    ) -> int:
        """Delete active cells along a ray segment starting at a picked cell centre."""
        previous_revision = int(self.topology_revision)
        self.delete_cells_by_ray_segment_from_cell_async(
            particle_q,
            start_cell_ids,
            ray_direction,
            depth,
            material_cuttable=material_cuttable,
            padding=padding,
        )
        return self._finish_sync_delete(previous_revision)

    def validate_device_state(self) -> None:
        """Synchronize a small error counter and raise if device state is inconsistent."""
        self._validation_error_count.zero_()
        self._validation_first_error.zero_()
        wp.launch(
            validate_cell_active_kernel,
            dim=int(self.aux.num_cells),
            inputs=[self.aux.cell_active, self._validation_error_count, self._validation_first_error],
            device=self.model.device,
        )
        wp.launch(
            validate_node_state_kernel,
            dim=int(self.model.particle_count),
            inputs=[
                self.aux.node_support_count,
                self.node_mass_device,
                self.locked_node_mask_device,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.model.particle_flags,
                self._validation_error_count,
                self._validation_first_error,
            ],
            device=self.model.device,
        )
        wp.launch(
            validate_spring_state_kernel,
            dim=int(self.model.spring_count),
            inputs=[
                self.aux.spring_support_count,
                self.aux.spring_enabled,
                self.model.spring_stiffness,
                self.model.spring_damping,
                self._validation_error_count,
                self._validation_first_error,
            ],
            device=self.model.device,
        )
        for clusters in (
            self.aux.shape_matching_clusters,
            None if self.aux.shape_matching_hierarchy is None else self.aux.shape_matching_hierarchy.outer8,
            None if self.aux.shape_matching_hierarchy is None else self.aux.shape_matching_hierarchy.full27,
            None if self.aux.shape_matching_hierarchy is None else self.aux.shape_matching_hierarchy.l2_outer8,
        ):
            if clusters is None:
                continue
            wp.launch(
                validate_cluster_state_kernel,
                dim=max(int(clusters.num_clusters), int(self.model.particle_count)),
                inputs=[
                    clusters.active,
                    clusters.particle_cluster_counts,
                    clusters.particle_cluster_inv_weights,
                    self._validation_error_count,
                    self._validation_first_error,
                ],
                device=self.model.device,
            )
        errors = int(self._validation_error_count.numpy()[0])
        if errors > 0:
            code = int(self._validation_first_error.numpy()[0])
            raise RuntimeError(f"device deletion state validation failed: {errors} errors, first code {code}")


def make_corner_deletion_state(
    model: newton.Model,
    aux: CornerGridAuxState,
    delete_padding: float | None = None,
) -> CornerDeletionState:
    """Build the host-side deletion state for a corner lattice."""
    if delete_padding is None:
        delete_padding = aux.voxel_size * np.sqrt(3.0) * 0.5
    return CornerDeletionState(
        model=model,
        aux=aux,
        delete_padding=float(delete_padding),
    )


__all__ = [
    "CornerDeletionState",
    "DeviceDeletionResult",
    "make_corner_deletion_state",
]

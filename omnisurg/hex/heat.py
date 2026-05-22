# SPDX-License-Identifier: Apache-2.0
"""Host-side cell-centred heat driver for hex-grid diathermy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from .deletion import HexDeletionState, DeviceDeletionResult
from .hex_grid import HexGridAuxState
from .kernels.cell_heat import (
    apply_diathermy_spheres_kernel,
    diffuse_cell_heat_kernel,
    gather_heat_overlay_kernel,
    reset_deleted_cell_heat_kernel,
    select_capsule_cells_kernel,
    select_overheated_cells_kernel,
)


@dataclass(frozen=True)
class HexHeatSelection:
    """Device-resident selected cell candidates."""

    cell_ids: wp.array
    count_device: wp.array
    capacity: int


@dataclass
class HexHeatState:
    """Mutable per-cell heat buffers for the hex particle lattice."""

    model: newton.Model
    aux: HexGridAuxState
    material_resistance: wp.array
    material_conductivity: wp.array
    cell_heat_a: wp.array
    cell_heat_b: wp.array
    cell_burn: wp.array
    fulguration: float = 0.05

    def __post_init__(self) -> None:
        device = self.model.device
        self._candidate_cells = wp.zeros(self.aux.num_cells, dtype=wp.int32, device=device)
        self._candidate_count = wp.zeros(1, dtype=wp.int32, device=device)
        self._dummy_int = wp.zeros(1, dtype=wp.int32, device=device)
        self._dummy_float = wp.zeros(1, dtype=wp.float32, device=device)
        self._overlay_points = wp.zeros(self.aux.num_cells, dtype=wp.vec3, device=device)
        self._overlay_colors = wp.zeros(self.aux.num_cells, dtype=wp.vec3, device=device)
        self._overlay_radii = wp.zeros(self.aux.num_cells, dtype=wp.float32, device=device)
        self._overlay_count = wp.zeros(1, dtype=wp.int32, device=device)
        self._overlay_last_radius: float | None = None
        self.last_heat_min: float = 0.0
        self.last_heat_max: float = 0.0

    @property
    def device(self):
        return self.model.device

    def reset(self) -> None:
        """Clear thermal state without touching topology."""
        self.cell_heat_a.zero_()
        self.cell_heat_b.zero_()
        self.cell_burn.zero_()
        self._candidate_count.zero_()
        self._overlay_count.zero_()
        self.last_heat_min = 0.0
        self.last_heat_max = 0.0

    def apply_diathermy_spheres(
        self,
        particle_q: wp.array,
        sphere_q: wp.array,
        sphere_enabled: wp.array,
        sphere_count: int,
        sphere_radius: float,
        *,
        power: float,
        dt: float,
        sphere_power: wp.array | None = None,
        material_cuttable: wp.array | None = None,
    ) -> None:
        """Add heat to active cuttable cells touched by enabled instrument spheres."""
        sphere_count = int(sphere_count)
        if sphere_count <= 0:
            return
        heat_delta = max(0.0, float(dt))
        if sphere_power is None:
            heat_delta *= max(0.0, float(power))
        if heat_delta <= 0.0 or float(sphere_radius) <= 0.0:
            return
        wp.launch(
            apply_diathermy_spheres_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.cell_material,
                self.aux.cell_active,
                material_cuttable if material_cuttable is not None else self._dummy_int,
                int(material_cuttable is not None),
                particle_q,
                sphere_q,
                sphere_enabled,
                sphere_power if sphere_power is not None else self._dummy_float,
                int(sphere_power is not None),
                sphere_count,
                float(sphere_radius),
                float(heat_delta),
            ],
            outputs=[self.cell_heat_a],
            device=self.device,
        )

    def diffuse(self, *, dt: float, diffusion: float, cooling: float, substeps: int = 1) -> None:
        """Diffuse and cool cell heat across active 6-neighbour cells."""
        substeps = max(1, int(substeps))
        sub_dt = max(0.0, float(dt)) / float(substeps)
        diffusion = max(0.0, float(diffusion))
        cooling = max(0.0, float(cooling))
        nx, ny, nz = (int(v) for v in self.aux.grid_shape)
        for _ in range(substeps):
            wp.launch(
                diffuse_cell_heat_kernel,
                dim=int(self.aux.num_cells),
                inputs=[
                    int(self.aux.num_cells),
                    self.aux.cell_grid_xyz,
                    self.aux.cell_material,
                    self.aux.cell_active,
                    self.aux.grid_to_cell,
                    self.material_conductivity,
                    self.cell_heat_a,
                    float(sub_dt),
                    diffusion,
                    cooling,
                    nx,
                    ny,
                    nz,
                ],
                outputs=[self.cell_heat_b],
                device=self.device,
            )
            self.cell_heat_a, self.cell_heat_b = self.cell_heat_b, self.cell_heat_a

    def select_overheated_cells(self, material_cuttable: wp.array | None = None) -> HexHeatSelection:
        """Select active cuttable cells above material heat resistance."""
        self._candidate_count.zero_()
        wp.launch(
            select_overheated_cells_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_material,
                self.aux.cell_active,
                self.material_resistance,
                material_cuttable if material_cuttable is not None else self._dummy_int,
                int(material_cuttable is not None),
                self.cell_heat_a,
                float(self.fulguration),
                self.cell_burn,
            ],
            outputs=[self._candidate_cells, self._candidate_count],
            device=self.device,
        )
        return HexHeatSelection(
            cell_ids=self._candidate_cells,
            count_device=self._candidate_count,
            capacity=int(self.aux.num_cells),
        )

    def delete_overheated_cells_async(
        self,
        delete_state: HexDeletionState,
        material_cuttable: wp.array | None = None,
    ) -> DeviceDeletionResult:
        """Select overheated cells and commit them through ``HexDeletionState``."""
        selection = self.select_overheated_cells(material_cuttable=material_cuttable)
        result = delete_state.delete_device_cells_async(
            selection.cell_ids,
            selection.capacity,
            candidate_count_device=selection.count_device,
        )
        self.reset_deleted_cells()
        return result

    def step_diathermy_async(
        self,
        delete_state: HexDeletionState,
        particle_q: wp.array,
        sphere_q: wp.array,
        sphere_enabled: wp.array,
        sphere_count: int,
        sphere_radius: float,
        *,
        dt: float,
        power: float,
        diffusion: float,
        cooling: float,
        substeps: int = 1,
        sphere_power: wp.array | None = None,
        material_cuttable: wp.array | None = None,
    ) -> DeviceDeletionResult:
        """Run heat injection, diffusion/cooling, and overheated-cell deletion."""
        substeps = max(1, int(substeps))
        sub_dt = max(0.0, float(dt)) / float(substeps)
        for _ in range(substeps):
            self.apply_diathermy_spheres(
                particle_q,
                sphere_q,
                sphere_enabled,
                sphere_count,
                sphere_radius,
                power=power,
                dt=sub_dt,
                sphere_power=sphere_power,
                material_cuttable=material_cuttable,
            )
            self.diffuse(dt=sub_dt, diffusion=diffusion, cooling=cooling, substeps=1)
        return self.delete_overheated_cells_async(delete_state, material_cuttable=material_cuttable)

    def select_capsule_cells(
        self,
        particle_q: wp.array,
        p0,
        p1,
        radius: float,
        material_cuttable: wp.array | None = None,
    ) -> HexHeatSelection:
        """Select active cuttable cells intersecting a world-space capsule."""
        start = np.asarray(p0, dtype=np.float32).reshape(3)
        end = np.asarray(p1, dtype=np.float32).reshape(3)
        self._candidate_count.zero_()
        wp.launch(
            select_capsule_cells_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_nodes,
                self.aux.cell_material,
                self.aux.cell_active,
                material_cuttable if material_cuttable is not None else self._dummy_int,
                int(material_cuttable is not None),
                particle_q,
                wp.vec3(float(start[0]), float(start[1]), float(start[2])),
                wp.vec3(float(end[0]), float(end[1]), float(end[2])),
                max(0.0, float(radius)),
            ],
            outputs=[self._candidate_cells, self._candidate_count],
            device=self.device,
        )
        return HexHeatSelection(
            cell_ids=self._candidate_cells,
            count_device=self._candidate_count,
            capacity=int(self.aux.num_cells),
        )

    def delete_cells_by_capsule_async(
        self,
        delete_state: HexDeletionState,
        particle_q: wp.array,
        p0,
        p1,
        radius: float,
        material_cuttable: wp.array | None = None,
    ) -> DeviceDeletionResult:
        """Delete active cuttable cells intersecting a capsule through ``delete_state``."""
        selection = self.select_capsule_cells(
            particle_q,
            p0,
            p1,
            radius,
            material_cuttable=material_cuttable,
        )
        result = delete_state.delete_device_cells_async(
            selection.cell_ids,
            selection.capacity,
            candidate_count_device=selection.count_device,
        )
        self.reset_deleted_cells()
        return result

    def reset_deleted_cells(self) -> None:
        """Clear heat/burn on cells already removed from the hex grid."""
        wp.launch(
            reset_deleted_cell_heat_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_active,
            ],
            outputs=[self.cell_heat_a, self.cell_heat_b, self.cell_burn],
            device=self.device,
        )

    def sync_heat_min_max(self) -> tuple[float, float]:
        """Synchronize heat to host and update the last min/max diagnostics."""
        heat = self.cell_heat_a.numpy()
        if heat.size == 0:
            self.last_heat_min = 0.0
            self.last_heat_max = 0.0
        else:
            self.last_heat_min = float(np.min(heat))
            self.last_heat_max = float(np.max(heat))
        return self.last_heat_min, self.last_heat_max

    def update_heat_overlay(
        self,
        viewer: Any,
        *,
        radius: float,
        hidden: bool = False,
        min_visible_heat: float = 1.0e-5,
    ) -> int:
        """Log heat-coloured cell centre points for debugging."""
        if hidden:
            viewer.draw_points(name="/hex_grid/heat", points=None, hidden=True)
            return 0
        _, heat_max = self.sync_heat_min_max()
        self._overlay_count.zero_()
        wp.launch(
            gather_heat_overlay_kernel,
            dim=int(self.aux.num_cells),
            inputs=[
                int(self.aux.num_cells),
                self.aux.cell_center_q,
                self.aux.cell_active,
                self.cell_heat_a,
                float(heat_max),
                float(min_visible_heat),
            ],
            outputs=[self._overlay_points, self._overlay_colors, self._overlay_count],
            device=self.device,
        )
        count = int(self._overlay_count.numpy()[0])
        if count <= 0:
            viewer.draw_points(name="/hex_grid/heat", points=None, hidden=True)
            return 0
        if self._overlay_last_radius != float(radius):
            self._overlay_radii.fill_(float(radius))
            self._overlay_last_radius = float(radius)
        viewer.draw_points(
            name="/hex_grid/heat",
            points=self._overlay_points[:count],
            radii=self._overlay_radii[:count],
            colors=self._overlay_colors[:count],
            hidden=False,
        )
        return count


def make_hex_heat_state(
    model: newton.Model,
    aux: HexGridAuxState,
    fulguration: float = 0.05,
) -> HexHeatState:
    """Build a :class:`HexHeatState` from a hex grid."""
    device = model.device
    mats = aux.materials
    return HexHeatState(
        model=model,
        aux=aux,
        material_resistance=wp.array(mats.resistance, dtype=wp.float32, device=device),
        material_conductivity=wp.array(mats.conductivity, dtype=wp.float32, device=device),
        cell_heat_a=wp.zeros(aux.num_cells, dtype=wp.float32, device=device),
        cell_heat_b=wp.zeros(aux.num_cells, dtype=wp.float32, device=device),
        cell_burn=wp.zeros(aux.num_cells, dtype=wp.float32, device=device),
        fulguration=fulguration,
    )


__all__ = [
    "HexHeatSelection",
    "HexHeatState",
    "make_hex_heat_state",
]

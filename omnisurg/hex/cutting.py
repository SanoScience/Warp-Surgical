# SPDX-License-Identifier: Apache-2.0
"""Per-frame driver for the electrocautery cutting pipeline.

Owns the double-buffered heat arrays and a snapshot of the original spring
stiffnesses (restored or zeroed by :func:`.kernels.heat.disable_cut_springs_kernel`).
A :class:`CuttingState` is cheap to construct from an existing
:class:`.grid.ParticleGrid` and should live for the whole simulation run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .grid import GridAuxState
from .kernels.heat import (
    apply_damage_kernel,
    disable_cut_springs_kernel,
    heat_apply_kernel,
    heat_diffuse_kernel,
)
from .tool import Tool


@dataclass
class CuttingState:
    """Mutable GPU state owned by the cutting pipeline.

    The only arrays that outlive a single step are ``heat_a``/``heat_b``, the
    ``spring_stiffness_base`` snapshot, and the ``material_stiffness_scale``
    slider-driven multiplier; the rest live on the Newton model or the grid
    aux state and are aliased here for convenience.
    """

    model: newton.Model
    aux: GridAuxState
    material_resistance: wp.array  # float32[n_materials]
    material_conductivity: wp.array  # float32[n_materials]
    material_stiffness_scale: wp.array  # float32[n_materials] - UI-driven, defaults to 1.0
    spring_stiffness_base: wp.array  # float32[num_springs] - original stiffness
    heat_a: wp.array  # float32[num_particles]
    heat_b: wp.array  # float32[num_particles]
    fulguration: float = 0.05
    device: wp.context.Device | None = None

    def set_material_stiffness_scale(self, values) -> None:
        """Upload a new per-material stiffness multiplier vector (length n_materials)."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != self.material_stiffness_scale.shape:
            raise ValueError(
                f"stiffness scale shape mismatch: expected {self.material_stiffness_scale.shape}, got {arr.shape}"
            )
        self.material_stiffness_scale.assign(arr)

    def step(self, tool: Tool, dt: float, particle_q: wp.array | None = None) -> None:
        """Run one frame of heat apply + diffuse + damage + spring pruning.

        Args:
            tool: tool pose + state to compare against particle positions.
            dt: frame time in seconds.
            particle_q: current particle positions (``State.particle_q``). When
                ``None`` the model's rest positions are used, which only makes
                sense for kinematic tests; in any live simulation pass the
                time-varying ``state.particle_q`` from the last solver step.
        """
        dev = self.device
        N = self.aux.num_particles
        q = particle_q if particle_q is not None else self.model.particle_q

        # 1. Inject heat where the electrode touches tissue.
        wp.launch(
            heat_apply_kernel,
            dim=N,
            inputs=[
                q,
                self.model.particle_flags,
                tool.segments.p0,
                tool.segments.p1,
                tool.segments.radius,
                tool.segments.role_electrode,
                int(tool.segments.num_segments),
                float(tool.power),
                int(1 if tool.active else 0),
            ],
            outputs=[self.heat_a],
            device=dev,
        )

        # 2. Diffuse heat across the 6-neighbour graph (paper Eq. 1).
        wp.launch(
            heat_diffuse_kernel,
            dim=N,
            inputs=[
                self.model.particle_flags,
                self.aux.particle_neighbors,
                self.aux.particle_material,
                self.material_conductivity,
                self.heat_a,
                float(dt),
            ],
            outputs=[self.heat_b],
            device=dev,
        )
        self.heat_a, self.heat_b = self.heat_b, self.heat_a

        # 3. Clear ACTIVE bit on particles above threshold + accumulate burn.
        wp.launch(
            apply_damage_kernel,
            dim=N,
            inputs=[
                self.aux.particle_material,
                self.material_resistance,
                self.heat_a,
                self.aux.particle_burnt,
                float(self.fulguration),
            ],
            outputs=[self.model.particle_flags],
            device=dev,
        )

        # 4. Update spring stiffness each frame: freshly-cut springs are zeroed
        #    once and never re-enabled; live springs are modulated by the
        #    current per-material stiffness multiplier. Skip when there are
        #    no springs (e.g. single-particle unit tests).
        if self.aux.num_springs > 0:
            wp.launch(
                disable_cut_springs_kernel,
                dim=self.aux.num_springs,
                inputs=[
                    self.model.spring_indices,
                    self.model.particle_flags,
                    self.aux.particle_material,
                    self.material_stiffness_scale,
                    self.spring_stiffness_base,
                ],
                outputs=[
                    self.aux.spring_enabled,
                    self.model.spring_stiffness,
                ],
                device=dev,
            )


def make_cutting_state(
    model: newton.Model,
    aux: GridAuxState,
    fulguration: float = 0.05,
) -> CuttingState:
    """Build a :class:`CuttingState` from a finalised model + grid aux."""
    device = model.device
    mats = aux.materials
    # Snapshot original spring stiffness so we can distinguish a PBD-time
    # value of 0 (never set) from a cut-induced 0 (was nonzero, then cleared).
    # A model with zero springs has ``spring_stiffness is None``; allocate an
    # empty array so kernel launches of size 0 remain no-ops.
    if model.spring_stiffness is None:
        base = wp.zeros(0, dtype=wp.float32, device=device)
    else:
        base = wp.clone(model.spring_stiffness)
    # Per-material stiffness multiplier - all 1.0 by default so step() is a
    # no-op relative to the pre-slider pipeline until the UI writes to it.
    scale = wp.array(np.ones(len(mats), dtype=np.float32), dtype=wp.float32, device=device)
    return CuttingState(
        model=model,
        aux=aux,
        material_resistance=wp.array(mats.resistance, dtype=wp.float32, device=device),
        material_conductivity=wp.array(mats.conductivity, dtype=wp.float32, device=device),
        material_stiffness_scale=scale,
        spring_stiffness_base=base,
        heat_a=wp.zeros(aux.num_particles, dtype=wp.float32, device=device),
        heat_b=wp.zeros(aux.num_particles, dtype=wp.float32, device=device),
        fulguration=fulguration,
        device=device,
    )

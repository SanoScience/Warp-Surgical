import warp as wp

from omnisurg.physics.base import SimulationSystem
from omnisurg.physics.kernels import (
    apply_deltas_and_zero_accumulators,
    apply_tri_points_constraints_jacobian,
    bounds_collision,
    solve_distance_constraints,
    solve_volume_constraints,
)


class BoundsCollisionSystem(SimulationSystem):
    def __init__(
        self,
        bounds_min,
        bounds_max,
        restitution: float = 0.0,
        friction: float = 0.0,
        priority: int = 100,
    ):
        super().__init__(priority=priority)
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.restitution = restitution
        self.friction = friction

    def solve_constraints(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        body_q,
        body_qd,
        body_deltas,
        dt,
        iteration,
    ):
        if model.particle_count <= 0:
            return

        wp.launch(
            kernel=bounds_collision,
            dim=model.particle_count,
            inputs=[
                particle_q,
                particle_qd,
                model.particle_inv_mass,
                self.bounds_min,
                self.bounds_max,
                self.restitution,
                self.friction,
                dt,
            ],
            device=model.device,
        )


class DistanceConstraintSystem(SimulationSystem):
    def __init__(self, priority: int = 50):
        super().__init__(priority=priority)
        self.particle_deltas_accumulator = None
        self.particle_deltas_count = None
        self.spring_constraint_lambdas = None

    def get_accumulators(self):
        if self.particle_deltas_accumulator is None:
            return None
        return (self.particle_deltas_accumulator, self.particle_deltas_count)

    def initialize(self, model):
        self.particle_deltas_accumulator = wp.zeros(
            model.particle_count,
            dtype=wp.vec3f,
            device=model.device,
        )
        self.particle_deltas_count = wp.zeros(
            model.particle_count,
            dtype=wp.int32,
            device=model.device,
        )
        if model.spring_count:
            self.spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)

    def solve_constraints(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        body_q,
        body_qd,
        body_deltas,
        dt,
        iteration,
    ):
        if not model.spring_count or self.spring_constraint_lambdas is None:
            return

        self.spring_constraint_lambdas.zero_()

        wp.launch(
            kernel=solve_distance_constraints,
            dim=model.spring_count,
            inputs=[
                particle_q,
                particle_qd,
                model.particle_inv_mass,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
                dt,
                self.spring_constraint_lambdas,
            ],
            outputs=[self.particle_deltas_accumulator, self.particle_deltas_count],
            device=model.device,
        )

        if not self.deferred_apply:
            wp.launch(
                kernel=apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[self.particle_deltas_accumulator, self.particle_deltas_count],
                outputs=[particle_deltas],
                device=model.device,
            )


class VolumeConstraintSystem(SimulationSystem):
    """Volume-preservation constraints for the active tetrahedra."""

    def __init__(self, stiffness: float = 0.1, priority: int = 60):
        super().__init__(priority=priority)
        self.stiffness = stiffness
        self.particle_deltas_accumulator = None
        self.particle_deltas_count = None

    def get_accumulators(self):
        if self.particle_deltas_accumulator is None:
            return None
        return (self.particle_deltas_accumulator, self.particle_deltas_count)

    def initialize(self, model):
        self.particle_deltas_accumulator = wp.zeros(
            model.particle_count,
            dtype=wp.vec3f,
            device=model.device,
        )
        self.particle_deltas_count = wp.zeros(
            model.particle_count,
            dtype=wp.int32,
            device=model.device,
        )

    def solve_constraints(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        body_q,
        body_qd,
        body_deltas,
        dt,
        iteration,
    ):
        if not hasattr(model, "tetrahedra_wp") or len(model.tetrahedra_wp) == 0:
            return

        wp.launch(
            kernel=solve_volume_constraints,
            dim=len(model.tetrahedra_wp),
            inputs=[
                particle_q,
                model.particle_inv_mass,
                model.tetrahedra_wp,
                model.tet_active,
                self.stiffness,
            ],
            outputs=[self.particle_deltas_accumulator, self.particle_deltas_count],
            device=model.device,
        )

        if not self.deferred_apply:
            wp.launch(
                kernel=apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[self.particle_deltas_accumulator, self.particle_deltas_count],
                outputs=[particle_deltas],
                device=model.device,
            )


class TrianglePointConstraintSystem(SimulationSystem):
    """Triangle-point connector constraints for multi-organ attachments."""

    def __init__(self, priority: int = 70):
        super().__init__(priority=priority)

    def solve_constraints(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        body_q,
        body_qd,
        body_deltas,
        dt,
        iteration,
    ):
        if not hasattr(model, "tri_points_connectors") or len(model.tri_points_connectors) == 0:
            return

        wp.launch(
            kernel=apply_tri_points_constraints_jacobian,
            dim=len(model.tri_points_connectors),
            inputs=[particle_q, model.tri_points_connectors],
            outputs=[particle_deltas],
            device=model.device,
        )

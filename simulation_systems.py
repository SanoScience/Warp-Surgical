import warp as wp
import numpy as np
from simulation_system import SimulationSystem
from simulation_kernels import (
    bounds_collision,
    solve_distance_constraints,
    apply_deltas_and_zero_accumulators,
    solve_volume_constraints,
    apply_tri_points_constraints_jacobian,
)
from collision_kernels import (
    collide_triangles_vs_spheres,
)


class BoundsCollisionSystem(SimulationSystem):
    def __init__(self, bounds_min, bounds_max, restitution=0.0, friction=0.0, priority=100):
        super().__init__(priority=priority)
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.restitution = restitution
        self.friction = friction

    def solve_constraints(self, model, state_in, state_out,
                         particle_q, particle_qd, particle_deltas,
                         body_q, body_qd, body_deltas, dt, iteration):
        if model.particle_count > 0:
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
                    dt
                ],
                device=model.device,
            )


class DistanceConstraintSystem(SimulationSystem):
    def __init__(self, priority=50):
        super().__init__(priority=priority)
        self.particle_deltas_accumulator = None
        self.particle_deltas_count = None
        self.spring_constraint_lambdas = None

    def initialize(self, model):
        self.particle_deltas_accumulator = wp.zeros(
            model.particle_count, dtype=wp.vec3f, device=wp.get_device()
        )
        self.particle_deltas_count = wp.zeros(
            model.particle_count, dtype=wp.int32, device=wp.get_device()
        )
        if model.spring_count:
            self.spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)

    def solve_constraints(self, model, state_in, state_out,
                         particle_q, particle_qd, particle_deltas,
                         body_q, body_qd, body_deltas, dt, iteration):
        if model.spring_count and self.spring_constraint_lambdas is not None:
            self.spring_constraint_lambdas.zero_()
            self.particle_deltas_accumulator.zero_()
            self.particle_deltas_count.zero_()

            wp.launch(
                kernel=solve_distance_constraints,
                dim=model.spring_count,
                inputs=[
                    particle_q, particle_qd,
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

            wp.launch(
                kernel=apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[self.particle_deltas_accumulator, self.particle_deltas_count],
                outputs=[particle_deltas],
                device=model.device,
            )


class CustomCollisionSystem(SimulationSystem):
    """Base class for custom collision systems.

    Subclass this and override solve_constraints to implement custom collision logic.
    """

    def __init__(self, priority=90):
        super().__init__(priority=priority)
        self.collision_radius = 0.1
        self.collision_position = wp.vec3(0.0, 0.0, 0.0)

    def set_collision_parameters(self, position, radius):
        if isinstance(position, (list, tuple)):
            self.collision_position = wp.vec3(position[0], position[1], position[2])
        else:
            self.collision_position = position
        self.collision_radius = radius


class VolumeConstraintSystem(SimulationSystem):
    """Volume preservation constraints for tetrahedra."""

    def __init__(self, stiffness=0.1, priority=60):
        super().__init__(priority=priority)
        self.stiffness = stiffness
        self.particle_deltas_accumulator = None
        self.particle_deltas_count = None

    def initialize(self, model):
        self.particle_deltas_accumulator = wp.zeros(
            model.particle_count, dtype=wp.vec3f, device=wp.get_device()
        )
        self.particle_deltas_count = wp.zeros(
            model.particle_count, dtype=wp.int32, device=wp.get_device()
        )

    def solve_constraints(self, model, state_in, state_out,
                         particle_q, particle_qd, particle_deltas,
                         body_q, body_qd, body_deltas, dt, iteration):
        if hasattr(model, 'tetrahedra_wp') and len(model.tetrahedra_wp) > 0:
            self.particle_deltas_accumulator.zero_()
            self.particle_deltas_count.zero_()

            wp.launch(
                kernel=solve_volume_constraints,
                dim=len(model.tetrahedra_wp),
                inputs=[
                    particle_q,
                    model.particle_inv_mass,
                    model.tetrahedra_wp,
                    model.tet_active,
                    self.stiffness
                ],
                outputs=[
                    self.particle_deltas_accumulator,
                    self.particle_deltas_count,
                ],
                device=model.device,
            )

            wp.launch(
                kernel=apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[
                    self.particle_deltas_accumulator,
                    self.particle_deltas_count,
                ],
                outputs=[particle_deltas],
                device=model.device,
            )


class TrianglePointConstraintSystem(SimulationSystem):
    """Triangle-point connector constraints."""

    def __init__(self, priority=70):
        super().__init__(priority=priority)

    def solve_constraints(self, model, state_in, state_out,
                         particle_q, particle_qd, particle_deltas,
                         body_q, body_qd, body_deltas, dt, iteration):
        if hasattr(model, 'tri_points_connectors') and len(model.tri_points_connectors) > 0:
            wp.launch(
                kernel=apply_tri_points_constraints_jacobian,
                dim=len(model.tri_points_connectors),
                inputs=[particle_q, model.tri_points_connectors],
                outputs=[particle_deltas],
                device=model.device,
            )


class ExternalSphereCollisionSystem(SimulationSystem):
    """Collision with external sphere colliders (e.g., jaw colliders)."""

    def __init__(self, restitution=0.0, priority=80):
        super().__init__(priority=priority)
        self.external_sphere_centers = None
        self.external_sphere_radii = None
        self.external_sphere_count = 0
        self.restitution = restitution
        self.particle_deltas_accumulator = None
        self.particle_deltas_count = None

    def initialize(self, model):
        self.particle_deltas_accumulator = wp.zeros(
            model.particle_count, dtype=wp.vec3f, device=wp.get_device()
        )
        self.particle_deltas_count = wp.zeros(
            model.particle_count, dtype=wp.int32, device=wp.get_device()
        )

    def set_external_sphere_colliders(self, model, centers, radii):
        """Register external sphere colliders for collision handling."""
        if centers is None or len(centers) == 0:
            self.external_sphere_centers = None
            self.external_sphere_radii = None
            self.external_sphere_count = 0
            return

        centers_np = np.asarray(centers, dtype=np.float32)
        if centers_np.size == 0:
            self.external_sphere_centers = None
            self.external_sphere_radii = None
            self.external_sphere_count = 0
            return

        centers_np = centers_np.reshape(-1, 3)
        radii_np = np.asarray(radii, dtype=np.float32).reshape(-1)

        if centers_np.shape[0] != radii_np.shape[0]:
            raise ValueError("Sphere centers and radii must have matching counts.")

        self.external_sphere_centers = wp.array(centers_np, dtype=wp.vec3f, device=model.device)
        self.external_sphere_radii = wp.array(radii_np, dtype=wp.float32, device=model.device)
        self.external_sphere_count = centers_np.shape[0]

    def solve_constraints(self, model, state_in, state_out,
                         particle_q, particle_qd, particle_deltas,
                         body_q, body_qd, body_deltas, dt, iteration):
        if (self.external_sphere_count > 0 and
            self.external_sphere_centers is not None and
            self.external_sphere_radii is not None and
            model.tri_count > 0):

            self.particle_deltas_accumulator.zero_()
            self.particle_deltas_count.zero_()

            wp.launch(
                kernel=collide_triangles_vs_spheres,
                dim=model.tri_count * self.external_sphere_count,
                inputs=[
                    particle_q,
                    particle_qd,
                    model.particle_inv_mass,
                    model.tri_indices,
                    self.external_sphere_centers,
                    self.external_sphere_radii,
                    self.external_sphere_count,
                    self.restitution,
                    dt,
                ],
                outputs=[
                    self.particle_deltas_accumulator,
                    self.particle_deltas_count,
                ],
                device=model.device,
            )

            wp.launch(
                kernel=apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[
                    self.particle_deltas_accumulator,
                    self.particle_deltas_count,
                ],
                outputs=[particle_deltas],
                device=model.device,
            )

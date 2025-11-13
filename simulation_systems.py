import warp as wp
from simulation_system import SimulationSystem
from simulation_kernels import (
    bounds_collision,
    solve_distance_constraints,
    apply_deltas_and_zero_accumulators,
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

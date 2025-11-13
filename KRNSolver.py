import numpy as np
import warp as wp

from newton._src.core.types import override


from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers import SolverBase
from newton._src.solvers.xpbd.kernels import (
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_shape_restitution,
    apply_rigid_restitution,
    bending_constraint,
    solve_body_contact_positions,
    solve_body_joints,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    # solve_simple_body_joints,
    solve_springs,
    solve_tetrahedra,
    update_body_velocities,
)

from simulation_kernels import (
    apply_tri_points_constraints_jacobian,
    solve_volume_constraints,
    solve_distance_constraints,
    apply_deltas,
    apply_deltas_and_zero_accumulators,
    clear_jacobian_accumulator,
    bounds_collision
)

from collision_kernels import (
    collide_particles_vs_sphere,
    collide_triangles_vs_sphere,
    vertex_triangle_collision_det,
)


class KRNSolver(SolverBase):


    def __init__(
        self,
        model: Model,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        angular_damping: float = 0.0,
        enable_restitution: bool = False,
    ):
        super().__init__(model=model)
        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation


        self.angular_damping = angular_damping

        self.enable_restitution = enable_restitution

        self.compute_body_velocity_from_position_delta = False

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        # System registry for callback-based extensibility
        self.systems = []

    def register_system(self, system):
        """Register a simulation system.

        Args:
            system: SimulationSystem instance to register
        """
        system.initialize(self.model)
        self.systems.append(system)
        # Sort by priority (lower priority runs first)
        self.systems.sort(key=lambda s: s.priority)

    def unregister_system(self, system):
        """Remove a simulation system.

        Args:
            system: SimulationSystem instance to remove
        """
        if system in self.systems:
            self.systems.remove(system)

    def apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    def apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.body_q = new_body_q
                state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_deltas = None

        rigid_contact_inv_weight = None

        # if contacts:
        #     if self.rigid_contact_con_weighting:
        #         rigid_contact_inv_weight = wp.zeros_like(contacts.rigid_contact_thickness0)
        #     rigid_contact_inv_weight_init = None

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            # Call pre_integrate callbacks
            for system in self.systems:
                if system.enabled:
                    system.pre_integrate(model, state_in, dt)

            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd

                self.particle_q_init = wp.clone(state_in.particle_q)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd)

              

                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        if model.particle_max_radius > 0.0 and model.particle_count > 1:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=solve_springs,
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
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )


                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # tetrahedral FEM
                        if model.tet_count:
                            wp.launch(
                                kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tet_indices,
                                    model.tet_poses,
                                    model.tet_activations,
                                    model.tet_materials,
                                    dt,
                                    self.soft_body_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # Call system constraint solvers
                        for system in self.systems:
                            if system.enabled:
                                system.solve_constraints(
                                    model, state_in, state_out,
                                    particle_q, particle_qd, particle_deltas,
                                    body_q, body_qd, body_deltas,
                                    dt, i
                                )

                        particle_q, particle_qd = self.apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                    

                        body_q, body_qd = self.apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            # Call post_solve callbacks
            for system in self.systems:
                if system.enabled:
                    system.post_solve(model, state_out, dt)

            return state_out

    def set_external_sphere_colliders(self, centers, radii):
        """Register external sphere colliders (e.g., jaw colliders) for collision handling."""
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

        self.external_sphere_centers = wp.array(centers_np, dtype=wp.vec3f, device=self.model.device)
        self.external_sphere_radii = wp.array(radii_np, dtype=wp.float32, device=self.model.device)
        self.external_sphere_count = centers_np.shape[0]

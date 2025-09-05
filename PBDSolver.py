import warp as wp
from newton.sim import Contacts, Control, Model, State
from newton.solvers.xpbd.solver_xpbd import XPBDSolver
from newton.solvers.xpbd.kernels import (
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

from newton.solvers.vbd.tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)

NUM_THREADS_PER_COLLISION_PRIMITIVE = 4

class PBDSolver(XPBDSolver):
    def __init__(self, model: Model, **kwargs):
        super().__init__(model, **kwargs)
        self.volCnstrs = True
        self.dev_pos_buffer = None
        self.self_contact_radius: float = 0.002
        self.self_contact_margin: float = 0.002
        
        # self.trimesh_collision_detector = TriMeshCollisionDetector(
        #     self.model,
        #     vertex_collision_buffer_pre_alloc=32,
        #     edge_collision_buffer_pre_alloc=64,
        #     edge_edge_parallel_epsilon=1e-5,
        # )

        # self.trimesh_collision_info = wp.array(
        #     [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
        # )
        
        soft_contact_max = model.shape_count * model.particle_count
        self.collision_evaluation_kernel_launch_size = max(
            self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
            self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
            soft_contact_max,
        )

    def collison_detection(self, particle_q: wp.array):
        """Perform collision detection for the current particle positions."""
        # Update trimesh collision detector with new particle positions
        self.trimesh_collision_detector.refit(particle_q)
        
        # Perform vertex-triangle collision detection
        #self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)
        query_radius = self.self_contact_margin
        self.trimesh_collision_detector.triangle_colliding_vertices_min_dist.fill_(query_radius)
        wp.launch(
            kernel=vertex_triangle_collision_det,
            inputs=[
                query_radius,
                self.trimesh_collision_detector.bvh_tris.id,
                self.trimesh_collision_detector.vertex_positions,
                self.model.tri_indices,
                self.trimesh_collision_detector.vertex_colliding_triangles_offsets,
                self.trimesh_collision_detector.vertex_colliding_triangles_buffer_sizes,
            ],
            outputs=[
                self.trimesh_collision_detector.vertex_colliding_triangles,
                self.trimesh_collision_detector.vertex_colliding_triangles_count,
                self.trimesh_collision_detector.vertex_colliding_triangles_min_dist,
                self.trimesh_collision_detector.triangle_colliding_vertices_min_dist,
                self.trimesh_collision_detector.resize_flags,
            ],
            dim=self.model.particle_count,
            device=self.model.device,
            block_dim=self.trimesh_collision_detector.collision_detection_block_size,
        )
        
        # Perform edge-edge collision detection
        #self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)

    def step(self, model: Model, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        particle_q = None
        particle_qd = None
        particle_deltas = None
        particle_deltas_accumulator = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        particle_deltas_count = wp.zeros(self.model.particle_count, dtype=wp.int32, device=wp.get_device())


        body_q = None
        body_qd = None
        body_deltas = None

        rigid_contact_inv_weight = None


        if contacts:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros_like(contacts.rigid_contact_thickness0)
            rigid_contact_inv_weight_init = None

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
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

                if model.joint_count:
                    wp.launch(
                        kernel=apply_joint_forces,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_axis,
                            control.joint_f,
                        ],
                        outputs=[state_in.body_f],
                        device=model.device,
                    )

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
                            particle_deltas_accumulator.zero_()
                            particle_deltas_count.zero_()

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
                                    model.shape_materials,
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
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas_accumulator, particle_deltas_count],
                                device=model.device,
                            )
                            
                            # wp.launch(
                            #     kernel=apply_deltas,
                            #     dim=model.particle_count,
                            #     inputs=[
                            #         particle_deltas_accumulator,
                            #         particle_deltas_count,
                            #     ],
                            #     outputs=[particle_deltas],
                            #     device=model.device,
                            # )

                            # wp.launch(
                            #     kernel=clear_jacobian_accumulator,
                            #     dim=model.particle_count,
                            #     inputs=[
                            #         particle_deltas_accumulator,
                            #         particle_deltas_count,
                            #     ],
                            #     device=model.device,
                            # )

                        

                        # bending constraints
                        # if model.edge_count:
                        #     edge_constraint_lambdas.zero_()
                        #     wp.launch(
                        #         kernel=bending_constraint,
                        #         dim=model.edge_count,
                        #         inputs=[
                        #             particle_q,
                        #             particle_qd,
                        #             model.particle_inv_mass,
                        #             model.edge_indices,
                        #             model.edge_rest_angle,
                        #             model.edge_bending_properties,
                        #             dt,
                        #             edge_constraint_lambdas,
                        #         ],
                        #         outputs=[particle_deltas],
                        #         device=model.device,
                        #     )

                        # tetrahedral FEM
                        # if model.tet_count:
                        #     wp.launch(
                        #         kernel=solve_tetrahedra,
                        #         dim=model.tet_count,
                        #         inputs=[
                        #             particle_q,
                        #             particle_qd,
                        #             model.particle_inv_mass,
                        #             model.tet_indices,
                        #             model.tet_poses,
                        #             model.tet_activations,
                        #             model.tet_materials,
                        #             dt,
                        #             self.soft_body_relaxation,
                        #         ],
                        #         outputs=[particle_deltas],
                        #         device=model.device,
                        #     )

                            # # solve volume constraints
                            
                            if self.volCnstrs:
                                wp.launch(
                                    solve_volume_constraints,
                                    dim=len(model.tetrahedra_wp),
                                    inputs=[
                                        particle_q,
                                        model.particle_inv_mass,
                                        model.tetrahedra_wp,
                                        model.tet_active,
                                        0.1  # stiffness
                                    ],
                                    outputs=[
                                        particle_deltas_accumulator,
                                        particle_deltas_count,
                                    ],
                                    device=model.device,
                                )
                            
                            wp.launch(
                                kernel=apply_deltas_and_zero_accumulators,
                                dim=model.particle_count,
                                inputs=[
                                    particle_deltas_accumulator,
                                    particle_deltas_count,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )




                        # CUSTOM CONSTRAINTS

                        wp.launch(
                            apply_tri_points_constraints_jacobian,
                            dim=len(model.tri_points_connectors),
                            inputs=[particle_q,model.tri_points_connectors],
                            outputs=[particle_deltas],
                            device=model.device,
                        )

                        



                        #if haptic_proxy_pos is not None:
                        # wp.launch(
                        #     kernel=collide_particles_vs_sphere,
                        #     dim=model.particle_count,
                        #     inputs=[
                        #         particle_q,
                        #         particle_qd,
                        #         model.particle_inv_mass,
                        #         self.dev_pos_buffer,  # sphere position
                        #         0.2,  # sphere radius
                        #         0.0,  # sphere restitution
                        #         dt
                        #     ],
                        #     outputs=[particle_deltas],
                        #     device=model.device,
                        # )

                        wp.launch(
                            kernel=collide_triangles_vs_sphere,
                            dim=model.tri_count,
                            inputs=[
                                particle_q,
                                particle_qd,
                                model.particle_inv_mass,
                                model.tri_indices,
                                self.dev_pos_buffer,  # sphere position
                                0.05,  # sphere radius
                                0.0,  # sphere restitution
                                dt
                            ],
                            outputs=[
                                particle_deltas_accumulator,
                                particle_deltas_count,
                            ],
                            device=model.device,
                        )

                        wp.launch(
                            kernel=apply_deltas_and_zero_accumulators,
                            dim=model.particle_count,
                            inputs=[
                                particle_deltas_accumulator,
                                particle_deltas_count,
                            ],
                            outputs=[particle_deltas],
                            device=model.device,
                        )

                        wp.launch(
                            bounds_collision,
                                  dim=model.particle_count,
                                  inputs=[
                                      particle_q,
                                      particle_qd,
                                      model.particle_inv_mass,
                                      wp.vec3(-2.0, 0.0, -8.0),
                                      wp.vec3(2.0, 10.0, -3.0),
                                      0.0,
                                      0.0,
                                      dt
                                  ],
                                  device=model.device,
                            )

                        particle_q, particle_qd = self.apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                    # handle rigid bodies
                    # ----------------------------

                    if model.joint_count:
                        # wp.launch(
                        #     kernel=solve_simple_body_joints,
                        #     dim=model.joint_count,
                        #     inputs=[
                        #         body_q,
                        #         body_qd,
                        #         model.body_com,
                        #         model.body_inv_mass,
                        #         model.body_inv_inertia,
                        #         model.joint_type,
                        #         model.joint_enabled,
                        #         model.joint_parent,
                        #         model.joint_child,
                        #         model.joint_X_p,
                        #         model.joint_X_c,
                        #         model.joint_limit_lower,
                        #         model.joint_limit_upper,
                        #         model.joint_qd_start,
                        #         model.joint_dof_dim,
                        #         model.joint_dof_mode,
                        #         model.joint_axis,
                        #         control.joint_target,
                        #         model.joint_target_ke,
                        #         model.joint_target_kd,
                        #         self.joint_linear_compliance,
                        #         self.joint_angular_compliance,
                        #         self.joint_angular_relaxation,
                        #         self.joint_linear_relaxation,
                        #         dt,
                        #     ],
                        #     outputs=[body_deltas],
                        #     device=model.device,
                        # )

                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.joint_type,
                                model.joint_enabled,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_X_p,
                                model.joint_X_c,
                                model.joint_limit_lower,
                                model.joint_limit_upper,
                                model.joint_qd_start,
                                model.joint_dof_dim,
                                model.joint_dof_mode,
                                model.joint_axis,
                                control.joint_target,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                self.joint_linear_compliance,
                                self.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas],
                            device=model.device,
                        )

                        body_q, body_qd = self.apply_body_deltas(model, state_in, state_out, body_deltas, dt)

                    # Solve rigid contact constraints
                    if model.body_count and contacts is not None:
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()
                        body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=contacts.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.shape_body,
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_point0,
                                contacts.rigid_contact_point1,
                                contacts.rigid_contact_offset0,
                                contacts.rigid_contact_offset1,
                                contacts.rigid_contact_normal,
                                contacts.rigid_contact_thickness0,
                                contacts.rigid_contact_thickness1,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                model.shape_materials,
                                self.rigid_contact_relaxation,
                                dt,
                                model.rigid_contact_torsional_friction,
                                model.rigid_contact_rolling_friction,
                            ],
                            outputs=[
                                body_deltas,
                                rigid_contact_inv_weight,
                            ],
                            device=model.device,
                        )

                        # if model.rigid_contact_count.numpy()[0] > 0:
                        #     print("rigid_contact_count:", model.rigid_contact_count.numpy().flatten())
                        #     # print("rigid_active_contact_distance:", rigid_active_contact_distance.numpy().flatten())
                        #     # print("rigid_active_contact_point0:", rigid_active_contact_point0.numpy().flatten())
                        #     # print("rigid_active_contact_point1:", rigid_active_contact_point1.numpy().flatten())
                        #     print("body_deltas:", body_deltas.numpy().flatten())

                        # print(rigid_active_contact_distance.numpy().flatten())

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

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

            if self.enable_restitution and contacts is not None:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=model.particle_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.shape_materials,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
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
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )

                if model.body_count:
                    body_deltas.zero_()
                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_materials,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            contacts.rigid_contact_thickness,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            return state_out
from grasping import grasp_end, grasp_process, grasp_start
from heating import heating_active_process, heating_conduction_process, heating_end, heating_start, paint_vertices_near_haptic_proxy, set_paint_strength
from stretching import stretching_breaking_process
from surface_reconstruction import extract_surface_triangles_bucketed
import warp as wp
import newton

from newton.utils.render import SimRendererOpenGL
from newton.solvers import XPBDSolver
from newton.solvers import VBDSolver
import numpy as np
import math

from PBDSolver import PBDSolver
from render_surgsim_opengl import SurgSimRendererOpenGL

from mesh_loader import Tetrahedron, load_mesh_and_build_model
from render_opengl import CustomOpenGLRenderer
from simulation_kernels import (
    set_body_position,
)



@wp.kernel
def set_active_tets_near_haptic(
    tet_active: wp.array(dtype=wp.int32),         # [num_tets]
    tets: wp.array(dtype=Tetrahedron),            # [num_tets]
    particle_q: wp.array(dtype=wp.vec3f),         # [num_particles]
    haptic_pos: wp.array(dtype=wp.vec3f),         # [1]
    radius: float,
    num_tets: int
):
    tid = wp.tid()
    if tid >= num_tets:
        return

    tet = tets[tid]
    
    # Compute centroid of the tet
    c = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(4):
        c += particle_q[tet.ids[i]]
    c *= 0.25

    hpos = haptic_pos[0] * 0.01
    dist = wp.length(c - hpos)
    if dist < radius:
        tet_active[tid] = 0

@wp.kernel
def build_vertex_neighbor_table(
    tet_active: wp.array(dtype=wp.int32),         # [num_tets]
    tets: wp.array(dtype=Tetrahedron),            # [num_tets]
    vertex_neighbors: wp.array(dtype=wp.int32, ndim = 2),   # [num_vertices, max_neighbors]
    vertex_neighbor_counts: wp.array(dtype=wp.int32),       # [num_vertices]
    num_tets: int,
    max_neighbors: int
):
    tid = wp.tid()
    if tid >= num_tets:
        return

    if tet_active[tid] == 0:
        return

    tet = tets[tid]
    for i in range(4):
        v = tet.ids[i]

        for j in range(4):
            if i == j:
                continue

            n = tet.ids[j]

            # Atomically add neighbor if not already present
            # (no duplicate check)
            idx = wp.atomic_add(vertex_neighbor_counts, v, 1)
            if idx < max_neighbors:
                vertex_neighbors[v, idx] = n


@wp.kernel
def build_tet_edge_table(
    tets: wp.array(dtype=Tetrahedron),                # [num_tets]
    springs: wp.array(dtype=wp.int32),                # [num_springs * 2], flat array
    tet_to_edges: wp.array(dtype=wp.int32, ndim=2),   # [num_tets, 6]
    tet_edge_counts: wp.array(dtype=wp.int32),        # [num_tets]
    num_tets: int,
    num_springs: int,
):
    tid = wp.tid()
    if tid >= num_tets:
        return

    tet = tets[tid]
    tet_ids = tet.ids
    count = int(0)
    for i in range(num_springs):

        spring_a = springs[i * 2 + 0]
        spring_b = springs[i * 2 + 1]
        # Edge 0: (0,1)
        a = tet_ids[0]
        b = tet_ids[1]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 1: (0,2)
        a = tet_ids[0]
        b = tet_ids[2]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 2: (0,3)
        a = tet_ids[0]
        b = tet_ids[3]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 3: (1,2)
        a = tet_ids[1]
        b = tet_ids[2]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 4: (1,3)
        a = tet_ids[1]
        b = tet_ids[3]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 5: (2,3)
        a = tet_ids[2]
        b = tet_ids[3]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1


class WarpSim:
    def __init__(self, stage_path="output.usd", num_frames=300, use_opengl=True):
        self.sim_substeps = 16
        self.num_frames = num_frames
        self.fps = 120

        self.frame_dt = 1.0 / self.fps
        self.substep_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_constraint_iterations = 1

        self.haptic_pos_right = None  # Haptic device position in simulation space

        self.radius_collision = 0.1
        self.radius_heating = 0.35
        self.radius_cutting = 0.25
        self.radius_grasping = 0.5

        self.particle_mass = 0.1

        self.cutting_active = False
        self.heating_active = False
        self.grasping_active = False

        print(f"Initializing WarpSim with {self.sim_substeps} substeps at {self.fps} FPS")
        print(f"Frame time: {self.frame_dt:.4f}s, Substep time: {self.substep_dt:.4f}s")

        # Initialize model
        self._build_model()
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        
        # Connectivity setup
        vertex_count = self.model.particle_count
        vertex_neighbour_count = 32

        self.vertex_to_vneighbours = wp.zeros((vertex_count, vertex_neighbour_count), dtype=wp.int32, device=wp.get_device())
        self.vertex_vneighbor_counts = wp.zeros(vertex_count, dtype=wp.int32, device=wp.get_device())
        self.vneighbours_max = vertex_neighbour_count

        # Tetrahedron to edge mapping setup
        num_tets = self.model.tetrahedra_wp.shape[0]
        num_springs = self.model.spring_indices.shape[0] // 2

        self.tet_to_edges = wp.zeros((num_tets, 6), dtype=wp.int32, device=wp.get_device())
        self.tet_edge_counts = wp.zeros(num_tets, dtype=wp.int32, device=wp.get_device())

        # Initialize simulation components
        self._setup_simulation()
        
        # Initialize rendering
        self._setup_renderer(stage_path, use_opengl)

        # Grasp setup
        self.grasp_capacity = 128
        self.grasped_particles_buffer = wp.zeros(self.grasp_capacity, dtype=wp.int32, device=wp.get_device())
        self.grasped_particles_counter = wp.zeros(1, dtype=wp.int32, device=wp.get_device())

        # Heating setup
        self.paint_color_buffer = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=wp.get_device())
        self.paint_strength_buffer = wp.array([0.0], dtype=wp.float32, device=wp.get_device()) 
        set_paint_strength(self, 0.0)
    
        # Load textures
        if self.use_opengl and self.renderer:
            self.liver_texture = self.renderer.load_texture("textures/liver-base.png")
            self.fat_texture = self.renderer.load_texture("textures/fat-base.png") 
            self.gallbladder_texture = self.renderer.load_texture("textures/gallbladder-base.png")

            self.liver_burn_texture = self.renderer.load_texture("textures/liver-burn.png")
            self.fat_burn_texture = self.renderer.load_texture("textures/fat-burn.png")
            self.gallbladder_burn_texture = self.renderer.load_texture("textures/gallbladder-burn.png")

            self.renderer.set_input_callbacks(
                on_key_press=self._on_key_press,
                on_key_release=self._on_key_release
            )

        # Setup CUDA graph if available
        self._setup_cuda_graph()

        wp.launch(
            build_tet_edge_table,
            dim=num_tets,
            inputs=[
                self.model.tetrahedra_wp,
                self.model.spring_indices,  # flat int32 array
                self.tet_to_edges,
                self.tet_edge_counts,
                num_tets,
                num_springs
            ],
            device=wp.get_device()
        )

    def _on_key_press(self, symbol, modifiers):
        from pyglet.window import key
        if symbol == key.C:
            self.cutting_active = True
        elif symbol == key.V:
            heating_start(self)
        elif symbol == key.B:
            grasp_start(self)
        elif symbol == key.Y:
            self.integrator.volCnstrs = not self.integrator.volCnstrs


    def _on_key_release(self, symbol, modifiers):
        from pyglet.window import key
        if symbol == key.C:
            self.cutting_active = False
        elif symbol == key.V:
            heating_end(self)
        elif symbol == key.B:
            grasp_end(self)

    def _build_model(self):
        """Build the simulation model with mesh and haptic device."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        
        spring_stiffness = 1.0
        spring_dampen = 0.2

        tetra_stiffness_mu = 1.0e4
        tetra_stiffness_lambda = 1.0e4
        tetra_dampen = 0.2

        # Import the mesh
        tri_points_connectors, self.surface_tris, uvs, self.mesh_ranges, tetrahedra_wp = load_mesh_and_build_model(builder,
            particle_mass=self.particle_mass, vertical_offset=-3.0, 
            spring_stiffness=spring_stiffness, spring_dampen=spring_dampen,
            tetra_stiffness_mu=tetra_stiffness_mu, tetra_stiffness_lambda=tetra_stiffness_lambda, tetra_dampen=tetra_dampen)
        
        self.surface_tris_wp = wp.array(self.surface_tris, dtype=wp.int32, device=wp.get_device())
        self.uvs_wp = wp.array(uvs, dtype=wp.vec2f, device=wp.get_device())

        for mesh_name, mesh_info in self.mesh_ranges.items():
                tet_count = mesh_info.get('tet_count', 0)
                if tet_count == 0:
                    continue

                # Allocate surface extraction buffers
                setattr(self, f'{mesh_name}_tet_surface_counter', wp.zeros(1, dtype=wp.int32, device=wp.get_device()))
                setattr(self, f'{mesh_name}_tet_surface_indices', wp.zeros((tet_count * 4, 3), dtype=wp.int32, device=wp.get_device()))
                
                # bucket buffers
                max_tri_count = tet_count * 4
                num_buckets = math.ceil(max_tri_count / 16)
                bucket_size = 64

                setattr(self, f'{mesh_name}_bucket_count', num_buckets)
                setattr(self, f'{mesh_name}_bucket_size', bucket_size)
                setattr(self, f'{mesh_name}_bucket_counters', wp.zeros(num_buckets, dtype=wp.int32, device=wp.get_device()))
                setattr(self, f'{mesh_name}_bucket_storage', wp.zeros((num_buckets, bucket_size, 3), dtype=wp.int32, device=wp.get_device()))

        # Add haptic device collision body
        self.haptic_body_id = builder.add_body(
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            mass=0.0,  # Zero mass makes it kinematic
            armature=0.0
        )

        builder.add_shape_sphere(
            body=self.haptic_body_id,
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            radius=self.radius_collision,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=10
            )
        )
        
        self.model = builder.finalize()
        self.model.tetrahedra_wp = tetrahedra_wp
        self.model.tet_active = wp.ones(self.model.tetrahedra_wp.shape[0], dtype=wp.int32, device=wp.get_device())
        self.model.tri_points_connectors = tri_points_connectors

        self.model.particle_max_velocity = 10.0

    def _setup_simulation(self):
        """Initialize simulation states and integrator."""
        self.integrator = PBDSolver(self.model, iterations=5)
        
        self.integrator.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.integrator.dev_pos_prev_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())

        self.rest = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.collide(self.state_0)
        
        # Create Jacobian accumulators
        # self.model.delta_accumulator = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        # self.model.count_accumulator = wp.zeros(self.model.particle_count, dtype=wp.int32, device=wp.get_device())

    def _setup_renderer(self, stage_path, use_opengl):
        """Initialize the appropriate renderer."""
        self.use_opengl = use_opengl
        
        if self.use_opengl:
            self.renderer = SurgSimRendererOpenGL(self.model, "Warp Surgical Simulation", scaling=1.0)
        elif stage_path:
            self.renderer = newton.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

        self.renderer._camera_pos = [0.2, 1.2, -1.0]
        #self.renderer.update_view_matrix()

    def _setup_cuda_graph(self):
        """Setup CUDA graph for performance optimization."""
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Run one simulation step with all substeps."""

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Update haptic device position
            wp.launch(
                set_body_position,
                dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.integrator.dev_pos_buffer, self.substep_dt],
                device=self.state_0.body_q.device,
            )

            # Run collision detection and integration
            #self.contacts = self.model.collide(self.state_0)
            #if self.contacts:
            #    print(f"Contacts detected: {self.contacts.soft_contact_normal}")

            self.integrator.step(self.model, self.state_0, self.state_1, None, self.contacts, self.substep_dt)
            
            # Swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        """Advance simulation by one frame."""
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            if self.use_opengl:
                self.renderer.begin_frame()

                self.renderer.render_sphere(
                    "haptic_proxy_sphere",
                    [self.haptic_pos_right[0] * 0.01, self.haptic_pos_right[1] * 0.01, self.haptic_pos_right[2] * 0.01],
                    [0.0, 0.0, 0.0, 1.0],
                    0.1,
                )

                # Grasping
                if self.grasping_active:
                    grasp_process(self)

                # Recompute connectivity
                wp.copy(self.vertex_vneighbor_counts, wp.zeros(self.model.particle_count, dtype=wp.int32, device=wp.get_device()))
                wp.launch(
                    build_vertex_neighbor_table,
                    dim=self.model.tetrahedra_wp.shape[0],
                    inputs=[
                        self.model.tet_active,
                        self.model.tetrahedra_wp,
                        self.vertex_to_vneighbours,
                        self.vertex_vneighbor_counts,
                        self.model.tetrahedra_wp.shape[0],
                        self.vneighbours_max
                    ],
                    device=wp.get_device()
                )

                # Heat conduction
                heating_conduction_process(self)
                stretching_breaking_process(self)

                for mesh_name, mesh_info in self.mesh_ranges.items():
                    tet_start = mesh_info.get('tet_start', 0)
                    tet_count = mesh_info.get('tet_count', 0)
                    if tet_count == 0:
                        continue

                    tet_surface_counter = getattr(self, f'{mesh_name}_tet_surface_counter')
                    tet_surface_indices = getattr(self, f'{mesh_name}_tet_surface_indices')

                    # Slice the tetrahedra for this mesh
                    tet_active = self.model.tet_active[tet_start:tet_start+tet_count]
                    mesh_tets = self.model.tetrahedra_wp[tet_start:tet_start+tet_count]

                    # Handle cutting
                    if self.cutting_active:
                        if tet_count == 0:
                            continue
                        if tet_active is None:
                            continue

                        wp.launch(
                            set_active_tets_near_haptic,
                            dim=tet_count,
                            inputs=[
                                tet_active,
                                mesh_tets,
                                self.state_0.particle_q,
                                self.integrator.dev_pos_buffer,
                                self.radius_cutting,
                                tet_count
                            ],
                            device=wp.get_device()
                        )

                    # Handle heating
                    if self.heating_active:
                        heating_active_process(self)

                    
                    # Extract surface
                    bucket_counters = getattr(self, f'{mesh_name}_bucket_counters')
                    bucket_storage = getattr(self, f'{mesh_name}_bucket_storage')
                    num_buckets = getattr(self, f'{mesh_name}_bucket_count')
                    bucket_size = getattr(self, f'{mesh_name}_bucket_size')

                    extract_surface_triangles_bucketed(
                        mesh_tets,
                        tet_active,
                        tet_surface_indices,
                        tet_surface_counter,
                        bucket_counters,
                        bucket_storage,
                        num_buckets,
                        bucket_size
                    )

                    num_triangles = int(tet_surface_counter.numpy()[0])
                    if num_triangles > 0:
                        texture = getattr(self, f"{mesh_name}_texture", None)
                        burn_texture = getattr(self, f"{mesh_name}_burn_texture", None)

                        self.renderer.render_mesh_warp_range(
                            name=f"{mesh_name}_mesh",
                            points=self.state_0.particle_q,
                            indices=tet_surface_indices,
                            texture_coords=self.uvs_wp,
                            texture=texture,
                            burn_texture=burn_texture,
                            colors=self.vertex_colors,
                            index_start=0,
                            index_count=num_triangles,
                            update_topology=True
                        )

                wp.copy(self.integrator.dev_pos_prev_buffer, self.integrator.dev_pos_buffer)
                self.renderer.end_frame()
            else:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()

    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        
        haptic_pos = wp.vec3(position[0], position[1] + 100.0, position[2] - 400.0)  # Offset to avoid collision with ground;
        self.haptic_pos_right = [haptic_pos[0], haptic_pos[1], haptic_pos[2]]

        wp.copy(self.integrator.dev_pos_buffer, wp.array([haptic_pos], dtype=wp.vec3, device=wp.get_device()))

    def is_running(self):
        """Check if the simulation should continue running."""
        return self.renderer.is_running() if self.use_opengl else True
    
    def save(self):
        """Save the simulation results."""
        if self.renderer:
            self.renderer.save()

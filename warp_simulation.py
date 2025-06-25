import warp as wp
import newton

from newton.utils.render import SimRendererOpenGL
from newton.solvers import XPBDSolver
from newton.solvers import VBDSolver
import numpy as np

from PBDSolver import PBDSolver
from render_surgsim_opengl import SurgSimRendererOpenGL

from mesh_loader import Tetrahedron, load_mesh_and_build_model
from render_opengl import CustomOpenGLRenderer
from simulation_kernels import (
    set_body_position,
    paint_vertices_near_haptic,
)

@wp.kernel
def extract_surface_from_tets(
    tets: wp.array(dtype=Tetrahedron),           # [num_tets]
    tet_active: wp.array(dtype=wp.int32),        # [num_tets], 1=keep, 0=skip
    counter: wp.array(dtype=wp.int32),           # [1], atomic counter
    out_indices: wp.array(dtype=wp.int32, ndim=2), # [max_triangles, 3]
):
    tid = wp.tid()
    if tid >= tets.shape[0]:
        return

    #if tet_active[tid] == 0:
    #    return

    tet = tets[tid]
    # Each tet has 4 faces (triangles)
    face1 = wp.vec3i(0, 1, 2)
    face2 = wp.vec3i(0, 1, 3)
    face3 = wp.vec3i(0, 2, 3)
    face4 = wp.vec3i(1, 2, 3)
    
    for f in range(4):
        face = face1
        if f == 1:
            face = face2
        elif f == 2:
            face = face3
        elif f == 3:
            face = face4


        tri_idx = wp.atomic_add(counter, 0, 1)
        i0 = tet.ids[face[0]]
        i1 = tet.ids[face[1]]
        i2 = tet.ids[face[2]]
        out_indices[tri_idx, 0] = i0
        out_indices[tri_idx, 1] = i1
        out_indices[tri_idx, 2] = i2

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

        print(f"Initializing WarpSim with {self.sim_substeps} substeps at {self.fps} FPS")
        print(f"Frame time: {self.frame_dt:.4f}s, Substep time: {self.substep_dt:.4f}s")

        # Initialize model
        self._build_model()
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        
        # Initialize simulation components
        self._setup_simulation()
        
        # Initialize rendering
        self._setup_renderer(stage_path, use_opengl)

        # Init component metadata
        #self.mesh_ranges = {
        #    'liver': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0},
        #    'fat': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0},
        #    'gallbladder': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0}
        #}

        self.paint_color_buffer = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=wp.get_device())  # Default red
        self.paint_strength_buffer = wp.array([0.1], dtype=wp.float32, device=wp.get_device())  # Default strength
        self.set_paint_strength(0.0)
    

        # Load textures
        if self.use_opengl and self.renderer:
            self.liver_texture = self.renderer.load_texture("textures/liver-base.png")
            self.fat_texture = self.renderer.load_texture("textures/fat-base.png") 
            self.gallbladder_texture = self.renderer.load_texture("textures/gallbladder-base.png")

            # self.renderer.set_input_callbacks(
            #     on_key_press=self._on_key_press,
            #     on_key_release=self._on_key_release
            # )

        # Setup CUDA graph if available
        self._setup_cuda_graph()

    def _on_key_press(self, symbol, modifiers):
        """Handle key press events."""
        from pyglet.window import key
        
        if symbol == key.R:
            self.set_paint_color([1.0, 0.0, 0.0])
            self.set_paint_strength(1.0)
        elif symbol == key.F:
            self.set_paint_color([0.0, 1.0, 0.0])
            self.set_paint_strength(1.0)

    def _on_key_release(self, symbol, modifiers):
        """Handle key release events."""
        from pyglet.window import key
        
        # Stop painting when any paint key is released
        if symbol in [key.R, key.F]:
            self.set_paint_strength(0.0)

    def _build_model(self):
        """Build the simulation model with mesh and haptic device."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        
        spring_stiffness = 1.0e3
        spring_dampen = 0.2

        tetra_stiffness_mu = 1.0e4
        tetra_stiffness_lambda = 1.0e4
        tetra_dampen = 0.2

        # Import the mesh
        tri_points_connectors, self.surface_tris, uvs, self.mesh_ranges, tetrahedra_wp = load_mesh_and_build_model(builder, vertical_offset=-3.0, 
            spring_stiffness=spring_stiffness, spring_dampen=spring_dampen,
            tetra_stiffness_mu=tetra_stiffness_mu, tetra_stiffness_lambda=tetra_stiffness_lambda, tetra_dampen=tetra_dampen)
        
        self.surface_tris_wp = wp.array(self.surface_tris, dtype=wp.int32, device=wp.get_device())
        self.uvs_wp = wp.array(uvs, dtype=wp.vec2f, device=wp.get_device())

        # Add haptic device collision body
        self.haptic_body_id = builder.add_body(
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            mass=0.0,  # Zero mass makes it kinematic
            armature=0.0
        )

        builder.add_shape_sphere(
            body=self.haptic_body_id,
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            radius=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=10
            )
        )
        


        self.model = builder.finalize()
        self.model.tetrahedra_wp = tetrahedra_wp
        self.model.tri_points_connectors = tri_points_connectors

    def _paint_vertices_near_haptic_proxy(self, paint_radius=0.25, falloff_power=2.0):
        """Paint vertex colors near the haptic proxy position."""
        wp.launch(
            paint_vertices_near_haptic,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,  # vertex positions
                self.state_0.body_q,      # haptic position
                self.vertex_colors,       # vertex colors to modify
                paint_radius,             # paint radius
                self.paint_color_buffer,  # paint color (from array)
                self.paint_strength_buffer,  # paint strength (from array)
                falloff_power             # falloff power for smooth edges
            ],
            device=wp.get_device()
        )

    def set_paint_color(self, color):
        """Set the paint color from CPU."""
        self.paint_color_buffer.assign([wp.vec3(color[0], color[1], color[2])])

    def set_paint_strength(self, strength):
        """Set the paint strength from CPU."""
        self.paint_strength_buffer.assign([strength])

    def _setup_simulation(self):
        """Initialize simulation states and integrator."""
        self.integrator = PBDSolver(self.model, iterations=5)
        
        self.integrator.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        
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

           # self._paint_vertices_near_haptic_proxy()

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
                
                #self.renderer.render_contacts(self.state_0, self.contacts)
                
                num_tets = self.model.tetrahedra_wp.shape[0]
                max_triangles = num_tets * 4

                # Allocate buffers for surface if not already done
                if not hasattr(self, 'tet_surface_counter'):
                    self.tet_surface_counter = wp.zeros(1, dtype=wp.int32, device=wp.get_device())
                    self.tet_surface_indices = wp.zeros((max_triangles, 3), dtype=wp.int32, device=wp.get_device())
                    self.tet_active = wp.ones(num_tets, dtype=wp.int32, device=wp.get_device())  # All active

                # Reset counter
                wp.copy(self.tet_surface_counter, wp.zeros(1, dtype=wp.int32, device=wp.get_device()))

                # Generate surface indices
                wp.launch(
                    extract_surface_from_tets,
                    dim=num_tets,
                    inputs=[self.model.tetrahedra_wp, self.tet_active, self.tet_surface_counter, self.tet_surface_indices],
                    device=wp.get_device()
                )

                # Get number of triangles written
                num_triangles = int(self.tet_surface_counter.numpy()[0])
                
                self.renderer.render_sphere(
                    "haptic_proxy_sphere",
                    [self.haptic_pos_right[0] * 0.01, self.haptic_pos_right[1] * 0.01, self.haptic_pos_right[2] * 0.01],
                    [0.0, 0.0, 0.0, 1.0],
                    0.2,
                )

                # self.renderer.render_sphere(
                #     "locking_sphere",
                #     [0.5, 1.5, -5.0],
                #     [0.0, 0.0, 0.0, 1.0],
                #     1,
                # )

                #print(num_triangles)
                if num_triangles > 0:
                    # Slice the indices buffer to the number of triangles written
                    indices_to_render = self.tet_surface_indices[:num_triangles]
                    #self.renderer.render_mesh_warp(
                    #    name="tet_surface",
                    #    points=self.state_0.particle_q,
                    #    indices=indices_to_render,
                    #    colors=self.vertex_colors,
                    #    texture_coords=self.uvs_wp,
                    #    texture=None,
                    #    update_topology=False,
                    #    smooth_shading=True,
                    #)

                    self.renderer.render_mesh_warp_range(
                        name="tet_mesh",
                        points=self.state_0.particle_q,
                        indices=indices_to_render,
                        texture_coords=self.uvs_wp,
                        texture=None,
                        colors=self.vertex_colors,
                        index_start=0,
                        index_count=num_triangles,
                        update_topology=False
                    )

                '''
                # Render liver mesh
                if self.mesh_ranges['liver']['index_count'] > 0:
                    self.renderer.render_mesh_warp_range(
                        name="liver_mesh",
                        points=self.state_0.particle_q,
                        indices=self.surface_tris_wp,
                        texture_coords=self.uvs_wp,
                        texture=self.liver_texture,
                        colors=self.vertex_colors,
                        index_start=self.mesh_ranges['liver']['index_start'],
                        index_count=self.mesh_ranges['liver']['index_count'],
                        update_topology=False
                    )
                
                # Render fat mesh
                if self.mesh_ranges['fat']['index_count'] > 0:
                    self.renderer.render_mesh_warp_range(
                        name="fat_mesh",
                        points=self.state_0.particle_q,
                        indices=self.surface_tris_wp,
                        texture_coords=self.uvs_wp,
                        texture=self.fat_texture,
                        colors=self.vertex_colors,
                        index_start=self.mesh_ranges['fat']['index_start'],
                        index_count=self.mesh_ranges['fat']['index_count'],
                        update_topology=False
                    )
                
                # Render gallbladder mesh
                if self.mesh_ranges['gallbladder']['index_count'] > 0:
                    self.renderer.render_mesh_warp_range(
                        name="gallbladder_mesh",
                        points=self.state_0.particle_q,
                        indices=self.surface_tris_wp,
                        texture_coords=self.uvs_wp,
                        texture=self.gallbladder_texture,
                        colors=self.vertex_colors,
                        index_start=self.mesh_ranges['gallbladder']['index_start'],
                        index_count=self.mesh_ranges['gallbladder']['index_count'],
                        update_topology=False
                    )
                '''
                self.renderer.end_frame()
            else:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)

                self.renderer.end_frame()

    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        
        haptic_pos = wp.vec3(position[0], position[1], position[2] - 500.0)  # Offset to avoid collision with ground;
        self.haptic_pos_right = [haptic_pos[0], haptic_pos[1], haptic_pos[2]];

        wp.copy(self.integrator.dev_pos_buffer, wp.array([haptic_pos], dtype=wp.vec3, device=wp.get_device()))

    def is_running(self):
        """Check if the simulation should continue running."""
        return self.renderer.is_running() if self.use_opengl else True
    
    def save(self):
        """Save the simulation results."""
        if self.renderer:
            self.renderer.save()

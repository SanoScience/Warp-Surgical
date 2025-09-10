import warp as wp
import newton

# from newton.utils.render import SimRendererOpenGL
# from newton.solvers import XPBDSolver

import numpy as np
import math

from PBDSolver import PBDSolver
from render_surgsim_opengl import SurgSimRendererOpenGL

from mesh_loader import Tetrahedron, load_cgal_and_build_model
from render_opengl import CustomOpenGLRenderer
from simulation_kernels import (
    set_body_position,
)



@wp.kernel
def transform_mesh_vertices(
    vertices: wp.array(dtype=wp.vec3f),
    origin: wp.vec3f,
    spacing: wp.float32,
    transformed_vertices: wp.array(dtype=wp.vec3f)
):
    tid = wp.tid()
    if tid >= len(vertices):
        return
    
    # Transform from grid space to world space
    grid_pos = vertices[tid]
    world_pos = origin + grid_pos * spacing
    transformed_vertices[tid] = world_pos




class WarpSim:
    #region Initialization
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
        self.radius_heating = 0.1
        self.radius_clipping = 0.1
        self.radius_cutting = 0.075
        self.radius_grasping = 0.075

        self.particle_mass = 0.1

        self.cutting_active = False
        self.heating_active = False
        self.grasping_active = False
        self.clipping = False


        print(f"Initializing WarpSim with {self.sim_substeps} substeps at {self.fps} FPS")
        print(f"Frame time: {self.frame_dt:.4f}s, Substep time: {self.substep_dt:.4f}s")

        # Initialize model
        self._build_model()
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec4f, device=wp.get_device())
        
        # Connectivity setup
        vertex_count = self.model.particle_count
        

        # Tetrahedron to edge mapping setup
        num_tets = self.model.tetrahedra_wp.shape[0]
        num_springs = self.model.spring_indices.shape[0] // 2

        self.tet_to_edges = wp.zeros((num_tets, 6), dtype=wp.int32, device=wp.get_device())
        self.tet_edge_counts = wp.zeros(num_tets, dtype=wp.int32, device=wp.get_device())



        # Initialize simulation components
        self._setup_simulation()
        
        # Initialize rendering
        self._setup_renderer(stage_path, use_opengl)

        # Setup CUDA graph if available
        self._setup_cuda_graph()


#endregion
    def _on_key_press(self, symbol, modifiers):
        from pyglet.window import key
        if symbol == key.C:
            self.cutting_active = True


    def _on_key_release(self, symbol, modifiers):
        from pyglet.window import key
        if symbol == key.C:
            self.cutting_active = False

#region  Model Setup
    def _build_model(self):
        """Build the simulation model with mesh and haptic device."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        
        spring_stiffness = 1.0
        spring_dampen = 0.2

        tetra_stiffness_mu = 1.0e4
        tetra_stiffness_lambda = 1.0e4
        tetra_dampen = 0.2

        # Import the mesh
        tri_points_connectors, self.surface_tris, uvs, self.mesh_ranges, tetrahedra_wp = load_cgal_and_build_model(builder,
            particle_mass=self.particle_mass, vertical_offset=-3.0, 
            spring_stiffness=spring_stiffness, spring_dampen=spring_dampen,
            tetra_stiffness_mu=tetra_stiffness_mu, tetra_stiffness_lambda=tetra_stiffness_lambda, tetra_dampen=tetra_dampen)
        
        self.surface_tris_wp = wp.array(self.surface_tris, dtype=wp.int32, device=wp.get_device())




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
        

    def _setup_renderer(self, stage_path, use_opengl):
        """Initialize the appropriate renderer."""
        self.use_opengl = use_opengl
        
        if self.use_opengl:
            #self.renderer = SurgSimRendererOpenGL(self.model, "Warp Surgical Simulation", scaling=1.0, near_plane=0.05, far_plane = 25)
            self.renderer = SurgSimRendererOpenGL(self.model, "Warp Surgical Simulation", scaling=1.0, near_plane=0.05, far_plane = 25)
        elif stage_path:
            self.renderer = newton.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

        #self.renderer._camera_pos = [0.2, 1.2, -1.0]
        #self.renderer.update_view_matrix()

    def _setup_cuda_graph(self):
        """Setup CUDA graph for performance optimization."""
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
#endregion
#region Simulation Loop
    def simulate(self):
        """Run one simulation step with all substeps."""

        #self.integrator.collison_detection(self.state_0.particle_q)

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
#endregion
    # region Rendering
    def render(self):
        """Render the current simulation state."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            if self.use_opengl:
                self.renderer.begin_frame()

                # self.renderer.render_sphere(
                #     "haptic_proxy_sphere",
                #     [self.haptic_pos_right[0] * 0.01, self.haptic_pos_right[1] * 0.01, self.haptic_pos_right[2] * 0.01],
                #     [0.0, 0.0, 0.0, 1.0],
                #     0.025,
                # )


                # self.renderer.render_points(
                #     "particles", self.model.particle_q, radius=self.model.particle_radius.numpy(), colors=(0.8, 0.3, 0.2)
                # )


                # Create per-vertex colors for multi-label visualization (only once)
                if not hasattr(self, '_vertex_colors') or True:  # Force regeneration for spatial coloring
                    self._vertex_colors = self._create_vertex_colors_for_labels()
                
                self.renderer.render_mesh_warp(
                    "surface",
                    self.model.particle_q,
                    self.model.tri_indices,
                    vertex_colors=self._vertex_colors,
                    basic_color=(1.0, 1.0, 1.0),  # White base to let vertex colors show through
                )

                # # render springs
                # self.render_line_list(
                #     "springs", self.model.particle_q, self.model.spring_indices.numpy().flatten(), (0.25, 0.5, 0.25), 0.02
                # )

                #wp.copy(self.integrator.dev_pos_prev_buffer, self.integrator.dev_pos_buffer)
                self.renderer.end_frame()
            else:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()


#endregion
    def _create_vertex_colors_for_labels(self):
        """Create per-vertex colors for multi-label visualization based on spatial regions."""
        import numpy as np
        import warp as wp
        
        # Get vertex positions
        positions = self.model.particle_q.numpy()
        num_vertices = positions.shape[0]
        
        # Define colors for each label (RGB + Alpha)
        label_colors = [
            (0.8, 0.2, 0.2, 1.0),  # Red for label 1
            (0.2, 0.8, 0.2, 1.0),  # Green for label 2  
            (0.2, 0.2, 0.8, 1.0),  # Blue for label 3
        ]
        
        # Create vertex colors array
        vertex_colors = np.zeros((num_vertices, 4), dtype=np.float32)
        
        # Analyze spatial distribution
        x_coords = positions[:, 0]
        y_coords = positions[:, 1] 
        z_coords = positions[:, 2]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        print(f"Mesh bounds: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}], Z[{z_min:.3f}, {z_max:.3f}]")
        
        # Assign colors based on spatial regions (using Y-axis for vertical slicing)
        # This assumes the liver is oriented vertically and we want horizontal slices
        y_range = y_max - y_min
        y_third = y_range / 3
        
        label_counts = [0, 0, 0]
        
        for i in range(num_vertices):
            y = y_coords[i]
            
            if y < y_min + y_third:
                label_idx = 0  # Bottom third - Red
            elif y < y_min + 2 * y_third:
                label_idx = 1  # Middle third - Green  
            else:
                label_idx = 2  # Top third - Blue
                
            vertex_colors[i] = label_colors[label_idx]
            label_counts[label_idx] += 1
        
        # Debug: Print spatial color statistics
        print(f"Spatial vertex coloring: {num_vertices} vertices")
        print(f"Y-axis slicing: Bottom[{y_min:.3f}, {y_min + y_third:.3f}] = {label_counts[0]} Red vertices")
        print(f"                Middle[{y_min + y_third:.3f}, {y_min + 2*y_third:.3f}] = {label_counts[1]} Green vertices")
        print(f"                Top[{y_min + 2*y_third:.3f}, {y_max:.3f}] = {label_counts[2]} Blue vertices")
        
        return wp.array(vertex_colors, dtype=wp.vec4, device=wp.get_device())

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


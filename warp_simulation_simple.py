import warp as wp
import warp
import warp.sim
import warp.sim.render
import newton

# from newton.utils.render import SimRendererOpenGL
# from newton.solvers import XPBDSolver

import numpy as np
import math
import colorsys

from PBDSolver import PBDSolver
from render_surgsim_opengl import SurgSimRendererOpenGL

from mesh_loader import Tetrahedron, load_mesh_and_build_model
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
    def __init__(self, stage_path="output.usd", num_frames=300, use_opengl=True, mesh_path=None, coloring_mode: str = "subdomain"):
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
        self.mesh_path = mesh_path
        self.coloring_mode = coloring_mode or "subdomain"

        self.cutting_active = False
        self.heating_active = False
        self.grasping_active = False
        self.clipping = False


        print(f"Initializing WarpSim with {self.sim_substeps} substeps at {self.fps} FPS")
        print(f"Frame time: {self.frame_dt:.4f}s, Substep time: {self.substep_dt:.4f}s")

        # Initialize model
        self._build_model()
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        # Precompute and cache vertex colors (avoid per-frame regeneration)
        try:
            self._vertex_colors = self._create_vertex_colors_for_labels()
        except Exception:
            self._vertex_colors = None
        # Expose to renderer/model for dynamic per-vertex color rendering
        try:
            if self._vertex_colors is not None:
                self.model.vertex_colors = self._vertex_colors
        except Exception:
            pass
        # Prepare vec4 colors for renderers that expect RGBA/weights
        try:
            if self._vertex_colors is not None:
                import numpy as _np
                cols = self._vertex_colors.numpy()
                if cols.ndim == 2 and cols.shape[1] == 3:
                    _w = _np.zeros((cols.shape[0], 1), dtype=cols.dtype)
                    _cols4 = _np.concatenate([cols, _w], axis=1)
                    self._vertex_colors4 = wp.array(_cols4, dtype=wp.vec4, device=wp.get_device())
                elif cols.ndim == 2 and cols.shape[1] == 4:
                    self._vertex_colors4 = self._vertex_colors
                else:
                    self._vertex_colors4 = wp.zeros(self.model.particle_count, dtype=wp.vec4, device=wp.get_device())
            else:
                self._vertex_colors4 = wp.zeros(self.model.particle_count, dtype=wp.vec4, device=wp.get_device())
        except Exception:
            self._vertex_colors4 = wp.zeros(self.model.particle_count, dtype=wp.vec4, device=wp.get_device())
        
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

        # Optional UI draw callback (e.g., ImGui overlay)
        self.ui_draw_callback = None


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
        tri_points_connectors, self.surface_tris, uvs, self.mesh_ranges, tetrahedra_wp = load_mesh_and_build_model(
            builder,
            particle_mass=self.particle_mass,
            vertical_offset=-3.0,
            spring_stiffness=spring_stiffness,
            spring_dampen=spring_dampen,
            tetra_stiffness_mu=tetra_stiffness_mu,
            tetra_stiffness_lambda=tetra_stiffness_lambda,
            tetra_dampen=tetra_dampen,
            warp_mesh_path=self.mesh_path,
        )
        
        self.tet_subdomain_labels = None
        self.label_ranges = None
        self.subdomain_color_map = {}
        if isinstance(self.mesh_ranges, dict):
            subdomains = self.mesh_ranges.get('_tet_subdomains')
            if subdomains:
                self.tet_subdomain_labels = np.asarray(subdomains, dtype=np.int32)
                unique_labels = np.unique(self.tet_subdomain_labels)
                print(f"Loaded {len(unique_labels)} tetrahedron subdomains: {unique_labels.tolist()}")
                for idx, label in enumerate(sorted(unique_labels)):
                    hue = (idx / max(len(unique_labels), 1)) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
                    self.subdomain_color_map[int(label)] = (float(r), float(g), float(b))
            label_ranges = self.mesh_ranges.get('_label_ranges')
            if label_ranges:
                self.label_ranges = label_ranges

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

        builder.add_particle_grid(
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=0.2,
            cell_y=0.2,
            cell_z=0.2,
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=1.0,
            jitter=0.001,
        )
        
        self.model = builder.finalize()
        self.model.tetrahedra_wp = tetrahedra_wp
        self.model.tet_active = wp.ones(self.model.tetrahedra_wp.shape[0], dtype=wp.int32, device=wp.get_device())
        self.model.tri_points_connectors = tri_points_connectors
        self.model.particle_max_velocity = 10.0

        # Build UV array to match particle_count (after model is finalized)
        try:
            import numpy as _np
            total = int(self.model.particle_count)
            uvs_np = _np.zeros((total, 2), dtype=_np.float32)
            if isinstance(uvs, (list, tuple)) and len(uvs) > 0:
                src = _np.asarray(uvs, dtype=_np.float32)
                count = min(src.shape[0], total)
                uvs_np[:count, :] = src[:count, :]
            self.uvs_wp = wp.array(uvs_np, dtype=wp.vec2f, device=wp.get_device())
        except Exception:
            self.uvs_wp = wp.zeros(self.model.particle_count, dtype=wp.vec2f, device=wp.get_device())


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
            # Use the surgical OpenGL renderer which supports vertex colors and
            # robust shading for meshes without bound textures.
            self.renderer = SurgSimRendererOpenGL(self.model,"Warp Surgical Simulation", scaling=1.0, near_plane=0.01, far_plane=100.0)
            #self.renderer = newton.viewer.ViewerGL()
            #self.renderer.set_model(self.model)
           
        elif stage_path:
            self.renderer = newton.viewer.RendererUsd(self.model, stage_path, scaling=1.0)
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
            # wp.launch(
            #     set_body_position,
            #     dim=1,
            #     inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.integrator.dev_pos_buffer, self.substep_dt],
            #     device=self.state_0.body_q.device,
            # )

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

    # region Runtime Parameter Setters
    def set_ui_draw_callback(self, callback):
        """Register a callable executed inside render() between begin/end frame."""
        self.ui_draw_callback = callback

    def set_substeps(self, substeps: int):
        try:
            substeps = int(max(1, substeps))
        except Exception:
            return
        self.sim_substeps = substeps
        self.substep_dt = self.frame_dt / float(self.sim_substeps)

    def set_iterations(self, iterations: int):
        try:
            iterations = int(max(1, iterations))
        except Exception:
            return
        if hasattr(self, "integrator"):
            self.integrator.iterations = iterations

    def set_distance_stiffness(self, stiffness: float):
        try:
            val = float(max(0.0, stiffness))
        except Exception:
            return
        try:
            if hasattr(self.model, "spring_stiffness") and self.model.spring_stiffness is not None:
                self.model.spring_stiffness.fill_(val)
        except Exception:
            pass

    def set_distance_damping(self, damping: float):
        try:
            val = float(max(0.0, damping))
        except Exception:
            return
        try:
            if hasattr(self.model, "spring_damping") and self.model.spring_damping is not None:
                self.model.spring_damping.fill_(val)
        except Exception:
            pass

    def set_volume_enabled(self, enabled: bool):
        try:
            if hasattr(self, "integrator"):
                self.integrator.volCnstrs = bool(enabled)
        except Exception:
            pass

    def set_volume_stiffness(self, stiffness: float):
        try:
            val = float(max(0.0, stiffness))
        except Exception:
            return
        try:
            if hasattr(self, "integrator") and hasattr(self.integrator, "vol_stiffness"):
                self.integrator.vol_stiffness = val
        except Exception:
            pass
#endregion
    # region Rendering
    def render(self):
        """Render the current simulation state."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            if self.use_opengl:
                self.renderer.begin_frame(self.sim_time)

                # self.renderer.render_sphere(
                #     "haptic_proxy_sphere",
                #     [self.haptic_pos_right[0] * 0.01, self.haptic_pos_right[1] * 0.01, self.haptic_pos_right[2] * 0.01],
                #     [0.0, 0.0, 0.0, 1.0],
                #     0.025,
                # )
                #self.renderer.render(self.state_0)

                try:
                    pr = float(self.model.particle_radius.numpy())
                except Exception:
                    pr = 0.0
                # Ensure a visible point radius in case model radius is unset/zero
                render_radius = pr if pr and pr > 0.0 else 0.05
                # self.renderer.render_points(
                #     "particles", self.model.particle_q, radius=render_radius, colors=(0.8, 0.3, 0.2)
                # )

                #self.renderer.log_points("particles", self.state_0.particle_q, self.model.particle_radius, self.model.particle_colors)

                # Create per-vertex colors for multi-label visualization (cached)
                # if not hasattr(self, '_vertex_colors') or self._vertex_colors is None:
                #     self._vertex_colors = self._create_vertex_colors_for_labels()
                
                # Explicit mesh render (manual path), driven by current simulation state
                self.renderer.render_mesh_warp(
                    name="surface",
                    points=self.state_0.particle_q,
                    indices=self.surface_tris_wp,
                    texture_coords=self.uvs_wp,
                    vertex_colors=self._vertex_colors4,
                    basic_color=(1.0, 1.0, 1.0),
                    smooth_shading=True,
                    update_topology=True,
                )

                # Draw overlay UI if provided (execute after scene, before swap)
                try:
                    if callable(self.ui_draw_callback):
                        self.ui_draw_callback()
                except Exception:
                    pass

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
        """Create per-vertex colors.

        - 'subdomain': use tetra subdomain labels to color vertices
        - 'spatial': fallback spatial bands by Y
        - 'none': uniform white
        """
        import numpy as np
        import warp as wp

        positions = self.model.particle_q.numpy()
        num_vertices = positions.shape[0]

        vertex_colors = np.zeros((num_vertices, 3), dtype=np.float32)

        # None: return white
        if str(self.coloring_mode).lower() == 'none':
            vertex_colors[:, 0:3] = (1.0, 1.0, 1.0)
            return wp.array(vertex_colors, dtype=wp.vec3, device=wp.get_device())

        # Subdomain coloring when labels available
        if str(self.coloring_mode).lower() == 'subdomain' and getattr(self, 'tet_subdomain_labels', None) is not None and len(self.tet_subdomain_labels):
            tetra_array = self.model.tetrahedra_wp
            tet_indices = None
            if tetra_array is not None and tetra_array.shape[0] > 0:
                tet_np = tetra_array.numpy()
                if tet_np is not None:
                    if getattr(tet_np, 'dtype', None) is not None and tet_np.dtype.fields and 'ids' in tet_np.dtype.fields:
                        tet_indices = np.asarray(tet_np['ids'], dtype=np.int64)
                    else:
                        tet_indices = np.asarray(tet_np, dtype=np.int64)
                    if tet_indices.ndim == 1:
                        tet_indices = tet_indices.reshape((-1, 4))

            if tet_indices is not None and len(tet_indices) == len(self.tet_subdomain_labels):
                vertex_label_lists = [[] for _ in range(num_vertices)]
                for tet_idx, tet in enumerate(tet_indices):
                    label = int(self.tet_subdomain_labels[tet_idx])
                    for vid in np.asarray(tet, dtype=np.int64):
                        if 0 <= vid < num_vertices:
                            vertex_label_lists[int(vid)].append(label)

                vertex_labels = np.zeros(num_vertices, dtype=np.int32)
                for vid, candidates in enumerate(vertex_label_lists):
                    if candidates:
                        counts = np.bincount(np.asarray(candidates, dtype=np.int32))
                        vertex_labels[vid] = int(counts.argmax())

                color_map = getattr(self, 'subdomain_color_map', {}) or {}
                for vid, label in enumerate(vertex_labels):
                    rgb = color_map.get(int(label))
                    if rgb is None:
                        hue = (int(label) * 0.61803398875) % 1.0
                        rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
                        color_map[int(label)] = rgb
                    vertex_colors[vid, 0:3] = rgb

                self.subdomain_color_map = color_map
                # Cache on device once; no per-frame printing
                return wp.array(vertex_colors, dtype=wp.vec3, device=wp.get_device())
            else:
                print('Warning: Unable to map subdomain labels to tetrahedra; using spatial coloring fallback')

        # Spatial coloring fallback or explicit request
        if str(self.coloring_mode).lower() not in ('subdomain',) or getattr(self, 'tet_subdomain_labels', None) is None:
            pass

        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        z_coords = positions[:, 2]

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)

        # Bounds useful for debugging but noisy; skip in production path

        label_colors = [
            (0.8, 0.2, 0.2),
            (0.2, 0.8, 0.2),
            (0.2, 0.2, 0.8),
        ]

        y_range = max(y_max - y_min, 1e-6)
        y_third = y_range / 3.0
        label_counts = [0, 0, 0]

        for i in range(num_vertices):
            y = y_coords[i]
            if y < y_min + y_third:
                label_idx = 0
            elif y < y_min + 2.0 * y_third:
                label_idx = 1
            else:
                label_idx = 2
            vertex_colors[i] = label_colors[label_idx]
            label_counts[label_idx] += 1

        # Return cached device array without per-frame logging
        return wp.array(vertex_colors, dtype=wp.vec3, device=wp.get_device())

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


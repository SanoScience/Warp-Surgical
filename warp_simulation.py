import KRNSolver
from centrelines import CentrelinePointInfo, ClampConstraint, attach_clip_to_nearest_centreline, check_centreline_leaks, compute_centreline_positions, cut_centrelines_near_haptic, emit_bleed_particles, update_bleed_particles, update_centreline_leaks
from connectivity import generate_connectivity, recompute_connectivity, setup_connectivity
from grasping import grasp_end, grasp_process, grasp_start
from heating import heating_active_process, heating_conduction_process, heating_end, heating_start, paint_vertices_near_haptic_proxy, set_paint_strength
from integrator_pbf import PBFIntegrator
from stretching import stretching_breaking_process
from surface_reconstruction import extract_surface_triangles_bucketed
from simulation_systems import BoundsCollisionSystem, DistanceConstraintSystem
import warp as wp
import newton
from pxr import Usd, UsdGeom

#from newton.utils.render import SimRendererOpenGL
#from newton.solvers import XPBDSolver
#from newton.solvers import VBDSolver
from newton import ParticleFlags
import numpy as np
import math

from PBDSolver import PBDSolver
from render_surgsim_opengl import SurgSimRendererOpenGL
import fluids
import instruments

from mesh_loader import Tetrahedron, load_background_mesh, load_mesh_and_build_model, parse_centreline_file
from render_opengl import CustomOpenGLRenderer
from simulation_kernels import (
    set_body_position,
)

def axis_angle_to_quat(axis, angle):
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    return [
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        np.cos(half_angle)
    ]


def multiply_quaternions(q1, q2):
    """Multiply two quaternions: q1 * q2. Format: [x, y, z, w]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
        w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
    ]

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

@wp.kernel
def interpolate_haptic_position(
    haptic_pos_src: wp.array(dtype=wp.vec3f),         # [1]
    haptic_pos_dst: wp.array(dtype=wp.vec3f),         # [1]
    haptic_pos_result: wp.array(dtype=wp.vec3f),      # [1]
    factor: float,
):
    tid = wp.tid()
    if tid >= 1:
        return

    pos_src = haptic_pos_src[0]
    pos_dst = haptic_pos_dst[0]

    result = wp.lerp(pos_src, pos_dst, factor)
    haptic_pos_result[0] = result



class WarpSim:
    #region Initialization
    def __init__(self, stage_path="output.usd", num_frames=300, use_opengl=True, jaw_collider_profiles=None):
        self.sim_substeps = 6
        self.num_frames = num_frames
        self.fps = 60

        self.frame_dt = 1.0 / self.fps
        self.substep_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_constraint_iterations = 16

        self.load_textures = False
        self.allow_cuda_graph = False

        self.haptic_pos_right = None  # Haptic device position in simulation space
        self.haptic_rot_right = [0.0, 0.0, 0.0, 1.0]  # Haptic device rotation as quaternion

        self.radius_collision = 0.1
        self.radius_heating = 0.2
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

        default_jaw_collider_profiles = {
            "default": [
                {"offset": (0.0, 0.0, 0.4), "radius": 0.1},
            ]
        }
        self.jaw_collider_profiles = jaw_collider_profiles or default_jaw_collider_profiles
        self.jaw_colliders = []

        # Initialize model
        self._build_model()
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec4f, device=wp.get_device())
        
        # Connectivity setup
        setup_connectivity(self)

        self.max_clips = 64
        self.clip_attached = wp.zeros(self.centreline_points.shape[0], dtype=wp.int32, device=wp.get_device())
        self.clip_indices = wp.zeros(self.max_clips, dtype=wp.int32, device=wp.get_device())
        self.clip_count = wp.zeros(1, dtype=wp.int32, device=wp.get_device())

        self.centreline_cut_flags = wp.zeros(self.centreline_points.shape[0], dtype=wp.int32, device=wp.get_device())

        fluids.setup_fluids_data(self)
        fluids.setup_fluids_rendering(self)

        # Initialize simulation components
        self._setup_simulation()
        
        # Initialize rendering
        self._setup_renderer(stage_path, use_opengl)

        # Grasp setup
        self.grasp_capacity = 1024
        self.grasped_particles_buffer = wp.zeros(self.grasp_capacity, dtype=wp.int32, device=wp.get_device())
        self.grasped_particles_counter = wp.zeros(1, dtype=wp.int32, device=wp.get_device())

        # Heating setup
        self.paint_color_buffer = wp.array([wp.vec4(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=wp.get_device())
        self.paint_strength_buffer = wp.array([0.0], dtype=wp.float32, device=wp.get_device()) 
        set_paint_strength(self, 0.0)
    
        self._load_background_mesh()

        # Load textures
        if self.use_opengl and self.renderer:

            self.background_diffuse = [self.renderer.load_texture("textures/cavity_diffuse.tga")]
            self.background_normal = [self.renderer.load_texture("textures/cavity_normals.tga")]
            self.background_spec = [self.renderer.load_texture("textures/cavity_spec.png")]

            if self.load_textures:
                for mesh_name, _ in self.mesh_ranges.items():
                    setattr(self, f"{mesh_name}_diffuse_maps", [self.renderer.load_texture(f"textures/{mesh_name}/diffuse-base.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/diffuse-coag.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/diffuse-damage.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/diffuse-blood.png")])
                    
                    setattr(self, f"{mesh_name}_normal_maps", [self.renderer.load_texture(f"textures/{mesh_name}/normal-base.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/normal-coag.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/normal-damage.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/normal-blood.png")])
                    
                    setattr(self, f"{mesh_name}_specular_maps", [self.renderer.load_texture(f"textures/{mesh_name}/spec-base.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/spec-coag.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/spec-damage.png"),
                                                                self.renderer.load_texture(f"textures/{mesh_name}/spec-blood.png")])
            
            self.renderer.set_input_callbacks(
                on_key_press=self._on_key_press,
                on_key_release=self._on_key_release
            )

        # Setup CUDA graph if available
        self._setup_cuda_graph()

        generate_connectivity(self)

    
#endregion

    
    def _matrix_to_quaternion(self, matrix):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(matrix)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2  # s = 4 * qx
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2  # s = 4 * qy
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2  # s = 4 * qz
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
        
        return [x, y, z, w]

    def update_instrument_transform(self, instrument_id, position=None, rotation=None, scale=None):
        """Update instrument root transform and propagate to hierarchy"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return
        
        instrument = self.instruments[instrument_id]
        
        # Build root transform matrix from components
        if position is None:
            position = [0.0, 0.0, 0.0]
        if rotation is None:
            rotation = [0.0, 0.0, 0.0, 1.0]  # identity quaternion
        if scale is None:
            scale = [1.0, 1.0, 1.0]
        
        # Create transform and scale matrices
        pos = wp.vec3(position[0], position[1], position[2])
        rot = wp.quat(rotation[0], rotation[1], rotation[2], rotation[3])
        transform = wp.transform(pos, rot)
        transform_matrix = wp.transform_to_matrix(transform)
        
        # Apply scale
        instrument['root_transform_matrix'] = wp.mat44f(
            transform_matrix[0, 0] * scale[0], transform_matrix[0, 1], transform_matrix[0, 2], transform_matrix[0, 3],
            transform_matrix[1, 0], transform_matrix[1, 1] * scale[1], transform_matrix[1, 2], transform_matrix[1, 3],
            transform_matrix[2, 0], transform_matrix[2, 1], transform_matrix[2, 2] * scale[2], transform_matrix[2, 3],
            transform_matrix[3, 0], transform_matrix[3, 1], transform_matrix[3, 2], transform_matrix[3, 3]
        )
        
        # Update world transforms for entire hierarchy
        self._update_instrument_hierarchy_transforms(instrument_id)

    def update_piece_transform(self, instrument_id, piece_name, transform_matrix):
        """Update a specific piece's runtime local transform matrix"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return
        
        instrument = self.instruments[instrument_id]
        
        # Find piece by name
        piece_index = None
        for i, piece in enumerate(instrument['pieces']):
            if piece['name'] == piece_name:
                piece_index = i
                break
        
        if piece_index is None:
            print(f"Piece '{piece_name}' not found in instrument")
            return
        
        piece = instrument['pieces'][piece_index]
        
        # Update runtime local transform matrix
        piece['runtime_local_transform'] = transform_matrix
        
        # Update world transforms for this piece and its children
        self._update_piece_world_transform(instrument_id, piece_index)

    def _update_instrument_hierarchy_transforms(self, instrument_id):
        """Update world transforms for entire instrument hierarchy"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return
        
        instrument = self.instruments[instrument_id]
        
        # Start with root pieces
        for root_index in instrument['root_pieces']:
            self._update_piece_world_transform(instrument_id, root_index)

    def _update_piece_world_transform(self, instrument_id, piece_index):
        """Recursively update world transform for a piece and its children"""
        instrument = self.instruments[instrument_id]
        piece = instrument['pieces'][piece_index]
        
        # Get parent world transform matrix
        parent_world_matrix = instrument['root_transform_matrix']
        
        if piece['parent_index'] is not None:
            parent_piece = instrument['pieces'][piece['parent_index']]
            parent_world_matrix = parent_piece['world_transform_matrix']
        
        # Compute world transform
        combined_local = piece['runtime_local_transform'] * piece['usd_local_transform']
        piece['world_transform_matrix'] = parent_world_matrix * combined_local
        
        # Apply transform to vertices
        self._transform_piece_vertices(instrument_id, piece_index)
        
        # Recursively update children
        for child_index in piece['children_indices']:
            self._update_piece_world_transform(instrument_id, child_index)

    @wp.kernel
    def apply_matrix_transform_to_vertices(
        original_vertices: wp.array(dtype=wp.vec3f),
        transformed_vertices: wp.array(dtype=wp.vec3f),
        transform_matrix: wp.mat44f,
        num_vertices: int
    ):
        tid = wp.tid()
        if tid >= num_vertices:
            return
        
        # Get original vertex
        orig_vert = original_vertices[tid]
        
        # Apply transformation using homogeneous coordinates
        vert_homo = wp.vec4f(orig_vert[0], orig_vert[1], orig_vert[2], 1.0)
        transformed_homo = transform_matrix * vert_homo
        
        # Convert back to 3D
        transformed_vertices[tid] = wp.vec3f(
            transformed_homo[0], 
            transformed_homo[1], 
            transformed_homo[2]
        )

    def _transform_piece_vertices(self, instrument_id, piece_index):
        """Apply current transform to piece vertices"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return
        
        piece = self.instruments[instrument_id]['pieces'][piece_index]
        
        wp.launch(
            self.apply_matrix_transform_to_vertices,
            dim=piece['vertex_count'],
            inputs=[
                piece['original_vertices'],
                piece['vertices'],
                piece['world_transform_matrix'],
                piece['vertex_count']
            ],
            device=wp.get_device()
        )

    def set_instrument_visibility(self, instrument_id, visible):
        """Set instrument visibility"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return
        
        self.instruments[instrument_id]['visible'] = visible

    def set_piece_visibility(self, instrument_id, piece_name, visible):
        """Set visibility for a specific piece"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return
        
        instrument = self.instruments[instrument_id]
        for piece in instrument['pieces']:
            if piece['name'] == piece_name:
                piece['visible'] = visible
                break

    def get_piece_names(self, instrument_id):
        """Get list of piece names for an instrument"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return []
        
        return [piece['name'] for piece in self.instruments[instrument_id]['pieces']]


    def get_instrument_position(self, instrument_id):
        """Get instrument world position"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return None
        
        pos = self.instruments[instrument_id]['position']
        return [pos[0], pos[1], pos[2]]


    def _list_usd_primitives(self, usd_path, max_depth=10):
        """List all primitives in a USD file"""
        try:
            stage = Usd.Stage.Open(usd_path)
            if not stage:
                print(f"Failed to load USD file: {usd_path}")
                return []
            
            print(f"\n=== USD Primitives in {usd_path} ===")
            
            def print_prim_info(prim, depth=0):
                if depth > max_depth:
                    return
                    
                indent = "  " * depth
                type_name = prim.GetTypeName()
                path = prim.GetPath()
                
                # Check if it's a mesh
                is_mesh = prim.IsA(UsdGeom.Mesh)
                mesh_info = ""
                
                if is_mesh:
                    mesh = UsdGeom.Mesh(prim)
                    try:
                        points_attr = mesh.GetPointsAttr()
                        faces_attr = mesh.GetFaceVertexIndicesAttr()
                        
                        if points_attr and faces_attr:
                            points = points_attr.Get()
                            faces = faces_attr.Get()
                            if points and faces:
                                mesh_info = f" [MESH: {len(points)} vertices, {len(faces)//3} triangles]"
                    except:
                        mesh_info = " [MESH: error reading geometry]"
                

                geom_info = ""
                if prim.IsA(UsdGeom.Sphere):
                    geom_info = " [SPHERE]"
                elif prim.IsA(UsdGeom.Cube):
                    geom_info = " [CUBE]"
                elif prim.IsA(UsdGeom.Cylinder):
                    geom_info = " [CYLINDER]"
                elif prim.IsA(UsdGeom.Cone):
                    geom_info = " [CONE]"
                elif prim.IsA(UsdGeom.Xform):
                    geom_info = " [TRANSFORM]"
                
                print(f"{indent}{path} ({type_name}){mesh_info}{geom_info}")
                
                for child in prim.GetChildren():
                    print_prim_info(child, depth + 1)
            
            # Start from root
            root = stage.GetPseudoRoot()
            for child in root.GetChildren():
                print_prim_info(child)
                
            print("=== End USD Primitives ===\n")
            
            # Return paths found
            mesh_paths = []
            def collect_meshes(prim):
                if prim.IsA(UsdGeom.Mesh):
                    mesh_paths.append(str(prim.GetPath()))
                for child in prim.GetChildren():
                    collect_meshes(child)
            
            for child in root.GetChildren():
                collect_meshes(child)
                
            if mesh_paths:
                print("Found mesh primitives at paths:")
                for path in mesh_paths:
                    print(f"  - {path}")
            else:
                print("No mesh primitives found in USD file")
                
            return mesh_paths
            
        except Exception as e:
            print(f"Error reading USD file {usd_path}: {e}")
            return []

    def _on_key_press(self, symbol, modifiers):
        from pyglet.window import key
        if symbol == key.C:
            self.cutting_active = True
        elif symbol == key.V:
            heating_start(self)
        elif symbol == key.B:
            grasp_start(self)
        elif symbol == key.G:
            self.clipping = True
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

        # Centrelines
        cntr1_points, cntr1_clamp_cnstr, cntr1_edge_cnstr = parse_centreline_file(
            'meshes/centrelines/cystic_artery.cntr',
            self.mesh_ranges['gallbladder']['vertex_start'],
            self.mesh_ranges['gallbladder']['edge_start']
        )
        cntr2_points, cntr2_clamp_cnstr, cntr2_edge_cnstr = parse_centreline_file(
            'meshes/centrelines/cystic_duct.cntr',
            self.mesh_ranges['gallbladder']['vertex_start'],
            self.mesh_ranges['gallbladder']['edge_start']
        )

        # Merge lists
        merged_points = cntr1_points + cntr2_points
        merged_clamp_cnstr = cntr1_clamp_cnstr + cntr2_clamp_cnstr
        merged_edge_cnstr = cntr1_edge_cnstr + cntr2_edge_cnstr

        self.centreline_points = wp.array(merged_points, dtype=CentrelinePointInfo, device=wp.get_device())
        self.centreline_states = wp.zeros(len(merged_points), dtype=wp.int32, device=wp.get_device())
        self.centreline_clamp_cnstr = wp.array(merged_clamp_cnstr, dtype=ClampConstraint, device=wp.get_device())
        self.centreline_edge_conn = wp.array(merged_edge_cnstr, dtype=wp.int32, device=wp.get_device())
        self.centreline_avg_positions = wp.zeros(self.centreline_points.shape[0], dtype=wp.vec3f, device=wp.get_device())

        # Add haptic device collision body
        #self.haptic_body_id = builder.add_body(
        #    xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        #    mass=0.0,  # Zero mass makes it kinematic
        #    armature=0.0
        #)

        #builder.add_shape_sphere(
        #    body=self.haptic_body_id,
        #    xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        #    radius=self.radius_collision,
        #    cfg=newton.ModelBuilder.ShapeConfig(
        #        density=10
        #    )
        #)

        # Import instruments
        self.instrument_id = instruments.load_instrument_from_usd(self, "meshes/pgrasp.usdc", builder, "pgrasp")
        if self.instrument_id is not None:
            print(f"Successfully loaded instrument with ID: {self.instrument_id}")
        else:
            print("Failed to load instrument")
        
        self.model = builder.finalize()
        self.model.tetrahedra_wp = tetrahedra_wp
        self.model.tet_active = wp.ones(self.model.tetrahedra_wp.shape[0], dtype=wp.int32, device=wp.get_device())
        self.model.tri_points_connectors = tri_points_connectors

        self.model.particle_max_velocity = 10.0

    def _load_background_mesh(self):
        positions, tri_indices, uvs, vertex_colors = load_background_mesh()
        self.background_mesh = positions
        self.background_tri_indices = tri_indices
        self.background_uvs = uvs
        self.background_vertex_colors = vertex_colors

    def _setup_simulation(self):
        """Initialize simulation states and integrator."""
        self.integrator = KRNSolver.KRNSolver(self.model, iterations=self.sim_constraint_iterations)

        # Register simulation systems
        self.integrator.register_system(DistanceConstraintSystem(priority=50))
        self.integrator.register_system(BoundsCollisionSystem(
            bounds_min=wp.vec3(-2.0, 0.0, -8.0),
            bounds_max=wp.vec3(2.0, 10.0, -3.0),
            priority=100
        ))

        self.integrator.dev_pos_current_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.integrator.dev_pos_target_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.integrator.dev_pos_prev_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())

        self.rest = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.collide(self.state_0)
        
        instruments._update_jaw_colliders(self, states=[self.state_0, self.state_1], update_solver=True)
        
        # Create Jacobian accumulators
        # self.model.delta_accumulator = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        # self.model.count_accumulator = wp.zeros(self.model.particle_count, dtype=wp.int32, device=wp.get_device())

    def _setup_renderer(self, stage_path, use_opengl):
        """Initialize the appropriate renderer."""
        self.use_opengl = use_opengl
        
        if self.use_opengl:
            self.renderer = SurgSimRendererOpenGL(self.model, "Warp Surgical Simulation", scaling=1.0, near_plane=0.05, far_plane = 25)
        elif stage_path:
            self.renderer = newton.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

        self.renderer._camera_pos = [0.2, 1.2, -1.0]
        #self.renderer.update_view_matrix()

    def _setup_cuda_graph(self):
        """Setup CUDA graph for performance optimization."""
        self.use_cuda_graph = wp.get_device().is_cuda and self.allow_cuda_graph
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
#endregion
#region Simulation Loop
    def simulate(self):
        """Run one simulation step with all substeps."""

        #self.integrator.collison_detection(self.state_0.particle_q)

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            factor = float(i) / float(self.sim_substeps)
            wp.launch(
                interpolate_haptic_position,
                dim=1,
                inputs=[
                    self.integrator.dev_pos_prev_buffer,
                    self.integrator.dev_pos_target_buffer,
                    self.integrator.dev_pos_current_buffer,
                    factor
                ],
                device=wp.get_device()
            )

            instruments._update_jaw_colliders(self, states=[self.state_0, self.state_1], update_solver=True)

            # Update haptic device position
            #wp.launch(
            #    set_body_position,
            #    dim=1,
            #    inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.integrator.dev_pos_current_buffer, self.substep_dt],
            #    device=self.state_0.body_q.device,
            #)

            # Run collision detection and integration
            #self.contacts = self.model.collide(self.state_0)
            #if self.contacts:
            #    print(f"Contacts detected: {self.contacts.soft_contact_normal}")
            
            fluids.simulate_fluid(self)

            self.integrator.step(self.state_0, self.state_1, None, self.contacts, self.substep_dt)


            # Recompute connectivity
            recompute_connectivity(self)

            # Heat conduction
            heating_conduction_process(self)
            stretching_breaking_process(self)
            
            # Swap states
            instruments._update_jaw_colliders(self, states=[self.state_1], update_solver=False)
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

                #self.renderer.render_sphere(
                #    "haptic_proxy_sphere",
                #    [self.haptic_pos_right[0] * 0.01, self.haptic_pos_right[1] * 0.01, self.haptic_pos_right[2] * 0.01],
                #    [0.0, 0.0, 0.0, 1.0],
                #    0.025,
                #)

                # Render background mesh
                with wp.ScopedTimer("background"):
                    if self.background_mesh is not None and self.background_tri_indices is not None:
                        background_update = not self.background_mesh_uploaded
                        shape_id = self.renderer.render_mesh_warp(
                            name="background_mesh",
                            points=self.background_mesh,
                            indices=self.background_tri_indices,
                            texture_coords=self.background_uvs,
                            vertex_colors=self.background_vertex_colors,
                            diffuse_maps=self.background_diffuse,
                            normal_maps=self.background_normal,
                            specular_maps=self.background_spec,
                            pos=(0.0, 1.0, -5.4),
                            rot=(0.0, 0.0, 0.0, 1.0),
                            scale=(1.2, 1.2, 1.2),
                            update_topology=True,
                            smooth_shading=True,
                            visible=True
                        )
                        if background_update and shape_id is not None:
                            self.background_mesh_uploaded = True
                            self.background_mesh_shape_id = shape_id

                # detector = self.integrator.trimesh_collision_detector
                # num_collisions = int(detector.vertex_colliding_triangles_count.numpy().sum())
                # print(f"Vertex-triangle collisions detected: {num_collisions}")

                # Grasping
                with wp.ScopedTimer("grasping"):
                    if self.grasping_active:
                        grasp_process(self)


                # Centreline update
                with wp.ScopedTimer("centrelines"):
                    wp.launch(
                        compute_centreline_positions,
                        dim=self.centreline_points.shape[0],
                        inputs=[
                            self.centreline_points,
                            self.centreline_clamp_cnstr,
                            self.state_0.particle_q,
                            self.centreline_avg_positions
                        ],
                        device=wp.get_device()
                    )

                    if self.clipping:
                        wp.launch(
                            attach_clip_to_nearest_centreline,
                            dim=1,
                            inputs=[
                                self.centreline_points,
                                self.centreline_avg_positions,
                                self.integrator.dev_pos_current_buffer,
                                self.clip_attached,
                                self.clip_indices,
                                self.clip_count,
                                self.max_clips,
                                self.radius_clipping
                            ],
                            device=wp.get_device()
                        )
                        self.clipping = False

                    results = check_centreline_leaks(self.centreline_states, self.centreline_points.shape[0])
                    if results["clipping_ready_to_cut"]:
                        print("Ready to cut between:", results["valid_ids_to_cut"])
                    if results["clipping_done"]:
                        print("Clipping done!")
                    if results["clipping_error"]:
                        print("Clipping error detected!")

                    # Draw centrelines
                    clip_count = int(self.clip_count.numpy()[0])
                    clip_indices = self.clip_indices.numpy()[:clip_count]
                    centreline_positions = self.centreline_avg_positions.numpy()
                    for i in range(self.max_clips):
                        # Minor hack: for some reason, rendering warp meshes breaks registering instances,
                        # so all instances registered after the warp mesh is rendered for the first time are broken.
                        # To get around this, always render the instance (so they're registered from the start)
                        if i < clip_count:
                            idx = clip_indices[i]
                            pos = centreline_positions[idx]
                            self.renderer.render_sphere(
                                name=f"clip_{i}",
                                pos=[pos[0], pos[1], pos[2]],
                                rot=[0.0, 0.0, 0.0, 1.0],
                                color=[1.0, 0.2, 0.2], 
                                radius=0.018
                            )
                        else:
                            self.renderer.render_sphere(
                                name=f"clip_{i}",
                                pos=[0.0, 0.0, 0.0],
                                rot=[0.0, 0.0, 0.0, 1.0],
                                color=[1.0, 0.2, 0.2], 
                                radius=0.018
                            )

                with wp.ScopedTimer("rendering meshes"):
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
                                    self.integrator.dev_pos_current_buffer,
                                    self.radius_cutting,
                                    tet_count
                                ],
                                device=wp.get_device()
                            )

                            print("Cutting")
                            wp.launch(
                                cut_centrelines_near_haptic,
                                dim=self.centreline_points.shape[0],
                                inputs=[
                                    self.centreline_points,
                                    self.centreline_avg_positions,
                                    self.integrator.dev_pos_current_buffer,
                                    self.centreline_cut_flags,
                                    self.radius_cutting
                                ],
                                device=wp.get_device()
                            )

                        # Handle heating
                        if self.heating_active:
                            heating_active_process(self)

                        # Emit new bleed particles from cut centrelines
                        wp.launch(
                            emit_bleed_particles,
                            dim=self.centreline_points.shape[0],
                            inputs=[
                                self.centreline_avg_positions,
                                self.centreline_cut_flags,
                                self.fluid_positions,
                                self.fluid_velocities,
                                self.fluid_active,
                                self.bleed_next_id,
                                self.fluid_particle_count,
                                self.sim_time,
                                self.frame_dt
                            ],
                            device=wp.get_device()
                        )

                
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
                            diffuse_maps = getattr(self, f"{mesh_name}_diffuse_maps", None)
                            normal_maps = getattr(self, f"{mesh_name}_normal_maps", None)
                            specular_maps = getattr(self, f"{mesh_name}_specular_maps", None)


                            self.renderer.render_mesh_warp_range(
                                name=f"{mesh_name}_mesh",
                                points=self.state_0.particle_q,
                                indices=tet_surface_indices,
                                texture_coords=self.uvs_wp,
                                diffuse_maps=diffuse_maps,
                                normal_maps=normal_maps,
                                specular_maps=specular_maps,
                                colors=self.vertex_colors,
                                index_start=0,
                                index_count=num_triangles,
                                update_topology=True
                            )


                if not hasattr(self, "jaw_angle"):
                    self.jaw_angle = 0.0
                
                target_angle = 0.0 if self.grasping_active else 0.5
                self.jaw_angle = wp.lerp(self.jaw_angle, target_angle, self.frame_dt * 30.0)

                # Debug jaw collider render
                with wp.ScopedTimer("rendering instruments"):
                    for i, collider_info in enumerate(self.jaw_colliders):
                        world_pos = collider_info.get('world_position')
                        if world_pos is not None:
                            pos = [world_pos[0], world_pos[1], world_pos[2]]
                        else:
                            collider_transform = self.state_0.body_q.numpy()[collider_info['body_id']]
                            pos = [collider_transform[0], collider_transform[1], collider_transform[2]]

                        world_rot = collider_info.get('world_rotation')
                        if world_rot is not None:
                            rot = [world_rot[0], world_rot[1], world_rot[2], world_rot[3]]
                        else:
                            rot = [0.0, 0.0, 0.0, 1.0]
                        
                        # Render debug sphere for jaw collider
                        self.renderer.render_sphere(
                            name=f"debug_jaw_collider_{i}",
                            pos=pos,
                            rot=rot,
                            color=[0.0, 1.0, 0.0], 
                            radius=collider_info['radius']
                                )

                    if hasattr(self, 'instruments'):
                        for instrument_idx, instrument in enumerate(self.instruments):
                            if not instrument['visible']:
                                continue
                            
                            # Update instrument position and rotation to follow haptic device
                            if instrument_idx == 0:
                                haptic_pos = [
                                    self.haptic_pos_right[0] * 0.01, 
                                    self.haptic_pos_right[1] * 0.01, 
                                    self.haptic_pos_right[2] * 0.01
                                ]
                                
                                # Flip rotation so it's pointing away from the device
                                flip_rotation = axis_angle_to_quat([0.0, 1.0, 0.0], np.pi)  # 180 degrees around Y
                                
                                haptic_quat = np.array(self.haptic_rot_right)
                                flip_quat = np.array(flip_rotation)
                                
                                combined_rotation = multiply_quaternions(haptic_quat, flip_quat)

                                # Extract forward direction from rotation quaternion
                                x, y, z, w = combined_rotation
                                forward_x = 2.0 * (x*z + w*y)
                                forward_y = 2.0 * (y*z - w*x)
                                forward_z = 2.0 * (z*z + w*w) - 1.0
                                
                                # Offset distance
                                offset_distance = 0.5
                                
                                offset_pos = [
                                    haptic_pos[0] - forward_x * offset_distance,
                                    haptic_pos[1] - forward_y * offset_distance,
                                    haptic_pos[2] - forward_z * offset_distance
                                ]

                                self.update_instrument_transform(
                                    instrument_idx, 
                                    position=offset_pos,
                                    rotation=combined_rotation
                                )
                                
                                
                                # Find and animate jaw pieces
                                piece_names = self.get_piece_names(instrument_idx)
                                for piece_name in piece_names:
                                    if 'jaw' in piece_name.lower() or 'grasp' in piece_name.lower():

                                        actual_angle = self.jaw_angle if 'left' in piece_name.lower() else -self.jaw_angle

                                        # Apply rotation around axis
                                        jaw_rot = axis_angle_to_quat([1.0, 0.0, 0.0], actual_angle)

                                        # Convert quaternion to 4x4 transformation matrix
                                        jaw_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), jaw_rot)
                                        jaw_matrix = wp.transform_to_matrix(jaw_transform)

                                        self.update_piece_transform(
                                            instrument_idx,
                                            piece_name,
                                            transform_matrix=jaw_matrix
                                        )
                                
                            
                            # Render each piece of the instrument
                            for piece_idx, piece in enumerate(instrument['pieces']):
                                if not piece['visible']:
                                    continue
                                
                                self.renderer.render_mesh_warp(
                                    name=f"instrument_{instrument_idx}_piece_{piece_idx}_{piece['name']}",
                                    points=piece['vertices'],
                                    indices=piece['indices'],
                                    pos=(0.0, 0.0, 0.0),
                                    rot=(0.0, 0.0, 0.0, 1.0),
                                    scale=(1.0, 1.0, 1.0),
                                    basic_color=(0.7, 0.7, 0.8),
                                    update_topology=False,
                                    smooth_shading=True,
                                    visible=True
                                )

                # Update all bleed particles
                '''
                wp.launch(
                    update_bleed_particles,
                    dim=self.max_bleed_particles,
                    inputs=[
                        self.bleed_positions,
                        self.bleed_velocities,
                        self.bleed_lifetimes,
                        self.bleed_active,
                        self.max_bleed_particles,
                        self.frame_dt
                    ],
                    device=wp.get_device()
                )
                '''

                # Generate bleeding mesh using marching cubes
                with wp.ScopedTimer("rendering fluids"):
                    fluid_mesh_data = fluids.generate_fluid_mesh(self)
                    if fluid_mesh_data is not None and fluid_mesh_data['triangle_count'] > 0:
                        source_vertices = fluid_mesh_data['vertices']
                        vertex_count = source_vertices.shape[0]
                        device = wp.get_device()

                        if self.fluid_mesh_vertices_world is None or self.fluid_mesh_vertices_world.shape[0] != vertex_count:
                            self.fluid_mesh_vertices_world = wp.zeros(vertex_count, dtype=wp.vec3f, device=device)

                        wp.launch(
                            transform_mesh_vertices,
                            dim=vertex_count,
                            inputs=[
                                source_vertices,
                                fluid_mesh_data['origin'],
                                fluid_mesh_data['spacing'],
                                self.fluid_mesh_vertices_world
                            ],
                            device=device
                        )

                        indices = fluid_mesh_data['indices']
                        index_count = indices.shape[0]
                        topology_changed = (
                            self.fluid_mesh_shape_id is None
                            or self.fluid_mesh_last_vertex_count != vertex_count
                            or self.fluid_mesh_last_index_count != index_count
                        )

                        shape_id = self.renderer.render_mesh_warp(
                            name="fluid_mesh",
                            points=self.fluid_mesh_vertices_world,
                            indices=indices,
                            pos=(0.0, 0.0, 0.0),
                            rot=(0.0, 0.0, 0.0, 1.0),
                            scale=(1.0, 1.0, 1.0),
                            basic_color=(0.1, 0.3, 0.8),  # Blue fluid color
                            update_topology=topology_changed,
                            smooth_shading=True,
                            visible=True
                        )

                        if shape_id is not None:
                            self.fluid_mesh_shape_id = shape_id
                        self.fluid_mesh_indices_current = indices
                        self.fluid_mesh_last_vertex_count = vertex_count
                        self.fluid_mesh_last_index_count = index_count
                    elif (
                        self.fluid_mesh_shape_id is not None
                        and self.fluid_mesh_vertices_world is not None
                        and self.fluid_mesh_indices_current is not None
                    ):
                        # Keep existing geometry but hide it
                        self.renderer.render_mesh_warp(
                            name="fluid_mesh",
                            points=self.fluid_mesh_vertices_world,
                            indices=self.fluid_mesh_indices_current,
                            pos=(0.0, 0.0, 0.0),
                            rot=(0.0, 0.0, 0.0, 1.0),
                            scale=(1.0, 1.0, 1.0),
                            basic_color=(0.1, 0.3, 0.8),
                            update_topology=False,
                            smooth_shading=True,
                            visible=False
                        )
                    else:
                        # Lazily create a tiny placeholder mesh for initial registration
                        if self._fluid_empty_vertices is None or self._fluid_empty_indices is None:
                            device = wp.get_device()
                            self._fluid_empty_vertices = wp.zeros(1, dtype=wp.vec3f, device=device)
                            self._fluid_empty_indices = wp.zeros(3, dtype=wp.int32, device=device)

                        shape_id = self.renderer.render_mesh_warp(
                            name="fluid_mesh",
                            points=self._fluid_empty_vertices,
                            indices=self._fluid_empty_indices,
                            pos=(0.0, 0.0, 0.0),
                            rot=(0.0, 0.0, 0.0, 1.0),
                            scale=(1.0, 1.0, 1.0),
                            basic_color=(0.1, 0.3, 0.8),
                            update_topology=self.fluid_mesh_shape_id is None,
                            smooth_shading=True,
                            visible=False
                        )

                        if self.fluid_mesh_shape_id is None and shape_id is not None:
                            self.fluid_mesh_shape_id = shape_id
                    
                    # Render individual bleed particles (debug)
                    '''
                    bleed_pos = self.bleed_positions.numpy()
                    bleed_active = self.bleed_active.numpy()
                    for i in range(self.max_bleed_particles):
                        if bleed_active[i]:
                            self.renderer.render_sphere(
                                name=f"bleed_{i}",
                                pos=bleed_pos[i],
                                rot=[0.0, 0.0, 0.0, 1.0],
                                color=[0.8, 0.0, 0.0],
                                radius=0.008
                            )
                        else:
                            self.renderer.render_sphere(
                                name=f"bleed_{i}",
                                pos=[0.0, 0.0, 0.0],
                                rot=[0.0, 0.0, 0.0, 1.0],
                                color=[0.8, 0.0, 0.0],
                                radius=0.008
                            )
                    '''   

                wp.copy(self.integrator.dev_pos_prev_buffer, self.integrator.dev_pos_target_buffer)
                self.renderer.end_frame()
            else:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()


#endregion
    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        haptic_pos = wp.vec3(position[0], position[1] + 100.0, position[2] - 400.0)  # Offset to avoid collision with ground
        self.haptic_pos_right = [haptic_pos[0], haptic_pos[1], haptic_pos[2]]
        wp.copy(self.integrator.dev_pos_target_buffer, wp.array([haptic_pos], dtype=wp.vec3, device=wp.get_device()))

    def update_haptic_rotation(self, rotation):
        """Update the haptic device rotation in the simulation."""
        self.haptic_rot_right = rotation.copy()

    def is_running(self):
        """Check if the simulation should continue running."""
        return self.renderer.is_running() if self.use_opengl else True
    
    def save(self):
        """Save the simulation results."""
        if self.renderer:
            self.renderer.save()


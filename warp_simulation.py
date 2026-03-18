import sys
import KRNSolver
from centrelines import CentrelinePointInfo, ClampConstraint, attach_clip_to_nearest_centreline, check_centreline_leaks, compute_centreline_positions, cut_centrelines_near_haptic, emit_bleed_particles, update_bleed_particles, update_centreline_leaks
from connectivity import generate_connectivity, recompute_connectivity, setup_connectivity
from grasping import grasp_end, grasp_start, grasp_process
from heating import heating_active_process, heating_conduction_process, heating_end, heating_start, paint_vertices_near_haptic_proxy, set_paint_strength
from stretching import stretching_breaking_process
from surface_reconstruction import extract_surface_triangles_bucketed
from simulation_systems import BoundsCollisionSystem, CustomCollisionSystem, DistanceConstraintSystem, ExternalSphereCollisionSystem, TrianglePointConstraintSystem, VolumeConstraintSystem
import warp as wp
import newton
from pxr import Usd, UsdGeom

import numpy as np
import math

from PBDSolver import PBDSolver
from render_surgsim_opengl import SurgSimRendererOpenGL
import fluids
import instruments

from mesh_loader import Tetrahedron, load_background_mesh, load_mesh_and_build_model, parse_centreline_file
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
def interpolate_haptic_position(
    haptic_pos_src: wp.array(dtype=wp.vec3f),
    haptic_pos_dst: wp.array(dtype=wp.vec3f),
    haptic_pos_result: wp.array(dtype=wp.vec3f),
    factor: float,
):
    tid = wp.tid()
    if tid >= 1:
        return

    pos_src = haptic_pos_src[0]
    pos_dst = haptic_pos_dst[0]

    result = wp.lerp(pos_src, pos_dst, factor)
    haptic_pos_result[0] = result

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
def fill_float32_3d(arr: wp.array(dtype=wp.float32, ndim=3), value: float):
    i, j, k = wp.tid()
    if i < arr.shape[0] and j < arr.shape[1] and k < arr.shape[2]:
        arr[i, j, k] = value

@wp.kernel
def reverse_triangle_winding(
    indices: wp.array(dtype=wp.int32),
    triangle_count: int
):
    tid = wp.tid()
    if tid >= triangle_count:
        return
    
    base_idx = tid * 3
    temp = indices[base_idx + 1]
    indices[base_idx + 1] = indices[base_idx + 2]
    indices[base_idx + 2] = temp

@wp.kernel
def compute_aabb_from_particles(
    positions: wp.array(dtype=wp.vec3f),
    active: wp.array(dtype=wp.int32),
    num_particles: int,
    aabb_min: wp.array(dtype=wp.float32),  # [3] - separate x,y,z components required for atomic_min/max
    aabb_max: wp.array(dtype=wp.float32)   # [3] - ^ as above
):
    tid = wp.tid()
    if tid >= num_particles:
        return
    
    if active[tid] == 0:
        return
    
    pos = positions[tid]
    
    # Atomic min/max operations on individual components
    wp.atomic_min(aabb_min, 0, pos[0])
    wp.atomic_min(aabb_min, 1, pos[1])
    wp.atomic_min(aabb_min, 2, pos[2])
    
    wp.atomic_max(aabb_max, 0, pos[0])
    wp.atomic_max(aabb_max, 1, pos[1])
    wp.atomic_max(aabb_max, 2, pos[2])

@wp.kernel
def compute_sdf_field(
    field: wp.array(dtype=wp.float32, ndim=3),
    field_dims: wp.vec3i,
    field_origin: wp.vec3f,
    field_spacing: wp.float32,
    particle_positions: wp.array(dtype=wp.vec3f),
    particle_active: wp.array(dtype=wp.int32),
    num_particles: int,
    particle_radius: wp.float32
):
    i, j, k = wp.tid()
    
    if i >= field_dims[0] or j >= field_dims[1] or k >= field_dims[2]:
        return
    
    # Convert grid coordinates to world position
    world_pos = field_origin + wp.vec3f(
        wp.float32(i) * field_spacing,
        wp.float32(j) * field_spacing,
        wp.float32(k) * field_spacing
    )
    
    # Find minimum distance to any active particle
    min_distance = float(1e6)
    
    for p in range(num_particles):
        if particle_active[p] == 0:
            continue
            
        particle_pos = particle_positions[p]
        dist = wp.length(world_pos - particle_pos)
        
        # SDF of sphere: distance to surface
        sdf_dist = dist - particle_radius
        
        if sdf_dist < min_distance:
            min_distance = sdf_dist
    
    # Store SDF
    field[i, j, k] = -min_distance

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
    def __init__(self, stage_path="output.usd", num_frames=300, use_opengl=True, viewer="gl", enable_textures=True):
        self.sim_substeps = 16
        self.num_frames = num_frames
        self.fps = 120
        self.enable_textures = enable_textures  # Global toggle for texture loading

        self.frame_dt = 1.0 / self.fps
        self.substep_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_constraint_iterations = 1

        self.haptic_pos_right = None  # Haptic device position in simulation space
        self.haptic_rot_right = [0.0, 0.0, 0.0, 1.0]  # Haptic device rotation as quaternion

        self.radius_collision = 0.1
        self.radius_heating = 0.2
        self.radius_clipping = 0.1
        self.radius_cutting = 0.2
        self.radius_grasping = 0.1

        self.particle_mass = 0.1

        self.cutting_active = False
        self.heating_active = False
        self.grasping_active = False
        self.clipping = False

        print(sys.version)
        print(f"Initializing WarpSim with {self.sim_substeps} substeps at {self.fps} FPS")
        print(f"Frame time: {self.frame_dt:.4f}s, Substep time: {self.substep_dt:.4f}s")

        self.jaw_colliders = []
        self.jaw_collider_offsets = {}

        # Initialize model
        self._build_model()
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec4f, device=wp.get_device())
        
        # Connectivity setup (from connectivity module)
        setup_connectivity(self)

        self.max_clips = 64
        self.clip_attached = wp.zeros(self.centreline_points.shape[0], dtype=wp.int32, device=wp.get_device())
        self.clip_indices = wp.zeros(self.max_clips, dtype=wp.int32, device=wp.get_device())
        self.clip_count = wp.zeros(1, dtype=wp.int32, device=wp.get_device())

        self.centreline_cut_flags = wp.zeros(self.centreline_points.shape[0], dtype=wp.int32, device=wp.get_device())

        # Bleeding
        self.max_bleed_particles = 4096 # Max number of particles used for bleeding
        self.bleed_positions = wp.zeros(self.max_bleed_particles, dtype=wp.vec3f, device=wp.get_device())
        self.bleed_velocities = wp.zeros(self.max_bleed_particles, dtype=wp.vec3f, device=wp.get_device())
        self.bleed_lifetimes = wp.zeros(self.max_bleed_particles, dtype=wp.float32, device=wp.get_device())
        self.bleed_active = wp.zeros(self.max_bleed_particles, dtype=wp.int32, device=wp.get_device())
        self.bleed_next_id = wp.zeros(1, dtype=wp.int32, device=wp.get_device())

        # Bleed marching cubes
        self.bleeding_field_resolution = 96  # Max grid resolution per axis
        self.bleeding_field_margin = 0.01    # Margin around AABB
        self.bleeding_particle_sdf_radius = 0.007  # Particle radius for SDF
        
        self.bleeding_field_aabb_min = wp.zeros(3, dtype=wp.float32, device=wp.get_device())
        self.bleeding_field_aabb_max = wp.zeros(3, dtype=wp.float32, device=wp.get_device())
        self.bleeding_scalar_field = None
        self.bleeding_field_dims = wp.vec3i(0, 0, 0)
        self.bleeding_field_origin = wp.vec3f(0.0, 0.0, 0.0)
        self.bleeding_field_spacing = 0.0

        self.bleeding_marching_cubes = None
        self.bleeding_mesh_vertices = None
        self.bleeding_mesh_indices = None
        self.bleeding_mesh_triangle_count = 0
        self.bleeding_isosurface_threshold = 0.0

        # Grasp setup
        self.grasp_capacity = 1024
        self.grasped_particles_buffer = wp.zeros(self.grasp_capacity, dtype=wp.int32, device=wp.get_device())
        self.grasped_particles_counter = wp.zeros(1, dtype=wp.int32, device=wp.get_device())
        self.grasp_offsets_buffer = wp.zeros(self.grasp_capacity, dtype=wp.vec3f, device=wp.get_device())
        self.grasp_stiffness = 1.0

        # Fluids setup (from fluids module)
        fluids.setup_fluids_data(self)
        fluids.setup_fluids_rendering(self)

        # Initialize simulation components
        self._setup_simulation()

        # Initialize rendering
        self._setup_renderer(stage_path, use_opengl, viewer)

        # Grasp setup
        self.grasp_capacity = 1024
        self.grasped_particles_buffer = wp.zeros(self.grasp_capacity, dtype=wp.int32, device=wp.get_device())
        self.grasped_particles_counter = wp.zeros(1, dtype=wp.int32, device=wp.get_device())

        # Heating setup
        self.paint_color_buffer = wp.array([wp.vec4(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=wp.get_device())
        self.paint_strength_buffer = wp.array([0.0], dtype=wp.float32, device=wp.get_device()) 
        set_paint_strength(self, 0.0)
    
        #self._load_background_mesh()
        self.background_mesh = None
        self.background_tri_indices = None

        '''
        # Load textures
        if self.use_opengl and self.renderer:
            
            self.background_diffuse = [self.renderer.load_texture("textures/cavity_diffuse.tga")]
            self.background_normal = [self.renderer.load_texture("textures/cavity_normals.tga")]
            self.background_spec = [self.renderer.load_texture("textures/cavity_spec.png")]


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
        '''

        if hasattr(self.renderer, 'renderer'):
            self.renderer.renderer.register_key_press(self._on_key_press)
            self.renderer.renderer.register_key_release(self._on_key_release)

        # Pre-load textures into memory to avoid disk I/O every frame
        self._load_textures()

        # Setup CUDA graph if available
        self._setup_cuda_graph()

        generate_connectivity(self)

#endregion

    def debug_instrument_transforms(self, instrument_id):
        instrument = self.instruments[instrument_id]
        print(f"\n--- Instrument {instrument_id} Transform Debug ---")
        
        for i, piece in enumerate(instrument['pieces']):
            print(f"Piece {i}: {piece['name']}")
            print(f"  USD Local Transform: {piece['usd_local_transform']}")
            print(f"  Runtime Local Transform: {piece['runtime_local_transform']}")
            print(f"  World Transform: {piece['world_transform_matrix']}")
            print(f"  Sample original vertex: {piece['original_vertices'].numpy()[0]}")
            print(f"  Sample transformed vertex: {piece['vertices'].numpy()[0]}")
            print("---")

    def _load_instrument_from_usd(self, usd_path, builder, name="instrument"):
        """Load surgical instrument mesh from USD file as a hierarchical instrument with separate pieces"""
        import numpy as np
        
        # Open USD stage
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"Failed to load USD file: {usd_path}")
            return None
        
        scale = 0.02
        mesh_pieces = []
        
        def collect_mesh_hierarchy(prim, parent_transform=None, parent_piece_index=None):
            """Recursively collect all mesh primitives with their hierarchy and transforms"""
            # Get local transform
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)
                local_matrix = xformable.GetLocalTransformation()
                print(f"  Local transform for {prim.GetPath()}:")
                print(f"    Raw USD matrix (column-major):\n{local_matrix}")
            else:
                local_matrix = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            
            # Convert to numpy array and ensure it's 4x4
            local_transform = np.array(local_matrix, dtype=np.float64)
            if local_transform.shape != (4, 4):
                local_transform = np.eye(4, dtype=np.float64)
            
            # USD matrices are column-major, transpose to row-major
            local_transform = local_transform.T
            
            print(f"    Corrected transform (row-major):\n{local_transform}")
            print(f"    Translation: [{local_transform[0,3]:.3f}, {local_transform[1,3]:.3f}, {local_transform[2,3]:.3f}]")
            
            # Compute the world transform for the current primitive
            if parent_transform is not None:
                world_transform = np.dot(parent_transform, local_transform)
            else:
                world_transform = local_transform.copy()
            
            current_piece_index = parent_piece_index  # Start with parent piece index
            
            # If this is a mesh, create a piece for it
            if prim.IsA(UsdGeom.Mesh):
                current_piece_index = len(mesh_pieces)
                
                # Get mesh geometry
                usd_geom = UsdGeom.Mesh(prim)
                points_attr = usd_geom.GetPointsAttr()
                face_indices_attr = usd_geom.GetFaceVertexIndicesAttr()
                face_counts_attr = usd_geom.GetFaceVertexCountsAttr()
                
                if not (points_attr and face_indices_attr and face_counts_attr):
                    print(f"  Warning: Mesh {prim.GetPath()} missing required attributes, skipping")
                    # Continue with children using parent's transform and piece index
                    for child in prim.GetChildren():
                        collect_mesh_hierarchy(child, parent_transform, parent_piece_index)
                    return
                    
                mesh_points = np.array(points_attr.Get(), dtype=np.float64)
                mesh_face_vertex_indices = np.array(face_indices_attr.Get())
                mesh_face_vertex_counts = np.array(face_counts_attr.Get())
                
                if len(mesh_points) == 0 or len(mesh_face_vertex_indices) == 0:
                    print(f"  Warning: Empty mesh {prim.GetPath()}, skipping")
                    # Continue with children using parent's transform and piece index
                    for child in prim.GetChildren():
                        collect_mesh_hierarchy(child, parent_transform, parent_piece_index)
                    return
                
                print(f"  Processing mesh: {prim.GetPath()}")
                print(f"  Points: {len(mesh_points)}")
                print(f"  World transform translation: [{world_transform[0,3]:.3f}, {world_transform[1,3]:.3f}, {world_transform[2,3]:.3f}]")
                
                original_vertices = mesh_points * scale
                initial_transformed_vertices = original_vertices
                
                print(f"  Sample original vertex: {original_vertices[0] if len(original_vertices) > 0 else 'N/A'}")
                
                # Triangulate faces
                triangulated_indices = []
                face_start = 0
                
                for face_vertex_count in mesh_face_vertex_counts:
                    if face_vertex_count < 3:
                        face_start += face_vertex_count
                        continue
                    elif face_vertex_count == 3:
                        triangulated_indices.extend([
                            mesh_face_vertex_indices[face_start],
                            mesh_face_vertex_indices[face_start + 1], 
                            mesh_face_vertex_indices[face_start + 2]
                        ])
                    else:
                        first_vertex = mesh_face_vertex_indices[face_start]
                        for j in range(1, face_vertex_count - 1):
                            triangulated_indices.extend([
                                first_vertex,
                                mesh_face_vertex_indices[face_start + j],
                                mesh_face_vertex_indices[face_start + j + 1]
                            ])
                    face_start += face_vertex_count
                
                # Convert to Warp format
                vertices = wp.array(np.array(initial_transformed_vertices, dtype=np.float32), dtype=wp.vec3f, device=wp.get_device())
                vertices_original = wp.array(np.array(original_vertices, dtype=np.float32), dtype=wp.vec3f, device=wp.get_device())
                indices = wp.array(np.array(triangulated_indices, dtype=np.int32), dtype=wp.int32, device=wp.get_device())
                
                # Store the complete world transform from USD as the "USD local transform"
                # (original positioning from USD)
                usd_world_transform_wp = wp.mat44f(
                    world_transform[0, 0], world_transform[0, 1], world_transform[0, 2], world_transform[0, 3] * scale,
                    world_transform[1, 0], world_transform[1, 1], world_transform[1, 2], world_transform[1, 3] * scale,
                    world_transform[2, 0], world_transform[2, 1], world_transform[2, 2], world_transform[2, 3] * scale,
                    world_transform[3, 0], world_transform[3, 1], world_transform[3, 2], world_transform[3, 3]
                )
                
                runtime_local_transform = wp.mat44f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                
                piece_data = {
                    'name': str(prim.GetPath()).split('/')[-1],
                    'path': str(prim.GetPath()),
                    'vertices': vertices,
                    'indices': indices,
                    'original_vertices': vertices_original,  # Mesh in original local space
                    'usd_local_transform': usd_world_transform_wp,
                    'runtime_local_transform': runtime_local_transform,
                    'world_transform_matrix': wp.mat44f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                    'parent_index': parent_piece_index,  # parent piece index (None for root)
                    'children_indices': [],
                    'visible': True,
                    'vertex_count': len(vertices),
                    'triangle_count': len(triangulated_indices) // 3
                }

                # Temporary HACK: hide shaft, there is something wrong with handling its local translation
                if piece_data['name'] == "shaft_color_001":
                    piece_data["visible"] = False
                
                mesh_pieces.append(piece_data)
                
                # Add this piece as a child to its parent (if it has one)
                if parent_piece_index is not None:
                    mesh_pieces[parent_piece_index]['children_indices'].append(current_piece_index)
                
                print(f"  Created piece '{piece_data['name']}' with parent_index={parent_piece_index}")
            
            # Recurse to children, passing this node's world transform as the new parent transform
            for child in prim.GetChildren():
                collect_mesh_hierarchy(child, world_transform, current_piece_index)
        
        # Start from root and collect all meshes with hierarchy
        root = stage.GetPseudoRoot()
        for child in root.GetChildren():
            collect_mesh_hierarchy(child)
        
        if not mesh_pieces:
            print("No mesh primitives found in USD file")
            return None
        
        print(f"\nFound {len(mesh_pieces)} mesh pieces:")
        for i, piece in enumerate(mesh_pieces):
            parent_name = mesh_pieces[piece['parent_index']]['name'] if piece['parent_index'] is not None else "None"
            children_names = [mesh_pieces[idx]['name'] for idx in piece['children_indices']]
            print(f"  {i}: '{piece['name']}' (parent: {parent_name}, children: {children_names})")
        
        instrument_data = {
            'name': name,
            'pieces': mesh_pieces,
            'root_pieces': [i for i, piece in enumerate(mesh_pieces) if piece['parent_index'] is None],
            'root_transform_matrix': wp.mat44f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            'visible': True
        }
        
        if not hasattr(self, 'instruments'):
            self.instruments = []
        
        self.instruments.append(instrument_data)
        
        # Update world transforms for all pieces (this will apply the root transform on top of USD transforms)
        self._update_instrument_hierarchy_transforms(len(self.instruments) - 1)
        
        total_vertices = sum(piece['vertex_count'] for piece in mesh_pieces)
        total_triangles = sum(piece['triangle_count'] for piece in mesh_pieces)
        print(f"Successfully loaded instrument '{name}' with {len(mesh_pieces)} pieces, {total_vertices} total vertices and {total_triangles} total triangles")

        self.debug_instrument_transforms(len(self.instruments) - 1)
        #self._setup_jaw_colliders(len(self.instruments) - 1, builder)


        return len(self.instruments) - 1

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

    def _setup_jaw_colliders(self, instrument_id, builder):
        """Setup sphere colliders for jaw pieces"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return
        
        instrument = self.instruments[instrument_id]
        
        # Find jaw pieces and create colliders for them
        for piece_idx, piece in enumerate(instrument['pieces']):
            piece_name = piece['name'].lower()
            
            if 'jaw' in piece_name or 'grasp' in piece_name:
                # Create a collision body for this jaw
                jaw_body_id = builder.add_body(
                    xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
                    mass=0.0,  # Kinematic body
                    armature=0.0
                )
                
                # Add sphere shape to the body
                collider_radius = 0.015
                builder.add_shape_sphere(
                    body=jaw_body_id,
                    xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
                    radius=collider_radius,
                    cfg=newton.ModelBuilder.ShapeConfig(
                        density=10
                    )
                )
                
                # Store the collider info
                collider_info = {
                    'body_id': jaw_body_id,
                    'piece_index': piece_idx,
                    'piece_name': piece['name'],
                    'radius': collider_radius,
                    'instrument_id': instrument_id
                }
                
                self.jaw_colliders.append(collider_info)
                
                # Define hardcoded offsets for jaw tips
                # These offsets are in the jaw piece's local space
                if 'left' in piece_name:
                    self.jaw_collider_offsets[f"{instrument_id}_{piece_idx}"] = wp.vec3f(0.0, 0.0, 0.08)
                elif 'right' in piece_name:
                    self.jaw_collider_offsets[f"{instrument_id}_{piece_idx}"] = wp.vec3f(0.0, 0.0, 0.08)
                else:
                    self.jaw_collider_offsets[f"{instrument_id}_{piece_idx}"] = wp.vec3f(0.0, 0.0, 0.08)
                
                print(f"Created jaw collider for piece '{piece['name']}' with body ID {jaw_body_id}")

    @wp.kernel
    def update_jaw_collider_transform(
        body_positions: wp.array(dtype=wp.transform),
        body_velocities: wp.array(dtype=wp.spatial_vector),
        body_id: int,
        world_position: wp.vec3f,
        world_rotation: wp.quat
    ):
        """Update jaw collider transform to follow jaw piece"""
        if wp.tid() == 0:
            # Set the body transform
            new_transform = wp.transform(world_position, world_rotation)
            body_positions[body_id] = new_transform
            
            # Zero out velocity for kinematic body
            body_velocities[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _compute_jaw_collider_world_transform(self, instrument_id, piece_index):
        """Compute world transform for a jaw collider based on its piece transform"""
        if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
            return None, None
        
        instrument = self.instruments[instrument_id]
        if piece_index >= len(instrument['pieces']):
            return None, None
        
        piece = instrument['pieces'][piece_index]
        
        # Get the offset for this jaw collider
        offset_key = f"{instrument_id}_{piece_index}"
        local_offset = self.jaw_collider_offsets.get(offset_key, wp.vec3f(0.0, 0.0, 0.0))
        
        # Transform the offset by the piece's world transform matrix
        offset_homo = wp.vec4f(local_offset[0], local_offset[1], local_offset[2], 1.0)
        world_offset_homo = piece['world_transform_matrix'] * offset_homo
        world_position = wp.vec3f(world_offset_homo[0], world_offset_homo[1], world_offset_homo[2])
        
        # Extract rotation from the world transform matrix
        transform_matrix = piece['world_transform_matrix']
        
        # Create rotation quaternion from the 3x3 rotation part of the matrix
        rotation_matrix = np.array([
            [float(transform_matrix[0, 0]), float(transform_matrix[0, 1]), float(transform_matrix[0, 2])],
            [float(transform_matrix[1, 0]), float(transform_matrix[1, 1]), float(transform_matrix[1, 2])],
            [float(transform_matrix[2, 0]), float(transform_matrix[2, 1]), float(transform_matrix[2, 2])]
        ])
        
        quat_components = self._matrix_to_quaternion(rotation_matrix)
        world_rotation = wp.quat(quat_components[0], quat_components[1], quat_components[2], quat_components[3])
        
        return world_position, world_rotation

    def _update_jaw_colliders(self):
        """Update all jaw collider positions to follow their respective jaw pieces"""
        for collider_info in self.jaw_colliders:
            instrument_id = collider_info['instrument_id']
            piece_index = collider_info['piece_index']
            body_id = collider_info['body_id']
            
            # Compute world transform for this jaw collider
            world_pos, world_rot = self._compute_jaw_collider_world_transform(instrument_id, piece_index)
            
            if world_pos is not None and world_rot is not None:
                # Update the collider body transform
                wp.launch(
                    self.update_jaw_collider_transform,
                    dim=1,
                    inputs=[
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        body_id,
                        world_pos,
                        world_rot
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

        # Import instruments
        self.instrument_id = self._load_instrument_from_usd("meshes/pgrasp.usdc", builder, "pgrasp")
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

    def _load_textures(self):
        """Pre-load all textures into memory to avoid disk I/O every frame."""
        from PIL import Image
        import os

        self.texture_cache = {}

        if not self.enable_textures:
            print("Textures disabled (enable_textures=False)")
            return

        # Load background mesh textures
        background_textures = {
            "diffuse": "textures/cavity_diffuse.tga",
            "normal": "textures/cavity_normals.tga",
            "specular": "textures/cavity_spec.png"
        }

        for tex_type, texture_path in background_textures.items():
            if os.path.exists(texture_path):
                try:
                    img = Image.open(texture_path).convert("RGBA")
                    self.texture_cache[f"background_{tex_type}"] = np.array(img)
                    print(f"Pre-loaded background texture: {texture_path}")
                except Exception as e:
                    print(f"Failed to load background texture {texture_path}: {e}")

        # Load textures for each mesh
        for mesh_name in self.mesh_ranges.keys():
            texture_path = f"textures/{mesh_name}/diffuse-base.png"

            if os.path.exists(texture_path):
                try:
                    img = Image.open(texture_path).convert("RGBA")
                    self.texture_cache[f"{mesh_name}_diffuse"] = np.array(img)
                    print(f"Pre-loaded texture: {texture_path}")
                except Exception as e:
                    print(f"Failed to load texture {texture_path}: {e}")
            else:
                print(f"Texture not found: {texture_path}")

        print(f"Loaded {len(self.texture_cache)} textures into memory")

    def _setup_simulation(self):
        """Initialize simulation states and integrator."""
        self.integrator = KRNSolver.KRNSolver(self.model, iterations=self.sim_constraint_iterations)

        # Register simulation systems (plugin architecture)
        self.integrator.register_system(DistanceConstraintSystem(priority=50))
        self.integrator.register_system(VolumeConstraintSystem(priority=60))
        self.integrator.register_system(TrianglePointConstraintSystem(priority=70))
        self.integrator.register_system(ExternalSphereCollisionSystem(priority=80))
        self.integrator.register_system(BoundsCollisionSystem(
            bounds_min=wp.vec3(-2.0, 0.0, -8.0),
            bounds_max=wp.vec3(2.0, 10.0, -3.0),
            priority=100
        ))

        self.integrator.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.integrator.dev_pos_target_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.integrator.dev_pos_prev_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())

        self.rest = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.collide(self.state_0)

        instruments._update_jaw_colliders(self, states=[self.state_0, self.state_1], update_solver=True)

    def _setup_renderer(self, stage_path, use_opengl, viewer="gl"):
        """Initialize the appropriate renderer."""
        self.use_opengl = use_opengl

        if self.use_opengl:
            if viewer == "surgsim":
                self.renderer = SurgSimRendererOpenGL(self.model, "Warp Surgical Simulation", scaling=1.0, near_plane=0.05, far_plane=25)
                self.renderer._camera_pos = [0.2, 1.2, -1.0]
            elif viewer == "rtx":
                self.renderer = newton.viewer.ViewerRTX()
                self.renderer.set_model(self.model)
                self.renderer.set_camera(wp.vec3f(0.2, 1.2, -1.0), 0, -90)
            else:
                self.renderer = newton.viewer.ViewerGL()
                self.renderer.set_model(self.model)
                self.renderer.set_camera(wp.vec3f(0.2, 1.2, -1.0), 0, -90)

        elif stage_path:
            self.renderer = newton.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

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

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Interpolate haptic position across substeps for smoother collision
            factor = float(i) / float(self.sim_substeps)
            wp.launch(
                interpolate_haptic_position,
                dim=1,
                inputs=[
                    self.integrator.dev_pos_prev_buffer,
                    self.integrator.dev_pos_target_buffer,
                    self.integrator.dev_pos_buffer,
                    factor
                ],
                device=wp.get_device()
            )

            # Update haptic device body position
            wp.launch(
                set_body_position,
                dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.integrator.dev_pos_buffer, 0.01, self.substep_dt],
                device=self.state_0.body_q.device,
            )

            # Run collision detection and integration
            #self.contacts = self.model.collide(self.state_0)
            #if self.contacts:
            #    print(f"Contacts detected: {self.contacts.soft_contact_normal}")

            instruments._update_jaw_colliders(self, states=[self.state_0, self.state_1], update_solver=True)

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

        self.renderer.begin_frame(self.sim_time)

        # Render default state
        # self.renderer.log_state(self.state_0)

        with wp.ScopedTimer("render"): 
            if self.haptic_pos_right is not None:
                self.renderer.log_points(
                    "haptic_proxy_sphere",
                    wp.array([[self.haptic_pos_right[0] * 0.01, self.haptic_pos_right[1] * 0.01, self.haptic_pos_right[2] * 0.01]], dtype=wp.vec3f, device=wp.get_device()),
                    wp.array([0.025], dtype=wp.float32, device=wp.get_device()),
                    wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f, device=wp.get_device()),

                )
            # Render background mesh
            if self.background_mesh is not None and self.background_tri_indices is not None:
                background_texture = self.texture_cache.get("background_diffuse", None) if self.enable_textures else None
                use_texture = 1.0 if self.enable_textures else 0.0

                self.renderer.log_mesh(
                    name="background_mesh",
                    points=self.background_mesh,
                    indices=self.background_tri_indices,
                    texture=background_texture,
                    uvs=self.background_uvs,
                    
                    hidden=True
                )

                xforms = wp.array([wp.transform()], dtype=wp.transformf)
                scales = wp.array([1.2, 1.2, 1.2], dtype=wp.vec3)
                colors = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)
                mat = wp.array([wp.vec4(1.0, 0.0, 0.0, use_texture)], dtype=wp.vec4)

                self.renderer.log_instances("background_mesh_instance", "background_mesh", xforms, scales, colors, materials=mat)



            # Centreline update
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
                        self.integrator.dev_pos_buffer,
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

                    print("Cutting")
                    wp.launch(
                        cut_centrelines_near_haptic,
                        dim=self.centreline_points.shape[0],
                        inputs=[
                            self.centreline_points,
                            self.centreline_avg_positions,
                            self.integrator.dev_pos_buffer,
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
                        self.bleed_positions,
                        self.bleed_velocities,
                        self.bleed_lifetimes,
                        self.bleed_active,
                        self.bleed_next_id,
                        self.max_bleed_particles,
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
                    mesh_full_name = f"{mesh_name}_mesh"

                    # Use pre-loaded texture from cache instead of file path
                    texture_data = self.texture_cache.get(f"{mesh_name}_diffuse", None) if self.enable_textures else None
                    use_texture = 1.0 if self.enable_textures else 0.0

                    self.renderer.log_mesh(
                        name=mesh_full_name,
                        points=self.state_0.particle_q,
                        indices=tet_surface_indices.flatten(),
                        uvs=self.uvs_wp,
                        texture=texture_data,
                        hidden=True,
                    )

                    xforms = wp.array([wp.transform()], dtype=wp.transformf)
                    scales = wp.array([1.0, 1.0, 1.0], dtype=wp.vec3)
                    colors = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3)
                    mat_default = wp.array([wp.vec4(0.0, 0.0, 0.0, use_texture)], dtype=wp.vec4)

                    self.renderer.log_instances(f"{mesh_name}_instance", mesh_full_name, xforms, scales, colors, materials=mat_default)
                

            if not hasattr(self, "jaw_angle"):
                self.jaw_angle = 0.0
            
            target_angle = 0.0 if self.grasping_active else 0.5
            self.jaw_angle = wp.lerp(self.jaw_angle, target_angle, self.frame_dt * 30.0)
            #self._update_jaw_colliders()
            
            if hasattr(self, 'instruments'):
                for instrument_idx, instrument in enumerate(self.instruments):
                    if not instrument['visible']:
                        continue
                    
                    # Update instrument position and rotation to follow haptic device
                    if instrument_idx == 0 and self.haptic_pos_right is not None:
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
                        
                        name = f"instrument_{instrument_idx}_piece_{piece_idx}_{piece['name']}"

                        self.renderer.log_mesh(
                            name=name,
                            points=piece['vertices'],
                            indices=piece['indices'],
                            hidden=True
                        )


                        xforms = wp.array([wp.transform()], dtype=wp.transformf)
                        scales = wp.array([1.0, 1.0, 1.0], dtype=wp.vec3)
                        colors = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3)
                        mat_default = wp.array([wp.vec4(0.1, 1.0, 0.0, 0.0)], dtype=wp.vec4)

                        self.renderer.log_instances(f"{name}_instance", name, xforms, scales, colors, materials=mat_default)

            
            # Update all bleed particles
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

            # Generate bleeding mesh using marching cubes
            mesh_data = self.generate_bleeding_mesh()
            if mesh_data is not None and mesh_data['triangle_count'] > 0:
                # Transform vertices from grid space to world space
                vertex_count = mesh_data['vertices'].shape[0]
                transformed_vertices = wp.zeros(vertex_count, dtype=wp.vec3f, device=wp.get_device())
                
                wp.launch(
                    transform_mesh_vertices,
                    dim=vertex_count,
                    inputs=[
                        mesh_data['vertices'],
                        mesh_data['origin'],
                        mesh_data['spacing'],
                        transformed_vertices
                    ],
                    device=wp.get_device()
                )
                
                # Render the bleeding mesh with a blood-like color
                self.renderer.render_mesh_warp(
                    name="bleeding_mesh",
                    points=transformed_vertices,
                    indices=mesh_data['indices'],
                    pos=(0.0, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),
                    scale=(1.0, 1.0, 1.0),
                    basic_color=(0.35, 0.0, 0.05),
                    update_topology=True,
                    smooth_shading=True,
                    visible=True
                )
            else:
                # Render empty mesh when no bleeding
                empty_vertices = wp.zeros(1, dtype=wp.vec3f, device=wp.get_device())
                empty_indices = wp.zeros(3, dtype=wp.int32, device=wp.get_device())
                
                self.renderer.log_mesh(
                    name="bleeding_mesh",
                    points=empty_vertices,
                    indices=empty_indices,
                    #pos=(0.0, 0.0, 0.0),
                    #rot=(0.0, 0.0, 0.0, 1.0),
                    #scale=(1.0, 1.0, 1.0),
                    #basic_color=(0.35, 0.0, 0.05),
                    #update_topology=True,
                    #smooth_shading=True,
                    #visible=False
                )
            
            # Render individual bleed particles (debug)
            
            bleed_pos = self.bleed_positions
            #bleed_active = self.bleed_active.numpy()
            self.renderer.log_points(
                    f"bleed_points",
                    bleed_pos,
                    wp.full(bleed_pos.size, 0.008, dtype=wp.float32, device=wp.get_device()),
                    wp.full(bleed_pos.size, [0.0, 0.0, 0.0], dtype=wp.vec3f, device=wp.get_device()),
                )
                
            
            wp.copy(self.integrator.dev_pos_prev_buffer, self.integrator.dev_pos_buffer)

        self.renderer.end_frame()
        

    def compute_bleeding_field(self):
        """Compute scalar field from bleeding particles for marching cubes."""
        # Count active particles
        active_count = int(np.sum(self.bleed_active.numpy()))
        #print(f"Active bleeding particles: {active_count}")
        if active_count == 0:
            return None
        
        # Reset AABB with large/small values
        wp.copy(self.bleeding_field_aabb_min, wp.array([1e6, 1e6, 1e6], dtype=wp.float32, device=wp.get_device()))
        wp.copy(self.bleeding_field_aabb_max, wp.array([-1e6, -1e6, -1e6], dtype=wp.float32, device=wp.get_device()))
        
        # Compute AABB from active particles
        wp.launch(
            compute_aabb_from_particles,
            dim=self.max_bleed_particles,
            inputs=[
                self.bleed_positions,
                self.bleed_active,
                self.max_bleed_particles,
                self.bleeding_field_aabb_min,
                self.bleeding_field_aabb_max
            ],
            device=wp.get_device()
        )
        
        # AABB values
        aabb_min_array = self.bleeding_field_aabb_min.numpy()
        aabb_max_array = self.bleeding_field_aabb_max.numpy()
        
        print(f"AABB: min={aabb_min_array}, max={aabb_max_array}")
        
        aabb_min = wp.vec3f(aabb_min_array[0], aabb_min_array[1], aabb_min_array[2])
        aabb_max = wp.vec3f(aabb_max_array[0], aabb_max_array[1], aabb_max_array[2])
        
        # Add margin
        margin_vec = wp.vec3f(self.bleeding_field_margin, self.bleeding_field_margin, self.bleeding_field_margin)
        field_min = aabb_min - margin_vec
        field_max = aabb_max + margin_vec
        
        # Compute field dimensions and spacing
        field_size = field_max - field_min
        max_extent = max(field_size[0], field_size[1], field_size[2])
        
        print(f"Field size: {field_size}, max_extent: {max_extent}")
        
        if max_extent <= 0:
            print("Max extent <= 0, returning None")
            return None
            
        self.bleeding_field_spacing = max_extent / float(self.bleeding_field_resolution)
        
        # Calculate actual grid dimensions
        self.bleeding_field_dims = wp.vec3i(
            int(math.ceil(field_size[0] / self.bleeding_field_spacing)) + 1,
            int(math.ceil(field_size[1] / self.bleeding_field_spacing)) + 1,
            int(math.ceil(field_size[2] / self.bleeding_field_spacing)) + 1
        )
        
        print(f"Field dims: {self.bleeding_field_dims}, spacing: {self.bleeding_field_spacing}")
        
        self.bleeding_field_origin = field_min
        
        # Allocate field if needed
        field_shape = (self.bleeding_field_dims[0], self.bleeding_field_dims[1], self.bleeding_field_dims[2])
        if self.bleeding_scalar_field is None or self.bleeding_scalar_field.shape != field_shape:
            self.bleeding_scalar_field = wp.zeros(field_shape, dtype=wp.float32, device=wp.get_device())
            print(f"Allocated new field with shape: {field_shape}")
        else:
            # Clear existing field
            wp.launch(
                fill_float32_3d,
                dim=self.bleeding_scalar_field.shape,
                inputs=[self.bleeding_scalar_field, 0.0],
                device=wp.get_device()
            )
            print("Cleared existing field")
        
        # Compute SDF field
        wp.launch(
            compute_sdf_field,
            dim=self.bleeding_field_dims,
            inputs=[
                self.bleeding_scalar_field,
                self.bleeding_field_dims,
                self.bleeding_field_origin,
                self.bleeding_field_spacing,
                self.bleed_positions,
                self.bleed_active,
                self.max_bleed_particles,
                self.bleeding_particle_sdf_radius
            ],
            device=wp.get_device()
        )
        
        print("SDF field computed")
        
        return {
            'field': self.bleeding_scalar_field,
            'dims': self.bleeding_field_dims,
            'origin': self.bleeding_field_origin,
            'spacing': self.bleeding_field_spacing
        }

    def generate_bleeding_mesh(self):
        """Generate mesh from bleeding particles using marching cubes."""
        field_data = self.compute_bleeding_field()
        if field_data is None:
            self.bleeding_mesh_triangle_count = 0
            #print("No field data")
            return None
        
        field = field_data['field']
        dims = field_data['dims']
        
        print(f"Generating mesh with dims: {dims}")
        
        # Initialize/resize
        if (self.bleeding_marching_cubes is None or 
            self.bleeding_marching_cubes.nx != dims[0] - 1 or
            self.bleeding_marching_cubes.ny != dims[1] - 1 or
            self.bleeding_marching_cubes.nz != dims[2] - 1):
            
            # Estimate maximum vertices and triangles
            total_cubes = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
            max_verts = min(total_cubes * 12, 100000)  # Up to 12 vertices per cube, cap at 100k
            max_tris = min(total_cubes * 5, 200000)   # Up to 5 triangles per cube, cap at 200k
            
            print(f"Creating marching cubes: cubes={total_cubes}, max_verts={max_verts}, max_tris={max_tris}")
            
            if self.bleeding_marching_cubes is None:
                self.bleeding_marching_cubes = wp.MarchingCubes(
                    nx=dims[0],
                    ny=dims[1], 
                    nz=dims[2],
                    max_verts=max_verts,
                    max_tris=max_tris,
                    device=wp.get_device()
                )
                print("Created new marching cubes object")
            else:
                self.bleeding_marching_cubes.resize(
                    nx=dims[0],
                    ny=dims[1],
                    nz=dims[2],
                    max_verts=max_verts,
                    max_tris=max_tris
                )
                print("Resized existing marching cubes object")
        
        print(f"Running marching cubes with threshold: {self.bleeding_isosurface_threshold}")

        # Extract isosurface
        self.bleeding_marching_cubes.surface(field, self.bleeding_isosurface_threshold)
        
        # Get the generated mesh
        self.bleeding_mesh_vertices = self.bleeding_marching_cubes.verts
        self.bleeding_mesh_indices = self.bleeding_marching_cubes.indices
        
        # Count actual triangles generated
        indices_array = self.bleeding_mesh_indices.numpy()
        self.bleeding_mesh_triangle_count = len(indices_array) // 3
        
        print(f"Marching cubes result: {self.bleeding_mesh_triangle_count} triangles, {len(self.bleeding_mesh_vertices.numpy())} vertices")
        
        if self.bleeding_mesh_triangle_count > 0:
            '''
            wp.launch(
                reverse_triangle_winding,
                dim=self.bleeding_mesh_triangle_count,
                inputs=[
                    self.bleeding_mesh_indices,
                    self.bleeding_mesh_triangle_count
                ],
                device=wp.get_device()
            )
            '''

            return {
                'vertices': self.bleeding_mesh_vertices,
                'indices': self.bleeding_mesh_indices,
                'triangle_count': self.bleeding_mesh_triangle_count,
                'origin': field_data['origin'],
                'spacing': field_data['spacing']
            }
        else:
            self.bleeding_mesh_triangle_count = 0
            return None

#endregion
    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        haptic_pos = wp.vec3(position[0], position[1] + 100.0, position[2] - 400.0)  # Offset to avoid collision with ground
        self.haptic_pos_right = [haptic_pos[0], haptic_pos[1], haptic_pos[2]]
        wp.copy(self.integrator.dev_pos_prev_buffer, self.integrator.dev_pos_target_buffer)
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
            self.renderer.close()


import warp as wp
from pxr import Usd, UsdGeom
import newton
import numpy as np


def get_jaw_collider_specs(self, piece):
    """Return collider sphere specifications for a jaw piece."""
    piece_name = piece['name'].lower()

    for key, specs in self.jaw_collider_profiles.items():
        if key == "default":
            continue
        if key in piece_name:
            selected_specs = specs
            break
    else:
        selected_specs = self.jaw_collider_profiles.get("default", [])

    normalized_specs = []
    for raw_spec in selected_specs:
        if isinstance(raw_spec, dict):
            normalized_specs.append(dict(raw_spec))
        elif isinstance(raw_spec, (list, tuple)) and len(raw_spec) == 2:
            offset_value, radius_value = raw_spec
            normalized_specs.append({"offset": offset_value, "radius": radius_value})
        else:
            raise ValueError(f"Invalid jaw collider spec: {raw_spec}")

    return normalized_specs

def setup_jaw_colliders(self, instrument_id, builder):
    """Setup sphere colliders for jaw pieces"""
    if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
        return
    
    instrument = self.instruments[instrument_id]
    
    # Find jaw pieces and create colliders for them
    for piece_idx, piece in enumerate(instrument['pieces']):
        piece_name = piece['name'].lower()
        
        if 'jaw' in piece_name or 'grasp' in piece_name:
            sphere_specs = get_jaw_collider_specs(self, piece)
            if not sphere_specs:
                continue

            for sphere_idx, spec in enumerate(sphere_specs):
                radius = float(spec.get("radius", 0.4))
                offset_values = spec.get("offset", (0.0, 0.0, 0.0))
                offset_vec = wp.vec3f(
                    float(offset_values[0]),
                    float(offset_values[1]),
                    float(offset_values[2]),
                )

                jaw_body_id = builder.add_body(
                    xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
                    mass=0.0,  # Kinematic body
                    armature=0.0
                )

                builder.add_shape_sphere(
                    body=jaw_body_id,
                    xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
                    radius=radius,
                    cfg=newton.ModelBuilder.ShapeConfig(
                        density=10
                    )
                )

                collider_info = {
                    'body_id': jaw_body_id,
                    'piece_index': piece_idx,
                    'piece_name': piece['name'],
                    'radius': radius,
                    'instrument_id': instrument_id,
                    'offset': offset_vec,
                    'sphere_index': sphere_idx,
                    'world_position': (0.0, 0.0, 0.0),
                    'world_rotation': (0.0, 0.0, 0.0, 1.0)
                }

                self.jaw_colliders.append(collider_info)
                print(
                    f"Created jaw collider for piece '{piece['name']}' "
                    f"(sphere {sphere_idx}) with body ID {jaw_body_id}, "
                    f"radius {radius} and offset {offset_vec}"
                )

@wp.kernel
def update_jaw_collider_transform(
    body_positions: wp.array(dtype=wp.transformf),
    body_velocities: wp.array(dtype=wp.spatial_vectorf),
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

def compute_jaw_collider_world_transform(self, instrument_id, piece_index, local_offset):
    """Compute world transform for a jaw collider based on its piece transform"""
    if not hasattr(self, 'instruments') or instrument_id >= len(self.instruments):
        return None, None
    
    instrument = self.instruments[instrument_id]
    if piece_index >= len(instrument['pieces']):
        return None, None
    
    piece = instrument['pieces'][piece_index]
    
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

def _update_jaw_colliders(self, states=None, update_solver=True):
    """Update jaw collider transforms and synchronize solver collision buffers."""
    if not self.jaw_colliders:
        if update_solver and hasattr(self, "integrator") and self.integrator is not None:
            self.integrator.set_external_sphere_colliders([], [])
        return

    if states is None:
        states_to_update = []
    else:
        states_to_update = []
        seen_ids = set()
        for state in states:
            if state is None:
                continue
            state_id = id(state)
            if state_id in seen_ids:
                continue
            seen_ids.add(state_id)
            states_to_update.append(state)

    updated_centers = []
    updated_radii = []

    for collider_info in self.jaw_colliders:
        instrument_id = collider_info['instrument_id']
        piece_index = collider_info['piece_index']
        body_id = collider_info['body_id']
        
        # Compute world transform for this jaw collider
        world_pos, world_rot = compute_jaw_collider_world_transform(
            self,
            instrument_id,
            piece_index,
            collider_info['offset']
        )

        if world_pos is None or world_rot is None:
            continue

        for state in states_to_update:
            wp.launch(
                update_jaw_collider_transform,
                dim=1,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    body_id,
                    world_pos,
                    world_rot
                ],
                device=state.body_q.device,
            )

        center_tuple = (float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
        collider_info['world_position'] = center_tuple
        collider_info['world_rotation'] = (
            float(world_rot[0]),
            float(world_rot[1]),
            float(world_rot[2]),
            float(world_rot[3]),
        )

        updated_centers.append(center_tuple)
        updated_radii.append(float(collider_info['radius']))

    if update_solver and hasattr(self, "integrator") and self.integrator is not None:
        self.integrator.set_external_sphere_colliders(updated_centers, updated_radii)


#region Importing

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


def load_instrument_from_usd(self, usd_path, builder, name="instrument"):
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

    debug_instrument_transforms(self, len(self.instruments) - 1)
    setup_jaw_colliders(self, len(self.instruments) - 1, builder)


    return len(self.instruments) - 1


#endregion
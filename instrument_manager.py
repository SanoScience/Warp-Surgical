"""
Instrument management module.

This module handles loading, transforming, and rendering of surgical instruments
from USD files. It maintains a hierarchical transform system for instrument pieces.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

from config import SimulationConfig
from coordinate_system import CoordinateSystem


def axis_angle_to_quat(axis: List[float], angle: float) -> List[float]:
    """Convert axis-angle representation to quaternion [x, y, z, w]."""
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


def multiply_quaternions(q1: List[float], q2: List[float]) -> List[float]:
    """Multiply two quaternions: q1 * q2. Format: [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
        w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
    ]


def matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    trace = np.trace(matrix)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    return [x, y, z, w]


@wp.kernel
def apply_matrix_transform_to_vertices(
    original_vertices: wp.array(dtype=wp.vec3f),
    transformed_vertices: wp.array(dtype=wp.vec3f),
    transform_matrix: wp.mat44f,
    num_vertices: int
):
    """Apply 4x4 transform matrix to vertices."""
    tid = wp.tid()
    if tid >= num_vertices:
        return

    orig_vert = original_vertices[tid]
    vert_homo = wp.vec4f(orig_vert[0], orig_vert[1], orig_vert[2], 1.0)
    transformed_homo = transform_matrix * vert_homo

    transformed_vertices[tid] = wp.vec3f(
        transformed_homo[0],
        transformed_homo[1],
        transformed_homo[2]
    )


class InstrumentPiece:
    """A single piece (mesh) of an instrument hierarchy."""

    def __init__(
        self,
        name: str,
        path: str,
        vertices: wp.array,
        indices: wp.array,
        original_vertices: wp.array,
        usd_local_transform: wp.mat44f,
        parent_index: Optional[int] = None
    ):
        self.name = name
        self.path = path
        self.vertices = vertices
        self.indices = indices
        self.original_vertices = original_vertices
        self.usd_local_transform = usd_local_transform
        self.runtime_local_transform = wp.mat44f(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        )
        self.world_transform_matrix = wp.mat44f(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        )
        self.parent_index = parent_index
        self.children_indices: List[int] = []
        self.visible = True
        self.vertex_count = len(vertices)
        self.triangle_count = len(indices) // 3


class Instrument:
    """A surgical instrument consisting of multiple mesh pieces."""

    def __init__(self, name: str):
        self.name = name
        self.pieces: List[InstrumentPiece] = []
        self.root_pieces: List[int] = []
        self.root_transform_matrix = wp.mat44f(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        )
        self.visible = True

    def add_piece(self, piece: InstrumentPiece) -> int:
        """Add a piece to the instrument and return its index."""
        index = len(self.pieces)
        self.pieces.append(piece)

        if piece.parent_index is None:
            self.root_pieces.append(index)
        else:
            self.pieces[piece.parent_index].children_indices.append(index)

        return index

    def get_piece_names(self) -> List[str]:
        """Get list of all piece names."""
        return [piece.name for piece in self.pieces]


class InstrumentManager:
    """Manages surgical instruments: loading, transforms, and rendering.

    Handles USD file loading, hierarchical transforms, and provides
    methods for updating instrument positions and animations.
    """

    def __init__(self, config: SimulationConfig = None, coords: CoordinateSystem = None):
        """Initialize the instrument manager.

        Args:
            config: Simulation configuration.
            coords: Coordinate system for transformations.
        """
        self.config = config if config is not None else SimulationConfig()
        self.coords = coords if coords is not None else CoordinateSystem()
        self.instruments: List[Instrument] = []

    def load_from_usd(self, usd_path: str, name: str = "instrument") -> Optional[int]:
        """Load surgical instrument mesh from USD file.

        Args:
            usd_path: Path to the USD file.
            name: Name for the instrument.

        Returns:
            Instrument ID if successful, None otherwise.
        """
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"Failed to load USD file: {usd_path}")
            return None

        scale = self.coords.USD_TO_SIM_SCALE
        instrument = Instrument(name)
        mesh_pieces_data: List[Dict] = []

        def collect_mesh_hierarchy(prim, parent_transform=None, parent_piece_index=None):
            """Recursively collect all mesh primitives with their hierarchy."""
            # Get local transform
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)
                local_matrix = xformable.GetLocalTransformation()
            else:
                local_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

            local_transform = np.array(local_matrix, dtype=np.float64)
            if local_transform.shape != (4, 4):
                local_transform = np.eye(4, dtype=np.float64)

            # USD matrices are column-major, transpose to row-major
            local_transform = local_transform.T

            # Compute the world transform
            if parent_transform is not None:
                world_transform = np.dot(parent_transform, local_transform)
            else:
                world_transform = local_transform.copy()

            current_piece_index = parent_piece_index

            # If this is a mesh, create a piece for it
            if prim.IsA(UsdGeom.Mesh):
                current_piece_index = len(mesh_pieces_data)

                # Get mesh geometry
                usd_geom = UsdGeom.Mesh(prim)
                points_attr = usd_geom.GetPointsAttr()
                face_indices_attr = usd_geom.GetFaceVertexIndicesAttr()
                face_counts_attr = usd_geom.GetFaceVertexCountsAttr()

                if not (points_attr and face_indices_attr and face_counts_attr):
                    for child in prim.GetChildren():
                        collect_mesh_hierarchy(child, parent_transform, parent_piece_index)
                    return

                mesh_points = np.array(points_attr.Get(), dtype=np.float64)
                mesh_face_vertex_indices = np.array(face_indices_attr.Get())
                mesh_face_vertex_counts = np.array(face_counts_attr.Get())

                if len(mesh_points) == 0 or len(mesh_face_vertex_indices) == 0:
                    for child in prim.GetChildren():
                        collect_mesh_hierarchy(child, parent_transform, parent_piece_index)
                    return

                original_vertices = mesh_points * scale

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

                # Convert to Warp arrays
                vertices = wp.array(
                    np.array(original_vertices, dtype=np.float32),
                    dtype=wp.vec3f,
                    device=wp.get_device()
                )
                vertices_original = wp.array(
                    np.array(original_vertices, dtype=np.float32),
                    dtype=wp.vec3f,
                    device=wp.get_device()
                )
                indices = wp.array(
                    np.array(triangulated_indices, dtype=np.int32),
                    dtype=wp.int32,
                    device=wp.get_device()
                )

                # Create USD world transform with scale applied to translation
                usd_world_transform_wp = wp.mat44f(
                    world_transform[0, 0], world_transform[0, 1], world_transform[0, 2], world_transform[0, 3] * scale,
                    world_transform[1, 0], world_transform[1, 1], world_transform[1, 2], world_transform[1, 3] * scale,
                    world_transform[2, 0], world_transform[2, 1], world_transform[2, 2], world_transform[2, 3] * scale,
                    world_transform[3, 0], world_transform[3, 1], world_transform[3, 2], world_transform[3, 3]
                )

                piece_data = {
                    'name': str(prim.GetPath()).split('/')[-1],
                    'path': str(prim.GetPath()),
                    'vertices': vertices,
                    'indices': indices,
                    'original_vertices': vertices_original,
                    'usd_local_transform': usd_world_transform_wp,
                    'parent_index': parent_piece_index
                }

                # Hide problematic pieces
                if piece_data['name'] == "shaft_color_001":
                    piece_data["visible"] = False
                else:
                    piece_data["visible"] = True

                mesh_pieces_data.append(piece_data)

            # Recurse to children
            for child in prim.GetChildren():
                collect_mesh_hierarchy(child, world_transform, current_piece_index)

        # Start from root
        root = stage.GetPseudoRoot()
        for child in root.GetChildren():
            collect_mesh_hierarchy(child)

        if not mesh_pieces_data:
            print("No mesh primitives found in USD file")
            return None

        # Create InstrumentPiece objects
        for piece_data in mesh_pieces_data:
            piece = InstrumentPiece(
                name=piece_data['name'],
                path=piece_data['path'],
                vertices=piece_data['vertices'],
                indices=piece_data['indices'],
                original_vertices=piece_data['original_vertices'],
                usd_local_transform=piece_data['usd_local_transform'],
                parent_index=piece_data['parent_index']
            )
            piece.visible = piece_data.get('visible', True)
            instrument.add_piece(piece)

        self.instruments.append(instrument)
        instrument_id = len(self.instruments) - 1

        # Update world transforms
        self._update_hierarchy_transforms(instrument_id)

        total_vertices = sum(p.vertex_count for p in instrument.pieces)
        total_triangles = sum(p.triangle_count for p in instrument.pieces)
        print(f"Loaded instrument '{name}' with {len(instrument.pieces)} pieces, "
              f"{total_vertices} vertices, {total_triangles} triangles")

        return instrument_id

    def update_transform(
        self,
        instrument_id: int,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None
    ):
        """Update instrument root transform.

        Args:
            instrument_id: ID of the instrument to update.
            position: New position [x, y, z].
            rotation: New rotation as quaternion [x, y, z, w].
            scale: New scale [sx, sy, sz].
        """
        if instrument_id >= len(self.instruments):
            return

        instrument = self.instruments[instrument_id]

        if position is None:
            position = [0.0, 0.0, 0.0]
        if rotation is None:
            rotation = [0.0, 0.0, 0.0, 1.0]
        if scale is None:
            scale = [1.0, 1.0, 1.0]

        # Create transform matrix
        pos = wp.vec3(position[0], position[1], position[2])
        rot = wp.quat(rotation[0], rotation[1], rotation[2], rotation[3])
        transform = wp.transform(pos, rot)
        transform_matrix = wp.transform_to_matrix(transform)

        # Apply scale
        instrument.root_transform_matrix = wp.mat44f(
            transform_matrix[0, 0] * scale[0], transform_matrix[0, 1], transform_matrix[0, 2], transform_matrix[0, 3],
            transform_matrix[1, 0], transform_matrix[1, 1] * scale[1], transform_matrix[1, 2], transform_matrix[1, 3],
            transform_matrix[2, 0], transform_matrix[2, 1], transform_matrix[2, 2] * scale[2], transform_matrix[2, 3],
            transform_matrix[3, 0], transform_matrix[3, 1], transform_matrix[3, 2], transform_matrix[3, 3]
        )

        self._update_hierarchy_transforms(instrument_id)

    def update_piece_transform(
        self,
        instrument_id: int,
        piece_name: str,
        transform_matrix: wp.mat44f
    ):
        """Update a specific piece's runtime local transform.

        Args:
            instrument_id: ID of the instrument.
            piece_name: Name of the piece to update.
            transform_matrix: New local transform matrix.
        """
        if instrument_id >= len(self.instruments):
            return

        instrument = self.instruments[instrument_id]

        # Find piece by name
        piece_index = None
        for i, piece in enumerate(instrument.pieces):
            if piece.name == piece_name:
                piece_index = i
                break

        if piece_index is None:
            return

        piece = instrument.pieces[piece_index]
        piece.runtime_local_transform = transform_matrix

        self._update_piece_world_transform(instrument_id, piece_index)

    def _update_hierarchy_transforms(self, instrument_id: int):
        """Update world transforms for entire instrument hierarchy."""
        if instrument_id >= len(self.instruments):
            return

        instrument = self.instruments[instrument_id]
        for root_index in instrument.root_pieces:
            self._update_piece_world_transform(instrument_id, root_index)

    def _update_piece_world_transform(self, instrument_id: int, piece_index: int):
        """Recursively update world transform for a piece and its children."""
        instrument = self.instruments[instrument_id]
        piece = instrument.pieces[piece_index]

        # Get parent world transform
        if piece.parent_index is not None:
            parent_world_matrix = instrument.pieces[piece.parent_index].world_transform_matrix
        else:
            parent_world_matrix = instrument.root_transform_matrix

        # Compute world transform
        combined_local = piece.runtime_local_transform * piece.usd_local_transform
        piece.world_transform_matrix = parent_world_matrix * combined_local

        # Apply transform to vertices
        self._transform_piece_vertices(piece)

        # Recursively update children
        for child_index in piece.children_indices:
            self._update_piece_world_transform(instrument_id, child_index)

    def _transform_piece_vertices(self, piece: InstrumentPiece):
        """Apply current transform to piece vertices."""
        wp.launch(
            apply_matrix_transform_to_vertices,
            dim=piece.vertex_count,
            inputs=[
                piece.original_vertices,
                piece.vertices,
                piece.world_transform_matrix,
                piece.vertex_count
            ],
            device=wp.get_device()
        )

    def set_visibility(self, instrument_id: int, visible: bool):
        """Set instrument visibility."""
        if instrument_id < len(self.instruments):
            self.instruments[instrument_id].visible = visible

    def set_piece_visibility(self, instrument_id: int, piece_name: str, visible: bool):
        """Set visibility for a specific piece."""
        if instrument_id >= len(self.instruments):
            return

        for piece in self.instruments[instrument_id].pieces:
            if piece.name == piece_name:
                piece.visible = visible
                break

    def get_piece_names(self, instrument_id: int) -> List[str]:
        """Get list of piece names for an instrument."""
        if instrument_id >= len(self.instruments):
            return []
        return self.instruments[instrument_id].get_piece_names()

    def get_instrument(self, instrument_id: int) -> Optional[Instrument]:
        """Get instrument by ID."""
        if instrument_id < len(self.instruments):
            return self.instruments[instrument_id]
        return None

    def render(self, renderer, frame_dt: float = 0.0):
        """Render all visible instruments.

        Args:
            renderer: The renderer to use.
            frame_dt: Frame delta time for animations.
        """
        for instrument_idx, instrument in enumerate(self.instruments):
            if not instrument.visible:
                continue

            for piece_idx, piece in enumerate(instrument.pieces):
                if not piece.visible:
                    continue

                renderer.render_mesh_warp(
                    name=f"instrument_{instrument_idx}_piece_{piece_idx}_{piece.name}",
                    points=piece.vertices,
                    indices=piece.indices,
                    pos=(0.0, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),
                    scale=(1.0, 1.0, 1.0),
                    basic_color=(0.7, 0.7, 0.8),
                    update_topology=False,
                    smooth_shading=True,
                    visible=True
                )

    def debug_transforms(self, instrument_id: int):
        """Print debug information about instrument transforms."""
        if instrument_id >= len(self.instruments):
            return

        instrument = self.instruments[instrument_id]
        print(f"\n--- Instrument {instrument_id} Transform Debug ---")

        for i, piece in enumerate(instrument.pieces):
            print(f"Piece {i}: {piece.name}")
            print(f"  USD Local Transform: {piece.usd_local_transform}")
            print(f"  Runtime Local Transform: {piece.runtime_local_transform}")
            print(f"  World Transform: {piece.world_transform_matrix}")
            print(f"  Sample original vertex: {piece.original_vertices.numpy()[0]}")
            print(f"  Sample transformed vertex: {piece.vertices.numpy()[0]}")
            print("---")

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def transform_points_with_matrix(
    world_matrix: wp.mat44f,
    local_points: wp.array(dtype=wp.vec3f),
    world_points: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    world_points[tid] = wp.transform_point(world_matrix, local_points[tid])


@dataclass
class GrasperPiece:
    name: str
    local_points: wp.array
    world_points: wp.array
    indices: wp.array
    bind_matrix: np.ndarray
    color: tuple[float, float, float]
    jaw_sign: float = 0.0


@dataclass
class JawSphereChain:
    name: str
    local_points: wp.array
    world_points: wp.array
    radii: wp.array
    colors: wp.array
    bind_matrix: np.ndarray
    jaw_sign: float


class KinematicGrasper:
    def __init__(
        self,
        pieces: list[GrasperPiece],
        sphere_chains: list[JawSphereChain],
        device,
        jaw_open_angle: float = 0.5,
        jaw_closed_angle: float = 0.0,
        jaw_response: float = 20.0,
    ):
        self.pieces = pieces
        self.sphere_chains = sphere_chains
        self.device = device
        self.jaw_open_angle = jaw_open_angle
        self.jaw_closed_angle = jaw_closed_angle
        self.jaw_response = jaw_response
        self.jaw_angle = jaw_open_angle
        self._model_rotation_fix = _axis_angle_to_quaternion((0.0, 1.0, 0.0), math.pi)

    def advance(self, dt: float, grasp_button_pressed: bool):
        target_angle = self.jaw_closed_angle if grasp_button_pressed else self.jaw_open_angle
        blend = min(1.0, dt * self.jaw_response)
        self.jaw_angle += (target_angle - self.jaw_angle) * blend

    def update_geometry(self, root_position: np.ndarray, haptic_rotation_xyzw: np.ndarray):
        root_rotation = _normalize_quaternion(haptic_rotation_xyzw)
        root_rotation = _multiply_quaternions(root_rotation, self._model_rotation_fix)
        root_matrix = _compose_matrix(root_position, root_rotation)

        for piece in self.pieces:
            piece_matrix = root_matrix @ piece.bind_matrix
            if piece.jaw_sign != 0.0:
                jaw_matrix = _compose_matrix(
                    np.zeros(3, dtype=np.float32),
                    _axis_angle_to_quaternion((1.0, 0.0, 0.0), piece.jaw_sign * self.jaw_angle),
                )
                piece_matrix = piece_matrix @ jaw_matrix

            wp.launch(
                transform_points_with_matrix,
                dim=len(piece.local_points),
                inputs=[_mat44f_from_numpy(piece_matrix), piece.local_points],
                outputs=[piece.world_points],
                device=self.device,
            )

        for chain in self.sphere_chains:
            chain_matrix = root_matrix @ chain.bind_matrix
            jaw_matrix = _compose_matrix(
                np.zeros(3, dtype=np.float32),
                _axis_angle_to_quaternion((1.0, 0.0, 0.0), chain.jaw_sign * self.jaw_angle),
            )
            chain_matrix = chain_matrix @ jaw_matrix
            wp.launch(
                transform_points_with_matrix,
                dim=len(chain.local_points),
                inputs=[_mat44f_from_numpy(chain_matrix), chain.local_points],
                outputs=[chain.world_points],
                device=self.device,
            )

    def render(self, renderer, prefix: str = "grasper"):
        for piece in self.pieces:
            renderer.draw_mesh(
                f"{prefix}_{piece.name}",
                piece.world_points,
                piece.indices,
                color=piece.color,
            )

        for chain in self.sphere_chains:
            renderer.draw_points(
                f"{prefix}_{chain.name}_spheres",
                chain.world_points,
                chain.radii,
                chain.colors,
            )


PIECE_COLORS = {
    "shaft": (0.44, 0.47, 0.50),
    "body": (0.55, 0.58, 0.62),
    "jaw_left": (0.78, 0.80, 0.84),
    "jaw_right": (0.70, 0.73, 0.78),
}

CHAIN_COLORS = {
    "jaw_left": (1.0, 0.55, 0.20),
    "jaw_right": (0.18, 0.78, 1.0),
}


def load_kinematic_grasper(
    usd_path: str | Path,
    device,
    *,
    scale: float = 0.02,
    jaw_sphere_count: int = 16,
    jaw_sphere_radius: float = 0.018,
) -> KinematicGrasper | None:
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return None

    usd_path = Path(usd_path)
    if not usd_path.exists():
        return None

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return None

    pieces: list[GrasperPiece] = []

    def collect_meshes(prim, parent_world: np.ndarray):
        local_matrix = np.eye(4, dtype=np.float32)
        if prim.IsA(UsdGeom.Xformable):
            local_matrix = np.array(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=np.float32).T

        world_matrix = parent_world @ local_matrix

        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            face_indices_attr = mesh.GetFaceVertexIndicesAttr()
            face_counts_attr = mesh.GetFaceVertexCountsAttr()
            if points_attr and face_indices_attr and face_counts_attr:
                points = np.array(points_attr.Get(), dtype=np.float32) * scale
                indices = _triangulate_faces(
                    np.array(face_indices_attr.Get(), dtype=np.int32),
                    np.array(face_counts_attr.Get(), dtype=np.int32),
                )
                if len(points) > 0 and len(indices) > 0:
                    name = prim.GetName()
                    role = _piece_role(name)
                    jaw_sign = 0.0
                    if role == "jaw_left":
                        jaw_sign = 1.0
                    elif role == "jaw_right":
                        jaw_sign = -1.0

                    pieces.append(
                        GrasperPiece(
                            name=name,
                            local_points=wp.array(points, dtype=wp.vec3f, device=device),
                            world_points=wp.zeros(len(points), dtype=wp.vec3f, device=device),
                            indices=wp.array(indices, dtype=wp.int32, device=device),
                            bind_matrix=world_matrix,
                            color=PIECE_COLORS[role],
                            jaw_sign=jaw_sign,
                        )
                    )

        for child in prim.GetChildren():
            collect_meshes(child, world_matrix)

    for child in stage.GetPseudoRoot().GetChildren():
        collect_meshes(child, np.eye(4, dtype=np.float32))

    if not pieces:
        return None

    jaw_pieces = [piece for piece in pieces if piece.jaw_sign != 0.0]
    if not jaw_pieces:
        return KinematicGrasper(pieces, [], device=device)

    jaw_tip_extent = max(float(piece.local_points.numpy()[:, 2].max()) for piece in jaw_pieces)
    jaw_tip_extent = max(jaw_tip_extent, 0.25)
    chain_template = np.zeros((jaw_sphere_count, 3), dtype=np.float32)
    chain_template[:, 2] = np.linspace(0.0, jaw_tip_extent, jaw_sphere_count, dtype=np.float32)

    sphere_chains: list[JawSphereChain] = []
    for piece in jaw_pieces:
        role = "jaw_left" if piece.jaw_sign > 0.0 else "jaw_right"
        colors = np.repeat(np.array([CHAIN_COLORS[role]], dtype=np.float32), jaw_sphere_count, axis=0)
        sphere_chains.append(
            JawSphereChain(
                name=role,
                local_points=wp.array(chain_template, dtype=wp.vec3f, device=device),
                world_points=wp.zeros(jaw_sphere_count, dtype=wp.vec3f, device=device),
                radii=wp.full(jaw_sphere_count, jaw_sphere_radius, dtype=wp.float32, device=device),
                colors=wp.array(colors, dtype=wp.vec3f, device=device),
                bind_matrix=piece.bind_matrix,
                jaw_sign=piece.jaw_sign,
            )
        )

    return KinematicGrasper(pieces, sphere_chains, device=device)


def _triangulate_faces(face_vertex_indices: np.ndarray, face_vertex_counts: np.ndarray) -> np.ndarray:
    triangles: list[int] = []
    face_start = 0
    for face_vertex_count in face_vertex_counts:
        count = int(face_vertex_count)
        if count < 3:
            face_start += count
            continue
        if count == 3:
            triangles.extend(face_vertex_indices[face_start : face_start + 3].tolist())
        else:
            first = int(face_vertex_indices[face_start])
            for i in range(1, count - 1):
                triangles.extend(
                    [
                        first,
                        int(face_vertex_indices[face_start + i]),
                        int(face_vertex_indices[face_start + i + 1]),
                    ]
                )
        face_start += count
    return np.array(triangles, dtype=np.int32)


def _piece_role(name: str) -> str:
    lower = name.lower()
    if "jawleft" in lower:
        return "jaw_left"
    if "jawright" in lower:
        return "jaw_right"
    if "shaft" in lower:
        return "shaft"
    return "body"


def _axis_angle_to_quaternion(axis, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis /= norm
    half_angle = 0.5 * float(angle)
    sin_half = math.sin(half_angle)
    return np.array(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half_angle)],
        dtype=np.float32,
    )


def _normalize_quaternion(quat_xyzw) -> np.ndarray:
    quat = np.array(quat_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def _multiply_quaternions(q1_xyzw, q2_xyzw) -> np.ndarray:
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _quaternion_to_matrix(quat_xyzw) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(quat_xyzw)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _compose_matrix(translation_xyz, quat_xyzw) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quaternion_to_matrix(quat_xyzw)
    matrix[:3, 3] = np.array(translation_xyz, dtype=np.float32)
    return matrix


def _mat44f_from_numpy(matrix: np.ndarray) -> wp.mat44f:
    return wp.mat44f(
        float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3]),
        float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3]),
        float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3]),
        float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3]),
    )

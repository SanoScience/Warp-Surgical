import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def update_jaw_angle_state(
    grip_command: wp.array(dtype=wp.float32),
    jaw_angle: wp.array(dtype=wp.float32),
    dt: float,
    jaw_open_angle: float,
    jaw_closed_angle: float,
    jaw_response: float,
):
    if wp.tid() != 0:
        return

    grip = grip_command[0]
    if grip < 0.0:
        target_angle = jaw_open_angle
    else:
        closure = wp.min(wp.max(grip, 0.0), 1.0)
        target_angle = jaw_open_angle + closure * (jaw_closed_angle - jaw_open_angle)

    blend = wp.min(1.0, dt * jaw_response)
    jaw_angle[0] = jaw_angle[0] + (target_angle - jaw_angle[0]) * blend


@wp.kernel
def interpolate_grasper_position(
    root_position_prev: wp.array(dtype=wp.vec3f),
    root_position_target: wp.array(dtype=wp.vec3f),
    root_position_current: wp.array(dtype=wp.vec3f),
    factor: float,
):
    if wp.tid() != 0:
        return

    root_position_current[0] = wp.lerp(root_position_prev[0], root_position_target[0], factor)


@wp.func
def _rotate_point_about_jaw_axis(local_point: wp.vec3f, jaw_sign: float, jaw_angle: float) -> wp.vec3f:
    if jaw_sign == 0.0:
        return local_point

    jaw_rotation = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), jaw_sign * jaw_angle)
    return wp.quat_rotate(jaw_rotation, local_point)


@wp.kernel
def transform_grasper_points(
    root_position: wp.array(dtype=wp.vec3f),
    root_rotation: wp.array(dtype=wp.quatf),
    jaw_angle: wp.array(dtype=wp.float32),
    bind_matrix: wp.array(dtype=wp.mat44f),
    local_points: wp.array(dtype=wp.vec3f),
    world_points: wp.array(dtype=wp.vec3f),
    jaw_sign: float,
):
    tid = wp.tid()
    local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])
    bind_space_point = wp.transform_point(bind_matrix[0], local_point)
    world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)


@wp.kernel
def transform_grasper_spheres(
    root_position: wp.array(dtype=wp.vec3f),
    root_rotation: wp.array(dtype=wp.quatf),
    grip_command: wp.array(dtype=wp.float32),
    jaw_angle: wp.array(dtype=wp.float32),
    bind_matrix: wp.array(dtype=wp.mat44f),
    local_points: wp.array(dtype=wp.vec3f),
    world_points: wp.array(dtype=wp.vec3f),
    base_radii: wp.array(dtype=wp.float32),
    active_radii: wp.array(dtype=wp.float32),
    jaw_sign: float,
):
    tid = wp.tid()
    local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])
    bind_space_point = wp.transform_point(bind_matrix[0], local_point)
    world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)

    if grip_command[0] < 0.0:
        active_radii[tid] = 0.0
    else:
        active_radii[tid] = base_radii[tid]


@dataclass
class GrasperPiece:
    name: str
    local_points: wp.array
    world_points: wp.array
    indices: wp.array
    bind_matrix: wp.array
    color: tuple[float, float, float]
    jaw_sign: float = 0.0


@dataclass
class JawSphereChain:
    name: str
    local_points: wp.array
    world_points_prev: wp.array
    world_points: wp.array
    base_radii: wp.array
    radii: wp.array
    colors: wp.array
    bind_matrix: wp.array
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

        self.root_position_prev = wp.zeros(1, dtype=wp.vec3f, device=device)
        self.root_position_target = wp.zeros(1, dtype=wp.vec3f, device=device)
        self.root_position = wp.zeros(1, dtype=wp.vec3f, device=device)
        self.root_rotation = wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device=device)
        self.grip_command = wp.array([-1.0], dtype=wp.float32, device=device)
        self.jaw_angle_buffer = wp.array([jaw_open_angle], dtype=wp.float32, device=device)

        self._root_position_staging = wp.zeros(1, dtype=wp.vec3f, device="cpu")
        self._root_position_view = self._root_position_staging.numpy()
        self._root_rotation_staging = wp.zeros(1, dtype=wp.quatf, device="cpu")
        self._root_rotation_view = self._root_rotation_staging.numpy()
        self._grip_staging = wp.zeros(1, dtype=wp.float32, device="cpu")
        self._grip_view = self._grip_staging.numpy()

        self.set_root_pose(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def set_root_pose(self, root_position: np.ndarray, haptic_rotation_xyzw: np.ndarray):
        normalized_rotation = _normalize_quaternion(haptic_rotation_xyzw)
        self._root_position_view[0] = np.asarray(root_position, dtype=np.float32)
        self._root_rotation_view[0] = normalized_rotation
        wp.copy(self.root_position_prev, self.root_position_target)
        wp.copy(self.root_position_target, self._root_position_staging)
        wp.copy(self.root_rotation, self._root_rotation_staging)

    def update_substep_pose(self, factor: float):
        wp.launch(
            interpolate_grasper_position,
            dim=1,
            inputs=[
                self.root_position_prev,
                self.root_position_target,
                self.root_position,
                factor,
            ],
            device=self.device,
        )

    def set_grip_command(self, grasp_command: float | bool | None):
        command = -1.0 if grasp_command is None else _coerce_unit_interval(grasp_command)
        self._grip_view[0] = command
        wp.copy(self.grip_command, self._grip_staging)

    def advance(self, dt: float, grasp_command: float | bool):
        closure = _coerce_unit_interval(grasp_command)
        target_angle = self.jaw_open_angle + closure * (self.jaw_closed_angle - self.jaw_open_angle)
        blend = min(1.0, dt * self.jaw_response)
        self.jaw_angle += (target_angle - self.jaw_angle) * blend

        self.set_grip_command(grasp_command)
        wp.launch(
            update_jaw_angle_state,
            dim=1,
            inputs=[
                self.grip_command,
                self.jaw_angle_buffer,
                dt,
                self.jaw_open_angle,
                self.jaw_closed_angle,
                self.jaw_response,
            ],
            device=self.device,
        )

    def update_collision_geometry(self):
        for chain in self.sphere_chains:
            wp.copy(chain.world_points_prev, chain.world_points)
            wp.launch(
                transform_grasper_spheres,
                dim=len(chain.local_points),
                inputs=[
                    self.root_position,
                    self.root_rotation,
                    self.grip_command,
                    self.jaw_angle_buffer,
                    chain.bind_matrix,
                    chain.local_points,
                    chain.world_points,
                    chain.base_radii,
                    chain.radii,
                    chain.jaw_sign,
                ],
                device=self.device,
            )

    def update_render_geometry(self):
        for piece in self.pieces:
            wp.launch(
                transform_grasper_points,
                dim=len(piece.local_points),
                inputs=[
                    self.root_position,
                    self.root_rotation,
                    self.jaw_angle_buffer,
                    piece.bind_matrix,
                    piece.local_points,
                    piece.world_points,
                    piece.jaw_sign,
                ],
                device=self.device,
            )

    def render(self, renderer, prefix: str = "grasper", draw_collision_spheres: bool = False):
        for piece in self.pieces:
            renderer.draw_mesh(
                f"{prefix}_{piece.name}",
                piece.world_points,
                piece.indices,
                color=piece.color,
            )

        if not draw_collision_spheres:
            return

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

    model_fix_matrix = _compose_matrix(
        np.zeros(3, dtype=np.float32),
        _axis_angle_to_quaternion((0.0, 1.0, 0.0), math.pi),
    )

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

                    corrected_bind = model_fix_matrix @ world_matrix
                    pieces.append(
                        GrasperPiece(
                            name=name,
                            local_points=wp.array(points, dtype=wp.vec3f, device=device),
                            world_points=wp.zeros(len(points), dtype=wp.vec3f, device=device),
                            indices=wp.array(indices, dtype=wp.int32, device=device),
                            bind_matrix=wp.array(np.expand_dims(corrected_bind, 0), dtype=wp.mat44f, device=device),
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
        base_radii = wp.full(jaw_sphere_count, jaw_sphere_radius, dtype=wp.float32, device=device)
        sphere_chains.append(
            JawSphereChain(
                name=role,
                local_points=wp.array(chain_template, dtype=wp.vec3f, device=device),
                world_points_prev=wp.zeros(jaw_sphere_count, dtype=wp.vec3f, device=device),
                world_points=wp.zeros(jaw_sphere_count, dtype=wp.vec3f, device=device),
                base_radii=base_radii,
                radii=wp.zeros(jaw_sphere_count, dtype=wp.float32, device=device),
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


def _coerce_unit_interval(value: float | bool) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(numeric):
        return 0.0

    return float(np.clip(numeric, 0.0, 1.0))

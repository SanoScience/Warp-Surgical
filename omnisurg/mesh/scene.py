from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from omnisurg.config import HapticConfig, SceneConfig
from omnisurg.input.haptic_proxy import HapticProxyState, create_haptic_proxy_state
from omnisurg.mesh.assets import MeshRange, TetMeshAsset
from omnisurg.mesh.types import Tetrahedron, TriPointsConnector, compute_tet_volume


@dataclass
class SceneData:
    """Everything produced by the Phase 1 or Phase 2 scene builder."""

    model: newton.Model
    surface_tri_indices: wp.array
    surface_meshes: dict[str, wp.array]
    uvs: wp.array | None
    haptic_proxy: HapticProxyState
    tetrahedra_wp: wp.array
    mesh_ranges: dict[str, MeshRange]


def _is_pinned(pos: np.ndarray, pin_center: tuple | None, pin_radius: float) -> bool:
    if pin_center is None:
        return False
    center = np.array(pin_center, dtype=np.float32)
    return float(np.linalg.norm(pos - center)) < pin_radius


def build_scene(
    asset: TetMeshAsset,
    scene: SceneConfig,
    haptic: HapticConfig,
    device,
) -> SceneData:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    translation = np.array(scene.translation, dtype=np.float32)

    for i in range(len(asset.rest_positions)):
        translated = asset.rest_positions[i] + translation
        pinned = _is_pinned(translated, scene.pin_center, scene.pin_radius)
        mass = 0.0 if pinned else scene.particle_mass
        builder.add_particle(
            wp.vec3(float(translated[0]), float(translated[1]), float(translated[2])),
            wp.vec3(0.0, 0.0, 0.0),
            mass=mass,
            radius=scene.particle_radius,
        )

    for i in range(len(asset.edge_indices)):
        builder.add_spring(
            int(asset.edge_indices[i, 0]),
            int(asset.edge_indices[i, 1]),
            scene.spring_stiffness,
            scene.spring_dampen,
            0,
        )

    for i in range(len(asset.surface_tri_indices)):
        ids = asset.surface_tri_indices[i]
        builder.add_triangle(int(ids[0]), int(ids[1]), int(ids[2]))

    tetrahedra = []
    for i in range(len(asset.tet_indices)):
        ids = asset.tet_indices[i]
        verts = [wp.vec3(*(asset.rest_positions[j] + translation).tolist()) for j in ids]
        rest_vol = compute_tet_volume(verts[0], verts[1], verts[2], verts[3])

        tet = Tetrahedron()
        tet.ids = wp.vec4i(int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
        tet.rest_volume = rest_vol
        tetrahedra.append(tet)

        builder.add_tetrahedron(
            int(ids[0]),
            int(ids[1]),
            int(ids[2]),
            int(ids[3]),
            scene.tet_stiffness_mu,
            scene.tet_stiffness_lambda,
            scene.tet_dampen,
        )

    haptic_body_id = builder.add_body(
        xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        mass=0.0,
        armature=0.0,
    )
    builder.add_shape_sphere(
        body=haptic_body_id,
        xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        radius=haptic.collision_radius,
        cfg=newton.ModelBuilder.ShapeConfig(density=10),
    )

    model = builder.finalize()

    tetrahedra_wp = wp.array(tetrahedra, dtype=Tetrahedron, device=device)
    model.tetrahedra_wp = tetrahedra_wp
    model.tet_active = wp.ones(len(tetrahedra), dtype=wp.int32, device=device)
    model.particle_max_velocity = 10.0
    model.tri_points_connectors = (
        wp.array(list(asset.connectors), dtype=TriPointsConnector, device=device)
        if asset.connectors
        else wp.empty(0, dtype=TriPointsConnector, device=device)
    )

    surface_tri_wp = wp.array(
        asset.surface_tri_indices.flatten().tolist(),
        dtype=wp.int32,
        device=device,
    )
    surface_meshes = {}
    for mesh_name, mesh_range in (asset.mesh_ranges or {}).items():
        mesh_surface = asset.surface_tri_indices[
            mesh_range.tri_start : mesh_range.tri_start + mesh_range.tri_count
        ]
        surface_meshes[mesh_name] = wp.array(
            mesh_surface.flatten().tolist(),
            dtype=wp.int32,
            device=device,
        )

    uvs_wp = wp.array(asset.uvs, dtype=wp.vec2, device=device) if asset.uvs is not None else None

    max_tri_extent = 0.0
    translated_positions = asset.rest_positions + translation
    for i in range(len(asset.surface_tri_indices)):
        ids = asset.surface_tri_indices[i]
        pts = translated_positions[ids]
        centroid = pts.mean(axis=0)
        extent = float(np.max(np.linalg.norm(pts - centroid, axis=1)))
        if extent > max_tri_extent:
            max_tri_extent = extent
    max_tri_extent *= 2.0

    proxy = create_haptic_proxy_state(
        body_id=haptic_body_id,
        radius=haptic.collision_radius,
        device=device,
        max_tri_extent=max_tri_extent,
    )

    return SceneData(
        model=model,
        surface_tri_indices=surface_tri_wp,
        surface_meshes=surface_meshes,
        uvs=uvs_wp,
        haptic_proxy=proxy,
        tetrahedra_wp=tetrahedra_wp,
        mesh_ranges=asset.mesh_ranges or {},
    )

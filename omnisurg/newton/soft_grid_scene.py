from dataclasses import dataclass

import warp as wp

import newton

from omnisurg.config import HapticConfig


@dataclass(frozen=True)
class SoftGridSceneConfig:
    grid_origin: tuple[float, float, float] = (0.0, 1.0, 1.0)
    haptic_start: tuple[float, float, float] = (0.5, 1.2, 1.2)
    dim_x: int = 12
    dim_y: int = 4
    dim_z: int = 4
    cell_size: float = 0.1
    density: float = 1.0e2
    k_mu: float = 1.0e5
    k_lambda: float = 1.0e5
    k_damp: float = 1.0e-1
    fix_left: bool = True
    fix_right: bool = False
    soft_contact_ke: float = 1.0e5
    soft_contact_kd: float = 1.0e-4
    soft_contact_mu: float = 1.0


@dataclass(frozen=True)
class SoftGridScene:
    model: newton.Model
    haptic_body_id: int
    haptic_start: wp.vec3


def build_soft_grid_scene(
    scene_config: SoftGridSceneConfig,
    haptic_config: HapticConfig,
) -> SoftGridScene:
    """Build the Newton soft-grid scene used by the standalone example."""

    #builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    grid_origin = wp.vec3(*scene_config.grid_origin)
    haptic_start = wp.vec3(*scene_config.haptic_start)

    builder.add_soft_grid(
        pos=grid_origin,
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=scene_config.dim_x,
        dim_y=scene_config.dim_y,
        dim_z=scene_config.dim_z,
        cell_x=scene_config.cell_size,
        cell_y=scene_config.cell_size,
        cell_z=scene_config.cell_size,
        density=scene_config.density,
        k_mu=scene_config.k_mu,
        k_lambda=scene_config.k_lambda,
        k_damp=scene_config.k_damp,
        fix_left=scene_config.fix_left,
        fix_right=scene_config.fix_right,
        add_surface_mesh_edges=True,
        particle_radius=0.02,
    )

    haptic_body_id = builder.add_body(
        xform=wp.transform(haptic_start, wp.quat_identity()),
        mass=0.0,
        armature=0.0,
    )
    builder.add_shape_sphere(
        body=haptic_body_id,
        xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        radius=haptic_config.collision_radius,
        cfg=newton.ModelBuilder.ShapeConfig(density=10),
    )

    builder.color()

    model = builder.finalize()
    model.soft_contact_ke = scene_config.soft_contact_ke
    model.soft_contact_kd = scene_config.soft_contact_kd
    model.soft_contact_mu = scene_config.soft_contact_mu

    model.shape_material_ke.fill_(scene_config.soft_contact_ke)
    model.shape_material_kd.fill_(scene_config.soft_contact_kd)
    model.shape_material_mu.fill_(scene_config.soft_contact_mu)

    return SoftGridScene(
        model=model,
        haptic_body_id=haptic_body_id,
        haptic_start=haptic_start,
    )

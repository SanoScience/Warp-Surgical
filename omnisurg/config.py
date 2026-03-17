from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    substeps: int = 16
    fps: int = 120
    constraint_iterations: int = 1
    frame_dt: float = field(init=False)
    substep_dt: float = field(init=False)

    def __post_init__(self):
        self.frame_dt = 1.0 / self.fps
        self.substep_dt = self.frame_dt / self.substeps


@dataclass
class SceneConfig:
    asset_name: str = "liver"
    mesh_dir: str = "meshes"
    translation: tuple = (0.0, -3.0, 0.0)
    particle_mass: float = 0.1
    particle_radius: float = 0.01
    spring_stiffness: float = 1.0
    spring_dampen: float = 0.2
    tet_stiffness_mu: float = 1e4
    tet_stiffness_lambda: float = 1e4
    tet_dampen: float = 0.2
    volume_stiffness: float = 0.1
    pin_center: tuple | None = (0.5, 1.5, -5.0)
    pin_radius: float = 1.0


@dataclass
class HapticConfig:
    collision_radius: float = 0.1
    # Offset applied in raw device units BEFORE the kernels' built-in 0.01 scale
    position_offset: tuple = (0.0, 100.0, -400.0)


@dataclass
class ViewerConfig:
    backend: str = "gl"
    camera_pos: tuple = (0.2, 1.2, -1.0)


@dataclass
class BoundsConfig:
    bounds_min: tuple = (-2.0, 0.0, -8.0)
    bounds_max: tuple = (2.0, 10.0, -3.0)

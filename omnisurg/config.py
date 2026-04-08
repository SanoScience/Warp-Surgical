from dataclasses import dataclass, field


SCENE_PRESETS: tuple[str, ...] = ("single", "chole")


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
    scene_preset: str = "single"
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
    position_offset: tuple = (0.0, 100.0, -400.0)
    position_scale: float = 0.01


@dataclass
class ViewerConfig:
    backend: str = "gl"
    camera_pos: tuple = (0.2, 1.2, -1.0)
    vsync: bool = True
    textures_enabled: bool = True
    sky_enabled: bool = True
    shadows_enabled: bool = False
    msaa_samples: int = 0
    direct_render_enabled: bool = True


@dataclass
class BoundsConfig:
    bounds_min: tuple = (-2.0, 0.0, -8.0)
    bounds_max: tuple = (2.0, 10.0, -3.0)


SIMULATION_PRESETS: dict[str, SimulationConfig] = {
    "quality": SimulationConfig(substeps=16, fps=120),
    "balanced": SimulationConfig(substeps=8, fps=90),
    "performance": SimulationConfig(substeps=4, fps=60),
}

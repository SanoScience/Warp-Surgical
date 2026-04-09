from dataclasses import dataclass, field


SCENE_PRESETS: tuple[str, ...] = ("single", "chole")
DEFAULT_PIN_CENTER = (0.5, 1.5, -5.0)
DEFAULT_PIN_RADIUS = 1.0


def is_synthetic_patch_asset_name(asset_name: str) -> bool:
    return asset_name.startswith("cloth_regular_") or asset_name.startswith("cloth_irregular_")


@dataclass
class SimulationConfig:
    substeps: int = 16
    fps: int = 120
    constraint_iterations: int = 1
    grasper_collision_mode: str = "projection"
    grasper_collision_motion_samples: int = 4
    grasper_collision_margin: float = 0.002
    grasper_truncation_safety: float = 0.90
    grasper_truncate_prediction: bool = True
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
    pin_center: tuple | None = DEFAULT_PIN_CENTER
    pin_radius: float = DEFAULT_PIN_RADIUS
    pinned_vertex_ids: tuple[int, ...] = ()
    show_grasper_mesh: bool = True
    enable_grasper_collisions: bool = True

    def __post_init__(self):
        if (
            self.scene_preset == "single"
            and is_synthetic_patch_asset_name(self.asset_name)
            and self.pin_center == DEFAULT_PIN_CENTER
            and self.pin_radius == DEFAULT_PIN_RADIUS
            and not self.pinned_vertex_ids
        ):
            # Synthetic cloth patches use exact corner pins instead of the default liver pin sphere.
            self.pin_center = None
            self.show_grasper_mesh = False
            self.enable_grasper_collisions = False


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

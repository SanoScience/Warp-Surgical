import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp
from newton._src.geometry import ParticleFlags
from newton._src.geometry.kernels import triangle_closest_point

from omnisurg.config import BoundsConfig, HapticConfig, SceneConfig, SimulationConfig, ViewerConfig
from omnisurg.grasper_runtime import load_kinematic_grasper
from omnisurg.haptic_bench.algorithms import (
    HapticAlgorithm,
    HapticStepInput,
    create as create_haptic_algorithm,
)
from omnisurg.haptic_feedback import (
    DEFAULT_HAPTIC_PRESET_FILENAME,
    HapticFeedbackDiagnostics,
    HapticFeedbackSettings,
    HapticPresetRecord,
    copy_feedback_settings,
    load_haptic_preset,
    sanitize_preset_filename,
    save_haptic_preset,
    scan_haptic_presets,
    settings_almost_equal,
)
from omnisurg.input.haptic_collision import HapticSphereCollisionSystem, HapticSphereTruncationSystem
from omnisurg.input.haptic_proxy import HapticProxyState, create_vec3_staging_buffer, scale_position, update_haptic_proxy
from omnisurg.input.sources import InputRig, InputSource
from omnisurg.mesh.assets import load_scene_asset
from omnisurg.mesh.scene import build_scene
from omnisurg.physics.base import SimulationSystem, SolverStage
from omnisurg.physics.kernels import apply_deltas_and_zero_accumulators
from omnisurg.physics.solver import Phase1Solver
from omnisurg.physics.systems import (
    BoundsCollisionSystem,
    DistanceConstraintSystem,
    TrianglePointConstraintSystem,
    VolumeConstraintSystem,
)
from omnisurg.rendering.bridge import RenderBridge
from omnisurg.rendering.slang import TISSUE_DEBUG_MODE_LABELS


REPO_ROOT = Path(__file__).resolve().parents[1]
HAPTIC_PRESET_DIR = REPO_ROOT / "presets" / "haptics"
GRASPER_ASSET_PATH = REPO_ROOT / "meshes" / "pgrasp.usdc"
GRASPER_SCALE = 0.01
GRASPER_JAW_SPHERE_COUNT = 24
GRASPER_JAW_SPHERE_RADIUS = 0.018
GRASPER_COLLISION_MOTION_SAMPLES = 4
GRASPER_COLLISION_SWEEP_SAMPLES = GRASPER_COLLISION_MOTION_SAMPLES
GRASPER_COLLISION_MARGIN = 0.002
GRASPER_COLLISION_TRUNCATION_SAFETY = 0.90
GRASPER_COLLISION_MODES = ("projection", "truncation", "hybrid")
GRASPER_COLLISION_MODE_LABELS = ("Projection", "Truncation", "Hybrid")
HAPTIC_VISUAL_RADIUS_BASELINE = 0.025
PRIMARY_CONTROLLER_ID = "right"
TEXTURE_CANDIDATES = {
    "liver": (
        REPO_ROOT / "textures" / "liver" / "diffuse-base.png",
        REPO_ROOT / "textures" / "liver_mesh_diffuse.png",
    ),
    "fat": (
        REPO_ROOT / "textures" / "fat" / "diffuse-base.png",
        REPO_ROOT / "textures" / "fat_mesh_diffuse.png",
    ),
    "gallbladder": (
        REPO_ROOT / "textures" / "gallbladder" / "diffuse-base.png",
        REPO_ROOT / "textures" / "gallbladder_mesh_diffuse.png",
    ),
}

MESH_COLORS = {
    "liver": (0.63, 0.29, 0.24),
    "fat": (0.90, 0.82, 0.42),
    "gallbladder": (0.19, 0.54, 0.22),
    "tissue": (0.87, 0.83, 0.78),
}

CONTROLLER_ROOT_COLORS = {
    "left": (0.18, 0.78, 1.0),
}


@dataclass(frozen=True)
class ControllerBinding:
    controller_id: str
    uses_physics_proxy: bool


@dataclass
class ControllerRuntimeState:
    active: bool = False
    sample_present: bool = False
    button: bool = False
    grip: float = 0.0
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    scaled_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    render_position: wp.array | None = None
    staging: wp.array | None = None
    staging_view: np.ndarray | None = None


@wp.func
def _double_sided_fallback_dir(
    v0: wp.vec3f,
    v1: wp.vec3f,
    v2: wp.vec3f,
    contact_point: wp.vec3f,
    reference_dir: wp.vec3f,
) -> wp.vec3f:
    if wp.length_sq(reference_dir) > 1.0e-12:
        return wp.normalize(reference_dir)

    centroid = (v0 + v1 + v2) / 3.0
    centroid_dir = centroid - contact_point
    if wp.length_sq(centroid_dir) > 1.0e-12:
        return wp.normalize(centroid_dir)

    d0 = v0 - contact_point
    d1 = v1 - contact_point
    d2 = v2 - contact_point

    best = d0
    if wp.length_sq(d1) > wp.length_sq(best):
        best = d1
    if wp.length_sq(d2) > wp.length_sq(best):
        best = d2

    if wp.length_sq(best) > 1.0e-12:
        return wp.normalize(best)

    return wp.vec3f(1.0, 0.0, 0.0)


@wp.kernel
def collide_triangles_vs_spheres(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    sphere_centers_prev: wp.array(dtype=wp.vec3f),
    sphere_centers: wp.array(dtype=wp.vec3f),
    sphere_radii: wp.array(dtype=wp.float32),
    num_spheres: int,
    sweep_samples: int,
    radius_margin: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    cull_radius: wp.float32,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32),
    reaction_accumulator: wp.array(dtype=wp.vec3f),
    reaction_counter: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    tri_count = tri_indices.shape[0]
    if tri_count == 0 or sweep_samples <= 0:
        return

    work_items_per_sphere = tri_count * sweep_samples
    sphere_idx = tid // work_items_per_sphere
    if sphere_idx >= num_spheres:
        return

    sphere_work_idx = tid % work_items_per_sphere
    sample_idx = sphere_work_idx // tri_count
    tri_idx = sphere_work_idx % tri_count

    sphere_radius = sphere_radii[sphere_idx] + radius_margin
    if sphere_radius <= 0.0:
        return

    t1 = tri_indices[tri_idx, 0]
    t2 = tri_indices[tri_idx, 1]
    t3 = tri_indices[tri_idx, 2]

    p1 = positions[t1]
    p2 = positions[t2]
    p3 = positions[t3]

    w1 = inv_masses[t1]
    w2 = inv_masses[t2]
    w3 = inv_masses[t3]
    weight = w1 + w2 + w3
    if weight <= 0.0:
        return

    sample_factor = wp.float32(sample_idx + 1) / wp.float32(sweep_samples)
    sphere_pos = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)
    if cull_radius > 0.0:
        centroid = (p1 + p2 + p3) / 3.0
        if wp.length(centroid - sphere_pos) > cull_radius:
            return

    closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)
    to_sphere = closest_p - sphere_pos
    dist = wp.length(to_sphere)
    if dist >= sphere_radius:
        return

    penetration = sphere_radius - dist
    if dist > 1e-8:
        correction_dir = to_sphere / dist
    else:
        correction_dir = _double_sided_fallback_dir(
            p1,
            p2,
            p3,
            sphere_pos,
            sphere_centers_prev[sphere_idx] - sphere_centers[sphere_idx],
        )

    total_correction = correction_dir * penetration
    d1 = total_correction * (w1 / weight)
    d2 = total_correction * (w2 / weight)
    d3 = total_correction * (w3 / weight)

    wp.atomic_add(delta_accumulator, t1, d1)
    wp.atomic_add(delta_accumulator, t2, d2)
    wp.atomic_add(delta_accumulator, t3, d3)
    wp.atomic_add(delta_counter, t1, 1)
    wp.atomic_add(delta_counter, t2, 1)
    wp.atomic_add(delta_counter, t3, 1)
    wp.atomic_add(reaction_accumulator, 0, -total_correction)
    wp.atomic_add(reaction_counter, 0, 1)


@wp.kernel
def fill_tissue_vertex_blend_colors(
    vertex_colors: wp.array(dtype=wp.vec4f),
    damage: wp.float32,
    coag: wp.float32,
    blood: wp.float32,
):
    tid = wp.tid()
    vertex_colors[tid] = wp.vec4f(damage, coag, blood, 0.0)


@wp.kernel
def compute_vertex_sphere_truncation_factors(
    particle_flags: wp.array(dtype=wp.int32),
    base_positions: wp.array(dtype=wp.vec3f),
    surface_vertex_ids: wp.array(dtype=wp.int32),
    displacement_in: wp.array(dtype=wp.vec3f),
    sphere_centers_prev: wp.array(dtype=wp.vec3f),
    sphere_centers: wp.array(dtype=wp.vec3f),
    sphere_radii: wp.array(dtype=wp.float32),
    num_spheres: int,
    motion_samples: int,
    radius_margin: wp.float32,
    safety: wp.float32,
    truncation_t_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    surface_vertex_count = surface_vertex_ids.shape[0]
    if surface_vertex_count == 0 or num_spheres <= 0 or motion_samples <= 0:
        return

    work_items_per_vertex = num_spheres * motion_samples
    vertex_idx = tid // work_items_per_vertex
    if vertex_idx >= surface_vertex_count:
        return

    local_idx = tid % work_items_per_vertex
    sphere_idx = local_idx // motion_samples
    sample_idx = local_idx % motion_samples
    vertex_id = surface_vertex_ids[vertex_idx]

    if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:
        return

    displacement = displacement_in[vertex_id]
    if wp.length_sq(displacement) <= 1.0e-12:
        return

    sphere_radius = sphere_radii[sphere_idx] + radius_margin
    if sphere_radius <= 0.0:
        return

    sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)
    sphere_center = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)

    x0 = base_positions[vertex_id]
    x1 = x0 + displacement

    normal = x1 - sphere_center
    if wp.length_sq(normal) <= 1.0e-12:
        normal = x0 - sphere_center
    if wp.length_sq(normal) <= 1.0e-12:
        normal = -displacement
    if wp.length_sq(normal) <= 1.0e-12:
        return

    n = wp.normalize(normal)
    plane_point = sphere_center + n * sphere_radius

    s0 = wp.dot(n, x0 - plane_point)
    s1 = wp.dot(n, x1 - plane_point)
    if s1 >= 0.0:
        return

    t = wp.float32(1.0)
    if s0 > 0.0:
        denom = s0 - s1
        if denom <= 1.0e-8:
            return
        crossing_t = s0 / denom
        t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)
    elif s1 < s0:
        t = 0.0
    else:
        return

    wp.atomic_min(truncation_t_out, vertex_id, t)


@wp.kernel
def apply_surface_vertex_truncation(
    surface_vertex_ids: wp.array(dtype=wp.int32),
    truncation_t: wp.array(dtype=wp.float32),
    displacements: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    if tid >= surface_vertex_ids.shape[0]:
        return

    vertex_id = surface_vertex_ids[tid]
    displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]


class GrasperSphereCollisionSystem(SimulationSystem):
    """Resolve tissue collisions against every sphere in the rendered grasper jaws."""

    def __init__(
        self,
        *,
        graspers: dict,
        proxy: HapticProxyState,
        motion_samples: int = GRASPER_COLLISION_MOTION_SAMPLES,
        contact_margin: float = GRASPER_COLLISION_MARGIN,
        priority: int = 85,
    ):
        super().__init__(priority=priority, stage=SolverStage.PROJECTION)
        self.graspers = graspers
        self.proxy = proxy
        self._accumulator: wp.array | None = None
        self._count: wp.array | None = None
        self._reaction_accumulators: dict[str, wp.array] = {}
        self._reaction_counts: dict[str, wp.array] = {}
        self.motion_samples = motion_samples
        self.contact_margin = contact_margin
        self._chain_cull_radii: dict[int, float] = {}
        self._rebuild_chain_cull_radii()

    def _rebuild_chain_cull_radii(self):
        self._chain_cull_radii = {}
        for grasper in self.graspers.values():
            if grasper is None:
                continue
            for chain in grasper.sphere_chains:
                self._chain_cull_radii[id(chain)] = (
                    float(chain.base_radii.numpy().max()) + float(self.proxy.max_tri_extent) + self.contact_margin
                )

    def get_accumulators(self):
        if self._accumulator is None:
            return None
        return (self._accumulator, self._count)

    def initialize(self, model):
        self._accumulator = wp.zeros(model.particle_count, dtype=wp.vec3f, device=model.device)
        self._count = wp.zeros(model.particle_count, dtype=wp.int32, device=model.device)
        self._reaction_accumulators = {
            controller_id: wp.zeros(1, dtype=wp.vec3f, device=model.device)
            for controller_id in self.graspers
        }
        self._reaction_counts = {
            controller_id: wp.zeros(1, dtype=wp.int32, device=model.device)
            for controller_id in self.graspers
        }

    def clear_reaction(self):
        for accumulator in self._reaction_accumulators.values():
            accumulator.zero_()
        for counter in self._reaction_counts.values():
            counter.zero_()

    def get_reaction_average(self, controller_id: str) -> tuple[np.ndarray, int]:
        reaction_accumulator = self._reaction_accumulators.get(controller_id)
        reaction_count = self._reaction_counts.get(controller_id)
        if reaction_accumulator is None or reaction_count is None:
            return np.zeros(3, dtype=np.float32), 0

        reaction = np.asarray(reaction_accumulator.numpy()[0], dtype=np.float32)
        count = int(reaction_count.numpy()[0])
        if count <= 0:
            return np.zeros(3, dtype=np.float32), 0

        return reaction / float(count), count

    def solve_constraints(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        body_q,
        body_qd,
        body_deltas,
        dt,
        iteration,
    ):
        if model.tri_count == 0:
            return

        for controller_id, grasper in self.graspers.items():
            if grasper is None:
                continue

            reaction_accumulator = self._reaction_accumulators.get(controller_id)
            reaction_count = self._reaction_counts.get(controller_id)
            if reaction_accumulator is None or reaction_count is None:
                continue

            for chain in grasper.sphere_chains:
                sphere_count = len(chain.world_points)
                if sphere_count == 0 or self.motion_samples <= 0:
                    continue

                wp.launch(
                    kernel=collide_triangles_vs_spheres,
                    dim=model.tri_count * sphere_count * self.motion_samples,
                    inputs=[
                        particle_q,
                        particle_qd,
                        model.particle_inv_mass,
                        model.tri_indices,
                        chain.world_points_prev,
                        chain.world_points,
                        chain.radii,
                        sphere_count,
                        self.motion_samples,
                        self.contact_margin,
                        0.0,
                        dt,
                        self._chain_cull_radii[id(chain)],
                    ],
                    outputs=[self._accumulator, self._count, reaction_accumulator, reaction_count],
                    device=model.device,
                )

        if not self.deferred_apply:
            wp.launch(
                kernel=apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[self._accumulator, self._count],
                outputs=[particle_deltas],
                device=model.device,
            )


class GrasperSphereTruncationSystem(SimulationSystem):
    """Truncate tissue motion against the rendered grasper jaw spheres before commit."""

    def __init__(
        self,
        *,
        graspers: dict,
        surface_vertex_ids: wp.array,
        motion_samples: int = GRASPER_COLLISION_MOTION_SAMPLES,
        contact_margin: float = GRASPER_COLLISION_MARGIN,
        safety: float = GRASPER_COLLISION_TRUNCATION_SAFETY,
        truncate_prediction: bool = True,
        priority: int = 85,
    ):
        super().__init__(priority=priority, stage=SolverStage.TRUNCATION)
        self.graspers = graspers
        self.surface_vertex_ids = surface_vertex_ids
        self.motion_samples = motion_samples
        self.contact_margin = contact_margin
        self.safety = safety
        self.truncate_prediction_enabled = truncate_prediction
        self._truncation_t: wp.array | None = None

    def initialize(self, model):
        self._truncation_t = wp.zeros(model.particle_count, dtype=wp.float32, device=model.device)

    def truncate_prediction(
        self,
        model,
        state_in,
        state_out,
        base_positions,
        displacements,
        dt,
    ):
        if not self.truncate_prediction_enabled or model.tri_count == 0 or self._truncation_t is None:
            return

        self._apply_truncation(
            model,
            base_positions=base_positions,
            displacements=displacements,
            use_motion_samples=True,
        )

    def truncate_deltas(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        dt,
        iteration,
    ):
        if model.tri_count == 0 or self._truncation_t is None:
            return

        self._apply_truncation(
            model,
            base_positions=particle_q,
            displacements=particle_deltas,
            use_motion_samples=False,
        )

    def _apply_truncation(
        self,
        model,
        *,
        base_positions: wp.array,
        displacements: wp.array,
        use_motion_samples: bool,
    ):
        if self._truncation_t is None:
            return

        self._truncation_t.fill_(1.0)
        sample_count = self.motion_samples if use_motion_samples else 1
        if sample_count <= 0 or len(self.surface_vertex_ids) == 0:
            return

        for grasper in self.graspers.values():
            if grasper is None:
                continue

            for chain in grasper.sphere_chains:
                sphere_count = len(chain.world_points)
                if sphere_count == 0:
                    continue

                centers_prev = chain.world_points_prev if use_motion_samples else chain.world_points
                centers = chain.world_points
                wp.launch(
                    kernel=compute_vertex_sphere_truncation_factors,
                    dim=len(self.surface_vertex_ids) * sphere_count * sample_count,
                    inputs=[
                        model.particle_flags,
                        base_positions,
                        self.surface_vertex_ids,
                        displacements,
                        centers_prev,
                        centers,
                        chain.radii,
                        sphere_count,
                        sample_count,
                        self.contact_margin,
                        self.safety,
                    ],
                    outputs=[self._truncation_t],
                    device=model.device,
                )

        wp.launch(
            kernel=apply_surface_vertex_truncation,
            dim=len(self.surface_vertex_ids),
            inputs=[self.surface_vertex_ids, self._truncation_t],
            outputs=[displacements],
            device=model.device,
        )


CONTROLLER_BINDINGS = {
    "right": ControllerBinding(controller_id="right", uses_physics_proxy=True),
    "left": ControllerBinding(controller_id="left", uses_physics_proxy=False),
}


def _resolve_texture_path(mesh_name: str) -> str | None:
    for candidate in TEXTURE_CANDIDATES.get(mesh_name, ()): 
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_mesh_textures(mesh_names) -> dict[str, str]:
    textures = {}
    for mesh_name in mesh_names:
        texture = _resolve_texture_path(mesh_name)
        if texture is not None:
            textures[mesh_name] = texture
    return textures


def _float_changed(current: float, target: float, eps: float = 1.0e-6) -> bool:
    return abs(float(current) - float(target)) > eps


def _clamp_grasper_collision_mode(mode: str) -> str:
    return mode if mode in GRASPER_COLLISION_MODES else GRASPER_COLLISION_MODES[0]


def _grasper_collision_mode_index(mode: str) -> int:
    return GRASPER_COLLISION_MODES.index(_clamp_grasper_collision_mode(mode))


def _grasper_collision_mode_from_index(index: int) -> str:
    clamped = int(np.clip(index, 0, len(GRASPER_COLLISION_MODES) - 1))
    return GRASPER_COLLISION_MODES[clamped]


def _ui_section_expanded(ui, label: str) -> bool:
    collapsing_header = getattr(ui, "collapsing_header", None)
    if collapsing_header is None:
        ui.text(label)
        return True

    state = collapsing_header(label)
    if isinstance(state, tuple):
        return bool(state[0])
    return bool(state)


def _format_vec3_readout(value) -> str:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 3:
        return "(0.0000, 0.0000, 0.0000)"
    return f"({array[0]:.4f}, {array[1]:.4f}, {array[2]:.4f})"


def _vec3_magnitude(value) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float32)))


class Runtime:
    """Phase 1/2 runtime for soft-body tissue scenes."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        scene_config: SceneConfig,
        haptic_config: HapticConfig,
        viewer_config: ViewerConfig,
        bounds_config: BoundsConfig,
    ):
        self.sim_config = sim_config
        self.scene_config = scene_config
        self.haptic_config = haptic_config
        self.device = wp.get_device()
        self.textures_enabled = viewer_config.textures_enabled
        self.show_tissue = True
        self.show_grasper_mesh = bool(scene_config.show_grasper_mesh)
        self.enable_grasper_collisions = bool(scene_config.enable_grasper_collisions)
        self.sky_enabled = bool(viewer_config.sky_enabled)
        self.shadows_enabled = bool(viewer_config.shadows_enabled)
        self.direct_render_enabled = bool(viewer_config.direct_render_enabled)
        self.controller_bindings = CONTROLLER_BINDINGS
        self.controller_states = self._create_controller_states()

        asset = load_scene_asset(scene_config)
        scene = build_scene(asset, scene_config, haptic_config, self.device)

        self.model = scene.model
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec4f, device=self.device)
        self.tissue_blend_damage = 0.0
        self.tissue_blend_coag = 0.0
        self.tissue_blend_blood = 0.0
        self._tissue_blend_dirty = False
        self.tissue_debug_mode = 0
        self.tissue_normal_strength = 0.65
        self.tissue_specular_scale = 0.35
        self.tissue_roughness_bias = 0.45
        self.tissue_ambient = 0.22
        self.tissue_rim_strength = 0.12
        self.tissue_wetness = 0.0
        self.tissue_wet_spec_scale = 1.0
        self.tissue_wet_roughness = 0.18
        self.tissue_subsurface_color = (0.8, 0.22, 0.16)
        self.tissue_subsurface_strength = 0.0
        self.tissue_blood_wetness = 1.0
        self.mesh_ranges = scene.mesh_ranges
        self.surface_indices = scene.surface_tri_indices
        self.surface_meshes = scene.surface_meshes
        self.uvs = scene.uvs
        self.mesh_textures = _resolve_mesh_textures(self.surface_meshes.keys())
        self.proxy: HapticProxyState = scene.haptic_proxy
        self.haptic_visual_radius = 0.5 * (HAPTIC_VISUAL_RADIUS_BASELINE + float(self.proxy.radius))
        surface_vertex_ids_np = np.unique(self.model.tri_indices.numpy().reshape(-1)).astype(np.int32, copy=False)
        self.surface_vertex_ids = wp.array(surface_vertex_ids_np, dtype=wp.int32, device=self.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sim_substeps = max(1, int(sim_config.substeps))
        self.iterations = max(1, int(sim_config.constraint_iterations))
        self.volume_stiffness = float(scene_config.volume_stiffness)
        self.spring_stiffness = float(scene_config.spring_stiffness)
        self.spring_damping = float(scene_config.spring_dampen)
        self.particle_max_velocity = float(self.model.particle_max_velocity)
        self.grasper_collision_mode = _clamp_grasper_collision_mode(sim_config.grasper_collision_mode)
        self.grasper_collision_motion_samples = max(1, int(sim_config.grasper_collision_motion_samples))
        self.grasper_collision_margin = max(0.0, float(sim_config.grasper_collision_margin))
        self.grasper_truncation_safety = float(np.clip(sim_config.grasper_truncation_safety, 0.5, 1.0))
        self.grasper_truncate_prediction = bool(sim_config.grasper_truncate_prediction)
        self._pending_substeps = self.sim_substeps
        self._pending_constraint_iterations = self.iterations
        self._pending_volume_stiffness = self.volume_stiffness
        self._pending_spring_stiffness = self.spring_stiffness
        self._pending_spring_damping = self.spring_damping
        self._pending_particle_max_velocity = self.particle_max_velocity
        self._pending_enable_grasper_collisions = self.enable_grasper_collisions
        self._pending_grasper_collision_mode = self.grasper_collision_mode
        self._pending_grasper_collision_motion_samples = self.grasper_collision_motion_samples
        self._pending_grasper_collision_margin = self.grasper_collision_margin
        self._pending_grasper_truncation_safety = self.grasper_truncation_safety
        self._pending_grasper_truncate_prediction = self.grasper_truncate_prediction
        self.sim_config.substeps = self.sim_substeps
        self.sim_config.constraint_iterations = self.iterations
        self.sim_config.grasper_collision_mode = self.grasper_collision_mode
        self.sim_config.grasper_collision_motion_samples = self.grasper_collision_motion_samples
        self.sim_config.grasper_collision_margin = self.grasper_collision_margin
        self.sim_config.grasper_truncation_safety = self.grasper_truncation_safety
        self.sim_config.grasper_truncate_prediction = self.grasper_truncate_prediction
        self.sim_config.substep_dt = self.sim_config.frame_dt / float(self.sim_substeps)

        self.solver = Phase1Solver(self.model, iterations=self.iterations)
        self.distance_system = DistanceConstraintSystem(priority=50)
        self.volume_system = VolumeConstraintSystem(stiffness=self.volume_stiffness, priority=60)
        self.haptic_collision_system = HapticSphereCollisionSystem(self.proxy, priority=80)
        self.haptic_truncation_system = HapticSphereTruncationSystem(
            self.proxy,
            surface_vertex_ids=self.surface_vertex_ids,
            motion_samples=self.grasper_collision_motion_samples,
            contact_margin=self.grasper_collision_margin,
            safety=self.grasper_truncation_safety,
            truncate_prediction=self.grasper_truncate_prediction,
            priority=80,
        )
        self.bounds_system = BoundsCollisionSystem(
            bounds_min=wp.vec3(*bounds_config.bounds_min),
            bounds_max=wp.vec3(*bounds_config.bounds_max),
            priority=100,
        )
        self.solver.register_system(self.distance_system)
        self.solver.register_system(self.volume_system)
        if len(self.model.tri_points_connectors) > 0:
            self.triangle_point_system = TrianglePointConstraintSystem(priority=70)
            self.solver.register_system(self.triangle_point_system)
        else:
            self.triangle_point_system = None
        self.solver.register_system(self.haptic_collision_system)
        self.solver.register_system(self.haptic_truncation_system)
        self.solver.register_system(self.bounds_system)

        self._extra_key_press_hooks: list = []
        self.renderer = RenderBridge(viewer_config, self.model, self.device)
        self._sync_tissue_material_params()
        self.renderer.set_input_callbacks(on_key_press=self._on_key_press)
        self.renderer.register_ui_callback(self.gui, position="side")
        self.graspers = {
            controller_id: load_kinematic_grasper(
                GRASPER_ASSET_PATH,
                self.device,
                scale=GRASPER_SCALE,
                jaw_sphere_count=GRASPER_JAW_SPHERE_COUNT,
                jaw_sphere_radius=GRASPER_JAW_SPHERE_RADIUS,
            )
            for controller_id in self.controller_bindings
        }
        self.grasper_collision_system = GrasperSphereCollisionSystem(
            graspers=self.graspers,
            proxy=self.proxy,
            motion_samples=self.grasper_collision_motion_samples,
            contact_margin=self.grasper_collision_margin,
            priority=85,
        )
        self.grasper_truncation_system = GrasperSphereTruncationSystem(
            graspers=self.graspers,
            surface_vertex_ids=self.surface_vertex_ids,
            motion_samples=self.grasper_collision_motion_samples,
            contact_margin=self.grasper_collision_margin,
            safety=self.grasper_truncation_safety,
            truncate_prediction=self.grasper_truncate_prediction,
            priority=85,
        )
        self.solver.register_system(self.grasper_truncation_system)
        self.solver.register_system(self.grasper_collision_system)
        self._apply_grasper_collision_mode(self.grasper_collision_mode)
        self._sync_grasper_collision_settings()
        self.sim_time = 0.0

        self._haptic_staging, self._haptic_staging_view = create_vec3_staging_buffer()
        self._haptic_render_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._initialize_haptic_feedback()
        self._profile_samples = defaultdict(list)
        self.profiling_enabled = True
        self.profiling_synchronize = False
        self.profiling_console_enabled = True
        self.profile_window = 120
        self.profile_report_interval = 1.0
        self._last_profile_report_time = time.perf_counter()

        self.use_cuda_graph = self.device.is_cuda and viewer_config.backend != "headless"
        self.graph = None
        self._capture_graph()

        self._frame_start = time.perf_counter()

    def _create_controller_states(self) -> dict[str, ControllerRuntimeState]:
        states: dict[str, ControllerRuntimeState] = {}
        for controller_id, binding in self.controller_bindings.items():
            render_position = None
            staging = None
            staging_view = None
            if not binding.uses_physics_proxy:
                render_position = wp.zeros(1, dtype=wp.vec3, device=self.device)
                staging, staging_view = create_vec3_staging_buffer()
            states[controller_id] = ControllerRuntimeState(
                render_position=render_position,
                staging=staging,
                staging_view=staging_view,
            )
        return states

    def _initialize_haptic_feedback(self):
        self.haptic_feedback_settings = HapticFeedbackSettings()
        self.haptic_feedback_diagnostics = HapticFeedbackDiagnostics()
        self._haptic_loaded_settings = copy_feedback_settings(self.haptic_feedback_settings)
        self._haptic_presets: list[HapticPresetRecord] = []
        self._haptic_selected_preset_index = 0
        self._haptic_active_preset_path: Path | None = None
        self._haptic_active_preset_name: str | None = None
        self._haptic_preset_dir = HAPTIC_PRESET_DIR
        self._haptic_new_preset_name = "custom"
        self._haptic_feedback_available = False
        self._force_feedback_compute_always = False
        self._haptic_force_commands = {
            PRIMARY_CONTROLLER_ID: np.zeros(3, dtype=np.float32),
        }
        self._haptic_algorithm: HapticAlgorithm = create_haptic_algorithm(
            self.haptic_feedback_settings.algorithm, self.haptic_feedback_settings
        )

        self._refresh_haptic_presets(show_status=False)
        default_index = self._find_haptic_preset_index(filename=DEFAULT_HAPTIC_PRESET_FILENAME)
        if default_index is not None:
            self._load_haptic_preset_by_index(default_index, show_status=False)
        else:
            self._sync_haptic_feedback_metadata()

    def _set_haptic_status(self, message: str = "", *, is_error: bool = False):
        self.haptic_feedback_diagnostics.status_message = message
        self.haptic_feedback_diagnostics.status_is_error = bool(message) and is_error

    def _sync_haptic_feedback_metadata(self):
        preset_name = self._haptic_active_preset_name or "Unsaved"
        self.haptic_feedback_diagnostics.current_preset_name = preset_name
        self.haptic_feedback_diagnostics.dirty = not settings_almost_equal(
            self.haptic_feedback_settings,
            self._haptic_loaded_settings,
        )
        self.haptic_feedback_diagnostics.force_feedback_available = bool(self._haptic_feedback_available)

    def _find_haptic_preset_index(
        self,
        *,
        path: Path | None = None,
        filename: str | None = None,
    ) -> int | None:
        normalized_path = str(path.resolve()) if path is not None and path.exists() else None
        normalized_filename = filename.lower() if filename is not None else None
        for index, preset in enumerate(self._haptic_presets):
            if normalized_path is not None and preset.path.exists() and str(preset.path.resolve()) == normalized_path:
                return index
            if normalized_filename is not None and preset.path.name.lower() == normalized_filename:
                return index
        return None

    def _refresh_haptic_presets(self, *, show_status: bool = True):
        selected_path = None
        if self._haptic_presets and 0 <= self._haptic_selected_preset_index < len(self._haptic_presets):
            selected_path = self._haptic_presets[self._haptic_selected_preset_index].path

        presets, errors = scan_haptic_presets(self._haptic_preset_dir)
        self._haptic_presets = presets

        preferred_index = None
        if selected_path is not None:
            preferred_index = self._find_haptic_preset_index(path=selected_path)
        if preferred_index is None and self._haptic_active_preset_path is not None:
            preferred_index = self._find_haptic_preset_index(path=self._haptic_active_preset_path)
        if preferred_index is None:
            preferred_index = self._find_haptic_preset_index(filename=DEFAULT_HAPTIC_PRESET_FILENAME)
        self._haptic_selected_preset_index = preferred_index if preferred_index is not None else 0

        if not show_status:
            return

        if errors:
            summary = errors[0]
            if len(errors) > 1:
                summary += f" (+{len(errors) - 1} more)"
            self._set_haptic_status(summary, is_error=True)
            return

        self._set_haptic_status(f"Refreshed {len(self._haptic_presets)} haptics preset(s).")

    def _load_haptic_preset_path(self, path: Path, *, show_status: bool = True) -> bool:
        try:
            preset = load_haptic_preset(path)
        except ValueError as exc:
            self._set_haptic_status(str(exc), is_error=True)
            return False

        self.haptic_feedback_settings = copy_feedback_settings(preset.settings)
        self._haptic_loaded_settings = copy_feedback_settings(preset.settings)
        self._haptic_active_preset_path = preset.path
        self._haptic_active_preset_name = preset.name
        self._haptic_new_preset_name = preset.name

        if self._find_haptic_preset_index(path=preset.path) is None:
            self._refresh_haptic_presets(show_status=False)
        selected_index = self._find_haptic_preset_index(path=preset.path)
        if selected_index is not None:
            self._haptic_selected_preset_index = selected_index

        device_position = None
        state = self.controller_states.get(PRIMARY_CONTROLLER_ID)
        if state is not None and state.sample_present:
            device_position = np.asarray(state.scaled_position, dtype=np.float32)
        self._reset_haptic_force_state(device_position=device_position)
        self._sync_haptic_feedback_metadata()
        if show_status:
            self._set_haptic_status(f'Loaded preset "{preset.name}".')
        return True

    def _load_haptic_preset_by_index(self, index: int, *, show_status: bool = True) -> bool:
        if index < 0 or index >= len(self._haptic_presets):
            self._set_haptic_status("No haptics preset is selected.", is_error=True)
            return False
        return self._load_haptic_preset_path(self._haptic_presets[index].path, show_status=show_status)

    def _save_haptic_preset_to_path(self, path: Path, name: str, *, show_status: bool = True) -> bool:
        try:
            save_haptic_preset(path, name, self.haptic_feedback_settings)
        except OSError as exc:
            self._set_haptic_status(f'Failed to save preset "{name}": {exc}', is_error=True)
            return False

        self._refresh_haptic_presets(show_status=False)
        self._load_haptic_preset_path(path, show_status=False)
        if show_status:
            self._set_haptic_status(f'Saved preset "{name}".')
        return True

    def _save_haptic_selected_preset(self) -> bool:
        if not self._haptic_presets:
            self._set_haptic_status("No haptics preset is selected.", is_error=True)
            return False
        selected = self._haptic_presets[self._haptic_selected_preset_index]
        return self._save_haptic_preset_to_path(selected.path, selected.name)

    def _save_haptic_preset_as_new(self, preset_name: str) -> bool:
        display_name = preset_name.strip()
        if not display_name:
            self._set_haptic_status("Preset name cannot be empty.", is_error=True)
            return False
        path = self._haptic_preset_dir / sanitize_preset_filename(display_name)
        return self._save_haptic_preset_to_path(path, display_name)

    def _reset_haptic_force_state(self, device_position: np.ndarray | None = None):
        self._haptic_algorithm.reset(device_position)
        self._haptic_force_commands[PRIMARY_CONTROLLER_ID] = np.zeros(3, dtype=np.float32)

    def _ensure_haptic_algorithm(self) -> None:
        desired = self.haptic_feedback_settings.algorithm
        current_name = getattr(self._haptic_algorithm, "ALGORITHM_NAME", None)
        if current_name != desired:
            self._haptic_algorithm = create_haptic_algorithm(
                desired, self.haptic_feedback_settings
            )
        else:
            self._haptic_algorithm.update_settings(self.haptic_feedback_settings)

    def _supports_haptic_force_feedback(self, source: InputRig | InputSource | None) -> bool:
        if source is None:
            return False
        supports = getattr(source, "supports_force_feedback", None)
        if supports is None:
            return False
        try:
            return bool(supports(PRIMARY_CONTROLLER_ID))
        except TypeError:
            return bool(supports())

    def _update_haptic_feedback_availability(self, source: InputRig | InputSource | None):
        self._haptic_feedback_available = self._supports_haptic_force_feedback(source)
        self.haptic_feedback_diagnostics.force_feedback_available = bool(self._haptic_feedback_available)

    def set_force_feedback_compute_always(self, enabled: bool) -> None:
        """Run the force-feedback computation path even when no live device is
        attached. Forces still won't be dispatched, but diagnostics are
        populated so telemetry can plot/log them."""
        self._force_feedback_compute_always = bool(enabled)

    def _update_haptic_force_feedback(self, avg_reaction_offset, contact_count: int):
        zero = np.zeros(3, dtype=np.float32)
        diagnostics = self.haptic_feedback_diagnostics
        settings = self.haptic_feedback_settings
        state = self.controller_states[PRIMARY_CONTROLLER_ID]

        reaction = np.asarray(avg_reaction_offset, dtype=np.float32).copy()
        diagnostics.contact_count = int(contact_count)
        diagnostics.avg_reaction_offset = reaction.copy()
        diagnostics.force_feedback_available = bool(self._haptic_feedback_available)

        self._ensure_haptic_algorithm()

        device_position = (
            np.asarray(state.scaled_position, dtype=np.float32).copy()
            if state.sample_present
            else zero.copy()
        )
        compute_allowed = self._haptic_feedback_available or self._force_feedback_compute_always
        if (
            not settings.enabled
            or not compute_allowed
            or not state.sample_present
            or int(contact_count) <= 0
        ):
            self._reset_haptic_force_state(device_position if state.sample_present else None)
            diagnostics.proxy_offset = zero.copy()
            diagnostics.raw_force = zero.copy()
            diagnostics.filtered_force = zero.copy()
            diagnostics.final_force = zero.copy()
            diagnostics.clamp_active = False
            diagnostics.slew_active = False
            return

        step_input = HapticStepInput(
            device_position=device_position,
            avg_reaction_offset=reaction,
            contact_count=int(contact_count),
            dt=max(self.sim_config.frame_dt, 1.0e-6),
        )
        result = self._haptic_algorithm.compute(step_input)

        self._haptic_force_commands[PRIMARY_CONTROLLER_ID] = result.force.copy()

        diagnostics.proxy_offset = result.proxy_offset
        diagnostics.raw_force = result.raw_force
        diagnostics.filtered_force = result.filtered_force
        diagnostics.final_force = result.force
        diagnostics.clamp_active = result.clamp_active
        diagnostics.slew_active = result.slew_active

    def get_haptic_force_commands(self) -> dict[str, np.ndarray]:
        return {
            controller_id: np.asarray(force, dtype=np.float32).copy()
            for controller_id, force in self._haptic_force_commands.items()
        }

    def _get_combined_haptic_reaction(self) -> tuple[np.ndarray, int]:
        weighted_sum = np.zeros(3, dtype=np.float32)
        total_count = 0

        for avg_reaction, contact_count in (
            self.haptic_collision_system.get_reaction_average(),
            self.grasper_collision_system.get_reaction_average(PRIMARY_CONTROLLER_ID),
        ):
            count = int(contact_count)
            if count <= 0:
                continue
            weighted_sum += np.asarray(avg_reaction, dtype=np.float32) * float(count)
            total_count += count

        if total_count <= 0:
            return np.zeros(3, dtype=np.float32), 0

        return weighted_sum / float(total_count), total_count

    def _needs_haptic_reaction_readback(self) -> bool:
        state = self.controller_states.get(PRIMARY_CONTROLLER_ID)
        if state is None or not state.sample_present:
            return False
        return bool(self._haptic_feedback_available or self._force_feedback_compute_always)

    def _render_haptics_ui(self, ui):
        if not _ui_section_expanded(ui, "Haptics"):
            return

        settings = self.haptic_feedback_settings
        diagnostics = self.haptic_feedback_diagnostics
        settings_changed = False

        changed, enabled = ui.checkbox("Enable Force Feedback", settings.enabled)
        if changed:
            settings.enabled = enabled
            settings_changed = True

        changed, reaction_scale = ui.slider_float("Reaction Scale", settings.reaction_scale, 0.0, 1000.0, "%.3f")
        if changed:
            settings.reaction_scale = reaction_scale
            settings_changed = True

        changed, proxy_follow = ui.slider_float("Proxy Follow", settings.proxy_follow, 0.0, 1.0, "%.3f")
        if changed:
            settings.proxy_follow = proxy_follow
            settings_changed = True

        changed, max_proxy_offset = ui.slider_float("Max Proxy Offset", settings.max_proxy_offset, 0.0, 1.0, "%.4f")
        if changed:
            settings.max_proxy_offset = max_proxy_offset
            settings_changed = True

        changed, spring_k = ui.slider_float("Spring K", settings.spring_k, 0.0, 1000.0, "%.4f")
        if changed:
            settings.spring_k = spring_k
            settings_changed = True

        changed, damper_b = ui.slider_float("Damper B", settings.damper_b, 0.0, 10.0, "%.4f")
        if changed:
            settings.damper_b = damper_b
            settings_changed = True

        changed, deadband = ui.slider_float("Deadband", settings.deadband, 0.0, 0.1, "%.4f")
        if changed:
            settings.deadband = deadband
            settings_changed = True

        changed, lowpass_alpha = ui.slider_float("Lowpass Alpha", settings.lowpass_alpha, 0.0, 1.0, "%.3f")
        if changed:
            settings.lowpass_alpha = lowpass_alpha
            settings_changed = True

        changed, max_force = ui.slider_float("Max Force", settings.max_force, 0.0, 10.0, "%.4f")
        if changed:
            settings.max_force = max_force
            settings_changed = True

        changed, slew_rate_limit = ui.slider_float("Slew Rate Limit", settings.slew_rate_limit, 0.0, 1000.0, "%.3f")
        if changed:
            settings.slew_rate_limit = slew_rate_limit
            settings_changed = True

        if settings_changed:
            self._sync_haptic_feedback_metadata()

        ui.separator()
        ui.text("Presets")
        if self._haptic_presets:
            max_index = len(self._haptic_presets) - 1
            selected_index = int(np.clip(self._haptic_selected_preset_index, 0, max_index))
            changed, selected_index = ui.slider_int("Preset Index", selected_index, 0, max_index)
            if changed:
                self._haptic_selected_preset_index = selected_index
            selected = self._haptic_presets[self._haptic_selected_preset_index]
            ui.text(f'Selected: {selected.name} ({selected.path.name})')
        else:
            ui.text("Selected: none")

        same_line = getattr(ui, "same_line", None)

        if ui.button("Load Selected"):
            self._load_haptic_preset_by_index(self._haptic_selected_preset_index)
        if same_line is not None:
            same_line()
        if ui.button("Save Selected"):
            self._save_haptic_selected_preset()
        if same_line is not None:
            same_line()
        if ui.button("Refresh"):
            self._refresh_haptic_presets()

        input_text = getattr(ui, "input_text", None)
        if input_text is not None:
            changed, preset_name = input_text("New Preset Name", self._haptic_new_preset_name, 128)
            if changed:
                self._haptic_new_preset_name = preset_name
        else:
            ui.text(f"New Preset Name: {self._haptic_new_preset_name}")

        if ui.button("Save As New"):
            self._save_haptic_preset_as_new(self._haptic_new_preset_name)
        if same_line is not None:
            same_line()
        if ui.button("Revert"):
            self.haptic_feedback_settings = copy_feedback_settings(self._haptic_loaded_settings)
            state = self.controller_states.get(PRIMARY_CONTROLLER_ID)
            device_position = None
            if state is not None and state.sample_present:
                device_position = np.asarray(state.scaled_position, dtype=np.float32)
            self._reset_haptic_force_state(device_position=device_position)
            self._sync_haptic_feedback_metadata()
            self._set_haptic_status("Reverted unsaved haptics tuning changes.")
        if same_line is not None:
            same_line()
        if ui.button("Reset Defaults"):
            self.haptic_feedback_settings = HapticFeedbackSettings()
            state = self.controller_states.get(PRIMARY_CONTROLLER_ID)
            device_position = None
            if state is not None and state.sample_present:
                device_position = np.asarray(state.scaled_position, dtype=np.float32)
            self._reset_haptic_force_state(device_position=device_position)
            self._sync_haptic_feedback_metadata()
            self._set_haptic_status("Reset haptics tuning to defaults.")

        ui.separator()
        ui.text(f"Active Preset: {diagnostics.current_preset_name}")
        ui.text(f"Dirty: {'yes' if diagnostics.dirty else 'no'}")
        ui.text(f"Force Feedback Available: {'yes' if diagnostics.force_feedback_available else 'no'}")
        ui.text(f"Contact Count: {diagnostics.contact_count}")
        ui.text(
            f"Avg Reaction Offset: {_format_vec3_readout(diagnostics.avg_reaction_offset)} | "
            f"|r|={_vec3_magnitude(diagnostics.avg_reaction_offset):.4f}"
        )
        ui.text(
            f"Proxy Offset: {_format_vec3_readout(diagnostics.proxy_offset)} | "
            f"|p|={_vec3_magnitude(diagnostics.proxy_offset):.4f}"
        )
        ui.text(
            f"Raw Force: {_format_vec3_readout(diagnostics.raw_force)} | "
            f"|F|={_vec3_magnitude(diagnostics.raw_force):.4f}"
        )
        ui.text(
            f"Filtered Force: {_format_vec3_readout(diagnostics.filtered_force)} | "
            f"|F|={_vec3_magnitude(diagnostics.filtered_force):.4f}"
        )
        ui.text(
            f"Final Force: {_format_vec3_readout(diagnostics.final_force)} | "
            f"|F|={_vec3_magnitude(diagnostics.final_force):.4f}"
        )
        ui.text(f"Clamp Active: {'yes' if diagnostics.clamp_active else 'no'}")
        ui.text(f"Slew Active: {'yes' if diagnostics.slew_active else 'no'}")
        if diagnostics.status_message:
            prefix = "Error" if diagnostics.status_is_error else "Status"
            ui.text(f"{prefix}: {diagnostics.status_message}")

    def _apply_grasper_collision_mode(self, mode: str):
        self.grasper_collision_mode = _clamp_grasper_collision_mode(mode)
        projection_enabled = self.grasper_collision_mode in ("projection", "hybrid")
        truncation_enabled = self.grasper_collision_mode in ("truncation", "hybrid")
        self.haptic_collision_system.enabled = projection_enabled
        self.haptic_truncation_system.enabled = truncation_enabled
        self.grasper_collision_system.enabled = self.enable_grasper_collisions and projection_enabled
        self.grasper_truncation_system.enabled = self.enable_grasper_collisions and truncation_enabled

    def _sync_grasper_collision_settings(self):
        self.haptic_truncation_system.motion_samples = self.grasper_collision_motion_samples
        self.haptic_truncation_system.contact_margin = self.grasper_collision_margin
        self.haptic_truncation_system.safety = self.grasper_truncation_safety
        self.haptic_truncation_system.truncate_prediction_enabled = self.grasper_truncate_prediction
        self.grasper_collision_system.motion_samples = self.grasper_collision_motion_samples
        self.grasper_collision_system.contact_margin = self.grasper_collision_margin
        self.grasper_collision_system._rebuild_chain_cull_radii()
        self.grasper_truncation_system.motion_samples = self.grasper_collision_motion_samples
        self.grasper_truncation_system.contact_margin = self.grasper_collision_margin
        self.grasper_truncation_system.safety = self.grasper_truncation_safety
        self.grasper_truncation_system.truncate_prediction_enabled = self.grasper_truncate_prediction

    def _capture_graph(self):
        self.graph = None
        if not self.use_cuda_graph:
            return

        with wp.ScopedCapture() as capture:
            self._simulate_step()
        self.graph = capture.graph

    def _trim_profile_samples(self):
        for values in self._profile_samples.values():
            overflow = len(values) - self.profile_window
            if overflow > 0:
                del values[:overflow]

    def _profile_last_ms(self, name: str) -> float | None:
        values = self._profile_samples.get(name)
        if not values:
            return None
        return float(values[-1])

    def _profile_avg_ms(self, name: str) -> float | None:
        values = self._profile_samples.get(name)
        if not values:
            return None
        return float(sum(values) / len(values))

    def _report_profile_stats(self):
        if not self.profiling_enabled or not self.profiling_console_enabled:
            return

        now = time.perf_counter()
        if now - self._last_profile_report_time < self.profile_report_interval:
            return

        solver_latest = self._profile_last_ms("solver_loop")
        render_latest = self._profile_last_ms("render")
        if solver_latest is None or render_latest is None:
            return

        solver_avg = self._profile_avg_ms("solver_loop")
        render_avg = self._profile_avg_ms("render")
        frame_latest = solver_latest + render_latest
        frame_avg = (solver_avg or 0.0) + (render_avg or 0.0)
        print(
            "[profile] "
            f"solver {solver_latest:.2f} ms latest / {solver_avg:.2f} ms avg | "
            f"render {render_latest:.2f} ms latest / {render_avg:.2f} ms avg | "
            f"frame {frame_latest:.2f} ms latest / {frame_avg:.2f} ms avg"
        )
        self._last_profile_report_time = now

    def _has_pending_solver_changes(self) -> bool:
        return (
            self._pending_substeps != self.sim_substeps
            or self._pending_constraint_iterations != self.iterations
            or _float_changed(self._pending_volume_stiffness, self.volume_stiffness)
            or _float_changed(self._pending_spring_stiffness, self.spring_stiffness)
            or _float_changed(self._pending_spring_damping, self.spring_damping)
            or _float_changed(self._pending_particle_max_velocity, self.particle_max_velocity)
            or self._pending_enable_grasper_collisions != self.enable_grasper_collisions
            or self._pending_grasper_collision_mode != self.grasper_collision_mode
            or self._pending_grasper_collision_motion_samples != self.grasper_collision_motion_samples
            or _float_changed(self._pending_grasper_collision_margin, self.grasper_collision_margin)
            or _float_changed(self._pending_grasper_truncation_safety, self.grasper_truncation_safety)
            or self._pending_grasper_truncate_prediction != self.grasper_truncate_prediction
        )

    def _pending_graph_rebuild_needed(self) -> bool:
        return (
            self._pending_substeps != self.sim_substeps
            or self._pending_constraint_iterations != self.iterations
            or _float_changed(self._pending_volume_stiffness, self.volume_stiffness)
            or _float_changed(self._pending_particle_max_velocity, self.particle_max_velocity)
            or self._pending_enable_grasper_collisions != self.enable_grasper_collisions
            or self._pending_grasper_collision_mode != self.grasper_collision_mode
            or self._pending_grasper_collision_motion_samples != self.grasper_collision_motion_samples
            or _float_changed(self._pending_grasper_collision_margin, self.grasper_collision_margin)
            or _float_changed(self._pending_grasper_truncation_safety, self.grasper_truncation_safety)
            or self._pending_grasper_truncate_prediction != self.grasper_truncate_prediction
        )

    def _apply_pending_solver_settings(self):
        substeps = max(1, int(self._pending_substeps))
        iterations = max(1, int(self._pending_constraint_iterations))
        volume_stiffness = max(0.0, float(self._pending_volume_stiffness))
        spring_stiffness = max(0.0, float(self._pending_spring_stiffness))
        spring_damping = max(0.0, float(self._pending_spring_damping))
        particle_max_velocity = max(0.1, float(self._pending_particle_max_velocity))
        enable_grasper_collisions = bool(self._pending_enable_grasper_collisions)
        grasper_collision_mode = _clamp_grasper_collision_mode(self._pending_grasper_collision_mode)
        grasper_collision_motion_samples = max(1, int(self._pending_grasper_collision_motion_samples))
        grasper_collision_margin = max(0.0, float(self._pending_grasper_collision_margin))
        grasper_truncation_safety = float(np.clip(self._pending_grasper_truncation_safety, 0.5, 1.0))
        grasper_truncate_prediction = bool(self._pending_grasper_truncate_prediction)
        rebuild_graph = False
        sync_grasper_collision_settings = False

        if substeps != self.sim_substeps:
            self.sim_substeps = substeps
            self.sim_config.substeps = substeps
            self.sim_config.substep_dt = self.sim_config.frame_dt / float(substeps)
            rebuild_graph = True

        if iterations != self.iterations:
            self.iterations = iterations
            self.sim_config.constraint_iterations = iterations
            self.solver.iterations = iterations
            rebuild_graph = True

        if _float_changed(volume_stiffness, self.volume_stiffness):
            self.volume_stiffness = volume_stiffness
            self.scene_config.volume_stiffness = volume_stiffness
            self.volume_system.stiffness = volume_stiffness
            rebuild_graph = True

        if _float_changed(spring_stiffness, self.spring_stiffness):
            self.spring_stiffness = spring_stiffness
            self.scene_config.spring_stiffness = spring_stiffness
            if getattr(self.model, "spring_count", 0) > 0:
                self.model.spring_stiffness.fill_(spring_stiffness)

        if _float_changed(spring_damping, self.spring_damping):
            self.spring_damping = spring_damping
            self.scene_config.spring_dampen = spring_damping
            if getattr(self.model, "spring_count", 0) > 0:
                self.model.spring_damping.fill_(spring_damping)

        if _float_changed(particle_max_velocity, self.particle_max_velocity):
            self.particle_max_velocity = particle_max_velocity
            self.model.particle_max_velocity = particle_max_velocity
            rebuild_graph = True

        if enable_grasper_collisions != self.enable_grasper_collisions:
            self.enable_grasper_collisions = enable_grasper_collisions
            self.scene_config.enable_grasper_collisions = enable_grasper_collisions
            self._apply_grasper_collision_mode(self.grasper_collision_mode)
            rebuild_graph = True
            sync_grasper_collision_settings = True

        if grasper_collision_mode != self.grasper_collision_mode:
            self._apply_grasper_collision_mode(grasper_collision_mode)
            self.sim_config.grasper_collision_mode = self.grasper_collision_mode
            rebuild_graph = True
            sync_grasper_collision_settings = True

        if grasper_collision_motion_samples != self.grasper_collision_motion_samples:
            self.grasper_collision_motion_samples = grasper_collision_motion_samples
            self.sim_config.grasper_collision_motion_samples = grasper_collision_motion_samples
            rebuild_graph = True
            sync_grasper_collision_settings = True

        if _float_changed(grasper_collision_margin, self.grasper_collision_margin):
            self.grasper_collision_margin = grasper_collision_margin
            self.sim_config.grasper_collision_margin = grasper_collision_margin
            rebuild_graph = True
            sync_grasper_collision_settings = True

        if _float_changed(grasper_truncation_safety, self.grasper_truncation_safety):
            self.grasper_truncation_safety = grasper_truncation_safety
            self.sim_config.grasper_truncation_safety = grasper_truncation_safety
            rebuild_graph = True
            sync_grasper_collision_settings = True

        if grasper_truncate_prediction != self.grasper_truncate_prediction:
            self.grasper_truncate_prediction = grasper_truncate_prediction
            self.sim_config.grasper_truncate_prediction = grasper_truncate_prediction
            rebuild_graph = True
            sync_grasper_collision_settings = True

        if sync_grasper_collision_settings:
            self._sync_grasper_collision_settings()

        if rebuild_graph:
            self._capture_graph()

    def poll_input(self, source: InputRig | InputSource):
        self._update_haptic_feedback_availability(source)
        controller_samples = self._poll_controller_samples(source)
        for controller_id, binding in self.controller_bindings.items():
            self._apply_controller_sample(controller_id, binding, controller_samples.get(controller_id))

    def _poll_controller_samples(self, source: InputRig | InputSource):
        if isinstance(source, InputRig):
            return source.poll()

        from omnisurg.input.sources import ControllerSample

        sample = ControllerSample.from_sample_dict(source.poll())
        if sample.active or sample.button or sample.grip > 0.0:
            return {PRIMARY_CONTROLLER_ID: sample}
        return {}

    def _apply_controller_sample(self, controller_id: str, binding: ControllerBinding, sample):
        state = self.controller_states[controller_id]
        grasper = self.graspers.get(controller_id)

        if sample is None:
            state.sample_present = False
            if binding.uses_physics_proxy:
                wp.copy(self.proxy.center_prev, self.proxy.center_target)
            state.button = False
            state.grip = 0.0
            return

        state.sample_present = sample.position is not None
        state.button = bool(sample.button)
        state.grip = float(np.clip(sample.grip, 0.0, 1.0))
        if sample.rotation is not None:
            state.rotation = np.array(sample.rotation, dtype=np.float32)
            state.active = True

        if sample.position is None:
            if binding.uses_physics_proxy:
                wp.copy(self.proxy.center_prev, self.proxy.center_target)
            if grasper is not None and state.active:
                grasper.set_root_pose(state.scaled_position, state.rotation)
            return

        raw_with_offset, scaled = self._map_haptic_position(sample.position)
        state.scaled_position = scaled
        state.active = True

        if binding.uses_physics_proxy:
            wp.copy(self.proxy.center_prev, self.proxy.center_target)
            self._haptic_staging_view[0] = raw_with_offset.tolist()
            wp.copy(self.proxy.center_target, self._haptic_staging)
        else:
            state.staging_view[0] = scaled.tolist()
            wp.copy(state.render_position, state.staging)

        if grasper is not None:
            grasper.set_root_pose(state.scaled_position, state.rotation)

    def _map_haptic_position(self, raw_position) -> tuple[np.ndarray, np.ndarray]:
        offset = np.array(self.haptic_config.position_offset, dtype=np.float32)
        mapped = np.array(raw_position, dtype=np.float32) + offset
        scaled = mapped * float(self.haptic_config.position_scale)
        return mapped, scaled

    def _simulate_step(self):
        scale = self.haptic_config.position_scale
        for i in range(self.sim_config.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            factor = float(i) / float(self.sim_config.substeps)
            wp.launch(
                update_haptic_proxy,
                dim=1,
                inputs=[
                    self.proxy.center_prev,
                    self.proxy.center_target,
                    self.proxy.center_current,
                    self.proxy.center_scaled_prev,
                    self.proxy.center_scaled,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.proxy.body_id,
                    factor,
                    scale,
                    self.sim_config.substep_dt,
                ],
                device=self.device,
            )

            for grasper in self.graspers.values():
                if grasper is None:
                    continue
                grasper.update_substep_pose(factor)
                if self.enable_grasper_collisions:
                    grasper.update_collision_geometry()

            self.solver.step(self.state_0, self.state_1, None, None, self.sim_config.substep_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def register_key_press_hook(self, callback):
        self._extra_key_press_hooks.append(callback)

    def _on_key_press(self, symbol: int, modifiers: int):
        try:
            import pyglet
        except Exception:
            return

        if symbol == pyglet.window.key.T:
            self.toggle_textures()

        for hook in self._extra_key_press_hooks:
            try:
                hook(symbol, modifiers)
            except Exception as exc:
                print(f"Key press hook raised: {exc}")

    def toggle_textures(self):
        self.textures_enabled = not self.textures_enabled
        print(f"Textures {'on' if self.textures_enabled else 'off'}")

    def _set_tissue_blend_channel(self, attr: str, value: float) -> None:
        clamped = float(np.clip(value, 0.0, 1.0))
        if _float_changed(getattr(self, attr), clamped):
            setattr(self, attr, clamped)
            self._tissue_blend_dirty = True

    def _reset_tissue_blend_channels(self) -> None:
        if (
            _float_changed(self.tissue_blend_damage, 0.0)
            or _float_changed(self.tissue_blend_coag, 0.0)
            or _float_changed(self.tissue_blend_blood, 0.0)
        ):
            self.tissue_blend_damage = 0.0
            self.tissue_blend_coag = 0.0
            self.tissue_blend_blood = 0.0
            self._tissue_blend_dirty = True

    def _sync_tissue_blend_vertex_colors(self) -> None:
        if not self._tissue_blend_dirty or len(self.vertex_colors) == 0:
            return

        wp.launch(
            fill_tissue_vertex_blend_colors,
            dim=len(self.vertex_colors),
            inputs=[
                self.vertex_colors,
                np.float32(self.tissue_blend_damage),
                np.float32(self.tissue_blend_coag),
                np.float32(self.tissue_blend_blood),
            ],
            device=self.device,
        )
        self._tissue_blend_dirty = False

    def _sync_tissue_material_params(self) -> None:
        renderer = getattr(self, "renderer", None)
        if renderer is None:
            return
        renderer.set_tissue_material_params(
            debug_mode=self.tissue_debug_mode,
            normal_strength=self.tissue_normal_strength,
            specular_scale=self.tissue_specular_scale,
            roughness_bias=self.tissue_roughness_bias,
            ambient=self.tissue_ambient,
            rim_strength=self.tissue_rim_strength,
            wetness=self.tissue_wetness,
            wet_spec_scale=self.tissue_wet_spec_scale,
            wet_roughness=self.tissue_wet_roughness,
            subsurface_color=self.tissue_subsurface_color,
            subsurface_strength=self.tissue_subsurface_strength,
            blood_wetness=self.tissue_blood_wetness,
        )

    def _set_tissue_debug_mode(self, value: int) -> None:
        mode = int(np.clip(value, 0, len(TISSUE_DEBUG_MODE_LABELS) - 1))
        if mode != self.tissue_debug_mode:
            self.tissue_debug_mode = mode
            self._sync_tissue_material_params()

    def _set_tissue_material_float(self, attr: str, value: float, min_value: float, max_value: float) -> None:
        clamped = float(np.clip(value, min_value, max_value))
        if _float_changed(getattr(self, attr), clamped):
            setattr(self, attr, clamped)
            self._sync_tissue_material_params()

    def _render_tissue_debug_ui(self, ui) -> None:
        max_debug_mode = len(TISSUE_DEBUG_MODE_LABELS) - 1
        current_mode = int(np.clip(self.tissue_debug_mode, 0, max_debug_mode))
        if current_mode != self.tissue_debug_mode:
            self.tissue_debug_mode = current_mode
            self._sync_tissue_material_params()

        combo = getattr(ui, "combo", None)
        if callable(combo):
            changed, debug_mode = combo(
                "Tissue Debug View",
                current_mode,
                TISSUE_DEBUG_MODE_LABELS,
            )
        else:
            changed, debug_mode = ui.slider_int(
                "Tissue Debug View",
                current_mode,
                0,
                max_debug_mode,
            )

        if changed:
            self._set_tissue_debug_mode(debug_mode)
            current_mode = self.tissue_debug_mode

        ui.text(f"Active: {TISSUE_DEBUG_MODE_LABELS[current_mode]}")
        if current_mode != 0 and ui.button("Show Final Tissue"):
            self._set_tissue_debug_mode(0)

    def _render_tissue_material_ui(self, ui) -> None:
        ui.text("Tissue Material")
        self._render_tissue_debug_ui(ui)

        changed, normal_strength = ui.slider_float(
            "Normal Strength",
            self.tissue_normal_strength,
            0.0,
            2.0,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_normal_strength", normal_strength, 0.0, 2.0)

        changed, specular_scale = ui.slider_float(
            "Specular Scale",
            self.tissue_specular_scale,
            0.0,
            2.0,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_specular_scale", specular_scale, 0.0, 2.0)

        changed, roughness_bias = ui.slider_float(
            "Roughness Bias",
            self.tissue_roughness_bias,
            0.0,
            1.0,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_roughness_bias", roughness_bias, 0.0, 1.0)

        changed, wetness = ui.slider_float(
            "Wetness",
            self.tissue_wetness,
            0.0,
            1.0,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_wetness", wetness, 0.0, 1.0)

        changed, wet_spec_scale = ui.slider_float(
            "Wet Spec Scale",
            self.tissue_wet_spec_scale,
            0.0,
            4.0,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_wet_spec_scale", wet_spec_scale, 0.0, 4.0)

        changed, wet_roughness = ui.slider_float(
            "Wet Roughness",
            self.tissue_wet_roughness,
            0.02,
            0.6,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_wet_roughness", wet_roughness, 0.02, 0.6)

        changed, blood_wetness = ui.slider_float(
            "Blood Wetness",
            self.tissue_blood_wetness,
            0.0,
            2.0,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_blood_wetness", blood_wetness, 0.0, 2.0)

        changed, ambient = ui.slider_float("Ambient", self.tissue_ambient, 0.0, 1.0, "%.2f")
        if changed:
            self._set_tissue_material_float("tissue_ambient", ambient, 0.0, 1.0)

        changed, rim_strength = ui.slider_float(
            "Rim Strength",
            self.tissue_rim_strength,
            0.0,
            1.0,
            "%.2f",
        )
        if changed:
            self._set_tissue_material_float("tissue_rim_strength", rim_strength, 0.0, 1.0)

    def gui(self, ui):
        ui.text("Solver")
        changed, substeps = ui.slider_int("Substeps", self._pending_substeps, 1, 64)
        if changed:
            self._pending_substeps = substeps

        changed, iterations = ui.slider_int("Constraint Iterations", self._pending_constraint_iterations, 1, 32)
        if changed:
            self._pending_constraint_iterations = iterations

        ui.separator()
        ui.text("Constraints")
        changed, volume_stiffness = ui.slider_float("Volume Stiffness", self._pending_volume_stiffness, 0.0, 1.0, "%.3f")
        if changed:
            self._pending_volume_stiffness = volume_stiffness

        changed, spring_stiffness = ui.slider_float("Spring Stiffness", self._pending_spring_stiffness, 0.0, 5.0, "%.3f")
        if changed:
            self._pending_spring_stiffness = spring_stiffness

        changed, spring_damping = ui.slider_float("Spring Damping", self._pending_spring_damping, 0.0, 2.0, "%.3f")
        if changed:
            self._pending_spring_damping = spring_damping

        ui.separator()
        changed, particle_max_velocity = ui.slider_float("Particle Max Velocity", self._pending_particle_max_velocity, 0.1, 50.0, "%.2f")
        if changed:
            self._pending_particle_max_velocity = particle_max_velocity

        ui.separator()
        ui.text("Collision")
        changed, enable_grasper_collisions = ui.checkbox("Enable Jaw Collisions", self._pending_enable_grasper_collisions)
        if changed:
            self._pending_enable_grasper_collisions = enable_grasper_collisions
        if not self._pending_enable_grasper_collisions:
            ui.text("Using only the haptic proxy sphere.")

        changed, collision_mode_index = ui.slider_int(
            "Sphere Collision Mode",
            _grasper_collision_mode_index(self._pending_grasper_collision_mode),
            0,
            len(GRASPER_COLLISION_MODES) - 1,
        )
        if changed:
            self._pending_grasper_collision_mode = _grasper_collision_mode_from_index(collision_mode_index)
        ui.text("0 Projection | 1 Truncation | 2 Hybrid")
        ui.text(f"Current: {GRASPER_COLLISION_MODE_LABELS[_grasper_collision_mode_index(self._pending_grasper_collision_mode)]}")

        changed, grasper_motion_samples = ui.slider_int(
            "Sphere Motion Samples",
            self._pending_grasper_collision_motion_samples,
            1,
            16,
        )
        if changed:
            self._pending_grasper_collision_motion_samples = grasper_motion_samples

        changed, grasper_collision_margin = ui.slider_float(
            "Sphere Collision Margin",
            self._pending_grasper_collision_margin,
            0.0,
            0.02,
            "%.3f",
        )
        if changed:
            self._pending_grasper_collision_margin = grasper_collision_margin

        changed, grasper_truncation_safety = ui.slider_float(
            "Sphere Truncation Safety",
            self._pending_grasper_truncation_safety,
            0.5,
            1.0,
            "%.2f",
        )
        if changed:
            self._pending_grasper_truncation_safety = grasper_truncation_safety

        changed, truncate_prediction = ui.checkbox(
            "Truncate Prediction Step",
            self._pending_grasper_truncate_prediction,
        )
        if changed:
            self._pending_grasper_truncate_prediction = truncate_prediction

        preview_dt_ms = (self.sim_config.frame_dt / float(max(1, int(self._pending_substeps)))) * 1000.0
        ui.text(f"Substep dt: {preview_dt_ms:.3f} ms")
        if self._has_pending_solver_changes():
            message = "Changes apply on the next frame."
            if self.use_cuda_graph and self._pending_graph_rebuild_needed():
                message += " CUDA graph will be rebuilt."
            ui.text(message)

        ui.separator()
        self._render_haptics_ui(ui)

        ui.separator()
        ui.text("Rendering")
        changed, show_tissue = ui.checkbox("Show Tissue", self.show_tissue)
        if changed:
            self.show_tissue = show_tissue

        ui.text("Tissue Blend")
        changed, damage_blend = ui.slider_float("Damage Blend", self.tissue_blend_damage, 0.0, 1.0, "%.2f")
        if changed:
            self._set_tissue_blend_channel("tissue_blend_damage", damage_blend)

        changed, coag_blend = ui.slider_float("Coag Blend", self.tissue_blend_coag, 0.0, 1.0, "%.2f")
        if changed:
            self._set_tissue_blend_channel("tissue_blend_coag", coag_blend)

        changed, blood_blend = ui.slider_float("Blood Blend", self.tissue_blend_blood, 0.0, 1.0, "%.2f")
        if changed:
            self._set_tissue_blend_channel("tissue_blend_blood", blood_blend)

        if ui.button("Reset Tissue Blend"):
            self._reset_tissue_blend_channels()

        self._render_tissue_material_ui(ui)

        changed, sky_enabled = ui.checkbox("Sky", self.sky_enabled)
        if changed:
            self.sky_enabled = sky_enabled
            self.renderer.configure_render_quality(sky_enabled=self.sky_enabled)

        changed, shadows_enabled = ui.checkbox("Shadows", self.shadows_enabled)
        if changed:
            self.shadows_enabled = shadows_enabled
            self.renderer.configure_render_quality(
                shadows_enabled=self.shadows_enabled,
                direct_render_enabled=self.direct_render_enabled,
            )

        changed, direct_render_enabled = ui.checkbox("Direct Render", self.direct_render_enabled)
        if changed:
            self.direct_render_enabled = direct_render_enabled
            self.renderer.configure_render_quality(direct_render_enabled=self.direct_render_enabled)

        ui.separator()
        ui.text("Profiling")
        changed, profiling_enabled = ui.checkbox("Enable Timers", self.profiling_enabled)
        if changed:
            self.profiling_enabled = profiling_enabled

        changed, profiling_synchronize = ui.checkbox("Sync GPU Timers", self.profiling_synchronize)
        if changed:
            self.profiling_synchronize = profiling_synchronize

        changed, profiling_console_enabled = ui.checkbox("Console Output", self.profiling_console_enabled)
        if changed:
            self.profiling_console_enabled = profiling_console_enabled

        changed, profile_report_interval = ui.slider_float("Console Interval", self.profile_report_interval, 0.1, 5.0, "%.1f s")
        if changed:
            self.profile_report_interval = profile_report_interval

        ui.text(f"Window: {self.profile_window} frames")
        if self.use_cuda_graph:
            ui.text("Solver timer measures CUDA graph replay.")

        for label, timer_name in (("Solver Loop", "solver_loop"), ("Render", "render")):
            latest = self._profile_last_ms(timer_name)
            if latest is None:
                ui.text(f"{label}: waiting for samples")
                continue
            avg = self._profile_avg_ms(timer_name)
            ui.text(f"{label}: {latest:.2f} ms latest / {avg:.2f} ms avg")

    def step(self):
        self._apply_pending_solver_settings()
        for controller_id, grasper in self.graspers.items():
            state = self.controller_states[controller_id]
            if grasper is not None and state.active:
                grasper.advance(self.sim_config.frame_dt, state.grip)

        self.haptic_collision_system.clear_reaction()
        self.grasper_collision_system.clear_reaction()
        with wp.ScopedTimer(
            "solver_loop",
            active=self.profiling_enabled,
            print=False,
            dict=self._profile_samples,
            synchronize=self.profiling_synchronize,
        ):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self._simulate_step()

        if self._needs_haptic_reaction_readback():
            reaction_offset, contact_count = self._get_combined_haptic_reaction()
        else:
            reaction_offset, contact_count = np.zeros(3, dtype=np.float32), 0
        self._update_haptic_force_feedback(reaction_offset, contact_count)
        self._trim_profile_samples()
        self.sim_time += self.sim_config.frame_dt

    def render(self):
        with wp.ScopedTimer(
            "render",
            active=self.profiling_enabled,
            print=False,
            dict=self._profile_samples,
            synchronize=self.profiling_synchronize,
        ):
            wp.launch(
                scale_position,
                dim=1,
                inputs=[self.proxy.center_current, self._haptic_render_pos, self.haptic_config.position_scale],
                device=self.device,
            )

            for controller_id, binding in self.controller_bindings.items():
                state = self.controller_states[controller_id]
                grasper = self.graspers.get(controller_id)
                if not self.show_grasper_mesh or grasper is None or not state.active:
                    continue

                grasper.update_render_geometry()

            self._sync_tissue_blend_vertex_colors()
            self.renderer.begin_frame(self.sim_time)
            if self.show_tissue:
                if self.surface_meshes:
                    for mesh_name, indices in self.surface_meshes.items():
                        texture = self.mesh_textures.get(mesh_name) if self.textures_enabled else None
                        self.renderer.draw_mesh(
                            mesh_name,
                            self.state_0.particle_q,
                            indices,
                            color=None if texture else MESH_COLORS.get(mesh_name, MESH_COLORS["tissue"]),
                            uvs=self.uvs if texture else None,
                            texture=texture,
                            vertex_colors=self.vertex_colors if texture else None,
                        )
                else:
                    texture = self.mesh_textures.get("tissue") if self.textures_enabled else None
                    self.renderer.draw_mesh(
                        "tissue",
                        self.state_0.particle_q,
                        self.surface_indices,
                        color=MESH_COLORS["tissue"] if texture is None else None,
                        uvs=self.uvs if texture else None,
                        texture=texture,
                        vertex_colors=self.vertex_colors if texture else None,
                    )

            if self.show_grasper_mesh:
                for controller_id, grasper in self.graspers.items():
                    state = self.controller_states[controller_id]
                    if grasper is not None and state.active:
                        grasper.render(
                            self.renderer,
                            prefix=f"{controller_id}_grasper",
                            draw_collision_spheres=False,
                        )

            if self.controller_states[PRIMARY_CONTROLLER_ID].active:
                self.renderer.draw_haptic_sphere(self._haptic_render_pos, radius=self.haptic_visual_radius)
            for controller_id, color in CONTROLLER_ROOT_COLORS.items():
                state = self.controller_states[controller_id]
                if state.active and state.render_position is not None:
                    self.renderer.draw_points(
                        f"{controller_id}_controller_root",
                        state.render_position,
                        0.025,
                        color,
                    )
            self.renderer.end_frame()

        self._trim_profile_samples()
        self._report_profile_stats()

    def pace(self):
        deadline = self._frame_start + self.sim_config.frame_dt
        now = time.perf_counter()
        remaining = deadline - now
        if remaining > 1e-4:
            time.sleep(remaining)
        self._frame_start = time.perf_counter()

    def is_running(self) -> bool:
        return self.renderer.is_running()

    def close(self):
        self.renderer.close()


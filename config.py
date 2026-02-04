"""
Simulation configuration module.

This module provides a centralized configuration dataclass for the surgical simulation,
eliminating magic numbers and providing clear documentation for all parameters.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SimulationConfig:
    """Configuration for the surgical simulation.

    Attributes:
        fps: Target frames per second for the simulation.
        substeps: Number of physics substeps per frame for stability.

        radius_collision: Radius for haptic device collision detection.
        radius_heating: Radius for tissue heating effect.
        radius_clipping: Radius for clip placement detection.
        radius_cutting: Radius for tissue cutting effect.
        radius_grasping: Radius for tissue grasping.

        particle_mass: Default mass for soft body particles.
        volume_stiffness: Stiffness for volume preservation constraints.
        sphere_collision_radius: Radius for sphere-based collision detection.

        spring_stiffness: Stiffness for spring constraints.
        spring_damping: Damping for spring constraints.

        tetra_stiffness_mu: Shear modulus for tetrahedral elements.
        tetra_stiffness_lambda: Bulk modulus for tetrahedral elements.
        tetra_damping: Damping for tetrahedral elements.

        bounds_min: Minimum bounds for particle collision.
        bounds_max: Maximum bounds for particle collision.

        solver_iterations: Number of constraint solver iterations.

        max_clips: Maximum number of clips that can be placed.
        max_bleed_particles: Maximum number of bleeding particles.

        bleeding_field_resolution: Resolution for bleeding marching cubes field.
        bleeding_field_margin: Margin around bleeding particle AABB.
        bleeding_particle_sdf_radius: Particle radius for SDF computation.
        bleeding_isosurface_threshold: Threshold for marching cubes extraction.
    """

    # Timing
    fps: int = 120
    substeps: int = 16

    # Interaction radii (in simulation units - meters)
    radius_collision: float = 0.1
    radius_heating: float = 0.2
    radius_clipping: float = 0.1
    radius_cutting: float = 0.075
    radius_grasping: float = 0.075

    # Physics - particles
    particle_mass: float = 0.1
    particle_max_velocity: float = 10.0

    # Physics - constraints
    volume_stiffness: float = 0.1
    sphere_collision_radius: float = 0.05

    # Physics - springs
    spring_stiffness: float = 1.0
    spring_damping: float = 0.2

    # Physics - tetrahedra
    tetra_stiffness_mu: float = 1.0e4
    tetra_stiffness_lambda: float = 1.0e4
    tetra_damping: float = 0.2

    # Bounds (simulation space)
    bounds_min: Tuple[float, float, float] = (-2.0, 0.0, -8.0)
    bounds_max: Tuple[float, float, float] = (2.0, 10.0, -3.0)

    # Solver
    solver_iterations: int = 5

    # Capacities
    max_clips: int = 64
    max_bleed_particles: int = 4096
    grasp_capacity: int = 1024

    # Bleeding visualization
    bleeding_field_resolution: int = 96
    bleeding_field_margin: float = 0.01
    bleeding_particle_sdf_radius: float = 0.007
    bleeding_isosurface_threshold: float = 0.0

    # Self-collision
    self_contact_radius: float = 0.002
    self_contact_margin: float = 0.002

    # Mesh loading
    vertical_offset: float = -3.0

    @property
    def frame_dt(self) -> float:
        """Time step per frame in seconds."""
        return 1.0 / self.fps

    @property
    def substep_dt(self) -> float:
        """Time step per substep in seconds."""
        return self.frame_dt / self.substeps


@dataclass
class RenderConfig:
    """Configuration for rendering.

    Attributes:
        scaling: Scale factor for rendering.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        camera_pos: Initial camera position.
    """

    scaling: float = 1.0
    near_plane: float = 0.05
    far_plane: float = 25.0
    camera_pos: Tuple[float, float, float] = (0.2, 1.2, -1.0)


@dataclass
class HapticConfig:
    """Configuration for haptic device integration.

    Attributes:
        scale: Scale factor from haptic device to simulation space.
        position_offset: Offset applied to haptic position.
        invert_x: Invert X axis rotation.
        invert_y: Invert Y axis rotation.
        invert_z: Invert Z axis rotation.
        invert_w: Invert quaternion W component (inverts entire rotation).
    """

    scale: float = 1.0
    position_offset: Tuple[float, float, float] = (0.0, 100.0, -400.0)

    # Rotation axis inversions
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_w: bool = True


def create_default_config() -> SimulationConfig:
    """Create a default simulation configuration."""
    return SimulationConfig()


def create_high_quality_config() -> SimulationConfig:
    """Create a high-quality simulation configuration with more substeps."""
    return SimulationConfig(
        substeps=32,
        solver_iterations=10,
        bleeding_field_resolution=128
    )


def create_fast_config() -> SimulationConfig:
    """Create a fast simulation configuration for testing."""
    return SimulationConfig(
        substeps=8,
        solver_iterations=3,
        bleeding_field_resolution=48
    )

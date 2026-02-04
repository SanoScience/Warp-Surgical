"""
Surgical behaviors module.

This module provides a unified interface for surgical interaction behaviors:
- Heating/Cauterization
- Grasping
- Cutting
- Clipping

Each behavior can be activated/deactivated and processes simulation state
based on the haptic device position.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import warp as wp

from config import SimulationConfig

# Import existing behavior kernels
from heating import (
    heating_start as _heating_start,
    heating_end as _heating_end,
    heating_active_process as _heating_active_process,
    heating_conduction_process as _heating_conduction_process,
    paint_vertices_near_haptic_proxy,
    set_paint_strength
)
from grasping import (
    grasp_start as _grasp_start,
    grasp_end as _grasp_end,
    grasp_process as _grasp_process
)


@dataclass
class BehaviorState:
    """State tracking for a surgical behavior."""
    active: bool = False
    start_time: float = 0.0
    activation_count: int = 0


class HeatingBehavior:
    """Manages tissue heating/cauterization behavior.

    When active, applies heat to vertices near the haptic position.
    Heat conducts through the tissue and can cause burning/tissue damage.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = BehaviorState()
        self.heat_radius = config.radius_heating

    @property
    def active(self) -> bool:
        return self.state.active

    def start(self, sim):
        """Start heating behavior."""
        if not self.state.active:
            self.state.active = True
            self.state.activation_count += 1
            _heating_start(sim)

    def stop(self, sim):
        """Stop heating behavior."""
        if self.state.active:
            self.state.active = False
            _heating_end(sim)

    def process(self, sim):
        """Process heating while active."""
        if self.state.active:
            _heating_active_process(sim)

    def process_conduction(self, sim):
        """Process heat conduction (runs every frame regardless of active state)."""
        _heating_conduction_process(sim)


class GraspingBehavior:
    """Manages tissue grasping behavior.

    When started, finds particles near the haptic position and locks them.
    While active, moves locked particles with the haptic device.
    When stopped, unlocks all grasped particles.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = BehaviorState()
        self.grasp_radius = config.radius_grasping

    @property
    def active(self) -> bool:
        return self.state.active

    def start(self, sim):
        """Start grasping - find and lock nearby particles."""
        if not self.state.active:
            self.state.active = True
            self.state.activation_count += 1
            _grasp_start(sim)

    def stop(self, sim):
        """Stop grasping - unlock all grasped particles."""
        if self.state.active:
            self.state.active = False
            _grasp_end(sim)

    def process(self, sim):
        """Process grasping while active - move locked particles."""
        if self.state.active:
            _grasp_process(sim)


class CuttingBehavior:
    """Manages tissue cutting behavior.

    When active, deactivates tetrahedra near the haptic position,
    effectively cutting through the tissue.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = BehaviorState()
        self.cut_radius = config.radius_cutting

    @property
    def active(self) -> bool:
        return self.state.active

    def start(self):
        """Start cutting behavior."""
        if not self.state.active:
            self.state.active = True
            self.state.activation_count += 1

    def stop(self):
        """Stop cutting behavior."""
        self.state.active = False

    def process(self, sim, mesh_name: str, tet_active, mesh_tets):
        """Process cutting for a specific mesh.

        Note: The actual cutting kernel is launched from render() in WarpSim
        as it's tightly coupled with the mesh iteration. This method provides
        the interface for future refactoring.
        """
        pass  # Cutting is currently handled inline in render loop


class ClippingBehavior:
    """Manages clip placement behavior.

    Places clips on centrelines near the haptic position.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = BehaviorState()
        self.clip_radius = config.radius_clipping
        self.pending_clip = False

    @property
    def active(self) -> bool:
        return self.state.active

    def request_clip(self):
        """Request a clip placement at current haptic position."""
        self.pending_clip = True

    def process(self, sim):
        """Process clip placement if pending.

        Returns True if a clip was placed.
        """
        if self.pending_clip:
            self.pending_clip = False
            self.state.activation_count += 1
            return True
        return False


class SurgicalBehaviors:
    """Unified manager for all surgical behaviors.

    Provides a single interface to manage heating, grasping, cutting,
    and clipping behaviors. Handles activation/deactivation and
    per-frame processing.
    """

    def __init__(self, config: SimulationConfig = None):
        """Initialize surgical behaviors.

        Args:
            config: Simulation configuration.
        """
        self.config = config if config is not None else SimulationConfig()

        self.heating = HeatingBehavior(self.config)
        self.grasping = GraspingBehavior(self.config)
        self.cutting = CuttingBehavior(self.config)
        self.clipping = ClippingBehavior(self.config)

    def process_passive(self, sim):
        """Process behaviors that run every frame regardless of activation.

        This includes heat conduction, which always runs.
        """
        self.heating.process_conduction(sim)

    def process_active(self, sim):
        """Process all active behaviors.

        Call this once per frame to update all active surgical behaviors.
        """
        if self.heating.active:
            self.heating.process(sim)

        if self.grasping.active:
            self.grasping.process(sim)

        # Cutting and clipping are processed inline in render loop
        # due to mesh iteration requirements

    def start_heating(self, sim):
        """Start heating/cauterization."""
        self.heating.start(sim)

    def stop_heating(self, sim):
        """Stop heating/cauterization."""
        self.heating.stop(sim)

    def start_grasping(self, sim):
        """Start grasping tissue."""
        self.grasping.start(sim)

    def stop_grasping(self, sim):
        """Stop grasping and release tissue."""
        self.grasping.stop(sim)

    def start_cutting(self):
        """Start cutting mode."""
        self.cutting.start()

    def stop_cutting(self):
        """Stop cutting mode."""
        self.cutting.stop()

    def place_clip(self):
        """Request clip placement at current haptic position."""
        self.clipping.request_clip()

    def get_active_behaviors(self) -> List[str]:
        """Get list of currently active behavior names."""
        active = []
        if self.heating.active:
            active.append("heating")
        if self.grasping.active:
            active.append("grasping")
        if self.cutting.active:
            active.append("cutting")
        if self.clipping.pending_clip:
            active.append("clipping_pending")
        return active

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about behavior usage."""
        return {
            "heating_activations": self.heating.state.activation_count,
            "grasping_activations": self.grasping.state.activation_count,
            "cutting_activations": self.cutting.state.activation_count,
            "clips_placed": self.clipping.state.activation_count
        }

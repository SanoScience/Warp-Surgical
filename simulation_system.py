from abc import ABC, abstractmethod
from newton._src.sim import Model, State, Contacts
import warp as wp


class SimulationSystem(ABC):
    """Base class for callback simulation systems.
    """

    def __init__(self, priority: int = 0):
        """Initialize a simulation system.

        Args:
            priority: Execution order (lower numbers run first)
        """
        self.priority = priority
        self.enabled = True

    def initialize(self, model: Model):
        pass

    def pre_integrate(self, model: Model, state: State, dt: float):
        """Called before particle/body integration.
        """
        pass

    def solve_constraints(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_q: wp.array,
        particle_qd: wp.array,
        particle_deltas: wp.array,
        body_q: wp.array,
        body_qd: wp.array,
        body_deltas: wp.array,
        dt: float,
        iteration: int
    ):
        """Called during constraint solving iterations.

        This is where most constraint solving happens. Systems should write
        their constraint deltas to particle_deltas and/or body_deltas.
        """
        pass

    def post_solve(self, model: Model, state: State, dt: float):
        """Called after all constraint solving is complete.
        """
        pass

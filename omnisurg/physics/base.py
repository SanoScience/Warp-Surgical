from abc import ABC
from enum import Enum

import warp as wp
from newton._src.sim import Model, State


class SolverStage(Enum):
    ELASTIC = "elastic"
    PROJECTION = "projection"


class SimulationSystem(ABC):
    """Base class for callback-based Phase 1 simulation systems."""

    def __init__(self, priority: int = 0, stage: SolverStage = SolverStage.ELASTIC):
        self.priority = priority
        self.stage = stage
        self.enabled = True
        self.deferred_apply = False

    def get_accumulators(self) -> tuple[wp.array, wp.array] | None:
        return None

    def initialize(self, model: Model):
        pass

    def pre_integrate(self, model: Model, state: State, dt: float):
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
        iteration: int,
    ):
        pass

    def post_solve(self, model: Model, state: State, dt: float):
        pass

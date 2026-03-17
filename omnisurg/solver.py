import warp as wp
from newton._src.core.types import override
from newton._src.sim import Control, Contacts, Model, State
from newton._src.solvers import SolverBase
from newton._src.solvers.xpbd.kernels import apply_particle_deltas

from simulation_system import SimulationSystem


class Phase1Solver(SolverBase):
    """Minimal PBD solver for Phase 1: particles only, system-based constraints."""

    def __init__(self, model: Model, iterations: int = 1):
        super().__init__(model=model)
        self.iterations = iterations
        self._particle_delta_counter = 0
        self.systems: list[SimulationSystem] = []

    def register_system(self, system: SimulationSystem):
        system.initialize(self.model)
        self.systems.append(system)
        self.systems.sort(key=lambda s: s.priority)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ) -> State:
        self._particle_delta_counter = 0
        model = self.model

        for system in self.systems:
            if system.enabled:
                system.pre_integrate(model, state_in, dt)

        if not model.particle_count:
            return state_out

        self.particle_q_init = wp.clone(state_in.particle_q)
        particle_deltas = wp.empty_like(state_out.particle_qd)

        self.integrate_particles(model, state_in, state_out, dt)

        particle_q = state_out.particle_q
        particle_qd = state_out.particle_qd

        for i in range(self.iterations):
            particle_deltas.zero_()

            for system in self.systems:
                if system.enabled:
                    system.solve_constraints(
                        model, state_in, state_out,
                        particle_q, particle_qd, particle_deltas,
                        None, None, None,
                        dt, i,
                    )

            particle_q, particle_qd = self.apply_particle_deltas(
                model, state_in, state_out, particle_deltas, dt,
            )

        if particle_q.ptr != state_out.particle_q.ptr:
            state_out.particle_q.assign(particle_q)
            state_out.particle_qd.assign(particle_qd)

        for system in self.systems:
            if system.enabled:
                system.post_solve(model, state_out, dt)

        return state_out

    def apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if self._particle_delta_counter == 0:
            particle_q = state_out.particle_q
            new_particle_q = state_in.particle_q
            new_particle_qd = state_in.particle_qd
        else:
            particle_q = state_in.particle_q
            new_particle_q = state_out.particle_q
            new_particle_qd = state_out.particle_qd
        self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        return new_particle_q, new_particle_qd

import warp as wp
from newton._src.core.types import override
from newton._src.sim import Control, Contacts, Model, State
from newton._src.solvers import SolverBase
from newton._src.solvers.xpbd.kernels import apply_particle_deltas

from simulation_system import SimulationSystem
from simulation_kernels import apply_fused_3_accumulators, apply_deltas_and_zero_accumulators


class Phase1Solver(SolverBase):
    """Minimal PBD solver for Phase 1: particles only, system-based constraints."""

    def __init__(self, model: Model, iterations: int = 1):
        super().__init__(model=model)
        self.iterations = iterations
        self._particle_delta_counter = 0
        self.systems: list[SimulationSystem] = []
        self._fused_accumulators: list[tuple[wp.array, wp.array]] | None = None

        n = model.particle_count
        if n:
            self._particle_q_init = wp.zeros(n, dtype=wp.vec3f, device=model.device)
            self._particle_deltas = wp.zeros(n, dtype=wp.vec3f, device=model.device)

    def register_system(self, system: SimulationSystem):
        system.initialize(self.model)
        self.systems.append(system)
        self.systems.sort(key=lambda s: s.priority)
        self._fused_accumulators = None

    def _setup_fused_apply(self):
        accums = []
        for s in self.systems:
            a = s.get_accumulators()
            if a is not None:
                s.deferred_apply = True
                accums.append(a)
        self._fused_accumulators = accums

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ) -> State:
        if self._fused_accumulators is None:
            self._setup_fused_apply()

        self._particle_delta_counter = 0
        model = self.model

        for system in self.systems:
            if system.enabled:
                system.pre_integrate(model, state_in, dt)

        if not model.particle_count:
            return state_out

        wp.copy(self._particle_q_init, state_in.particle_q)
        self._particle_deltas.zero_()

        self.integrate_particles(model, state_in, state_out, dt)

        particle_q = state_out.particle_q
        particle_qd = state_out.particle_qd

        for i in range(self.iterations):
            self._particle_deltas.zero_()

            for system in self.systems:
                if system.enabled:
                    system.solve_constraints(
                        model, state_in, state_out,
                        particle_q, particle_qd, self._particle_deltas,
                        None, None, None,
                        dt, i,
                    )

            self._apply_fused(model)

            particle_q, particle_qd = self.apply_particle_deltas(
                model, state_in, state_out, self._particle_deltas, dt,
            )

        if particle_q.ptr != state_out.particle_q.ptr:
            state_out.particle_q.assign(particle_q)
            state_out.particle_qd.assign(particle_qd)

        for system in self.systems:
            if system.enabled:
                system.post_solve(model, state_out, dt)

        return state_out

    def _apply_fused(self, model: Model):
        accums = self._fused_accumulators
        if len(accums) == 3:
            wp.launch(
                apply_fused_3_accumulators,
                dim=model.particle_count,
                inputs=[
                    accums[0][0], accums[0][1],
                    accums[1][0], accums[1][1],
                    accums[2][0], accums[2][1],
                ],
                outputs=[self._particle_deltas],
                device=model.device,
            )
        else:
            for da, ca in accums:
                wp.launch(
                    apply_deltas_and_zero_accumulators,
                    dim=model.particle_count,
                    inputs=[da, ca],
                    outputs=[self._particle_deltas],
                    device=model.device,
                )

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
                self._particle_q_init,
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

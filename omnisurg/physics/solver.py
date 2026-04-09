import warp as wp
from newton._src.core.types import override
from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers import SolverBase
from newton._src.solvers.xpbd.kernels import apply_particle_deltas

from omnisurg.physics.base import SimulationSystem, SolverStage
from omnisurg.physics.kernels import (
    apply_deltas_and_zero_accumulators,
    apply_displacements_from_base,
    apply_fused_3_accumulators,
    compute_position_deltas,
)


class Phase1Solver(SolverBase):
    """Minimal particles-only PBD solver for the Phase 1 runtime."""

    def __init__(self, model: Model, iterations: int = 1):
        super().__init__(model=model)
        self.iterations = iterations
        self._particle_delta_counter = 0
        self.systems: list[SimulationSystem] = []
        self._elastic_systems: list[SimulationSystem] = []
        self._truncation_systems: list[SimulationSystem] = []
        self._projection_systems: list[SimulationSystem] = []
        self._fused_accumulators: list[tuple[wp.array, wp.array]] | None = None

        n = model.particle_count
        if n:
            self._particle_q_init = wp.zeros(n, dtype=wp.vec3f, device=model.device)
            self._particle_deltas = wp.zeros(n, dtype=wp.vec3f, device=model.device)
            self._prediction_displacements = wp.zeros(n, dtype=wp.vec3f, device=model.device)

    def register_system(self, system: SimulationSystem):
        system.initialize(self.model)
        self.systems.append(system)
        self.systems.sort(key=lambda item: item.priority)
        self._refresh_system_layout()

    def _refresh_system_layout(self):
        self._elastic_systems = []
        self._truncation_systems = []
        self._projection_systems = []
        elastic_accumulators = []

        for system in self.systems:
            values = system.get_accumulators()
            system.deferred_apply = values is not None
            if system.stage == SolverStage.PROJECTION:
                self._projection_systems.append(system)
            elif system.stage == SolverStage.TRUNCATION:
                self._truncation_systems.append(system)
            else:
                self._elastic_systems.append(system)
                if values is not None:
                    elastic_accumulators.append(values)

        self._fused_accumulators = elastic_accumulators

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
            self._refresh_system_layout()

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
        self._apply_prediction_truncation(model, state_in, state_out, dt)

        particle_q = state_out.particle_q
        particle_qd = state_out.particle_qd

        for iteration in range(self.iterations):
            self._particle_deltas.zero_()

            for system in self._elastic_systems:
                if system.enabled:
                    system.solve_constraints(
                        model,
                        state_in,
                        state_out,
                        particle_q,
                        particle_qd,
                        self._particle_deltas,
                        None,
                        None,
                        None,
                        dt,
                        iteration,
                    )

            self._apply_fused(model)
            self._apply_truncation_stage(
                model,
                state_in,
                state_out,
                particle_q,
                particle_qd,
                dt,
                iteration,
            )

            particle_q, particle_qd = self.apply_particle_deltas(
                model,
                state_in,
                state_out,
                self._particle_deltas,
                dt,
            )

            particle_q, particle_qd = self._apply_projection_stage(
                model,
                state_in,
                state_out,
                particle_q,
                particle_qd,
                dt,
                iteration,
            )

        if particle_q.ptr != state_out.particle_q.ptr:
            state_out.particle_q.assign(particle_q)
            state_out.particle_qd.assign(particle_qd)

        for system in self.systems:
            if system.enabled:
                system.post_solve(model, state_out, dt)

        return state_out

    def _apply_projection_stage(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_q: wp.array,
        particle_qd: wp.array,
        dt: float,
        iteration: int,
    ):
        for system in self._projection_systems:
            if not system.enabled:
                continue

            self._particle_deltas.zero_()
            system.solve_constraints(
                model,
                state_in,
                state_out,
                particle_q,
                particle_qd,
                self._particle_deltas,
                None,
                None,
                None,
                dt,
                iteration,
            )

            values = system.get_accumulators()
            if values is None:
                continue

            self._particle_deltas.zero_()
            wp.launch(
                apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[values[0], values[1]],
                outputs=[self._particle_deltas],
                device=model.device,
            )
            particle_q, particle_qd = self.apply_particle_deltas(
                model,
                state_in,
                state_out,
                self._particle_deltas,
                dt,
            )

        return particle_q, particle_qd

    def _apply_prediction_truncation(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ):
        if not self._truncation_systems or not any(system.enabled for system in self._truncation_systems):
            return

        wp.launch(
            kernel=compute_position_deltas,
            dim=model.particle_count,
            inputs=[self._particle_q_init, state_out.particle_q],
            outputs=[self._prediction_displacements],
            device=model.device,
        )

        for system in self._truncation_systems:
            if system.enabled:
                system.truncate_prediction(
                    model,
                    state_in,
                    state_out,
                    self._particle_q_init,
                    self._prediction_displacements,
                    dt,
                )

        self._rewrite_particle_state_from_base(model, state_out, self._prediction_displacements, dt)

    def _apply_truncation_stage(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_q: wp.array,
        particle_qd: wp.array,
        dt: float,
        iteration: int,
    ):
        if not self._truncation_systems or not any(system.enabled for system in self._truncation_systems):
            return

        for system in self._truncation_systems:
            if system.enabled:
                system.truncate_deltas(
                    model,
                    state_in,
                    state_out,
                    particle_q,
                    particle_qd,
                    self._particle_deltas,
                    dt,
                    iteration,
                )

    def _apply_fused(self, model: Model):
        accumulators = self._fused_accumulators or []
        if len(accumulators) == 3:
            wp.launch(
                apply_fused_3_accumulators,
                dim=model.particle_count,
                inputs=[
                    accumulators[0][0],
                    accumulators[0][1],
                    accumulators[1][0],
                    accumulators[1][1],
                    accumulators[2][0],
                    accumulators[2][1],
                ],
                outputs=[self._particle_deltas],
                device=model.device,
            )
            return

        for delta_accum, count_accum in accumulators:
            wp.launch(
                apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[delta_accum, count_accum],
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

    def _rewrite_particle_state_from_base(
        self,
        model: Model,
        state_out: State,
        displacements: wp.array,
        dt: float,
    ):
        wp.launch(
            kernel=apply_displacements_from_base,
            dim=model.particle_count,
            inputs=[
                self._particle_q_init,
                model.particle_flags,
                displacements,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )

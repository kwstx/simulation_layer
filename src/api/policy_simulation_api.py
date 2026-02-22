from __future__ import annotations

from statistics import mean
from typing import Sequence

from pydantic import BaseModel, Field

from src.causal_impact_propagation import (
    CausalImpactPropagationEngine,
    SynergyShiftAnalyzer,
    SynergyShiftHorizonProjection,
)
from src.models.cooperative_state_snapshot import CooperativeStateSnapshot
from src.models.intelligence_evolution_model import EvolutionMetrics, IntelligenceEvolutionModel
from src.models.policy import PolicySchema, TransformationOperator
from src.simulation.entropy_stress_test import EntropyCycleMetrics, EntropyStressTest
from src.simulation.negotiation_dynamics_simulator import GovernanceRuleRisk, NegotiationDynamicsSimulator


class DownstreamImpactDelta(BaseModel):
    horizon: int = Field(..., ge=1)
    projected_outcome_score: float
    impact_delta_vs_baseline: float


class NegotiationStabilityMetrics(BaseModel):
    converged: bool
    convergence_time: int = Field(..., ge=1)
    instability_score: float = Field(..., ge=0.0)
    coordination_friction_score: float = Field(..., ge=0.0)
    instability_detected: bool
    coordination_friction_detected: bool
    governance_risks: tuple[GovernanceRuleRisk, ...]


class CausalDriverContribution(BaseModel):
    driver: str
    contribution: float
    rationale: str


class CausalExplanationTrace(BaseModel):
    variable: str
    observed_delta: float
    interpretation: str
    drivers: tuple[CausalDriverContribution, ...]


class PolicySimulationOutput(BaseModel):
    policy_id: str
    projected_downstream_impact_deltas: tuple[DownstreamImpactDelta, ...]
    synergy_distribution_shifts: tuple[SynergyShiftHorizonProjection, ...]
    cooperative_intelligence_evolution_curves: tuple[EvolutionMetrics, ...]
    entropy_trajectory_forecasts: tuple[EntropyCycleMetrics, ...]
    negotiation_stability_metrics: NegotiationStabilityMetrics
    causal_explanation_traces: tuple[CausalExplanationTrace, ...]


class PolicySimulationAPI:
    """
    Composed simulation API that accepts a PolicySchema and returns a structured
    multi-domain policy impact report.
    """

    def __init__(
        self,
        baseline_snapshot: CooperativeStateSnapshot,
        horizons: Sequence[int] = (1, 5, 10),
        evolution_steps: int = 20,
        entropy_cycles: int = 12,
        negotiation_steps: int = 20,
    ) -> None:
        if not horizons:
            raise ValueError("horizons must contain at least one value")
        if min(horizons) < 1:
            raise ValueError("all horizons must be >= 1")
        if evolution_steps < 1:
            raise ValueError("evolution_steps must be >= 1")
        if entropy_cycles < 1:
            raise ValueError("entropy_cycles must be >= 1")
        if negotiation_steps < 1:
            raise ValueError("negotiation_steps must be >= 1")

        self._baseline_snapshot = baseline_snapshot
        self._horizons = tuple(sorted(set(int(h) for h in horizons)))
        self._evolution_steps = int(evolution_steps)
        self._entropy_cycles = int(entropy_cycles)
        self._negotiation_steps = int(negotiation_steps)

        self._impact_engine = CausalImpactPropagationEngine()
        self._synergy_analyzer = SynergyShiftAnalyzer()
        self._entropy_tester = EntropyStressTest()
        self._negotiation_simulator = NegotiationDynamicsSimulator()

    def simulate(self, policy: PolicySchema) -> PolicySimulationOutput:
        impact_report = self._impact_engine.simulate(
            policy=policy,
            baseline_snapshot=self._baseline_snapshot,
            horizons=self._horizons,
        )
        synergy_report = self._synergy_analyzer.analyze(
            policy=policy,
            baseline_snapshot=self._baseline_snapshot,
            horizons=self._horizons,
        )
        evolution_model = IntelligenceEvolutionModel(self._baseline_snapshot, policy)
        evolution_curve = tuple(evolution_model.evolve(self._evolution_steps))
        entropy_report = self._entropy_tester.evaluate(
            policy=policy,
            baseline_snapshot=self._baseline_snapshot,
            cycles=self._entropy_cycles,
        )
        negotiation_report = self._negotiation_simulator.simulate(
            policy=policy,
            baseline_snapshot=self._baseline_snapshot,
            max_steps=self._negotiation_steps,
        )

        downstream_deltas = tuple(
            DownstreamImpactDelta(
                horizon=point.horizon,
                projected_outcome_score=point.projected_outcome_score,
                impact_delta_vs_baseline=point.impact_delta_vs_baseline,
            )
            for point in impact_report.horizons
        )

        negotiation_metrics = NegotiationStabilityMetrics(
            converged=negotiation_report.converged,
            convergence_time=negotiation_report.convergence_time,
            instability_score=negotiation_report.instability_score,
            coordination_friction_score=negotiation_report.coordination_friction_score,
            instability_detected=negotiation_report.instability_detected,
            coordination_friction_detected=negotiation_report.coordination_friction_detected,
            governance_risks=negotiation_report.governance_risks,
        )

        traces = self._build_causal_traces(
            policy=policy,
            downstream_deltas=downstream_deltas,
            evolution_curve=evolution_curve,
            entropy_trajectory=tuple(entropy_report.trajectory),
            negotiation_metrics=negotiation_metrics,
            synergy_shifts=tuple(synergy_report.horizons),
        )

        return PolicySimulationOutput(
            policy_id=policy.policy_id,
            projected_downstream_impact_deltas=downstream_deltas,
            synergy_distribution_shifts=tuple(synergy_report.horizons),
            cooperative_intelligence_evolution_curves=evolution_curve,
            entropy_trajectory_forecasts=tuple(entropy_report.trajectory),
            negotiation_stability_metrics=negotiation_metrics,
            causal_explanation_traces=traces,
        )

    def _build_causal_traces(
        self,
        policy: PolicySchema,
        downstream_deltas: tuple[DownstreamImpactDelta, ...],
        evolution_curve: tuple[EvolutionMetrics, ...],
        entropy_trajectory: tuple[EntropyCycleMetrics, ...],
        negotiation_metrics: NegotiationStabilityMetrics,
        synergy_shifts: tuple[SynergyShiftHorizonProjection, ...],
    ) -> tuple[CausalExplanationTrace, ...]:
        transform_pressure = self._transformation_pressure(policy)
        entropy_pressure = self._entropy_pressure(policy)
        modifier_pressure = self._modifier_pressure(policy)
        temporal_pressure = self._temporal_pressure(policy)

        driver_values = {
            "transformations": transform_pressure,
            "entropy_adjustments": entropy_pressure,
            "impact_modifiers": modifier_pressure,
            "temporal_rules": temporal_pressure,
        }

        downstream_delta = mean(d.impact_delta_vs_baseline for d in downstream_deltas)
        evolution_delta = evolution_curve[-1].projected_impact - evolution_curve[0].projected_impact
        entropy_delta = (
            entropy_trajectory[-1].normalized_entropy - entropy_trajectory[0].normalized_entropy
        )
        negotiation_delta = -(
            0.6 * negotiation_metrics.instability_score
            + 0.4 * negotiation_metrics.coordination_friction_score
        )
        synergy_delta = self._average_synergy_distribution_shift(synergy_shifts)

        return (
            self._trace(
                variable="projected_downstream_impact",
                observed_delta=downstream_delta,
                interpretation=(
                    "Positive values indicate a net gain versus baseline in projected outcome score."
                ),
                weights={
                    "transformations": 0.45,
                    "impact_modifiers": 0.35,
                    "entropy_adjustments": 0.10,
                    "temporal_rules": 0.10,
                },
                drivers=driver_values,
            ),
            self._trace(
                variable="synergy_distribution",
                observed_delta=synergy_delta,
                interpretation="Captures average normalized tensor redistribution magnitude per horizon.",
                weights={
                    "transformations": 0.35,
                    "entropy_adjustments": 0.30,
                    "impact_modifiers": 0.25,
                    "temporal_rules": 0.10,
                },
                drivers=driver_values,
            ),
            self._trace(
                variable="cooperative_intelligence_evolution",
                observed_delta=evolution_delta,
                interpretation="Measures projected impact growth along the evolution curve.",
                weights={
                    "transformations": 0.40,
                    "impact_modifiers": 0.30,
                    "temporal_rules": 0.20,
                    "entropy_adjustments": 0.10,
                },
                drivers=driver_values,
            ),
            self._trace(
                variable="entropy_trajectory",
                observed_delta=entropy_delta,
                interpretation="Tracks normalized entropy change from cycle 0 to final cycle.",
                weights={
                    "entropy_adjustments": 0.45,
                    "transformations": 0.25,
                    "impact_modifiers": 0.20,
                    "temporal_rules": 0.10,
                },
                drivers=driver_values,
            ),
            self._trace(
                variable="negotiation_stability",
                observed_delta=negotiation_delta,
                interpretation="Higher negative values indicate higher instability/friction pressure.",
                weights={
                    "transformations": 0.40,
                    "entropy_adjustments": 0.25,
                    "impact_modifiers": 0.20,
                    "temporal_rules": 0.15,
                },
                drivers=driver_values,
            ),
        )

    def _trace(
        self,
        variable: str,
        observed_delta: float,
        interpretation: str,
        weights: dict[str, float],
        drivers: dict[str, float],
    ) -> CausalExplanationTrace:
        contributions = []
        for key, weight in weights.items():
            value = weight * drivers.get(key, 0.0)
            contributions.append(
                CausalDriverContribution(
                    driver=key,
                    contribution=value,
                    rationale=f"Weighted by {weight:.2f} for {variable}.",
                )
            )

        return CausalExplanationTrace(
            variable=variable,
            observed_delta=observed_delta,
            interpretation=interpretation,
            drivers=tuple(contributions),
        )

    @staticmethod
    def _coerce_numeric(value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    def _transformation_pressure(self, policy: PolicySchema) -> float:
        if not policy.transformations:
            return 0.0

        effects = []
        for transformation in policy.transformations:
            numeric = self._coerce_numeric(transformation.value)
            if transformation.operator == TransformationOperator.MULTIPLY:
                effects.append(numeric - 1.0)
            elif transformation.operator == TransformationOperator.ADD:
                effects.append(0.5 * numeric)
            elif transformation.operator == TransformationOperator.DECAY:
                effects.append(-numeric)
            elif transformation.operator == TransformationOperator.CLAMP:
                effects.append(-0.05)
            else:
                effects.append(0.0)
        return mean(effects)

    def _entropy_pressure(self, policy: PolicySchema) -> float:
        numeric_values = [
            float(v)
            for v in policy.entropy_adjustments.values()
            if isinstance(v, (int, float))
        ]
        return sum(numeric_values)

    def _modifier_pressure(self, policy: PolicySchema) -> float:
        numeric_values = [
            float(v) - 1.0
            for v in policy.impact_modifiers.values()
            if isinstance(v, (int, float))
        ]
        if not numeric_values:
            return 0.0
        return mean(numeric_values)

    def _temporal_pressure(self, policy: PolicySchema) -> float:
        persistence_weight = {
            "transient": 0.2,
            "sticky": 0.6,
            "permanent": 1.0,
        }.get(policy.temporal_rules.persistence_mode, 0.3)
        decay_penalty = max(0.0, float(policy.temporal_rules.auto_decay_coefficient))
        return persistence_weight - 0.5 * decay_penalty

    @staticmethod
    def _average_synergy_distribution_shift(
        synergy_shifts: tuple[SynergyShiftHorizonProjection, ...],
    ) -> float:
        values = []
        for horizon in synergy_shifts:
            for tensor in horizon.projected_synergy_distribution_delta_tensors:
                for row in tensor.delta_values:
                    values.extend(abs(float(cell)) for cell in row)
        if not values:
            return 0.0
        return mean(values)

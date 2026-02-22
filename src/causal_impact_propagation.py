from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List, Sequence

from pydantic import BaseModel, Field

from src.models.cooperative_state_snapshot import CooperativeStateSnapshot
from src.models.policy import PolicySchema, TransformationOperator


class BaselineSignals(BaseModel):
    predictive_synergy_density: float = Field(..., ge=0.0)
    cooperative_intelligence_amplification: float = Field(..., ge=0.0)
    trust_weighted_forecast_adjustment: float = Field(..., ge=0.0)
    baseline_outcome_score: float = Field(..., ge=0.0)


class HorizonImpactProjection(BaseModel):
    horizon: int = Field(..., ge=1)
    projected_predictive_synergy_density: float = Field(..., ge=0.0)
    projected_cooperative_intelligence_amplification: float = Field(..., ge=0.0)
    projected_trust_weighted_forecast_adjustment: float = Field(..., ge=0.0)
    projected_outcome_score: float = Field(..., ge=0.0)
    impact_delta_vs_baseline: float


class ImpactPropagationReport(BaseModel):
    policy_id: str
    baseline: BaselineSignals
    horizons: tuple[HorizonImpactProjection, ...]


class CausalImpactPropagationEngine:
    """
    Simulates policy-induced parameter shifts and propagates their impact across
    multiple future horizons relative to a baseline cooperative state snapshot.
    """

    def simulate(
        self,
        policy: PolicySchema,
        baseline_snapshot: CooperativeStateSnapshot,
        horizons: Sequence[int],
    ) -> ImpactPropagationReport:
        if not horizons:
            raise ValueError("horizons must contain at least one horizon value")

        normalized_horizons = tuple(sorted(set(horizons)))
        if normalized_horizons[0] < 1:
            raise ValueError("all horizons must be >= 1")

        baseline = self._compute_baseline_signals(baseline_snapshot)
        shift = self._compute_policy_shift(policy)

        projections: List[HorizonImpactProjection] = []
        for horizon in normalized_horizons:
            policy_effect = self._effective_policy_shift(policy, shift, horizon)

            projected_psd = baseline.predictive_synergy_density * (
                1.0 + policy_effect * (0.6 + 0.4 * self._trust_signal(baseline_snapshot))
            )
            projected_cia = baseline.cooperative_intelligence_amplification * (
                1.0 + policy_effect * (0.5 + 0.5 * self._calibration_quality(baseline_snapshot))
            )
            projected_tfa = baseline.trust_weighted_forecast_adjustment * (
                1.0 + policy_effect * (0.4 + 0.6 * self._trust_signal(baseline_snapshot))
            )

            projected_outcome = self._compose_outcome(projected_psd, projected_cia, projected_tfa)

            projections.append(
                HorizonImpactProjection(
                    horizon=horizon,
                    projected_predictive_synergy_density=max(0.0, projected_psd),
                    projected_cooperative_intelligence_amplification=max(0.0, projected_cia),
                    projected_trust_weighted_forecast_adjustment=max(0.0, projected_tfa),
                    projected_outcome_score=max(0.0, projected_outcome),
                    impact_delta_vs_baseline=projected_outcome - baseline.baseline_outcome_score,
                )
            )

        return ImpactPropagationReport(
            policy_id=policy.policy_id,
            baseline=baseline,
            horizons=tuple(projections),
        )

    def _compute_baseline_signals(self, snapshot: CooperativeStateSnapshot) -> BaselineSignals:
        synergy_density = self._synergy_density(snapshot)
        intelligence_density = self._intelligence_density(snapshot)
        trust_signal = self._trust_signal(snapshot)
        calibration_quality = self._calibration_quality(snapshot)

        predictive_synergy_density = synergy_density * (0.5 + 0.5 * calibration_quality)
        cooperative_intelligence_amplification = intelligence_density * (1.0 + 0.25 * synergy_density)
        trust_weighted_forecast_adjustment = trust_signal * calibration_quality

        baseline_outcome = self._compose_outcome(
            predictive_synergy_density,
            cooperative_intelligence_amplification,
            trust_weighted_forecast_adjustment,
        )

        return BaselineSignals(
            predictive_synergy_density=max(0.0, predictive_synergy_density),
            cooperative_intelligence_amplification=max(0.0, cooperative_intelligence_amplification),
            trust_weighted_forecast_adjustment=max(0.0, trust_weighted_forecast_adjustment),
            baseline_outcome_score=max(0.0, baseline_outcome),
        )

    @staticmethod
    def _compose_outcome(psd: float, cia: float, tfa: float) -> float:
        return (0.5 * psd) + (0.3 * cia) + (0.2 * tfa)

    def _compute_policy_shift(self, policy: PolicySchema) -> float:
        modifier_values = [v for v in policy.impact_modifiers.values() if isinstance(v, (int, float))]
        modifier_term = (mean(modifier_values) - 1.0) if modifier_values else 0.0

        entropy_term = sum(
            float(v) for v in policy.entropy_adjustments.values() if isinstance(v, (int, float))
        )

        transform_term = 0.0
        for transform in policy.transformations:
            numeric_value = self._coerce_numeric(transform.value)
            operator_weight = {
                TransformationOperator.ADD: 0.02,
                TransformationOperator.MULTIPLY: 0.05,
                TransformationOperator.CLAMP: 0.01,
                TransformationOperator.DECAY: -0.03,
                TransformationOperator.CUSTOM: 0.0,
            }[transform.operator]
            transform_term += operator_weight * numeric_value

        if policy.transformations:
            transform_term /= len(policy.transformations)

        constraint_term = 0.005 * len(policy.constraints)
        return modifier_term + entropy_term + transform_term + constraint_term

    def _effective_policy_shift(self, policy: PolicySchema, shift: float, horizon: int) -> float:
        temporal = policy.temporal_rules
        decay = max(0.0, float(temporal.auto_decay_coefficient))
        attenuation = math.exp(-decay * horizon)

        if temporal.duration_steps is None or horizon <= temporal.duration_steps:
            lifecycle_weight = 1.0
        elif temporal.persistence_mode == "transient":
            lifecycle_weight = 0.0
        elif temporal.persistence_mode == "sticky":
            lifecycle_weight = 0.5
        else:
            lifecycle_weight = 1.0

        return shift * attenuation * lifecycle_weight

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

    @staticmethod
    def _flatten(values: Iterable[Iterable[float]]) -> List[float]:
        flat: List[float] = []
        for seq in values:
            flat.extend(float(v) for v in seq)
        return flat

    def _synergy_density(self, snapshot: CooperativeStateSnapshot) -> float:
        if not snapshot.synergy_density_matrices:
            return 0.0

        all_values: List[float] = []
        for matrix in snapshot.synergy_density_matrices:
            all_values.extend(self._flatten(matrix.values))
        return mean(all_values) if all_values else 0.0

    def _intelligence_density(self, snapshot: CooperativeStateSnapshot) -> float:
        if not snapshot.cooperative_intelligence_distributions:
            return 0.0

        values = self._flatten(d.values for d in snapshot.cooperative_intelligence_distributions)
        return mean(values) if values else 0.0

    def _trust_signal(self, snapshot: CooperativeStateSnapshot) -> float:
        if not snapshot.trust_vectors:
            return 0.5

        values = self._flatten(v.values for v in snapshot.trust_vectors)
        return mean(values) if values else 0.5

    def _calibration_quality(self, snapshot: CooperativeStateSnapshot) -> float:
        curves = snapshot.predictive_calibration_curves
        if not curves:
            return 0.5

        errors: List[float] = []
        for curve in curves:
            for point in curve.points:
                errors.append(abs(point.predicted - point.observed))

        if not errors:
            return 0.5

        return max(0.0, min(1.0, 1.0 - mean(errors)))

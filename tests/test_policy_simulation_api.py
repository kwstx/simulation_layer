from src.api.policy_simulation_api import PolicySimulationAPI
from src.models.cooperative_state_snapshot import (
    CooperativeIntelligenceDistribution,
    CooperativeStateSnapshot,
    PredictiveCalibrationCurve,
    PredictiveCalibrationPoint,
    SynergyDensityMatrix,
    TrustVector,
)
from src.models.policy import PolicySchema


def _baseline_snapshot() -> CooperativeStateSnapshot:
    return CooperativeStateSnapshot(
        simulation_id="sim-api-001",
        capture_step=0,
        trust_vectors=(
            TrustVector(entity_id="a1", values=(0.85, 0.80)),
            TrustVector(entity_id="a2", values=(0.70, 0.75)),
            TrustVector(entity_id="a3", values=(0.60, 0.55)),
        ),
        cooperative_intelligence_distributions=(
            CooperativeIntelligenceDistribution(domain="planning", values=(0.5, 0.7, 0.8)),
        ),
        synergy_density_matrices=(
            SynergyDensityMatrix(
                matrix_id="synergy-main",
                row_labels=("a1", "a2", "a3"),
                col_labels=("a1", "a2", "a3"),
                values=(
                    (0.9, 0.4, 0.2),
                    (0.4, 0.8, 0.3),
                    (0.2, 0.3, 0.7),
                ),
            ),
        ),
        predictive_calibration_curves=(
            PredictiveCalibrationCurve(
                curve_id="calib-1",
                points=(
                    PredictiveCalibrationPoint(predicted=0.72, observed=0.68),
                    PredictiveCalibrationPoint(predicted=0.61, observed=0.60),
                ),
            ),
        ),
    )


def _policy() -> PolicySchema:
    return PolicySchema(
        policy_id="policy-api-001",
        name="Policy Simulation API Candidate",
        scope={"agent_categories": ["all"], "task_domains": ["planning"]},
        transformations=(
            {
                "metric_source": "trust_core",
                "operator": "multiply",
                "value": 1.2,
                "target_metric": "influence_weight",
            },
            {
                "metric_source": "consensus_boost",
                "operator": "add",
                "value": 0.15,
                "target_metric": "negotiation_alignment",
            },
        ),
        affected_metrics=("synergy_density", "collective_iq"),
        entropy_adjustments={"shannon_entropy_target": -0.03},
        impact_modifiers={"projected_real_world_impact": 1.08, "coordination_gain": 1.05},
        temporal_rules={"persistence_mode": "sticky", "auto_decay_coefficient": 0.01},
    )


def test_policy_simulation_api_returns_required_structured_outputs():
    api = PolicySimulationAPI(
        baseline_snapshot=_baseline_snapshot(),
        horizons=(1, 3, 7),
        evolution_steps=12,
        entropy_cycles=10,
        negotiation_steps=15,
    )

    result = api.simulate(_policy())

    assert result.policy_id == "policy-api-001"

    assert len(result.projected_downstream_impact_deltas) == 3
    assert tuple(p.horizon for p in result.projected_downstream_impact_deltas) == (1, 3, 7)

    assert len(result.synergy_distribution_shifts) == 3
    assert tuple(p.horizon for p in result.synergy_distribution_shifts) == (1, 3, 7)

    assert len(result.cooperative_intelligence_evolution_curves) == 12
    assert result.cooperative_intelligence_evolution_curves[-1].projected_impact > 0.0

    # cycles + baseline cycle 0
    assert len(result.entropy_trajectory_forecasts) == 11
    assert result.entropy_trajectory_forecasts[0].cycle == 0
    assert result.entropy_trajectory_forecasts[-1].cycle == 10

    assert result.negotiation_stability_metrics.convergence_time >= 1
    assert result.negotiation_stability_metrics.instability_score >= 0.0

    variables = {trace.variable for trace in result.causal_explanation_traces}
    assert variables == {
        "projected_downstream_impact",
        "synergy_distribution",
        "cooperative_intelligence_evolution",
        "entropy_trajectory",
        "negotiation_stability",
    }

    for trace in result.causal_explanation_traces:
        assert len(trace.drivers) == 4
        assert any(driver.driver == "transformations" for driver in trace.drivers)

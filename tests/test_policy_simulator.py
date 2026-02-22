import pytest
from src.models.policy import PolicySchema, TransformationOperator
from src.models.cooperative_state_snapshot import CooperativeStateSnapshot
from src.simulation.policy_simulator import PolicyInjectionSimulator

def test_simulator_injection_propagation():
    # 1. Setup a mock snapshot
    snapshot_data = {
        "simulation_id": "sim-test-001",
        "capture_step": 10,
        "trust_vectors": [
            {"entity_id": "agent_01", "values": (0.5, 0.8)},
            {"entity_id": "agent_02", "values": (0.6, 0.7)}
        ],
        "metadata": [
            ("negotiation_parameters", {"compromise_sensitivity": 0.5}),
            ("task_formation_probabilities", {"synergy_bias": 1.0}),
            ("reward_scaling_functions", {"base_multiplier": 1.0})
        ]
    }
    snapshot = CooperativeStateSnapshot(**snapshot_data)
    
    # 2. Define a candidate policy with multiple transformation paths
    policy_data = {
        "policy_id": "pol-test-001",
        "name": "Cooperative Boost",
        "version": "1.0.0",
        "scope": {"agent_categories": ["all"]},
        "transformations": [
            {
                "metric_source": "trust",
                "operator": "multiply",
                "value": 1.2,
                "target_metric": "trust_weight"
            },
            {
                "metric_source": "synergy",
                "operator": "add",
                "value": 0.15,
                "target_metric": "negotiation_alignment"
            },
            {
                "metric_source": "focus",
                "operator": "decay",
                "value": 0.05,
                "target_metric": "task_formation_exploration"
            }
        ],
        "affected_metrics": ["synergy_density"],
        "entropy_adjustments": {"global_entropy": -0.01},
        "impact_modifiers": {"efficiency": 1.05},
        "temporal_rules": {"persistence_mode": "transient"}
    }
    policy = PolicySchema(**policy_data)
    
    # 3. Run Simulation
    simulator = PolicyInjectionSimulator(snapshot)
    results = simulator.simulate(policy)
    
    # 4. Assertions
    assert results.status == "success"
    
    # Check Trust Propagation
    trust_updates = results.transformed_trust_vectors
    for tv in trust_updates:
        if tv["entity_id"] == "agent_01":
            # 0.5 * 1.2 = 0.6
            assert pytest.approx(tv["values"][0]) == 0.6
            
    # Check Negotiation Propagation
    neg_params = results.negotiation_parameters
    assert any("alignment" in k for k in neg_params)
    
    # Check Entropy Impact
    assert results.entropy_deltas["global_entropy"] == -0.01
    
    print("\nSimulation results verified successfully.")

if __name__ == "__main__":
    test_simulator_injection_propagation()

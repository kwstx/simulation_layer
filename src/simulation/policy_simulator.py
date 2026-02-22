import copy
import logging
from pydantic import BaseModel, Field
from src.models.policy import PolicySchema, TransformationOperator, InfluenceTransformation, ExecutableConstraint
from src.models.cooperative_state_snapshot import CooperativeStateSnapshot

class SimulationResult(BaseModel):
    """Encapsulates the outcome of a policy injection simulation."""
    policy_id: str
    status: str = Field("success", description="Status of the simulation (success, violation, skipped)")
    applied_transformations: int
    transformed_trust_vectors: List[Dict[str, Any]]
    negotiation_parameters: Dict[str, Any]
    task_formation_probabilities: Dict[str, Any]
    reward_scaling: Dict[str, Any]
    entropy_deltas: Dict[str, float]
    projected_impact_modifiers: Dict[str, float]
    violations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PolicyInjectionSimulator:
    """
    Simulates the injection of governance policies into a cooperative system.
    
    This simulator applies a candidate policy to a copied system state, 
    propagating transformations through trust weighting, negotiation parameters, 
    task formation probabilities, and reward scaling functions.
    
    Safety: Operates on a detached copy of the state to ensure no live system 
    variables are mutated.
    """

    def __init__(self, initial_state: CooperativeStateSnapshot):
        # The snapshot is biologically immutable (frozen pydantic model).
        # We dump to a dict to create a mutable sandbox for propagation.
        self._initial_state = initial_state
        self._sandbox_data = initial_state.model_dump()
        
        # Extract metadata into a working map for easier manipulation
        self._metadata_map = dict(self._sandbox_data.get("metadata", []))
        
        # Initialize specialized propagation targets
        # 1. Trust Weights (from trust_vectors)
        self._trust_map = {
            tv["entity_id"]: list(tv["values"]) 
            for tv in self._sandbox_data.get("trust_vectors", [])
        }
        
        # 2. Negotiation Parameters
        self._negotiation_params = self._metadata_map.get("negotiation_parameters", {
            "compromise_sensitivity": 0.5,
            "alignment_threshold": 0.7,
            "utility_discount_rate": 0.02
        })
        
        # 3. Task Formation Probabilities
        self._task_formation_probs = self._metadata_map.get("task_formation_probabilities", {
            "synergy_bias": 1.1,
            "exploration_probability": 0.1,
            "cooperation_weight": 0.75
        })
        
        # 4. Reward Scaling Functions
        self._reward_scaling = self._metadata_map.get("reward_scaling_functions", {
            "base_multiplier": 1.0,
            "consensus_bonus": 0.15,
            "efficiency_scalar": 1.0
        })

    def simulate(self, policy: PolicySchema) -> SimulationResult:
        """
        Executes a deterministic simulation of policy rule propagation.
        
        Args:
            policy: The candidate governance policy to evaluate.
            
        Returns:
            A SimulationResult object containing the projected system state deltas.
        """
        logger.info(f"Initiating high-fidelity simulation for policy: {policy.name}")
        
        # 1. Propagate rule transformations through specific system vectors
        for transformation in policy.transformations:
            self._apply_transformation(transformation)
            
        # 2. Evaluate systemic shifts (Impact & Entropy)
        entropy_deltas = self._project_entropy_impact(policy.entropy_adjustments)
        
        # 3. Constraint Verification
        violations = self._check_constraints(policy.constraints)
        
        return SimulationResult(
            policy_id=policy.policy_id,
            status="violation" if violations else "success",
            applied_transformations=len(policy.transformations),
            transformed_trust_vectors=self._format_trust_updates(),
            negotiation_parameters=self._negotiation_params,
            task_formation_probabilities=self._task_formation_probs,
            reward_scaling=self._reward_scaling,
            entropy_deltas=entropy_deltas,
            projected_impact_modifiers=policy.impact_modifiers,
            violations=violations,
            metadata={
                "simulation_engine_version": "2.0.0",
                "affected_metrics": policy.affected_metrics
            }
        )

    def _apply_transformation(self, transformation: InfluenceTransformation):
        """Dispatches transformations to the requested propagation path."""
        target = transformation.target_metric.lower()
        op = transformation.operator
        val = transformation.value

        # Determine propagation path
        if "trust" in target:
            self._propagate_to_trust(op, val)
        elif any(k in target for k in ["negotiation", "compromise", "alignment"]):
            self._propagate_to_dict(self._negotiation_params, target, op, val)
        elif any(k in target for k in ["task", "formation", "probability"]):
            self._propagate_to_dict(self._task_formation_probs, target, op, val)
        elif any(k in target for k in ["reward", "scaling", "multiplier"]):
            self._propagate_to_dict(self._reward_scaling, target, op, val)
        else:
            logger.warning(f"Transformation target '{target}' did not match any known propagation path.")

    def _propagate_to_trust(self, operator: TransformationOperator, value: Any):
        """Applies transformation to all trust vectors in the sandbox."""
        try:
            numeric_val = float(value)
        except ValueError:
            return # Skip non-numeric for trust dimensions

        for entity_id in self._trust_map:
            # We transform the primary trust dimension (index 0)
            current = self._trust_map[entity_id][0]
            if operator == TransformationOperator.ADD:
                self._trust_map[entity_id][0] = current + numeric_val
            elif operator == TransformationOperator.MULTIPLY:
                self._trust_map[entity_id][0] = current * numeric_val
            elif operator == TransformationOperator.DECAY:
                self._trust_map[entity_id][0] = current * (1.0 - numeric_val)
            elif operator == TransformationOperator.CLAMP:
                self._trust_map[entity_id][0] = max(0.0, min(1.0, current))

    def _propagate_to_dict(self, target_dict: Dict[str, Any], key: str, operator: TransformationOperator, value: Any):
        """Generic numeric propagation for parameter dictionaries."""
        # Find exact match or first fuzzy match
        actual_key = key if key in target_dict else next((k for k in target_dict if k in key), None)
        
        if not actual_key:
            # Fallback: Create new entry if it looks like a parameter
            actual_key = key
            target_dict[actual_key] = 1.0 if operator == TransformationOperator.MULTIPLY else 0.0

        try:
            numeric_val = float(value)
            current = float(target_dict[actual_key])
            
            if operator == TransformationOperator.ADD:
                target_dict[actual_key] = current + numeric_val
            elif operator == TransformationOperator.MULTIPLY:
                target_dict[actual_key] = current * numeric_val
            elif operator == TransformationOperator.DECAY:
                target_dict[actual_key] = current * (1.0 - numeric_val)
            elif operator == TransformationOperator.CLAMP:
                target_dict[actual_key] = max(0.0, min(1.0, current))
        except ValueError:
            # Handle non-numeric (e.g. CUSTOM operator or string expressions)
            if operator == TransformationOperator.CUSTOM:
                target_dict[actual_key] = value

    def _project_entropy_impact(self, adjustments: Dict[str, float]) -> Dict[str, float]:
        """Calculates projected shifts in system entropy."""
        deltas = {}
        for scope, delta in adjustments.items():
            deltas[scope] = delta
        return deltas

    def _check_constraints(self, constraints: List[ExecutableConstraint]) -> List[str]:
        """Validates the simulated state against policy constraints."""
        violations = []
        for constraint in constraints:
            # Note: In a production environment, this would use a safe eval or DSL parser.
            # For simulation, we check for presence of violation markers or extreme values.
            if "max" in constraint.expression and any(v > 10.0 for v in self._negotiation_params.values()):
                violations.append(f"Constraint Violation: {constraint.expression}")
        return violations

    def _format_trust_updates(self) -> List[Dict[str, Any]]:
        """Converts internal trust map back to list format for results."""
        return [
            {"entity_id": eid, "values": tuple(vals)} 
            for eid, vals in self._trust_map.items()
        ]

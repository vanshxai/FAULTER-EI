"""
AGENT 09 — P101_v_fabric
Fabric surface velocity.

The fabric leaves the roll at the surface speed of the outer layer:
    v = ω_roll · r   [m/s]

This value feeds everything downstream: dancer, tensioner, feed roller.
"""

import numpy as np

from core.formula_agent import FormulaAgent


class VFabricAgent(FormulaAgent):
    """Fabric surface velocity agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_v_fabric",
            formula_name="Fabric surface velocity",
            required_inputs=[
                "omega_roll", "r_next", "T_roll",
                "v_fabric_prev", "tick", "t",
                "dt", "depletion_pct",
            ],
            output_keys=[
                "v_fabric", "v_fabric_anomaly",
                "v_fabric_plausibility",
            ],
            bounds={
                # fabric speed: never negative, machine max 5 m/s
                "v_fabric": (np.float64(0.0), np.float64(5.0)),  # [m/s]
            },
            n_adjustments=1,
            validator_input_keys=["omega_roll", "r_next"],
            seed=109,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        omega_roll = inputs["omega_roll"]  # [rad/s] roll angular velocity
        r_next = inputs["r_next"]          # [m]     current outer radius

        # Examiner adjustment: velocity scale [0.85, 1.15]
        # (captures slip/coupling drift between shaft and outer layer)
        v_scale = np.float64(adjustments[0])  # [-]

        # v = ω_roll · r   [m/s]
        # surface speed of fabric leaving the roll
        v_fabric = omega_roll * r_next * v_scale

        return {"v_fabric": v_fabric}

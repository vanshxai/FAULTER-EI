"""
AGENT 03 — P101_j_roll
Roll moment of inertia.

The wound fabric is a hollow cylinder rotating about its axis:
    J = ½·m·(r_outer² + r_core²)   [kg·m²]
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import R_CORE


class JRollAgent(FormulaAgent):
    """Roll moment of inertia agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_j_roll",
            formula_name="Roll moment of inertia",
            required_inputs=[
                "m_roll", "r_next", "omega_roll",
                "T_roll", "tick", "t", "dt",
                "v_fabric_prev",
            ],
            output_keys=[
                "J_roll", "J_roll_anomaly",
                "J_roll_plausibility",
            ],
            bounds={
                # inertia of wound roll: 0 (empty) .. 40 kg·m²
                # full roll per spec geometry: ½·626·(0.3²+0.05²)
                # ≈ 29 kg·m², ceiling sits above with examiner headroom
                "J_roll": (np.float64(0.0), np.float64(40.0)),  # [kg·m²]
            },
            n_adjustments=1,
            validator_input_keys=["m_roll", "r_next"],
            seed=103,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        m_roll = inputs["m_roll"]  # [kg] mass of fabric on roll
        r_next = inputs["r_next"]  # [m]  current outer radius

        # Examiner adjustment: inertia scale, range [0.85, 1.15]
        J_scale = np.float64(adjustments[0])  # [-]

        # J = ½·m·(r_outer² + r_core²)   [kg·m²]
        # hollow cylinder about its rotation axis
        J_roll = (np.float64(0.5) * m_roll *
                  (r_next ** 2 + R_CORE ** 2) * J_scale)

        return {"J_roll": J_roll}

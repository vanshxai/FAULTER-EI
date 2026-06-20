"""
AGENT 02 — P101_m_roll
Roll mass.

The fabric remaining on the roll is a hollow cylinder between the
core radius and the current outer radius:
    m = ρ·π·(r_outer² - r_core²)·w   [kg]
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import R_CORE, W_FABRIC


class MRollAgent(FormulaAgent):
    """Roll mass agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_m_roll",
            formula_name="Roll mass",
            required_inputs=[
                "r_next", "rho_eff", "omega_roll",
                "T_roll", "tick", "t", "dt",
                "v_fabric_prev",
            ],
            output_keys=[
                "m_roll", "m_roll_anomaly",
                "m_roll_plausibility",
            ],
            bounds={
                # mass of fabric on roll: 0 (empty) .. 800 kg
                # full roll per spec geometry is ρ·π·(R_MAX²-R_CORE²)·w
                # ≈ 626 kg, so the hard ceiling sits above that with
                # headroom for the ±15% examiner density adjustment
                "m_roll": (np.float64(0.0), np.float64(800.0)),  # [kg]
            },
            n_adjustments=1,
            validator_input_keys=["r_next", "rho_eff"],
            seed=102,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        r_next = inputs["r_next"]    # [m]     current outer radius
        rho_eff = inputs["rho_eff"]  # [kg/m³] effective wound density

        # Examiner adjustment: density scale, range [0.85, 1.15]
        rho_scale = np.float64(adjustments[0])  # [-]

        # m = ρ·π·(r_outer² - r_core²)·w   [kg]
        # hollow cylinder of wound fabric, width w
        m_roll = (rho_eff * rho_scale * np.pi *
                  (r_next ** 2 - R_CORE ** 2) * W_FABRIC)

        return {"m_roll": m_roll}

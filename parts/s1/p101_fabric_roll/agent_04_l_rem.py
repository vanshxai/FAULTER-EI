"""
AGENT 04 — P101_l_rem
Fabric length remaining.

The wound fabric forms an Archimedes spiral. Its total length equals
the annulus cross-section area divided by the fabric thickness:
    L = π·(r² - r_core²) / t_fabric   [m]
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import R_CORE, T_FABRIC


class LRemAgent(FormulaAgent):
    """Fabric length remaining agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_l_rem",
            formula_name="Fabric length remaining",
            required_inputs=[
                "r_next", "depletion_pct", "N_layers",
                "tick", "t", "dt",
                "v_fabric_prev", "omega_roll",
            ],
            output_keys=[
                "L_rem", "L_rem_anomaly",
                "L_rem_plausibility",
            ],
            bounds={
                # remaining fabric length: 0 .. 80 km per roll
                "L_rem": (np.float64(0.0), np.float64(80000.0)),  # [m]
            },
            n_adjustments=1,
            validator_input_keys=["r_next", "depletion_pct"],
            seed=104,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        r_next = inputs["r_next"]  # [m] current outer radius

        # Examiner adjustment: effective fabric thickness scale [0.85, 1.15]
        # (real thickness drifts with winding compression)
        t_fabric_scale = np.float64(adjustments[0])  # [-]
        t_eff = T_FABRIC * t_fabric_scale            # [m] effective thickness

        # L = π·(r² - r_core²) / t_fabric   [m]
        # Archimedes spiral: annulus area / layer thickness
        L_rem = np.pi * (r_next ** 2 - R_CORE ** 2) / t_eff

        return {"L_rem": L_rem}

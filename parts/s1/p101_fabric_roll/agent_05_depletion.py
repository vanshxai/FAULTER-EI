"""
AGENT 05 — P101_depletion
Roll depletion percentage.

How much usable fabric radius remains:
    depletion_pct = (r - r_core) / (r_max - r_core) · 100   [%]
100% = full roll, 0% = empty (down to bare core).
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import R_MAX, R_CORE


class DepletionAgent(FormulaAgent):
    """Roll depletion percentage agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_depletion",
            formula_name="Roll depletion percentage",
            required_inputs=[
                "r_next", "dr_dt", "L_rem",
                "v_fabric_prev", "tick", "t",
                "dt", "omega_roll",
            ],
            output_keys=[
                "depletion_pct", "depletion_warning",
                "depletion_anomaly",
                "depletion_plausibility",
            ],
            bounds={
                # percentage is hard-bounded to [0, 100]
                "depletion_pct": (np.float64(0.0), np.float64(100.0)),  # [%]
            },
            n_adjustments=1,
            validator_input_keys=["r_next", "dr_dt"],
            seed=105,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        r_next = inputs["r_next"]  # [m] current outer radius

        # Examiner adjustment: scales the effective empty-core reference
        # radius (where "0% remaining" actually sits — cores vary, and
        # the last layers near the core are often unusable). [0.85, 1.15]
        depletion_scale = np.float64(adjustments[0])  # [-]
        r_core_eff = R_CORE * depletion_scale         # [m] effective core

        # depletion_pct = (r - r_core) / (r_max - r_core) · 100   [%]
        # fraction of usable radius remaining, as a percentage
        depletion_pct = ((r_next - r_core_eff) /
                         (R_MAX - r_core_eff) * np.float64(100.0))

        # warning flag: roll nearly empty below 10% remaining  [-]
        depletion_warning = (np.float64(1.0)
                             if depletion_pct < np.float64(10.0)
                             else np.float64(0.0))

        return {"depletion_pct": depletion_pct,
                "depletion_warning": depletion_warning}

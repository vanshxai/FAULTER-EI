"""
AGENT 08 — P101_sigma_res
Wound-in residual tension.

During factory winding, tension T_wind is locked into every layer.
The stored stress at the current outer surface follows a log law:
    σ = (T_wind / (w·t)) · ln(r_max / r)   [Pa]

As outer layers unwind they release this stored stress, causing a
slow tension drift downstream.
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import R_MAX, R_CORE, W_FABRIC, T_FABRIC, T_WIND


class SigmaResAgent(FormulaAgent):
    """Wound-in residual tension agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_sigma_res",
            formula_name="Wound-in residual tension",
            required_inputs=[
                "r_next", "T_roll", "omega_roll",
                "depletion_pct", "tick", "t",
                "dt", "v_fabric_prev",
            ],
            output_keys=[
                "sigma_res", "sigma_res_anomaly",
                "sigma_res_plausibility",
            ],
            bounds={
                # residual stress: 0 .. 500 kPa physical maximum
                "sigma_res": (np.float64(0.0), np.float64(500000.0)),  # [Pa]
            },
            n_adjustments=1,
            validator_input_keys=["r_next", "T_roll"],
            seed=108,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        r_next = inputs["r_next"]  # [m] current outer radius

        # Examiner adjustment: effective winding tension scale [0.85, 1.15]
        # (factory winding tension varies roll to roll)
        T_wind_scale = np.float64(adjustments[0])  # [-]
        T_wind_eff = T_WIND * T_wind_scale         # [N] effective wound tension

        # σ = (T_wind / (w·t)) · ln(r_max / r)   [Pa]
        # T_wind/(w·t): winding stress per cross-section  [N/m² = Pa]
        # ln(r_max/r):  log accumulation toward the core  [-]
        sigma_res = ((T_wind_eff / (W_FABRIC * T_FABRIC)) *
                     np.log(R_MAX / np.maximum(r_next, R_CORE)))

        return {"sigma_res": sigma_res}

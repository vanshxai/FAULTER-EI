"""
AGENT 07 — P101_rho_eff
Effective roll density.

A wound roll is not solid fiber — air is trapped between layers.
The effective density is a volume-weighted mix:
    ρ_eff = ρ_fiber·(1 - φ) + ρ_air·φ   [kg/m³]

The air fraction φ drifts with winding tension; the examiner
learns that drift (±15%).
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import RHO_FIBER, RHO_AIR, PHI_AIR


class RhoEffAgent(FormulaAgent):
    """Effective roll density agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_rho_eff",
            formula_name="Effective roll density",
            required_inputs=[
                "omega_roll", "T_roll", "r_next",
                "m_roll", "tick", "t",
                "dt", "v_fabric_prev",
            ],
            output_keys=[
                "rho_eff", "rho_eff_anomaly",
                "rho_eff_plausibility",
            ],
            bounds={
                # plausible wound-fabric density range
                "rho_eff": (np.float64(100.0), np.float64(2000.0)),  # [kg/m³]
            },
            n_adjustments=1,
            validator_input_keys=["omega_roll", "T_roll"],
            seed=107,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        # Examiner adjustment: effective air-fraction scale [0.85, 1.15]
        # (tighter winding squeezes air out; looser traps more)
        phi_air_scale = np.float64(adjustments[0])  # [-]
        phi_eff = PHI_AIR * phi_air_scale           # [-] effective air fraction

        # ρ_eff = ρ_fiber·(1 - φ) + ρ_air·φ   [kg/m³]
        # volume-weighted density of fiber + trapped air
        rho_eff = (RHO_FIBER * (np.float64(1.0) - phi_eff) +
                   RHO_AIR * phi_eff)

        return {"rho_eff": rho_eff}

"""
AGENT 06 — P101_n_layers
Number of fabric layers.

The radial build-up between core and outer surface, divided by one
fabric thickness, gives the layer count:
    N = (r_outer - r_core) / t_fabric   [-, count]
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import R_CORE, T_FABRIC


class NLayersAgent(FormulaAgent):
    """Number of fabric layers agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_n_layers",
            formula_name="Number of fabric layers",
            required_inputs=[
                "r_next", "depletion_pct", "L_rem",
                "T_roll", "tick", "t",
                "dt", "omega_roll",
            ],
            output_keys=[
                "N_layers", "N_layers_anomaly",
                "N_layers_plausibility",
            ],
            bounds={
                # layer count: 0 (bare core) .. 500 layers max
                "N_layers": (np.float64(0.0), np.float64(500.0)),  # [-]
            },
            n_adjustments=1,
            validator_input_keys=["r_next", "depletion_pct"],
            seed=106,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        r_next = inputs["r_next"]  # [m] current outer radius

        # Examiner adjustment: effective layer thickness scale [0.85, 1.15]
        # (wound layers compress under tension, thinning each layer)
        t_scale = np.float64(adjustments[0])  # [-]
        t_eff = T_FABRIC * t_scale            # [m] effective layer thickness

        # N = (r_outer - r_core) / t_fabric   [-, count of layers]
        N_layers = (r_next - R_CORE) / t_eff

        return {"N_layers": N_layers}

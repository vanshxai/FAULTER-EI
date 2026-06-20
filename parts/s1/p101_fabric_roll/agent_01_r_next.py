"""
AGENT 01 — P101_r_next
Roll radius shrinkage.

As fabric unwinds at surface speed v, the roll loses one fabric
thickness t_fabric per revolution. One revolution takes 2π·r/v
seconds, so the radius shrinks continuously:
    dr/dt = -v_fabric·t_fabric / (2π·r)   [m/s]
"""

import numpy as np

from core.formula_agent import FormulaAgent
from core.constants import R_MAX, R_CORE, T_FABRIC


class RNextAgent(FormulaAgent):
    """Roll radius shrinkage agent."""

    def __init__(self):
        super().__init__(
            agent_id="P101_r_next",
            formula_name="Roll radius shrinkage",
            required_inputs=[
                "r_current", "v_fabric_prev",
                "omega_roll", "tick", "t",
                "depletion_pct", "dt", "T_roll",
            ],
            output_keys=[
                "r_next", "dr_dt",
                "r_next_anomaly",
                "r_next_plausibility",
            ],
            bounds={
                # radius can never leave the physical roll envelope
                "r_next": (R_CORE, R_MAX),        # [m]
                # radius only shrinks while unwinding: dr/dt <= 0
                "dr_dt": (-np.inf, np.float64(0.0)),  # [m/s]
            },
            n_adjustments=1,
            validator_input_keys=["r_current", "v_fabric_prev"],
            seed=101,
        )

    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        r_current = inputs["r_current"]          # [m]   current roll radius
        v_fabric_prev = inputs["v_fabric_prev"]  # [m/s] fabric surface speed
        dt = inputs["dt"]                        # [s]   timestep

        # Examiner adjustment: scale factor on dr/dt, range [0.85, 1.15]
        dr_dt_scale = np.float64(adjustments[0])  # [-]

        # Guard the division: radius never below core in the denominator
        r_safe = np.maximum(r_current, R_CORE)   # [m]

        # dr/dt = -v_fabric·t_fabric / (2π·r)   [m/s]
        # one fabric-thickness lost per revolution (period 2π·r/v),
        # smeared continuously: [m/s]·[m] / [m] = [m/s]
        dr_dt = (-v_fabric_prev * T_FABRIC /
                 (np.float64(2.0) * np.pi * r_safe))
        dr_dt = dr_dt * dr_dt_scale              # examiner-scaled [m/s]

        # r_next = r + (dr/dt)·dt   [m]
        # Floor at R_CORE is enforced by the hard bound (clamp + anomaly),
        # so a below-core input is both corrected AND flagged.
        r_next = r_current + dr_dt * dt          # [m]

        return {"r_next": r_next, "dr_dt": dr_dt}

"""
Part 1.01 — Fabric Roll.
9 formula agents assembled into a 4-layer blackboard pipeline.

Execution groups (dependency order):
    Layer 1: rho_eff                       (constants only)
    Layer 2: r_next                        (needs v_fabric_prev)
    Layer 3: m_roll, sigma_res             (need r_next [, rho_eff])
    Layer 4: J_roll, L_rem, depletion,
             N_layers, v_fabric            (need r_next [, m_roll])
"""

import numpy as np

from core.agent_layer import AgentLayer
from core.agent_pipeline import AgentPipeline
from core.constants import R_MAX

from .agent_01_r_next import RNextAgent
from .agent_02_m_roll import MRollAgent
from .agent_03_j_roll import JRollAgent
from .agent_04_l_rem import LRemAgent
from .agent_05_depletion import DepletionAgent
from .agent_06_n_layers import NLayersAgent
from .agent_07_rho_eff import RhoEffAgent
from .agent_08_sigma_res import SigmaResAgent
from .agent_09_v_fabric import VFabricAgent


def build_p101_pipeline() -> AgentPipeline:
    """Instantiate all 9 agents and assemble the 4-layer pipeline."""
    layers = [
        # Layer 1 — no dependencies (constants only)
        AgentLayer(1, [RhoEffAgent()]),

        # Layer 2 — needs v_fabric_prev from state feedback
        AgentLayer(2, [RNextAgent()]),

        # Layer 3 — need fresh r_next (and rho_eff for mass)
        AgentLayer(3, [MRollAgent(), SigmaResAgent()]),

        # Layer 4 — need fresh r_next (and m_roll for inertia)
        AgentLayer(4, [JRollAgent(), LRemAgent(), DepletionAgent(),
                       NLayersAgent(), VFabricAgent()]),
    ]
    return AgentPipeline("P101_fabric_roll", layers)


def get_initial_state() -> dict:
    """Clean starting blackboard: a brand-new full roll at rest."""
    return {
        "tick": 0,
        "t": np.float64(0.0),              # [s]     simulation time
        "dt": np.float64(0.02),            # [s]     timestep (50 Hz)
        "r_current": np.float64(R_MAX),    # [m]     full roll = 0.300
        "omega_roll": np.float64(10.0),    # [rad/s] from motor
        "T_roll": np.float64(5.0),         # [Nm]    from gear
        "v_fabric_prev": np.float64(0.0),  # [m/s]   previous tick speed
        "depletion_pct": np.float64(100.0),  # [%]   full roll
        "N_layers": np.float64(250.0),     # [-]     full roll layer count
        "L_rem": np.float64(0.0),          # [m]     filled on first tick
        "m_roll": np.float64(0.0),         # [kg]    filled on first tick
        "J_roll": np.float64(0.0),         # [kg·m²] filled on first tick
        "rho_eff": np.float64(0.0),        # [kg/m³] filled on first tick
        "sigma_res": np.float64(0.0),      # [Pa]    filled on first tick
        "v_fabric": np.float64(0.0),       # [m/s]   filled on first tick
        "dr_dt": np.float64(0.0),          # [m/s]   filled on first tick
        "anomaly_map": {},                 # agent_id -> anomaly flag
    }

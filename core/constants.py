"""
Fixed physical constants for the Nassenger 8 digital twin.

These are material properties and machine geometry.
They are FIXED — not adjustable, not learned by any neural network.
Organized by part. float64 everywhere.
"""

import numpy as np

# =====================================================================
# PART 1.01 — FABRIC ROLL
# =====================================================================

# --- Fabric roll geometry ---
R_MAX = np.float64(0.300)      # m    full roll outer radius
R_CORE = np.float64(0.050)     # m    empty core radius
W_FABRIC = np.float64(1.850)   # m    fabric width
T_FABRIC = np.float64(0.001)   # m    fabric thickness (1 mm)

# --- Fabric material (default: cotton) ---
RHO_FIBER = np.float64(1540.0)  # kg/m³  cotton fiber density
RHO_AIR = np.float64(1.225)     # kg/m³  air density
PHI_AIR = np.float64(0.20)      # -      air volume fraction in wound roll

# Effective fabric density of the wound roll.
# Volume-weighted mix of fiber and trapped air.
# ρ_eff = ρ_fiber·(1 - φ) + ρ_air·φ   [kg/m³]
# Computed here, never hardcoded.
RHO_FABRIC = RHO_FIBER * (np.float64(1.0) - PHI_AIR) + RHO_AIR * PHI_AIR
#           = 1540·0.80 + 1.225·0.20
#           = 1232.0 + 0.245
#           = 1232.245 kg/m³

# --- Wound-in tension (from factory winding) ---
T_WIND = np.float64(50.0)      # N    winding tension applied at factory

# --- Gravity ---
G = np.float64(9.81)           # m/s²  gravitational acceleration

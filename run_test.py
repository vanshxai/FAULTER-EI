"""
P101 Fabric Roll — 500-tick test run with fault injection.

  - Motor spinup: omega_roll ramps 10 -> 50 rad/s
  - Tick 250: overspeed fault (omega_roll = 500 rad/s, impossible)
  - Tick 300: overspeed cleared
  - Tick 350: below-core radius fault (r_current = 0.049 m)

Run: python run_test.py
"""

import numpy as np

from parts.s1.p101_fabric_roll import (
    build_p101_pipeline,
    get_initial_state,
)


def main():
    pipeline = build_p101_pipeline()
    state = get_initial_state()

    print("P101 Fabric Roll — 9 agents built")
    print("=" * 100)
    header = (f"{'tick':>5} | {'r_next':>8} | {'m_roll':>8} | "
              f"{'v_fabric':>9} | {'depl_%':>7} | {'sigma_res':>9} | "
              f"{'rho_eff':>8} | {'anomalies':>9}")
    print(header)
    print("-" * 100)

    overspeed_fired = False
    core_clamp_ok = False
    core_anomaly_fired = False

    for tick in range(500):
        # --- state feedback (previous tick's outputs become inputs) ---
        if tick > 0:
            state["r_current"] = state["r_next"]
            state["v_fabric_prev"] = state["v_fabric"]

        # --- slowly increase omega, simulating motor spinup ---
        # omega = min(10 + 0.05·tick, 50)  [rad/s]
        state["omega_roll"] = np.float64(min(10.0 + tick * 0.05, 50.0))

        # --- FAULT 1: impossible overspeed at ticks 250-299 ---
        if 250 <= tick < 300:
            state["omega_roll"] = np.float64(500.0)  # [rad/s] impossible

        # --- FAULT 2: below-core radius at tick 350 ---
        if tick == 350:
            state["r_current"] = np.float64(0.049)   # [m] below R_CORE

        state = pipeline.step(state)

        # --- fault verification checks ---
        if 250 <= tick < 300 and state["v_fabric_anomaly"] > 0.3:
            if not overspeed_fired:
                print(f">>> tick {tick}: v_fabric_anomaly fired "
                      f"({state['v_fabric_anomaly']:.2f}) — "
                      f"overspeed detected")
            overspeed_fired = True

        if tick == 350:
            core_clamp_ok = abs(state["r_next"] - 0.050) < 1e-9
            core_anomaly_fired = state["r_next_anomaly"] > 0.3
            print(f">>> tick 350: r_next = {state['r_next']:.6f} m "
                  f"(clamped to R_CORE: {core_clamp_ok}), "
                  f"r_next_anomaly = {state['r_next_anomaly']:.2f} "
                  f"(fired: {core_anomaly_fired})")

        # --- periodic report every 50 ticks ---
        if (tick + 1) % 50 == 0:
            anomaly_count = sum(
                1 for flag in state["anomaly_map"].values() if flag > 0.3)
            print(f"{state['tick']:>5} | "
                  f"{state['r_next']:>8.4f} | "
                  f"{state['m_roll']:>8.2f} | "
                  f"{state['v_fabric']:>9.4f} | "
                  f"{state['depletion_pct']:>7.2f} | "
                  f"{state['sigma_res']:>9.0f} | "
                  f"{state['rho_eff']:>8.2f} | "
                  f"{anomaly_count:>9}")

    # --- final verification summary ---
    print("=" * 100)
    print("FAULT VERIFICATION")
    print(f"  overspeed  : v_fabric_anomaly fired ... "
          f"{'PASS' if overspeed_fired else 'FAIL'}")
    print(f"  below-core : r_next clamped to R_CORE . "
          f"{'PASS' if core_clamp_ok else 'FAIL'}")
    print(f"  below-core : r_next_anomaly fired ..... "
          f"{'PASS' if core_anomaly_fired else 'FAIL'}")

    # --- final anomaly report ---
    print("=" * 100)
    print("FINAL ANOMALY REPORT (anomaly > 0.3, sorted by severity)")
    report = pipeline.get_anomaly_report()
    if not report:
        print("  no agents above threshold")
    for agent_id, flag in report:
        print(f"  {agent_id:<20} {flag:.4f}")


if __name__ == "__main__":
    main()

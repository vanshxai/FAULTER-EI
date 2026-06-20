"""
FormulaAgent — the atomic neurosymbolic unit.

Three components:
  1. EXAMINER  (ExaminerNN)  — tweaks formula constants within ±15%
  2. FORMULA   (exact physics) — implemented by each subclass
  3. VALIDATOR (ValidatorNN) — plausibility + anomaly scoring

The formula is never approximated and never replaced.
Bounds are hard limits: violation = clamp + anomaly = 1.0.
"""

import numpy as np
from collections import deque

from .base_nn import ExaminerNN, ValidatorNN


class FormulaAgent:
    """
    Base class for all formula agents.

    Subclasses must:
      - implement _run_formula(inputs, adjustments) -> dict
      - pass agent_id, formula_name, required_inputs, output_keys, bounds

    Conventions:
      - output_keys[0] is the primary output (the physics value).
      - output_keys containing "anomaly"/"plausibility" are filled
        by the base class.
      - bounds maps output key -> (lo, hi) hard physical limits.
      - validator_input_keys names the 2 primary inputs fed to the
        validator (defaults to the first two required_inputs).
    """

    def __init__(self,
                 agent_id: str,
                 formula_name: str,
                 required_inputs: list,
                 output_keys: list,
                 bounds: dict,
                 dt: float = 0.02,
                 n_adjustments: int = 1,
                 validator_input_keys: list = None,
                 seed: int = 0):
        self.agent_id = agent_id
        self.formula_name = formula_name
        self.required_inputs = list(required_inputs)
        self.output_keys = list(output_keys)
        self.bounds = dict(bounds)
        self.dt = np.float64(dt)

        # Primary output is the first output key
        self.primary_key = self.output_keys[0]

        # Anomaly / plausibility key names (found in output_keys)
        self.anomaly_key = next(k for k in self.output_keys
                                if "anomaly" in k)
        self.plausibility_key = next(k for k in self.output_keys
                                     if "plausibility" in k)

        # Two primary inputs for the validator vector
        self.validator_input_keys = (validator_input_keys or
                                     self.required_inputs[:2])

        # History of primary output, last 300 ticks
        self.history = deque(maxlen=300)
        self.tick = 0
        self.prev_output = np.float64(0.0)

        # Per-key violation latch: log on entry into violation,
        # not on every tick it persists
        self._in_violation = {key: False for key in self.bounds}

        # UI telemetry — last internal state, exposed to server/main.py
        self.last_exam_input  = []       # list[float]
        self.last_exam_output = []       # list[float]
        self.last_formula_out = 0.0      # float
        self.last_val_output  = (0.5, 0.0)  # (plausibility, anomaly)
        self.fired_this_tick  = False    # True while current tick is live
        self.anomaly_flag     = 0.0      # current anomaly (mirrors last_val_output[1])

        # Neural networks — examiner sized to this agent's inputs
        self.examiner = ExaminerNN(n_in=len(self.required_inputs),
                                   n_out=n_adjustments,
                                   seed=seed)
        self.validator = ValidatorNN(n_in=5, seed=seed + 1000)

    # -----------------------------------------------------------------
    def _run_formula(self, inputs: dict, adjustments: np.ndarray) -> dict:
        """
        Exact symbolic physics. Implemented per agent.
        Returns dict of formula outputs (primary key first).
        """
        raise NotImplementedError

    # -----------------------------------------------------------------
    def forward(self, state: dict) -> dict:
        """
        Full neurosymbolic pass: examine -> formula -> validate -> bound.
        Never raises in normal operation; the layer catches anything else.
        """
        self.tick += 1

        # Step 1: extract required inputs from blackboard
        inputs = {}
        for key in self.required_inputs:
            inputs[key] = np.float64(state.get(key, 0.0))

        # Step 2: build examiner input vector (fixed order)
        x_exam = np.array([inputs[k] for k in self.required_inputs],
                          dtype=np.float64)

        # Step 3: examiner -> constant adjustments in [0.85, 1.15]
        adjustments = self.examiner.forward(x_exam)
        self.last_exam_input  = x_exam.tolist()
        self.last_exam_output = adjustments.tolist()

        # Step 4: exact formula with examiner-adjusted constants
        formula_out = self._run_formula(inputs, adjustments)
        output = np.float64(formula_out[self.primary_key])
        self.last_formula_out = float(output)

        # Step 5: rate of change of primary output [units/s]
        rate_of_change = (output - self.prev_output) / self.dt

        # Step 6: validator input vector
        x_val = np.array([
            output,
            self.prev_output,
            rate_of_change,
            inputs[self.validator_input_keys[0]],
            inputs[self.validator_input_keys[1]],
        ], dtype=np.float64)

        # Step 7: validator -> plausibility, anomaly
        plausibility, anomaly = self.validator.forward(x_val)
        self.last_val_output = (float(plausibility), float(anomaly))
        self.fired_this_tick = True

        # Step 8: hard bounds — clamp + force anomaly = 1.0 on violation
        for key, (lo, hi) in self.bounds.items():
            if key not in formula_out:
                continue
            val = np.float64(formula_out[key])
            if val < lo or val > hi:
                clamped = np.float64(np.clip(val, lo, hi))
                if not self._in_violation[key]:
                    print(f"[BOUND VIOLATION] tick {self.tick} "
                          f"{self.agent_id}.{key} = {float(val):.6f} "
                          f"outside [{float(lo)}, {float(hi)}] "
                          f"— clamped to {float(clamped):.6f}")
                    self._in_violation[key] = True
                formula_out[key] = clamped
                anomaly = np.float64(1.0)
                plausibility = np.float64(0.0)
            else:
                if self._in_violation[key]:
                    print(f"[BOUND CLEARED] tick {self.tick} "
                          f"{self.agent_id}.{key} back in range")
                    self._in_violation[key] = False

        # Re-read primary output after any clamping
        output = np.float64(formula_out[self.primary_key])

        # Step 9: update history with primary output
        self.history.append(float(output))
        self.prev_output = output
        self.anomaly_flag = float(anomaly)

        # Step 10: build output dict
        result = {k: float(v) for k, v in formula_out.items()}
        result[self.anomaly_key] = float(anomaly)
        result[self.plausibility_key] = float(plausibility)
        return result

    # -----------------------------------------------------------------
    def get_ui_state(self) -> dict:
        """Return latest internal state for the dashboard."""
        return {
            "agent_id":        self.agent_id,
            "formula_name":    self.formula_name,
            "examiner": {
                "input":  self.last_exam_input,
                "output": self.last_exam_output,
            },
            "formula": {
                "output": self.last_formula_out,
            },
            "validator": {
                "plausibility": self.last_val_output[0],
                "anomaly_flag": self.last_val_output[1],
            },
            "anomaly_flag":    self.anomaly_flag,
            "fired_this_tick": self.fired_this_tick,
        }

    # -----------------------------------------------------------------
    def reset(self):
        """Clear history and tick counter. Weights are kept."""
        self.history.clear()
        self.tick = 0
        self.prev_output = np.float64(0.0)
        self._in_violation = {key: False for key in self.bounds}
        self.last_exam_input  = []
        self.last_exam_output = []
        self.last_formula_out = 0.0
        self.last_val_output  = (0.5, 0.0)
        self.fired_this_tick  = False
        self.anomaly_flag     = 0.0

"""
Pure numpy neural networks: ExaminerNN and ValidatorNN.

NO PYTORCH. NO TENSORFLOW. NO FRAMEWORKS.
Manual weight matrices. Manual forward pass.
Manual GELU. Manual LayerNorm. Manual sigmoid.
He initialization. float64 everywhere.
"""

import numpy as np


# ---------------------------------------------------------------------
# Manual activation / normalization primitives
# ---------------------------------------------------------------------

# GELU tanh-approximation constant: sqrt(2/pi) = 0.7978845608
_GELU_C = np.float64(0.7978845608)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU: x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))"""
    x = np.asarray(x, dtype=np.float64)
    return x * np.float64(0.5) * (
        np.float64(1.0) + np.tanh(_GELU_C * (x + np.float64(0.044715) * x ** 3))
    )


def layer_norm(x: np.ndarray) -> np.ndarray:
    """LayerNorm: (x - mean) / (std + 1e-8). No learned gamma/beta."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    sigma = np.std(x) + np.float64(1e-8)
    return (x - mu) / sigma


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid: 1 / (1 + exp(-x)). Clipped for numerical stability."""
    x = np.clip(np.asarray(x, dtype=np.float64), -500.0, 500.0)
    return np.float64(1.0) / (np.float64(1.0) + np.exp(-x))


def _he_init(n_out: int, n_in: int, rng: np.random.Generator) -> np.ndarray:
    """He initialization: W = randn(out, in) * sqrt(2 / in). float64."""
    return (rng.standard_normal((n_out, n_in)) *
            np.sqrt(np.float64(2.0) / np.float64(n_in))).astype(np.float64)


# ---------------------------------------------------------------------
# ExaminerNN
# ---------------------------------------------------------------------

class ExaminerNN:
    """
    Examiner network. Watches blackboard values and outputs small
    constant adjustments in [0.85, 1.15] (±15%).

    Architecture:
        Input:  n_in values
        Hidden: n_hidden layers x `hidden` nodes
                Linear -> LayerNorm -> GELU per hidden layer
        Output: n_out adjustments
                Linear -> sigmoid, scaled to [0.85, 1.15]

    ~8,000 parameters for the default 8-in / 2x32 / 1-out shape.
    """

    def __init__(self, n_in: int, n_out: int,
                 hidden: int = 32, n_hidden: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_out = n_out

        # Hidden layer weights: list of (W, b)
        self.layers = []
        prev = n_in
        for _ in range(n_hidden):
            W = _he_init(hidden, prev, rng)
            b = np.zeros(hidden, dtype=np.float64)
            self.layers.append((W, b))
            prev = hidden

        # Output layer
        self.W_out = _he_init(n_out, prev, rng)
        self.b_out = np.zeros(n_out, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        x: (n_in,) float64 array of blackboard values.
        Returns: (n_out,) float64 adjustments in [0.85, 1.15].
        """
        h = np.asarray(x, dtype=np.float64)

        # Hidden layers: Linear -> LayerNorm -> GELU
        for W, b in self.layers:
            h = W @ h + b
            h = layer_norm(h)
            h = gelu(h)

        # Output: Linear -> sigmoid scaled to [0.85, 1.15]
        z = self.W_out @ h + self.b_out
        return np.float64(0.85) + sigmoid(z) * np.float64(0.30)


# ---------------------------------------------------------------------
# ValidatorNN
# ---------------------------------------------------------------------

class ValidatorNN:
    """
    Validator network. Checks output plausibility and rate of change.

    Architecture:
        Input:  5 values
                [output, prev_output, rate_of_change,
                 primary_input_1, primary_input_2]
        Hidden: 1 layer x 16 nodes, GELU
        Output: 2 values (plausibility, anomaly_flag), sigmoid

    ~2,000 parameters (well under, but tiny by design).
    """

    def __init__(self, n_in: int = 5, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in

        # Hidden layer
        self.W1 = _he_init(hidden, n_in, rng)
        self.b1 = np.zeros(hidden, dtype=np.float64)

        # Output layer: 2 values
        self.W2 = _he_init(2, hidden, rng)
        self.b2 = np.zeros(2, dtype=np.float64)

    def forward(self, x: np.ndarray) -> tuple:
        """
        Forward pass.
        x: (5,) float64 array.
        Returns: (plausibility, anomaly_flag), both float64 in [0, 1].
        """
        h = np.asarray(x, dtype=np.float64)

        # Hidden: Linear -> GELU
        h = gelu(self.W1 @ h + self.b1)

        # Output: Linear -> sigmoid
        out = sigmoid(self.W2 @ h + self.b2)
        return np.float64(out[0]), np.float64(out[1])

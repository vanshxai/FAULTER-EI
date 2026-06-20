"""
AgentPipeline — blackboard-driven execution engine.

Layers execute sequentially; agents within a layer execute in
parallel. After every tick the full blackboard is snapshotted to a
300-tick history buffer.
"""

import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class AgentPipeline:
    """Executes layers against the shared blackboard state."""

    def __init__(self, pipeline_id: str, layers: list, dt: float = 0.02):
        self.pipeline_id = pipeline_id
        self.layers = list(layers)
        self.dt = np.float64(dt)
        self.tick = 0
        self.history = deque(maxlen=300)
        self.executor = ThreadPoolExecutor(max_workers=8)

    def step(self, state: dict) -> dict:
        """Advance the simulation one tick (dt = 0.02 s)."""
        self.tick += 1
        state["tick"] = self.tick
        state["t"] = float(self.tick * self.dt)

        # Layers run sequentially; agents within a layer in parallel
        for layer in self.layers:
            state = layer.execute(state, self.executor)

        # Full blackboard snapshot to history
        self.history.append(state.copy())
        return state

    def get_anomaly_report(self) -> list:
        """
        All agents with anomaly > 0.3 in the latest tick,
        sorted by severity (highest first).
        Returns list of (agent_id, anomaly) tuples.
        """
        if not self.history:
            return []
        anomaly_map = self.history[-1].get("anomaly_map", {})
        flagged = [(aid, flag) for aid, flag in anomaly_map.items()
                   if flag > 0.3]
        return sorted(flagged, key=lambda item: item[1], reverse=True)

    def get_history(self, key: str, n: int = 300) -> np.ndarray:
        """Last n values of a blackboard key, as a float64 array."""
        values = [snap[key] for snap in self.history if key in snap]
        return np.array(values[-n:], dtype=np.float64)

    def reset(self):
        """Reset tick counter, history, and every agent (weights kept)."""
        self.tick = 0
        self.history.clear()
        for layer in self.layers:
            for agent in layer.agents:
                agent.reset()

"""
AgentLayer — one parallel execution group.

All agents in a layer are submitted to a shared ThreadPoolExecutor,
then a barrier waits for ALL futures before merging outputs back
into the blackboard. An agent exception never crashes the pipeline:
it is logged, the agent's anomaly is forced to 1.0, and the rest
continue.
"""

import traceback
from concurrent.futures import ThreadPoolExecutor


class AgentLayer:
    """One layer of agents executed in parallel."""

    def __init__(self, layer_id: int, agents: list):
        self.layer_id = layer_id
        self.agents = list(agents)

    def execute(self, state: dict, executor: ThreadPoolExecutor) -> dict:
        """
        Run every agent in this layer against a snapshot of the state,
        barrier-wait, then merge all outputs into the live state.
        Returns the enriched state.
        """
        # Submit all agents — each gets its own copy of the state
        futures = {
            executor.submit(agent.forward, state.copy()): agent
            for agent in self.agents
        }

        anomaly_map = state.setdefault("anomaly_map", {})

        # Barrier: wait for ALL futures, merge results
        for future, agent in futures.items():
            try:
                outputs = future.result()
                state.update(outputs)
                anomaly_map[agent.agent_id] = outputs.get(
                    agent.anomaly_key, 0.0)
            except Exception:
                # Never crash the pipeline — log, flag, continue
                print(f"[AGENT ERROR] layer {self.layer_id} "
                      f"agent {agent.agent_id}:")
                traceback.print_exc()
                anomaly_map[agent.agent_id] = 1.0
                state[agent.anomaly_key] = 1.0
                state[agent.plausibility_key] = 0.0

        return state

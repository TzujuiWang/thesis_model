import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from 论文.project.agents.base import AgentBase


class ReplacementEngine:
    """
    [Source 1 Section 4.1 & 4.2] 破产替换引擎。
    """

    def __init__(self,
                 config: Dict[str, Any],
                 agent_factory_func: Callable[[int, str], AgentBase]):

        self.config = config
        self.agent_factory = agent_factory_func

        rep_cfg = config.get('replacement_policy', {})
        self.mode = rep_cfg.get('mode', 'steady_state')
        self.threshold = float(rep_cfg.get('bankruptcy_threshold', 1e-9))
        self.id_strategy = rep_cfg.get('id_strategy', 'reuse')  # reuse or new

        # Newborn policy
        self.newborn_cfg = rep_cfg.get('newborn', {})

    def process_bankruptcies(self,
                             agents: List[AgentBase],
                             current_price: float,
                             current_fundamental: float,  # For completeness if wealth uses it
                             current_period: int) -> Tuple[List[AgentBase], Dict[str, Any]]:

        survivors = []
        stats = {"removed": 0, "replaced": 0, "by_type": {}}

        for agent in agents:
            # 1. Force Wealth Update (Ensure mark-to-market is current)
            # This handles pending settlements implicitly via update_wealth_stats logic
            agent.state.process_settlements(current_period)  # Defensive: ensure processed
            agent.state.update_wealth_stats(current_price)

            # 2. Check Bankruptcy
            # Delegate to state for definition consistency
            if agent.state.is_bankrupt(threshold=self.threshold):
                # Record Stats
                stats["by_type"][agent.type] = stats["by_type"].get(agent.type, 0) + 1

                if self.mode == 'survival':
                    # 4.1: Remove
                    stats["removed"] += 1
                    # Do not append to survivors

                elif self.mode == 'steady_state':
                    # 4.2: Replace
                    stats["replaced"] += 1

                    new_id = agent.id if self.id_strategy == 'reuse' else -1  # -1 -> logic for new ID needed
                    new_agent = self.agent_factory(new_id, agent.type)

                    # Apply Newborn Policy (e.g. init u_ewma from current market?)
                    self._apply_newborn_policy(new_agent, current_price, current_fundamental)

                    survivors.append(new_agent)
            else:
                survivors.append(agent)

        return survivors, stats

    def _apply_newborn_policy(self, agent: AgentBase, price: float, fundamental: float):
        """
        [Advanced Replication] 初始化新人的信念状态，使其不至于完全脱节。
        """
        init_u = self.newborn_cfg.get('init_u_from', 'initial_pd')  # default to config default

        if hasattr(agent, 'rules'):  # Learning Agent
            current_pd = price + 0.0  # D is not passed here? Assume P contains info or P+Dbar
            # Refinement: pass current_dividend to process_bankruptcies if needed.
            # Usually init from Price is enough approximation.

            if init_u == 'current_pd':
                for rule in agent.rules:
                    rule.u_ewma = current_pd  # Reset anchor to current market
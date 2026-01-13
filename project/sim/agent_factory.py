import numpy as np
from typing import Dict, Any, Tuple

from 论文.project.agents.base import AgentBase
from 论文.project.agents.learning import LearningAgent
from 论文.project.agents.noise import NoiseAgent
from 论文.project.learning.gp.tree import GPTreeFactory
from 论文.project.learning.evolution import EvolutionaryEngine


class AgentFactory:
    """
    负责创建 Agent 实例。
    【关键复现特性】：为每个 Agent 派生独立的 RNG 序列。
    """

    def __init__(self, config: Dict[str, Any], master_seed: int):
        self.config = config
        self.master_seed = master_seed

        # Prepare Config Subsets (Cleanup)
        self.gp_cfg = config.get('learning_gp', {})
        self.belief_cfg = config.get('beliefs', {})
        self.noise_cfg = config.get('noise_trader', {})

        # Prepare Factories (Shared logic, but distinct usage)
        func_names = self.gp_cfg.get('function_set', {}).get('functions', [])
        terminals_i = self.gp_cfg.get('terminal_set', {}).get('informed', [])
        terminals_u = self.gp_cfg.get('terminal_set', {}).get('uninformed', [])

        # Factories are stateless regarding RNG?
        # No, GPTreeFactory needs RNG.
        # We must create tree factories dynamically per agent or pass agent's RNG to them.
        # Current GPTreeFactory design takes RNG in __init__.
        # Strategy: GPTreeFactory should be lightweight. Create on demand?
        # Or modify GPTreeFactory to accept RNG in create_random_tree().
        # Let's modify usages: create new factories for each agent type setup is expensive? No.
        # But we need to use the AGENT's RNG.
        # Refactoring: AgentFactory will instantiate helpers using the Agent's specific RNG.
        self.func_names = func_names
        self.terminals_i = terminals_i
        self.terminals_u = terminals_u

    def _spawn_rng(self, agent_id: int, agent_type: str) -> np.random.Generator:
        """
        [Replication Key] Deterministic RNG spawning.
        Seed = f(Master, ID, TypeHash)
        """
        # Simple string hash for type
        type_hash = hash(agent_type) & 0xFFFFFFFF
        # Combine
        seed = (self.master_seed * 1000003 + agent_id * 31337 + type_hash) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def create_agent(self, agent_id: int, agent_type: str) -> AgentBase:
        rng = self._spawn_rng(agent_id, agent_type)

        endow = self.config['agents']['endowment']
        init_cash = endow['initial_bonds']
        init_stock = endow['initial_shares']

        if agent_type in ('informed', 'uninformed'):
            # Create specific helpers with Agent's RNG
            # Note: EvolutionaryEngine and TreeFactory share this RNG

            terminals = self.terminals_i if agent_type == 'informed' else self.terminals_u

            tree_factory = GPTreeFactory(
                self.func_names, terminals,
                self.gp_cfg.get('config', {}), rng
            )

            evo_engine = EvolutionaryEngine(
                tree_factory, rng, self.gp_cfg
            )

            return LearningAgent(
                agent_id, agent_type, init_cash, init_stock,
                self.config, rng,
                tree_factory, evo_engine
            )

        elif agent_type == 'noise':
            return NoiseAgent(
                agent_id, init_cash, init_stock,
                self.config, rng
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
from typing import Dict, Any, List, Optional
import numpy as np

from .base import AgentBase, Order
from project.learning.gp.tree import GPTreeFactory
from project.learning.forecasting import ForecastingRule
from project.learning.evolution import EvolutionaryEngine


class LearningAgent(AgentBase):
    """
    Informed / Uninformed Trader with GP.

    [Correction]
    - begin_period: 确定本期的"信念" (Expectation & Variance)，这是静态的。
    - generate_order: 结合静态信念和"动态持仓"计算 P^R。
    """

    def __init__(self,
                 agent_id: int,
                 agent_type: str,
                 init_cash: float,
                 init_stock: int,
                 config: Dict[str, Any],
                 rng: np.random.Generator,
                 tree_factory: GPTreeFactory,
                 evo_engine: EvolutionaryEngine):

        super().__init__(agent_id, agent_type, init_cash, init_stock, config, rng)
        self.evo_engine = evo_engine

        # GP Setup
        gp_cfg = config.get('learning_gp', {})
        gp_init_cfg = gp_cfg.get('init_params', {})
        self.max_depth = gp_init_cfg.get('max_depth', 4)
        self.init_method = gp_init_cfg.get('method', 'grow')
        self.n_rules = gp_cfg.get('rules_per_trader', 2)

        # Rules Init
        belief_cfg = config.get('beliefs', {})
        self.rules: List[ForecastingRule] = []
        for _ in range(self.n_rules):
            tree = tree_factory.create_random_tree(
                max_depth=self.max_depth,
                method=self.init_method
            )
            rule = ForecastingRule(gp_tree=tree, config=belief_cfg)
            self.rules.append(rule)

        # --- State Variables ---
        self.active_rule: Optional[ForecastingRule] = None
        self._rule_to_update: Optional[ForecastingRule] = None

        # [Fix] Cache Beliefs, NOT Price
        self._cached_expectation: Optional[float] = None
        self._cached_variance: Optional[float] = None

    def begin_period(self, market_info: Dict[str, Any], anchor_p_d: float):
        """
        [Period Start]
        只确定信念 (Expectation & Variance)，不锁定价格。
        """
        # 1. Select Active Rule
        self.active_rule = self.evo_engine.select_active_rule(self.rules)

        # 2. Form Beliefs (Static for this period)
        context = self._build_context(market_info)
        self._cached_expectation = self.active_rule.predict(context, anchor_p_d)
        self._cached_variance = self.active_rule.est_variance

        # 注意：这里不再计算 calculate_reservation_price

    def generate_order(self, market_info: Dict[str, Any], current_p_d: float) -> Optional[Order]:
        """
        [Trading Round]
        P^R = f(Static Belief, Dynamic Holdings)
        """
        # Defensive check
        if self._cached_expectation is None:
            self.begin_period(market_info, current_p_d)

        # [Fix] Calculate PR dynamically based on CURRENT holdings
        # 每笔交易后，state.get_holdings() 都会变，从而改变 PR
        rf = self.config['assets']['risk_free']['rf']

        pr = self.calculate_reservation_price(
            expected_payoff=self._cached_expectation,
            variance=self._cached_variance,
            rf_rate=rf
        )

        bb = market_info.get('best_bid')
        ba = market_info.get('best_ask')
        return self._make_cda_decision(pr, bb, ba)

    def end_period_update(self, realized_p_d: float):
        """[Period End] Update accuracy & Shift pointers"""
        if self._rule_to_update:
            self._rule_to_update.update_metrics(realized_p_d)

        self._rule_to_update = self.active_rule

        # Reset per-period state
        self.active_rule = None
        self._cached_expectation = None
        self._cached_variance = None

    def evolve_strategies(self):
        self.rules = self.evo_engine.evolve_rules(self.rules)
        self.active_rule = None
        self._cached_expectation = None
        self._cached_variance = None

    def _build_context(self, market_info: Dict[str, Any]) -> Dict[str, float]:
        history = market_info['history']

        def get_lag(key, lag):
            s = history.get(key, [])
            return s[-lag] if len(s) >= lag else 0.0

        ctx = {}
        for k in range(1, 6):
            ctx[f"P_t_{k}"] = get_lag('prices', k)

        if self.type == 'informed':
            ctx["Pf_t"] = market_info['current_fundamental']
            ctx["D_t"] = market_info['current_dividend']
            for k in range(1, 6):
                ctx[f"D_t_{k}"] = get_lag('dividends', k)
        else:
            ctx["Pf_t_1"] = get_lag('fundamentals', 1)
            for k in range(1, 6):
                ctx[f"D_t_{k}"] = get_lag('dividends', k)

        return ctx
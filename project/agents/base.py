from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, NamedTuple
import numpy as np

from project.core.state import AccountState
from project.preference.reservation_price import PolicyFactory, PolicyParams, ReservationPriceContext


class Order(NamedTuple):
    agent_id: int
    direction: int  # 1=Buy, -1=Sell
    order_type: str  # "limit" or "market"
    price: float  # Limit price
    quantity: int


class AgentBase(ABC):
    """交易者基类"""

    def __init__(self,
                 agent_id: int,
                 agent_type: str,
                 init_cash: float,
                 init_stock: int,
                 config: Dict[str, Any],
                 rng: np.random.Generator):

        self.id = agent_id
        self.type = agent_type
        self.rng = rng
        self.config = config

        # 1. State Config
        reg_cfg = config.get('regulation', {})
        # [Fix] Explicit Budget Policy
        budget_policy = reg_cfg.get('budget_policy', 'include_pending')

        sc_enabled = reg_cfg.get('settlement_cycle', {}).get('enabled', False)
        # 结算延迟以 Period 为单位
        self.settlement_lag = 1 if sc_enabled else 0

        self.state = AccountState(
            initial_cash=init_cash,
            initial_stock=init_stock,
            budget_policy=budget_policy,
            equity_includes_pending=True,
            exposure_includes_pending=True
        )

        # 2. Preference Config
        pref_cfg = config.get('preference', {})
        self.res_price_policy = PolicyFactory.get_policy(pref_cfg.get('policy', 'BaselineEq3Policy'))
        self.policy_params = PolicyParams(
            risk_aversion=config.get('risk_aversion', {}).get('lambda', 0.5),
            custom_params=pref_cfg.get('custom_params', {})
        )

        # 3. Order Params
        self.order_size = config.get('market', {}).get('trade_unit', 1)

    def on_period_start(self, current_period: int):
        # [Fix] Explicit interpretation: release_time is period index
        self.state.process_settlements(current_period)

    def calculate_reservation_price(self, exp_payoff: float, variance: float, rf: float) -> float:
        h_t = self.state.get_holdings()
        ctx = ReservationPriceContext(exp_payoff, variance, h_t, 1 + rf)
        return self.res_price_policy.calculate(ctx, self.policy_params)

    @abstractmethod
    def generate_order(self, market_info: Dict[str, Any], current_p_d: float) -> Optional[Order]:
        pass

    def _make_cda_decision(self, pr: float, bb: Optional[float], ba: Optional[float]) -> Optional[Order]:
        """
        通用 CDA 决策逻辑。
        """
        # Market Buy
        if ba is not None and pr > ba:
            return Order(self.id, 1, "market", 0.0, self.order_size)

        # Market Sell
        if bb is not None and pr < bb:
            return Order(self.id, -1, "market", 0.0, self.order_size)

        # Limit Order
        # [Fix] Robust mid price calculation
        if bb is not None and ba is not None:
            mid = 0.5 * (bb + ba)
        else:
            mid = pr

        direction = 1 if pr >= mid else -1
        return Order(self.id, direction, "limit", pr, self.order_size)
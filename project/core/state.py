from typing import List, NamedTuple, Dict, Optional


class PendingTransfer(NamedTuple):
    """
    待结算交易记录。
    release_time: 资金/股票到账的绝对时间 (period index)
    cash_delta: 资金变动 (负数代表支出)
    stock_delta: 股票变动 (负数代表卖出)
    """
    release_time: int
    cash_delta: float
    stock_delta: int


class AccountState:
    """
    管理单个交易者的资产状态 (Cash, Stock, Wealth)。
    支持 T+0 (即时结算) 和 T+1 (延迟结算) 模式。
    """

    def __init__(self,
                 initial_cash: float,
                 initial_stock: int,
                 budget_policy: str = "include_pending",
                 equity_includes_pending: bool = True,
                 exposure_includes_pending: bool = True):
        """
        :param budget_policy: 预算约束检查策略
            - 'strict_available': 仅使用当前已结算资产进行检查 (保守)
            - 'include_pending': 允许使用待结算资金抵扣支出 (宽松, ASM 常用)
        :param equity_includes_pending: 计算财富(Wealth)时是否包含待结算资产
        :param exposure_includes_pending: 计算持仓(Holdings)时是否包含待结算股票
        """
        self.cash = float(initial_cash)
        self.stock = int(initial_stock)

        self.pending_transfers: List[PendingTransfer] = []

        # Policies
        self.budget_policy = budget_policy
        self.equity_includes_pending = equity_includes_pending
        self.exposure_includes_pending = exposure_includes_pending

        # State Cache
        self.last_wealth: Optional[float] = None

    def check_budget_constraint(self,
                                cash_delta: float,
                                stock_delta: int) -> bool:
        """
        [Budget Constraint Check]
        检查本次交易是否会导致违约 (Cash < 0 或 Stock < 0)。
        """
        # 1. 确定基准资源 (Based on Policy)
        if self.budget_policy == "strict_available":
            base_cash = self.cash
            base_stock = self.stock
        elif self.budget_policy == "include_pending":
            base_cash = self.cash + sum(t.cash_delta for t in self.pending_transfers)
            base_stock = self.stock + sum(t.stock_delta for t in self.pending_transfers)
        else:
            raise ValueError(f"Unknown budget_policy: {self.budget_policy}")

        # 2. 叠加本次交易影响
        # 注意：无论 T+0 还是 T+1，一旦下单，这笔资源就被承诺出去了，必须做减法检查
        final_cash = base_cash + cash_delta
        final_stock = base_stock + stock_delta

        # 3. 判定 (若要支持融资融券，需在此处修改阈值)
        return (final_cash >= -1e-9) and (final_stock >= 0)

    def apply_trade(self,
                    cash_delta: float,
                    stock_delta: int,
                    current_time: int,
                    settlement_lag: int = 0,
                    enforce_budget: bool = True):
        """
        应用交易结果。

        :param enforce_budget: 默认强制检查预算，防止上层逻辑遗漏导致隐性透支。
        """
        # 防御性断言
        if enforce_budget:
            if not self.check_budget_constraint(cash_delta, stock_delta):
                raise ValueError(
                    f"Budget constraint violated! "
                    f"Policy={self.budget_policy}, "
                    f"CashDelta={cash_delta}, StockDelta={stock_delta}, "
                    f"CurrentCash={self.cash}, CurrentStock={self.stock}"
                )

        if settlement_lag == 0:
            self.cash += cash_delta
            self.stock += stock_delta
        else:
            release_time = current_time + settlement_lag
            transfer = PendingTransfer(release_time, cash_delta, stock_delta)
            self.pending_transfers.append(transfer)

    def process_settlements(self, current_time: int):
        """
        [Timing Convention]
        必须在每个 Period 开始时 (Start of Period t) 调用。
        先结算之前的挂单，更新 Cash/Stock，然后再进行当期的交易决策。
        """
        if not self.pending_transfers:
            return

        remaining = []
        for transfer in self.pending_transfers:
            if current_time >= transfer.release_time:
                self.cash += transfer.cash_delta
                self.stock += transfer.stock_delta
            else:
                remaining.append(transfer)

        self.pending_transfers = remaining

    def update_wealth_stats(self, current_price: float) -> float:
        """更新财富统计 (Mark-to-Market)"""
        eff_cash = self.cash
        eff_stock = self.stock

        if self.equity_includes_pending:
            eff_cash += sum(t.cash_delta for t in self.pending_transfers)
            eff_stock += sum(t.stock_delta for t in self.pending_transfers)

        self.last_wealth = eff_cash + eff_stock * current_price
        return self.last_wealth

    def is_bankrupt(self, threshold: float = 0.0) -> bool:
        """
        权威破产判定。
        基于 Mark-to-Market Wealth。
        """
        if self.last_wealth is None:
            return False
        return self.last_wealth <= threshold
    def get_holdings(self) -> int:
        """获取用于决策的需求函数输入变量 h_{i,t}"""
        if self.exposure_includes_pending:
            pending_stock = sum(t.stock_delta for t in self.pending_transfers)
            return self.stock + pending_stock
        return self.stock

    def to_dict(self) -> Dict:
        return {
            "cash": self.cash,
            "stock": self.stock,
            "wealth": self.last_wealth,
            "pending_count": len(self.pending_transfers)
        }
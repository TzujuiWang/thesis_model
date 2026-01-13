from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Type, Optional


@dataclass(frozen=True)
class ReservationPriceContext:
    """
    [Input Context] 计算预留价所需的市场状态快照。
    使用 dataclass 封装，保证接口签名稳定，不受策略参数扩展影响。
    """
    expectation_payoff: float  # E_i,t(P_{t+1} + D_{t+1}) (Source: GP/Forecast)
    conditional_variance: float  # V_i,t(R_{t+1}) (Source: Variance Model)
    current_holdings: int  # h_{i,t} (Source: AccountState)
    risk_free_R: float  # R = 1 + rf (Source: AssetModel)


@dataclass(frozen=True)
class PolicyParams:
    """
    [Configuration] 策略所需的静态参数。
    """
    risk_aversion: float  # lambda (CARA coefficient)
    min_variance_eps: float = 0.0  # 数值稳定阈值 (Default 0.0 = Strictly follow Eq 3)
    # 预留给未来策略（如前景理论）的额外参数字典
    custom_params: Dict[str, Any] = field(default_factory=dict)


class ReservationPricePolicy(ABC):
    """
    [Extension Point] 预留价策略基类。
    实现类应该是无状态的 (Stateless)，所有状态通过 Context 和 Params 传入。
    """

    @abstractmethod
    def calculate(self, ctx: ReservationPriceContext, params: PolicyParams) -> float:
        """计算保留价格 P^R_{i,t}"""
        pass


class BaselineEq3Policy(ReservationPricePolicy):
    """
    [Source 1 Eq 3] 默认 CARA 效用策略。

    Formula:
      P^R = [ E(Payoff) - lambda * h * Var ] / R

    Math Interpretation:
      - 风险惩罚项 (Risk Penalty) = lambda * h * Var
      - 若 h > 0 (净多头): 惩罚项 > 0 => P^R < E(Payoff)/R (折价卖出/买入)
      - 若 h < 0 (净空头): 惩罚项 < 0 => P^R > E(Payoff)/R (溢价买入/卖出)
    """

    def calculate(self, ctx: ReservationPriceContext, params: PolicyParams) -> float:
        var = ctx.conditional_variance

        # 数值稳定性处理（仅在配置显式允许时生效）
        if params.min_variance_eps > 0.0:
            var = max(var, params.min_variance_eps)

        # Eq(3) 核心计算
        risk_penalty = params.risk_aversion * ctx.current_holdings * var
        numerator = ctx.expectation_payoff - risk_penalty

        # P^R = Numerator / R
        return numerator / ctx.risk_free_R


# 策略注册表
_POLICY_REGISTRY: Dict[str, Type[ReservationPricePolicy]] = {
    "BaselineEq3Policy": BaselineEq3Policy,
    # Future: "ProspectTheoryPolicy": ProspectTheoryPolicy
}


class PolicyFactory:
    """工厂类：根据配置名称实例化策略对象"""

    @staticmethod
    def get_policy(config_name: str) -> ReservationPricePolicy:
        try:
            # 实例化策略类 (策略类通常是无状态的单例模式，此处每次新建实例开销可忽略)
            return _POLICY_REGISTRY[config_name]()
        except KeyError:
            raise ValueError(
                f"Unknown Reservation Price Policy: '{config_name}'. "
                f"Available: {list(_POLICY_REGISTRY.keys())}"
            )
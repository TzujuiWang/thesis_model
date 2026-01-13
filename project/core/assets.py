import numpy as np
from typing import Dict


class AssetModel:
    """
    管理无风险资产与风险资产的随机过程。

    复现约束：
    1. 不猜测参数含义：必须显式指定 mu_param_type (std/variance)。
    2. 可复现性：严禁使用 numpy 全局随机状态，必须使用传入的 rng 实例。
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        """
        :param config: base.yaml 中的 'assets' 部分
        :param rng: 由 Runner 统一管理的随机数生成器 (Source of Randomness)
        """
        self.rng = rng

        # 1. 无风险资产
        self.rf = config['risk_free']['rf']
        self.R = 1 + self.rf

        # 2. 风险资产配置
        risky_cfg = config['risky_dividend_ar1']
        self.rho = risky_cfg['rho']
        self.D_bar = risky_cfg['D_bar']
        self.mu_param = risky_cfg['mu_dist_param']

        # [Strict Replication] 解析参数类型，拒绝猜测
        param_type = risky_cfg.get('mu_param_type')
        if param_type == 'std':
            self.mu_std = self.mu_param
        elif param_type == 'variance':
            self.mu_std = np.sqrt(self.mu_param)
        else:
            raise ValueError(
                f"Configuration Error: 'mu_param_type' in assets.risky_dividend_ar1 "
                f"must be explicitly set to 'std' or 'variance'. Got: {param_type}"
            )

        # 3. 初始化股息 D_t
        # [Strict Replication] 显式策略
        init_policy = risky_cfg.get('initialization_policy')
        if init_policy == 'steady_state_mean':
            self.current_dividend = self.D_bar
        elif init_policy == 'explicit_value':
            val = risky_cfg.get('initial_value')
            if val is None:
                raise ValueError("initialization_policy is 'explicit_value' but 'initial_value' is None")
            self.current_dividend = val
        else:
            raise ValueError(f"Unknown initialization_policy: {init_policy}")

    def step(self) -> float:
        """
        推进一个时间步 (Period)。
        Eq(1): D_{t+1} = D_bar + rho * (D_t - D_bar) + mu_{t+1}
        """
        # 使用传入的 RNG 生成噪声
        mu = self.rng.normal(0, self.mu_std)

        D_next = self.D_bar + self.rho * (self.current_dividend - self.D_bar) + mu
        self.current_dividend = D_next

        return self.current_dividend

    def get_fundamental_value(self) -> float:
        """Eq: Pf = D_t / rf"""
        if self.rf == 0:
            return float('inf')
        return self.current_dividend / self.rf

    def get_risk_free_rate(self) -> float:
        return self.rf

    def get_gross_return_rate(self) -> float:
        return self.R
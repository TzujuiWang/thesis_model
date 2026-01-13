import numpy as np
from typing import Dict, Optional, Any
from .gp.tree import GPNode


class ForecastingRule:
    """
    [Source 1 Section 3.2.4 & Source 2 Yeh & Yang Eq 8-10]
    封装 GP 树为预测规则。

    【时间索引约定 (Time Indexing Convention)】
    -------------------------------------------------------
    Period t (Decision):
      - Info Available:
          Informed:   P_{t-1}, D_{t-1}, ..., D_t (Private)
          Uninformed: P_{t-1}, D_{t-1}, ...
      - Action: Call predict(context_t)
      - Output: E_{i,t}(P_{t+1} + D_{t+1})  <-- 预测的是 t+1 的总价值
      - Cache: last_pred_payoff

    Period t+1 (Update):
      - Info Available: Realized P_{t+1}, D_{t+1}
      - Action: Call update_metrics(realized_P_{t+1} + realized_D_{t+1})
      - Update: Variance(t+1), Strength(t+1) based on Error(t)
    -------------------------------------------------------
    """

    def __init__(self,
                 gp_tree: GPNode,
                 config: Dict[str, Any] = None,
                 init_p_d: float = 25.01,
                 init_variance: float = 0.0004):

        self.gp_tree = gp_tree
        config = config or {}

        # --- Eq(4) Variant Switch ---
        # "dai_abs_f_minus_1": ln(|-1 + f|) [Source 1 Text, Default]
        # "yeh_abs_1_plus_f":  ln(|1 + f|)  [Source 2 / Common Sense]
        self.eq4_variant = config.get('eq4_negative_log_variant', 'dai_abs_f_minus_1')

        # --- Parameters ---
        self.omega = float(config.get('mapping_omega', 15.0))
        self.theta_0 = float(config.get('theta_0', 0.2))

        # [Source 2 Table 2] Variance updating
        self.theta_1 = float(config.get('theta_1', 0.01))
        self.theta_2 = float(config.get('theta_2', 0.001))

        # --- State Variables ---
        self.u_ewma: float = init_p_d
        self.est_variance: float = init_variance
        self.last_pred_payoff: Optional[float] = None

        # Debugging
        self.last_f_value: float = 0.0

    @property
    def strength(self) -> float:
        """[Source 2] Strength = -Variance"""
        return -self.est_variance

    def predict(self, context: Dict[str, float], current_p_d: float) -> float:
        """
        [Source 1 Eq 4] 计算预期总回报。
        Context 必须严格匹配 Terminal Set (不能包含未来信息)。
        """
        f_val = self.gp_tree.evaluate(context)
        self.last_f_value = f_val

        epsilon = 1e-9

        # --- Eq(4) Mapping ---
        if f_val >= 0:
            # Positive branch: ln(1 + f)
            log_term = np.log(1.0 + f_val + epsilon)
        else:
            # Negative branch: controlled by config
            if self.eq4_variant == 'yeh_abs_1_plus_f':
                # Yeh & Yang style: ln(|1 + f|)
                log_term = np.log(abs(1.0 + f_val) + epsilon)
            else:
                # Dai (Source 1) style: ln(|-1 + f|)
                # Note: |-1 + f| is |f - 1|.
                log_term = np.log(abs(f_val - 1.0) + epsilon)

        scaled_x = log_term / self.omega
        mapped_rate = np.tanh(self.theta_0 * scaled_x)

        # E_{i,t} = (P_t + D_t) * (1 + mapped_rate)
        expected_payoff = current_p_d * (1.0 + mapped_rate)

        self.last_pred_payoff = expected_payoff
        return expected_payoff

    def update_metrics(self, realized_p_d_next: float):
        """
        [Source 2 Eq 8 & 9]
        在 t+1 时刻调用，使用 realized_p_d_next (即 P_{t+1} + D_{t+1}) 更新状态。
        """
        if self.last_pred_payoff is None:
            return

        # Snapshot old u (u_{t}) used for variance update
        old_u = self.u_ewma

        # 1. Update u (Eq 9) -> u_{t+1}
        # u_{t+1} = (1 - th1)*u_t + th1*(P_{t+1} + D_{t+1})
        self.u_ewma = (1 - self.theta_1) * old_u + self.theta_1 * realized_p_d_next

        # 2. Update Variance (Eq 8) -> sigma^2_{t+1}
        # Term B: (P_{t+1} + D_{t+1} - u_t)^2
        # Term C: (P_{t+1} + D_{t+1} - E_{i,t})^2

        term_a = (1 - self.theta_1 - self.theta_2) * self.est_variance

        diff_u = realized_p_d_next - old_u
        term_b = self.theta_1 * (diff_u ** 2)

        diff_pred = realized_p_d_next - self.last_pred_payoff
        term_c = self.theta_2 * (diff_pred ** 2)

        self.est_variance = term_a + term_b + term_c

        # Reset cache
        self.last_pred_payoff = None

    def clone(self) -> 'ForecastingRule':
        """Clone with state inheritance"""
        new_tree = self.gp_tree.clone()

        # Pass current config and params
        cfg = {
            'eq4_negative_log_variant': self.eq4_variant,
            'mapping_omega': self.omega,
            'theta_0': self.theta_0,
            'theta_1': self.theta_1,
            'theta_2': self.theta_2
        }

        new_rule = ForecastingRule(
            new_tree,
            config=cfg,
            init_p_d=self.u_ewma,  # Inherit EWMA
            init_variance=self.est_variance  # Inherit Strength
        )
        return new_rule
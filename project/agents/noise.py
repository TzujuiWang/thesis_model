from typing import Dict, Any, Optional
import numpy as np
from .base import AgentBase, Order


class NoiseAgent(AgentBase):
    """Noise Trader with Configurable Bias Mode."""

    def __init__(self,
                 agent_id: int,
                 init_cash: float,
                 init_stock: int,
                 config: Dict[str, Any],
                 rng: np.random.Generator):

        super().__init__(agent_id, "noise", init_cash, init_stock, config, rng)

        noise_cfg = config.get('noise_trader', {})
        self.sigma = noise_cfg.get('sigma_epsilon', 0.05)  # Need calibration
        self.mode = noise_cfg.get('noise_mode', 'additive')  # additive or multiplicative

        # Fixed perceived variance
        self.perceived_var = noise_cfg.get('perceived_variance', 0.0004)

    def generate_order(self, market_info: Dict[str, Any], current_p_d: float) -> Optional[Order]:
        epsilon = self.rng.normal(0, self.sigma)

        # [Fix] Dimensionality Check
        if self.mode == 'multiplicative':
            # E = (P+D) * (1 + eps)
            exp_payoff = current_p_d * (1.0 + epsilon)
        else:
            # E = (P+D) + eps
            # Ensure sigma is scaled to price level (e.g. ~25)
            exp_payoff = current_p_d + epsilon

        rf = self.config['assets']['risk_free']['rf']
        pr = self.calculate_reservation_price(exp_payoff, self.perceived_var, rf)

        bb = market_info.get('best_bid')
        ba = market_info.get('best_ask')
        return self._make_cda_decision(pr, bb, ba)
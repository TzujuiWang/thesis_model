import numpy as np
from typing import List, Dict, Any


class MarketRecorder:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Time Series
        self.prices: List[float] = []
        self.fundamentals: List[float] = []
        self.dividends: List[float] = []
        self.volumes: List[int] = []

        # Analysis Data
        self.market_returns: List[float] = []
        self.noise_avg_returns: List[float] = []

        # Cache for Noise Risk Calc: {agent_id: last_wealth}
        self._prev_noise_wealth: Dict[int, float] = {}

    def record_period(self,
                      price: float,
                      fundamental: float,
                      dividend: float,
                      volume: int,
                      noise_agents: List[Any]):  # Pass actual agent objects

        # 1. Basic
        self.prices.append(price)
        self.fundamentals.append(fundamental)
        self.dividends.append(dividend)
        self.volumes.append(volume)

        # 2. Market Return
        if len(self.prices) > 1:
            prev_p = self.prices[-2]
            # Configurable Return Type
            rtype = self.config.get('metrics', {}).get('market_return_type', 'log')

            if rtype == 'simple':
                ret = (price + dividend) / prev_p - 1.0
            else:
                ret = np.log((price + dividend) / prev_p)
            self.market_returns.append(ret)

        # 3. Noise Trader Return (Individual Avg)
        current_period_returns = []
        for agent in noise_agents:
            w = agent.state.last_wealth
            if w is None: continue

            prev_w = self._prev_noise_wealth.get(agent.id)
            if prev_w is not None and prev_w > 0:
                r = (w - prev_w) / prev_w
                current_period_returns.append(r)

            # Update cache
            self._prev_noise_wealth[agent.id] = w

        if current_period_returns:
            avg_ret = np.mean(current_period_returns)
            self.noise_avg_returns.append(avg_ret)
        else:
            # No noise traders or first period
            if len(self.noise_avg_returns) > 0:
                self.noise_avg_returns.append(0.0)  # Survivor bias?

    def calculate_statistics(self) -> Dict[str, float]:
        # Discard warmup? Assume Engine handles or slicing happens here.
        # Let's calculate on full recorded history.

        pv = np.std(self.market_returns, ddof=1) if len(self.market_returns) > 1 else 0.0

        distortions = []
        for p, pf in zip(self.prices, self.fundamentals):
            if pf != 0: distortions.append(abs(p - pf) / pf)
        pd_val = np.mean(distortions) if distortions else 0.0

        vol = np.mean(self.volumes) if self.volumes else 0.0

        nt_risk = np.std(self.noise_avg_returns, ddof=1) if len(self.noise_avg_returns) > 1 else 0.0

        return {
            "PV": pv,
            "PD": pd_val,
            "Volume": vol,
            "NoiseRisk": nt_risk
        }

    def get_history(self) -> Dict[str, List[float]]:
        # Return period-level history (for GP context)
        # Note: GP needs period closes.
        return {
            "prices": self.prices,
            "dividends": self.dividends,
            "fundamentals": self.fundamentals
        }
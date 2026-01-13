from typing import Dict, Any, Optional

class PriceLimitRule:
    """[Source 1 Section 3.3] 10% 涨跌停规则"""
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get('regulation', {}).get('price_limit', {})
        self.enabled = bool(cfg.get('enabled', False))
        self.threshold = float(cfg.get('threshold', 0.10))

    def is_valid_price(self, price: float, ref_price: Optional[float]) -> bool:
        if not self.enabled or ref_price is None:
            return True
        lower = ref_price * (1.0 - self.threshold)
        upper = ref_price * (1.0 + self.threshold)
        return lower <= price <= upper

class TransactionTaxRule:
    """[Source 1 Section 3.3] 0.1% 交易税规则"""
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get('regulation', {}).get('transaction_tax', {})
        self.enabled = bool(cfg.get('enabled', False))
        self.rate = float(cfg.get('rate', 0.001))

    def calculate_tax(self, price: float, quantity: int) -> float:
        if not self.enabled: return 0.0
        return price * quantity * self.rate

class SettlementCycleRule:
    """[Source 1 Section 3.3] T+1 结算规则"""
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get('regulation', {}).get('settlement_cycle', {})
        self.enabled = bool(cfg.get('enabled', False))
        self.cycle_type = cfg.get('type', 'T+0')

    @property
    def settlement_lag(self) -> int:
        # Lag unit: period
        return 1 if (self.enabled and self.cycle_type == 'T+1') else 0
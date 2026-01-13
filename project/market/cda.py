from typing import List, Dict, Any, Optional, NamedTuple
from .orderbook import OrderBook, BookEntry
from project.agents.base import Order
from project.regulation.rules import PriceLimitRule, TransactionTaxRule


class Trade(NamedTuple):
    """
    成交记录 (不可变)
    """
    price: float
    quantity: int
    buyer_id: int
    seller_id: int
    tax_amount: float
    timestamp: int  # 撮合序号 (intra-period trade id)


class CDAEngine:
    """
        连续双向拍卖引擎 (注入 Rule 对象版)。
    """

    def __init__(self,
                 config: Dict[str, Any],
                 pl_rule: PriceLimitRule,
                 tt_rule: TransactionTaxRule):

        self.config = config
        self.book = OrderBook()

        # Injected Rules (Single Source of Truth)
        self.pl_rule = pl_rule
        self.tt_rule = tt_rule

        self.last_close_price: Optional[float] = config.get("market", {}).get("initial_price", None)
        self._trade_counter = 0
        self.last_transaction_price: Optional[float] = None
        self.trades_this_period: List[Trade] = []

    def set_reference_price_and_reset_book(self, last_close: float):
        self.last_close_price = last_close
        self.book.clear()
        self.trades_this_period.clear()
        self.last_transaction_price = None

    def get_market_snapshot(self) -> Dict[str, Any]:
        return self.book.snapshot()

    def process_order(self, order: Order) -> List[Trade]:
        """
        处理订单并返回成交列表。
        """
        # 1. Validation
        if order.quantity <= 0: return []
        if order.order_type not in ("market", "limit"): return []
        if order.direction not in (1, -1): return []
        if order.order_type == "limit" and order.price <= 0: return []

        # 2. Price Limit Check on SUBMISSION (Limit Orders only)
        # 如果限价单直接超出涨跌停，拒绝入场
        if order.order_type == "limit":
            if not self.pl_rule.is_valid_price(order.price, self.last_close_price):
                return []

        remaining = order.quantity
        trades: List[Trade] = []



        # 3. Matching Loop
        while remaining > 0:
            # Determine opposite side
            if order.direction == 1:  # Buy
                best_entry = self.book.peek_best_ask()
                match_price = self.book.best_ask()
            else:  # Sell
                best_entry = self.book.peek_best_bid()
                match_price = self.book.best_bid()

            # No liquidity?
            if best_entry is None or match_price is None:
                break

            # Limit Order Crossing Check
            if order.order_type == "limit":
                if order.direction == 1 and order.price < match_price:
                    break  # Buy limit too low
                if order.direction == -1 and order.price > match_price:
                    break  # Sell limit too high

            # [Regulation] Price Limit Check on EXECUTION
            # 即使 Limit Price 合规，成交价（对方挂单价）也必须合规
            if not self.pl_rule.is_valid_price(match_price, self.last_close_price):
                break

            # Execute Trade
            qty = min(remaining, best_entry.remaining)

            # Calculate Tax
            tax = self.tt_rule.calculate_tax(match_price, qty)

            self._trade_counter += 1

            # Determine Buyer/Seller IDs for record
            buyer_id = order.agent_id if order.direction == 1 else best_entry.order.agent_id
            seller_id = best_entry.order.agent_id if order.direction == 1 else order.agent_id

            trade = Trade(
                price=match_price,
                quantity=qty,
                buyer_id=buyer_id,
                seller_id=seller_id,
                tax_amount=tax,
                timestamp=self._trade_counter
            )

            trades.append(trade)

            self.last_transaction_price = match_price
            self.trades_this_period.append(trade)

            # Update Quantities
            remaining -= qty

            # Update Book (Opposite Side)
            # Pop the best entry
            popped: BookEntry = self.book.pop_best_ask() if order.direction == 1 else self.book.pop_best_bid()
            popped.remaining -= qty

            # If partially filled, put back with ORIGINAL timestamp
            if popped.remaining > 0:
                self.book.reinsert(popped)

        # 4. Post-Match: Resting Order
        if self.pl_rule.is_valid_price(order.price, self.last_close_price):
            self.book.add_limit(order._replace(quantity=remaining))

        # Market Order remainder is implicitly cancelled here (Standard CDA)

        return trades

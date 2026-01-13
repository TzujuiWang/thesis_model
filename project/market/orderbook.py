import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from 论文.project.agents.base import Order


@dataclass(order=False)
class BookEntry:
    """
    订单簿条目。
    封装原始 Order 对象，并维护剩余数量和原始时间戳。
    """
    price: float
    timestamp: int
    order: Order
    remaining: int


class OrderBook:
    """
    Limit Order Book with strictly enforced Price-Time (FIFO) priority.

    Structure:
    - Bids: Max-Heap (stored as -price, timestamp, entry)
    - Asks: Min-Heap (stored as  price, timestamp, entry)
    """

    def __init__(self):
        # Heap elements: (priority_price, timestamp, entry)
        # Bids uses negative price for max-heap behavior
        self.bids: List[Tuple[float, int, BookEntry]] = []
        self.asks: List[Tuple[float, int, BookEntry]] = []

        # Monotonic counter for strict time priority
        self._counter = 0

    def clear(self):
        """Reset the book (e.g., at start of period)"""
        self.bids.clear()
        self.asks.clear()
        self._counter = 0

    def _next_ts(self) -> int:
        self._counter += 1
        return self._counter

    def add_limit(self, order: Order):
        """Add a new Limit Order to the book."""
        # 1. Validation
        if order.quantity <= 0:
            return
        if order.price <= 0:
            return
        # Note: Market orders should not reach here

        # 2. Create Entry with NEW timestamp
        ts = self._next_ts()
        entry = BookEntry(
            price=order.price,
            timestamp=ts,
            order=order,
            remaining=order.quantity
        )

        # 3. Push to Heap
        if order.direction == 1:  # Buy
            heapq.heappush(self.bids, (-entry.price, entry.timestamp, entry))
        else:  # Sell
            heapq.heappush(self.asks, (entry.price, entry.timestamp, entry))

    def reinsert(self, entry: BookEntry):
        """
        Re-insert an existing entry that was partially filled.
        CRITICAL: Preserves original timestamp to maintain FIFO priority.
        """
        if entry.remaining <= 0:
            return

        if entry.order.direction == 1:  # Buy
            heapq.heappush(self.bids, (-entry.price, entry.timestamp, entry))
        else:  # Sell
            heapq.heappush(self.asks, (entry.price, entry.timestamp, entry))

    # --- Query Methods ---

    def best_bid(self) -> Optional[float]:
        return -self.bids[0][0] if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    def peek_best_bid(self) -> Optional[BookEntry]:
        return self.bids[0][2] if self.bids else None

    def peek_best_ask(self) -> Optional[BookEntry]:
        return self.asks[0][2] if self.asks else None

    # --- Operation Methods ---

    def pop_best_bid(self) -> BookEntry:
        _, _, e = heapq.heappop(self.bids)
        return e

    def pop_best_ask(self) -> BookEntry:
        _, _, e = heapq.heappop(self.asks)
        return e

    def snapshot(self) -> dict:
        """Provide B_b and B_a for agent decision making."""
        return {
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask()
        }
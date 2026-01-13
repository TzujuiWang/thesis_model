from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TimeIndex:
    period: int
    round: int


class Timeline:
    """
    [Core Time Driver]
    State Transitions:
    - Settlement (Round 0)
    - Trading (Round 1..N_R)
    - Period End (Round N_R + 1)
    - Next Period Settlement ...
    """

    def __init__(self, total_periods: int, rounds_per_period: int):
        self.total_periods = total_periods
        self.rounds_per_period = rounds_per_period

        self.current_period = 1
        self.current_round = -1  # 0=Settlement
        self.finished = False

    def step(self) -> Literal["settlement", "trading", "period_end", "finished"]:
        if self.finished:
            return "finished"

        # 1. Check if we just finished period_end of prev period
        # Logic: We are at state X, we want to move to next logical state.

        # Current: Settlement (0) -> Trading (1)
        if self.current_round == 0:
            self.current_round = 1
            return "trading"

        # Current: Trading (1..N-1) -> Trading (next)
        if 1 <= self.current_round < self.rounds_per_period:
            self.current_round += 1
            return "trading"

        # Current: Trading (N) -> Period End (N+1)
        if self.current_round == self.rounds_per_period:
            self.current_round += 1
            return "period_end"

        # Current: Period End (N+1) -> Next Period Settlement (0) or Finish
        if self.current_round > self.rounds_per_period:
            if self.current_period >= self.total_periods:
                self.finished = True
                return "finished"
            else:
                self.current_period += 1
                self.current_round = 0
                return "settlement"

        return "finished"  # Should not reach

    def is_evolution_time(self, evolve_interval: int) -> bool:
        return (self.current_round > self.rounds_per_period) and \
            (self.current_period % evolve_interval == 0)
import numpy as np
from typing import Dict, Any, List
import copy

from project.core.config_loader import ConfigLoader
from project.core.timeline import Timeline
from project.core.assets import AssetModel
from project.regulation.rules import PriceLimitRule, TransactionTaxRule, SettlementCycleRule
from project.market.cda import CDAEngine
from project.sim.agent_factory import AgentFactory
from project.sim.initialization import MarketInitializer
from project.sim.replacement import ReplacementEngine
from project.metrics.market_quality import MarketRecorder


class SimulationEngine:
    """
    [Core] 仿真主引擎。
    负责调度时间、资产更新、市场撮合、代理人决策与进化。
    """

    def __init__(self, config: Dict[str, Any], run_id: int):
        self.config = config
        self.run_id = run_id

        # 1. Randomness (Master Seed)
        base_seed = config['random']['seed']
        self.master_seed = base_seed + run_id
        self.rng = np.random.default_rng(self.master_seed)

        # 2. Infrastructure
        periods = config['time']['periods_total']
        rounds = config['time']['rounds_per_period']
        self.timeline = Timeline(periods, rounds)

        # 3. Rules (Single Source of Truth)
        self.pl_rule = PriceLimitRule(config)
        self.tt_rule = TransactionTaxRule(config)
        self.sc_rule = SettlementCycleRule(config)

        # 4. Components
        # [Asset]
        # Generate a dedicated RNG for asset to ensure macro-environment consistency across runs?
        # Ideally asset path depends on seed.
        asset_rng = np.random.default_rng(self.master_seed + 1000)
        self.assets = AssetModel(config['assets'], asset_rng)

        # [Market]
        self.market = CDAEngine(config, self.pl_rule, self.tt_rule)

        # [Factory]
        self.agent_factory = AgentFactory(config, self.master_seed)

        # [Agents Init]
        # Use MarketInitializer to determine counts/types
        initializer = MarketInitializer(config, self.rng)
        agent_dicts = initializer.create_agents()

        self.agents = []
        for ad in agent_dicts:
            new_agent = self.agent_factory.create_agent(ad['id'], ad['type'])
            self.agents.append(new_agent)

        # [Replacement]
        self.replacement = ReplacementEngine(config, self.agent_factory.create_agent)

        # [Recorder]
        self.recorder = MarketRecorder()

        # State Cache
        self.current_period_volume = 0

    def run(self):
        """执行仿真主循环"""
        print(f"Run {self.run_id}: Start ({self.timeline.total_periods} periods)")

        while not self.timeline.finished:
            event = self.timeline.step()

            if event == "settlement":
                self._handle_settlement_phase()
            elif event == "trading":
                self._handle_trading_round()
            elif event == "period_end":
                self._handle_period_end()
            elif event == "finished":
                break

        print(f"Run {self.run_id}: Finished.")
        return self.recorder.calculate_statistics()

    def _handle_settlement_phase(self):
        p = self.timeline.current_period

        # 1. Macro Update
        self.assets.step()

        # 2. Market Reset
        last_close = self.recorder.prices[-1] if self.recorder.prices else self.config['market']['initial_price']
        self.market.set_reference_price_and_reset_book(last_close)
        self.current_period_volume = 0

        # 3. Agent Settlement & Pre-calculation
        # Info for Context
        market_info = {
            'history': self.recorder.get_history(),
            'current_fundamental': self.assets.get_fundamental_value(),
            'current_dividend': self.assets.current_dividend
        }
        anchor = last_close + self.assets.current_dividend

        for agent in self.agents:
            agent.on_period_start(p)
            # [Learning Phase 1] Select Rule & Calc Pr
            if hasattr(agent, 'begin_period'):
                agent.begin_period(market_info, anchor)

    def _handle_trading_round(self):
        self.rng.shuffle(self.agents)

        # Snapshot for order generation (Best Bid/Ask)
        snapshot = self.market.get_market_snapshot()

        # Note: GP Context needs history (handled in begin_period),
        # but CDA decision needs current Bb/Ba (passed here).
        # We pass snapshot + history? No, Agent.begin_period cached the Pr.
        # generate_order only needs snapshot.

        anchor = 0.0  # Not used if Pr cached, but pass for Noise agents if they don't cache?
        # Noise agents might not cache Pr if they don't have begin_period logic.
        # Refactoring: Add begin_period to NoiseAgent (to form belief once per period? or per round?)
        # Source 1: "Noise traders... biased belief... random epsilon".
        # Usually noise changes per period or per round? "epsilon_{i,t}" implies per period t?
        # If per period, move Noise belief formation to begin_period too.
        # Assume Noise changes per period for consistency with t index.

        # For now, pass anchor just in case.

        for agent in self.agents:
            order = agent.generate_order(snapshot, anchor)
            if order:
                trades = self.market.process_order(order)
                for trade in trades:
                    self._apply_trade_to_agents(trade)

    def _apply_trade_to_agents(self, trade):
        self.current_period_volume += trade.quantity

        buyer = self.agent_map[trade.buyer_id]
        seller = self.agent_map[trade.seller_id]

        # Tax Logic
        tax = trade.tax_amount
        buyer_pay = 0.0
        seller_pay = 0.0

        if self.tax_payer == 'seller':
            seller_pay = tax
        elif self.tax_payer == 'buyer':
            buyer_pay = tax
        elif self.tax_payer == 'split':
            buyer_pay = tax / 2
            seller_pay = tax / 2

        # Apply
        lag = self.sc_rule.settlement_lag
        p_now = self.timeline.current_period

        buyer.state.apply_trade(-(trade.price * trade.quantity + buyer_pay), trade.quantity, p_now, lag)
        seller.state.apply_trade((trade.price * trade.quantity - seller_pay), -trade.quantity, p_now, lag)

    def _handle_period_end(self):
        # 1. Determine Close
        if self.market.last_transaction_price is not None:
            close_price = self.market.last_transaction_price
        else:
            close_price = self.recorder.prices[-1] if self.recorder.prices else self.config['market']['initial_price']

        # 2. Update Wealth (Mark-to-Market)
        # MUST happen before Replacement & Metrics
        for agent in self.agents:
            agent.state.update_wealth_stats(close_price)

        # 3. Learning Update (t -> t+1)
        # Realized P+D for prediction at t-1 (which predicted t) is Current Close + Current Div
        # Wait. Timeline:
        # At start of period t, agents predicted P_{t+1} + D_{t+1}?
        # No. Agents predict Return of holding asset from t to t+1.
        # Payoff = P_{t+1} + D_{t+1}.
        # So at end of period t, we have P_t + D_t.
        # This realizes the prediction made at t-1.
        realized_p_d = close_price + self.assets.current_dividend

        for agent in self.agents:
            if hasattr(agent, 'end_period_update'):
                agent.end_period_update(realized_p_d)

        # 4. Evolution
        if self.timeline.is_evolution_time(self.config['learning_gp']['config']['evolution_cycle']):
            for agent in self.agents:
                if hasattr(agent, 'evolve_strategies'):
                    agent.evolve_strategies()

        # 5. Replacement
        new_agents, stats = self.replacement.process_bankruptcies(
            self.agents, close_price, self.assets.get_fundamental_value(), self.timeline.current_period
        )
        self.agents = new_agents
        # Update Map
        self.agent_map = {a.id: a for a in self.agents}

        # 6. Record Metrics (Last step)
        noise_group = [a for a in self.agents if a.type == 'noise']
        self.recorder.record_period(
            close_price,
            self.assets.get_fundamental_value(),
            self.assets.current_dividend,
            self.current_period_volume,
            noise_group
        )

    def _get_agent(self, agent_id: int):
        # Optimized lookup?
        # List comprehension is slow for 100 agents * 50 rounds.
        # Better: dict map or direct index if ids are 0..N-1 and stable?
        # Replacement reuses IDs or creates new?
        # If reuse, index match is fine.
        for a in self.agents:
            if a.id == agent_id: return a
        raise ValueError(f"Agent {agent_id} not found")
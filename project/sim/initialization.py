from typing import Dict, List, Any, Optional
import numpy as np


class MarketInitializer:
    """
    负责根据配置生成交易者种群 (Population)。
    严格复现：
    1. 保证种群总数精确等于 N (处理浮点比例截断误差)。
    2. 校验参数合法性 (Shares in [0,1])。
    3. 统一 Agent Schema (包含 id, type, cash, stock, alive, pending_settlement)。
    """

    def __init__(self, full_config: Dict, rng: Optional[np.random.Generator] = None):
        self.config = full_config
        self.agent_cfg = full_config["agents"]
        self.total_pop = int(self.agent_cfg["population_total"])
        # [Strict Replication] 必须注入 RNG，避免全局随机状态污染
        self.rng = rng or np.random.default_rng()

        comp = self.agent_cfg["composition"]
        self.informed_share = float(comp["informed_share"])
        self.noise_share = float(comp["noise_share"])

        # ---- 1. Validation (防静默跑偏) ----
        if not (0.0 <= self.informed_share <= 1.0 and 0.0 <= self.noise_share <= 1.0):
            raise ValueError("Shares must be in [0,1].")
        if self.informed_share + self.noise_share > 1.0 + 1e-12:
            raise ValueError(f"informed_share ({self.informed_share}) + noise_share ({self.noise_share}) > 1.0")

        # ---- 2. Robust Rounding (保证总数严格为 N) ----
        raw_informed = self.total_pop * self.informed_share
        raw_noise = self.total_pop * self.noise_share
        n_informed = int(round(raw_informed))
        n_noise = int(round(raw_noise))

        # 通过补差法计算 uninformed，修复四舍五入带来的漂移
        n_uninformed = self.total_pop - n_informed - n_noise

        # 极端情况处理：如果漂移导致人数为负 (极少见，但需防御)
        if n_uninformed < 0:
            # 回补给人数较多的一方，保持总数不变
            if n_noise >= n_informed:
                n_noise += n_uninformed
            else:
                n_informed += n_uninformed
            n_uninformed = 0

        self.n_informed = n_informed
        self.n_noise = n_noise
        self.n_uninformed = n_uninformed

        self.init_cash = self.agent_cfg["endowment"]["initial_bonds"]
        self.init_stock = self.agent_cfg["endowment"]["initial_shares"]

    def create_agents(self) -> List[Dict[str, Any]]:
        """
        生成标准化的 Agent 字典列表。
        Schema: {id, type, cash, stock, alive, pending_settlement}
        """
        agents: List[Dict[str, Any]] = []
        agent_id = 0

        def make_agent(agent_type: str) -> Dict[str, Any]:
            return {
                "id": agent_id,
                "type": agent_type,  # "informed" | "uninformed" | "noise"
                "cash": self.init_cash,
                "stock": self.init_stock,
                "alive": True,  # 4.1 生存实验关键字段
                "pending_settlement": [],  # 3.3 T+1 结算关键字段 (预留)
            }

        # 按顺序生成 (Informed -> Uninformed -> Noise)
        # 顺序不影响后续逻辑，因为每期交易都会 shuffle
        for _ in range(self.n_informed):
            agents.append(make_agent("informed"))
            agent_id += 1

        for _ in range(self.n_uninformed):
            agents.append(make_agent("uninformed"))
            agent_id += 1

        for _ in range(self.n_noise):
            agents.append(make_agent("noise"))
            agent_id += 1

        return agents

    def summary(self) -> Dict[str, Any]:
        """返回结构化统计信息，便于 Logger 记录"""
        return {
            "total": self.total_pop,
            "counts": {
                "informed": self.n_informed,
                "uninformed": self.n_uninformed,
                "noise": self.n_noise,
            },
            "shares": {
                "informed": round(self.n_informed / self.total_pop, 4),
                "uninformed": round(self.n_uninformed / self.total_pop, 4),
                "noise": round(self.n_noise / self.total_pop, 4),
            },
            "endowment": {"cash": self.init_cash, "stock": self.init_stock},
        }
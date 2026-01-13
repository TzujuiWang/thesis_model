import numpy as np
from typing import List, Tuple, Dict, Any
from .gp.tree import GPTreeFactory, GPNode
from .forecasting import ForecastingRule


class EvolutionaryEngine:
    """
    [Source 2 Steady-State Evolution]
    """

    def __init__(self,
                 tree_factory: GPTreeFactory,
                 rng: np.random.Generator,
                 config: Dict[str, Any]):

        self.factory = tree_factory
        self.rng = rng
        self.config = config  # Save full config for new rules

        probs = config.get('evolution_probs', {})
        self.p_cross = probs.get('crossover', 0.7)
        self.p_mut = probs.get('mutation', 0.2)
        # Immigration is implicit remainder

        self.tourn_size = config.get('selection', {}).get('tournament_size', 2)

        # [Strategy] New Rule Initialization
        # "default": use ForecastingRule default init (high variance)
        # "parent_mean": use average variance of parents (Source 2 hint)
        self.init_strategy = config.get('new_rule_init', 'default')

    def select_active_rule(self, rules: List[ForecastingRule]) -> ForecastingRule:
        """
        [Added Method] 从规则库中选择当前强度 (Strength) 最高的规则。
        LearningAgent 在每个 Period 开始时调用此方法。
        """
        if not rules:
            raise ValueError("No rules available to select.")

        # Strength = -Variance (方差越小，强度越大)
        # 直接返回强度最大的规则
        return max(rules, key=lambda r: r.strength)

    def evolve_rules(self, rules: List[ForecastingRule]) -> List[ForecastingRule]:
        """Replace the worst rule in the list."""
        if len(rules) < 2: return rules

        # 1. Identify Worst (lowest strength = highest variance)
        sorted_indices = sorted(range(len(rules)), key=lambda i: rules[i].strength)
        worst_idx = sorted_indices[0]

        survivors = [rules[i] for i in sorted_indices[1:]]
        if not survivors: survivors = rules

        r = self.rng.random()
        new_rule = None

        if r < self.p_cross:
            p1 = self._tournament_select(survivors)
            p2 = self._tournament_select(survivors)
            child, _ = self._crossover(p1, p2)  # Only keep one
            new_rule = child

            # Init Variance Strategy
            if self.init_strategy == 'parent_mean':
                avg_var = (p1.est_variance + p2.est_variance) / 2
                new_rule.est_variance = avg_var

        elif r < self.p_cross + self.p_mut:
            p1 = self._tournament_select(survivors)
            new_rule = self._mutation(p1)

            if self.init_strategy == 'parent_mean':
                new_rule.est_variance = p1.est_variance

        else:
            new_rule = self._immigration()
            # Immigration usually starts fresh (default variance)

        rules[worst_idx] = new_rule
        return rules

    def _tournament_select(self, pool: List[ForecastingRule]) -> ForecastingRule:
        size = min(len(pool), self.tourn_size)
        candidates = self.rng.choice(pool, size=size, replace=True)
        return max(candidates, key=lambda r: r.strength)

    def _crossover(self, p1: ForecastingRule, p2: ForecastingRule) -> Tuple[ForecastingRule, ForecastingRule]:
        c1 = p1.clone()
        c2 = p2.clone()

        nodes1 = c1.gp_tree.get_all_nodes()
        nodes2 = c2.gp_tree.get_all_nodes()

        if nodes1 and nodes2:
            pt1 = self.rng.choice(nodes1)
            pt2 = self.rng.choice(nodes2)

            # [Fix] Explicitly update root
            c1.gp_tree = self._do_swap(c1.gp_tree, pt1, pt2.clone())
            c2.gp_tree = self._do_swap(c2.gp_tree, pt2, pt1.clone())

        return c1, c2

    def _mutation(self, p: ForecastingRule) -> ForecastingRule:
        c = p.clone()
        nodes = c.gp_tree.get_all_nodes()
        if nodes:
            pt = self.rng.choice(nodes)
            new_sub = self.factory.create_random_tree(max_depth=2, method='grow')
            # [Fix] Explicitly update root
            c.gp_tree = self._do_swap(c.gp_tree, pt, new_sub)
        return c

    def _do_swap(self, root: GPNode, old_node: GPNode, new_node: GPNode) -> GPNode:
        """
        Replace old_node with new_node.
        Returns the new root (essential if root was replaced).
        """
        if root is old_node:
            return new_node

        if old_node.parent:
            old_node.parent.replace_child(old_node, new_node)
            return root

        # Should not happen if tree integrity is maintained
        raise ValueError("Tree structure corruption: old_node has no parent but is not root.")

    def _immigration(self) -> ForecastingRule:
        # Pass config to ensure Eq4 variant consistency
        tree = self.factory.create_random_tree(max_depth=4, method='grow')
        return ForecastingRule(tree, config=self.config)
import yaml
import os
import copy
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    加载 base.yaml + experiments.yaml，并按:
    Base -> Experiment Defaults -> Scenario Overrides -> (Optional Mode Overrides)
    做深度合并。
    不做隐式类型转换；list/标量一律覆盖；dict 递归合并。
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.base_config = self._load_yaml("base.yaml")
        self.experiments_config = self._load_yaml("experiments.yaml")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")

    def get_base_params(self) -> Dict[str, Any]:
        # 返回深拷贝，避免调用方意外修改内部状态
        return copy.deepcopy(self.base_config)

    def get_scenario_config(self, scenario_id: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        获取特定实验场景的完整配置。
        mode: 可选，例如 "survival_experiment_overrides"（对应 experiments.yaml optional_modes 的 key）
        """
        # 1) 深拷贝 base，避免污染
        config = copy.deepcopy(self.base_config)

        # 2) defaults
        exp_defaults = self.experiments_config.get("defaults", {})
        self._deep_update(config, exp_defaults)

        # 3) scenario overrides
        scenario = self._find_scenario(scenario_id)
        overrides = scenario.get("overrides", {})
        self._deep_update(config, overrides)

        # 4) optional mode overrides
        if mode is not None:
            mode_overrides = (self.experiments_config.get("optional_modes", {}) or {}).get(mode, {})
            if not mode_overrides:
                raise ValueError(f"Mode '{mode}' not found or empty in experiments.yaml optional_modes")
            self._deep_update(config, mode_overrides)

        # annotate
        config.setdefault("meta", {})
        if not isinstance(config["meta"], dict):
            raise TypeError("base.yaml meta must be a dict if present")
        config["meta"]["current_scenario_id"] = scenario_id
        if mode is not None:
            config["meta"]["current_mode"] = mode

        return config

    def _find_scenario(self, scenario_id: str) -> Dict[str, Any]:
        groups = self.experiments_config.get("groups", {}) or {}
        for _, group_data in groups.items():
            for sc in group_data.get("scenarios", []) or []:
                if sc.get("id") == scenario_id:
                    return sc
        raise ValueError(f"Scenario ID '{scenario_id}' not found in experiments.yaml")

    def _deep_update(self, source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则：
        - dict: 递归合并
        - list / scalar / None: 直接覆盖（不拼接）
        """
        if overrides is None:
            return source
        if not isinstance(overrides, dict):
            raise TypeError(f"Overrides must be a dict, got {type(overrides)}")

        for key, value in overrides.items():
            if isinstance(value, dict):
                existing = source.get(key)
                if isinstance(existing, dict):
                    source[key] = self._deep_update(existing, value)
                else:
                    # 源不是 dict，但 override 是 dict：用空 dict 起步合并
                    source[key] = self._deep_update({}, value)
            else:
                # list/scalar/None: 覆盖
                source[key] = value
        return source

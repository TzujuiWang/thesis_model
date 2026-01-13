import yaml
import os
import copy
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    加载 base.yaml + experiments.yaml
    支持 'groups' 嵌套结构和 'optional_modes'。
    """

    def __init__(self):
        # 1. 路径设置
        current_file_path = os.path.abspath(__file__)
        core_dir = os.path.dirname(current_file_path)
        project_root = os.path.dirname(core_dir)
        self.config_dir = os.path.join(project_root, "configs")

        # 2. 加载 YAML
        self.base_config = self._load_yaml("base.yaml")
        self.experiments_config = self._load_yaml("experiments.yaml")

        # 3. [关键新增] 预加载所有场景到缓存，处理 groups 结构
        self._scenario_cache = {}
        self._preload_scenarios()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")

    def _preload_scenarios(self):
        """
        解析 experiments.yaml，将所有场景扁平化存入 _scenario_cache。
        支持两种格式：
        1. 根目录 scenarios: { ID: { overrides: ... } } (字典格式)
        2. groups: { GroupName: { scenarios: [ {id: ID, overrides: ...} ] } } (列表格式，即你现在的格式)
        """
        # 格式 1: Flat Dict
        flat_scenarios = self.experiments_config.get("scenarios", {})
        if isinstance(flat_scenarios, dict):
            self._scenario_cache.update(flat_scenarios)

        # 格式 2: Groups List
        groups = self.experiments_config.get("groups", {})
        if isinstance(groups, dict):
            for group_name, group_data in groups.items():
                # 获取该组下的 scenario 列表
                sc_list = group_data.get("scenarios", [])
                if isinstance(sc_list, list):
                    for sc in sc_list:
                        sc_id = sc.get("id")
                        if sc_id:
                            # 存入缓存
                            self._scenario_cache[sc_id] = sc

    def get_base_params(self) -> Dict[str, Any]:
        return copy.deepcopy(self.base_config)

    def get_scenario_config(self, scenario_id: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        获取特定实验场景的完整配置。
        """
        # 1) 深拷贝 base
        config = copy.deepcopy(self.base_config)

        # 2) 应用 defaults
        exp_defaults = self.experiments_config.get("defaults", {})
        self._deep_update(config, exp_defaults)

        # 3) 获取场景配置 (从预加载的缓存中取)
        if scenario_id not in self._scenario_cache:
            raise KeyError(
                f"Scenario ID '{scenario_id}' not found in experiments.yaml (checked {len(self._scenario_cache)} scenarios)")

        scenario_cfg = self._scenario_cache[scenario_id]

        # 4) 合并 Scenario Overrides
        # 你的 yaml 结构是: {id: "B0", overrides: {...}}
        overrides = scenario_cfg.get("overrides", {})
        self._deep_update(config, overrides)

        # 5) mode overrides (可选，用于生存实验)
        if mode is not None:
            mode_overrides = self.experiments_config.get("optional_modes", {}).get(mode, {})
            if not mode_overrides:
                print(f"Warning: Mode '{mode}' not found in experiments.yaml optional_modes")
            else:
                # 例如 survival 模式会覆盖 replacement_policy
                # 你的 YAML 里: survival_experiment_overrides -> replacement_policy -> overrides
                # 注意：你的 YAML 写法是直接写 replacement_policy，不是 overrides 嵌套
                # 所以直接合并即可
                self._deep_update(config, mode_overrides)

        return config

    def _deep_update(self, source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        if overrides is None:
            return source
        if not isinstance(overrides, dict):
            return source

        for key, value in overrides.items():
            if isinstance(value, dict):
                existing = source.get(key)
                if isinstance(existing, dict):
                    source[key] = self._deep_update(existing, value)
                else:
                    source[key] = self._deep_update({}, value)
            else:
                source[key] = value
        return source
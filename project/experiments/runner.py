import os
import yaml
import pandas as pd
from project.core.config_loader import ConfigLoader
from project.sim.engine import SimulationEngine


class ExperimentRunner:
    def __init__(self, output_dir: str = "output"):
        self.loader = ConfigLoader()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_scenario(self, scenario_id: str, mode: str = None):
        print(f"=== Scenario: {scenario_id} (Mode: {mode}) ===")
        results = []

        # Dump Config for Reproducibility
        # Just dump the first run config (assuming identical except seed)
        sample_cfg = self.loader.get_scenario_config(scenario_id, mode)
        with open(os.path.join(self.output_dir, f"{scenario_id}_config.yaml"), 'w') as f:
            yaml.dump(sample_cfg, f)

        for run_idx in range(10):  # 10 runs
            config = self.loader.get_scenario_config(scenario_id, mode)
            engine = SimulationEngine(config, run_idx)
            stats = engine.run()

            stats.update({
                'scenario': scenario_id,
                'run_id': run_idx,
                'mode': mode,
                'seed': engine.master_seed
            })
            results.append(stats)

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, f"{scenario_id}_{mode or 'default'}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        print(df.mean(numeric_only=True))


if __name__ == "__main__":
    runner = ExperimentRunner()
    # Example: Run Baseline B0
    runner.run_scenario("B0")
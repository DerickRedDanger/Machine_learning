from titanic_ml.paths import EXPERIMENT_RESULTS_FILE, EXPERIMENT_CONFIGS_FILE
import pandas as pd

def save_results(results_df, path=EXPERIMENT_RESULTS_FILE, append=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        old_results = pd.read_csv(path)
        results_df = pd.concat([old_results, results_df], ignore_index=True)

        # Avoid duplicate experiment rows if you rerun the same experiment
        results_df = results_df.drop_duplicates(
            subset=["experiment"],
            keep="last",
        )

    results_df.to_csv(path, index=False)

    return results_df


def load_results(path=EXPERIMENT_RESULTS_FILE):
    return pd.read_csv(path)

import json
from pathlib import Path


def make_config_json_safe(config):
    safe_config = {}

    for key, value in config.items():
        if key == "feature_engineering" and isinstance(value, list):
            safe_config[key] = [
                fn.__name__ if callable(fn) else str(fn)
                for fn in value
            ]
        else:
            safe_config[key] = value

    return safe_config


def save_configs(configs, path=EXPERIMENT_CONFIGS_FILE, append=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_configs = {
        config["name"]: make_config_json_safe(config)
        for config in configs
    }

    if append and path.exists():
        with open(path, "r", encoding="utf-8") as file:
            old_configs = json.load(file)
    else:
        old_configs = {}

    old_configs.update(new_configs)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(old_configs, file, indent=2)

    return old_configs


def load_configs(path=EXPERIMENT_CONFIGS_FILE):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
    
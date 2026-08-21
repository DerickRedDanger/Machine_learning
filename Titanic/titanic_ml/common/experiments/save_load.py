from titanic_ml.paths import EXPERIMENT_RESULTS_FILE, EXPERIMENT_CONFIGS_FILE, EXPERIMENT_FEATURE_EFFECT
import pandas as pd
import json
from sklearn.base import BaseEstimator
import numpy as np
from pathlib import Path

# temporary test overwrite
EXPERIMENT_RESULTS_FILE = "test_results.csv"
EXPERIMENT_CONFIGS_FILE = "test_configs.json"
EXPERIMENT_FEATURE_EFFECT = "test_feature_effects.json"

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




def make_json_safe(value):
    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, BaseEstimator):
        return {
            "class": value.__class__.__name__,
            "params": make_json_safe(
                value.get_params(deep=False)
            ),
        }

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if callable(value):
        return getattr(
            value,
            "__name__",
            str(value),
        )

    return value

def make_config_json_safe(config):
    return make_json_safe(config)

def _config_values(configs):
    if isinstance(configs, dict):
        # Single experiment config
        if "name" in configs and "model_name" in configs:
            return [configs]

        # Experiment group
        return configs.values()

    if isinstance(configs, (list, tuple)):
        return configs

    raise TypeError(
        "configs must be a configuration dictionary, "
        "a dictionary of configurations, or a list/tuple "
        "of configurations."
    )

def save_configs(
    configs,
    path=EXPERIMENT_CONFIGS_FILE,
    append=True,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_configs = {
        config["name"]:
            make_config_json_safe(config)

        for config in _config_values(configs)
    }

    if append and path.exists():
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            old_configs = json.load(file)
    else:
        old_configs = {}

    old_configs.update(new_configs)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            old_configs,
            file,
            indent=2,
        )

    return old_configs

def load_configs(path=EXPERIMENT_CONFIGS_FILE):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_feature_effects(
    feature_effects,
    path=EXPERIMENT_FEATURE_EFFECT,
    append=True,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        with open(path, "r", encoding="utf-8") as file:
            existing = json.load(file)
    else:
        existing = {}

    for effect in feature_effects:
        group = effect["compare_group"]
        existing[group] = effect

    with open(path, "w", encoding="utf-8") as file:
        json.dump(existing, file, indent=2)

    return existing


def load_feature_effects(path=EXPERIMENT_FEATURE_EFFECT):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
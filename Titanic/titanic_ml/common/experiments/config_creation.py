import copy
import json

VALID_EXPERIMENT_KEYS = {
    "name",
    "features",
    "feature_engineering",
    "preprocessing",
    "model_name",
    "model_params",
    "evaluation",
    "notes",
}

def validate_experiment_override(override):
    unknown_keys = set(override) - VALID_EXPERIMENT_KEYS

    if unknown_keys:
        raise ValueError(f"Unknown override keys: {unknown_keys}")

    if "features" in override and not isinstance(override["features"], list):
        raise TypeError("'features' must be a list.")

    if "feature_engineering" in override:
        feature_engineering = override["feature_engineering"]

        if not isinstance(feature_engineering, list):
            raise TypeError("'feature_engineering' must be a list of functions.")

        for fn in feature_engineering:
            if not callable(fn):
                raise TypeError(f"Feature engineering item is not callable: {fn}")

    if "preprocessing" in override and not isinstance(override["preprocessing"], dict):
        raise TypeError("'preprocessing' must be a dictionary.")

    if "model_params" in override and not isinstance(override["model_params"], dict):
        raise TypeError("'model_params' must be a dictionary.")

    if "evaluation" in override and not isinstance(override["evaluation"], dict):
        raise TypeError("'evaluation' must be a dictionary.")


def create_experiment_group(stage, feature_group, base_configs, group_override=None, domain=None):
    if not stage:
        raise ValueError("Experiment group needs a stage, like 'fe01'.")

    if not feature_group:
        raise ValueError("Experiment group needs a feature group, like 'family'.")

    if not base_configs:
        raise ValueError("No base configs provided.")

    group_override = group_override or {}

    validate_experiment_override(group_override)

    configs = copy.deepcopy(base_configs)

    for exp in configs:
        model_name = exp["model_name"]

        exp.update(group_override)

        exp["name"] = f"{stage}__{feature_group}__{model_name}"
        exp["stage"] = stage
        exp["feature_group"] = feature_group
        exp["group"] = f"{stage}__{feature_group}"
        exp["domain"] = domain or feature_group


    return configs
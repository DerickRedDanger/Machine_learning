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

def features_from_preprocessing(preprocessing):
    feature_groups = [
        "numeric_features",
        "onehot_features",
        "ordinal_features",
    ]

    features = []
    seen = set()

    for group in feature_groups:
        for feature in preprocessing.get(group, []):
            if feature in seen:
                raise ValueError(
                    f"Feature '{feature}' appears in more than one "
                    "preprocessing group."
                )

            seen.add(feature)
            features.append(feature)

    return features

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

    validate_feature_configuration(configs)

    for exp in configs:
        exp["features"] = features_from_preprocessing(
            exp["preprocessing"]
        )

    return configs

def validate_feature_configuration(
    config,
    target="Survived",
):
    preprocessing = config["preprocessing"]

    groups = {
        "numeric_features":
            preprocessing.get("numeric_features", []),

        "onehot_features":
            preprocessing.get("onehot_features", []),

        "ordinal_features":
            preprocessing.get("ordinal_features", []),
    }

    occurrences = {}

    for group_name, features in groups.items():

        # Duplicates inside the same group
        if len(features) != len(set(features)):
            raise ValueError(
                f"Duplicate feature found in "
                f"'{group_name}'."
            )

        for feature in features:
            occurrences.setdefault(
                feature, []
            ).append(group_name)

    # Same feature assigned to multiple transformers
    duplicates = {
        feature: locations
        for feature, locations in occurrences.items()
        if len(locations) > 1
    }

    if duplicates:
        raise ValueError(
            "Features assigned to multiple "
            f"preprocessing groups: {duplicates}"
        )

    if target in occurrences:
        raise ValueError(
            f"Target '{target}' cannot be used "
            "as an input feature."
        )

    return True
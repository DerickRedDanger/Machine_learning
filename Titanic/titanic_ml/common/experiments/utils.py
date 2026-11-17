def get_experiment_configs(experiment_configs):
    if isinstance(experiment_configs, dict):
        return experiment_configs.values()

    return experiment_configs

def get_config_group(experiment_configs):
    if not experiment_configs:
        raise ValueError(
            "No experiment configs provided."
        )

    groups = {
        exp.get("group")
        for exp in get_experiment_configs(experiment_configs)
    }

    if len(groups) != 1:
        raise ValueError(
            f"Expected one group, found: {groups}"
        )

    return groups.pop()


def index_experiments_by_name(experiment_groups):
    """
    Build a flat lookup of experiment configurations by name.

    Returns references to the original configurations,
    without copying them.

    Raises ValueError if duplicate experiment names exist.
    """
    indexed = {}

    for group in experiment_groups.values():
        for exp in group.values():
            name = exp["name"]

            if name in indexed:
                raise ValueError(
                    f"Duplicate experiment name: '{name}'."
                )

            indexed[name] = exp

    return indexed

def validate_threshold_config_references(
    ALL_THRESHOLD_CONFIGS,
    HPO_CONFIGS_BY_NAME, debug=False
):
    for threshold_key, threshold_config in ALL_THRESHOLD_CONFIGS.items():
        threshold_name = threshold_config["name"]
        source_name = threshold_config["source_experiment"]

        if source_name not in HPO_CONFIGS_BY_NAME:
            raise ValueError(
                f"Threshold experiment '{threshold_name}' references "
                f"unknown HPO experiment '{source_name}'."
            )

        source_config = HPO_CONFIGS_BY_NAME[source_name]

        expected_threshold = source_config.get("threshold_experiment")

        if expected_threshold != threshold_name:
            raise ValueError(
                f"Threshold configuration mismatch:\n"
                f"  Registry key: {threshold_key}\n"
                f"  Threshold name: {threshold_name}\n"
                f"  Source experiment: {source_name}\n"
                f"  Source references: {expected_threshold}"
            )

        if threshold_key != threshold_name:
            raise ValueError(
                f"Threshold registry key '{threshold_key}' "
                f"does not match configuration name '{threshold_name}'."
            )
        if debug:
            print(f"{threshold_name} -> {source_name}")
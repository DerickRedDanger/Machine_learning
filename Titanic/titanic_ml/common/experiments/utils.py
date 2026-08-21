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
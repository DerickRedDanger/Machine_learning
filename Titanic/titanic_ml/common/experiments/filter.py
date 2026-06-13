def filter_results(
    results_df,
    stage=None,
    feature_group=None,
    model=None,
    group=None,
):
    df = results_df.copy()

    if stage is not None:
        df = df[df["stage"] == stage]

    if feature_group is not None:
        df = df[df["feature_group"] == feature_group]

    if model is not None:
        df = df[df["model_name"] == model]

    if group is not None:
        df = df[df["group"] == group]

    return df

def filter_configs(configs,
    stage=None,
    feature_group=None,
    model=None,
    group=None,
    ):
    filtered = {}

    for name, config in configs.items():

        if stage is not None and config["stage"] != stage:
            continue

        if feature_group is not None and config["feature_group"] != feature_group:
            continue

        if model is not None and config["model_name"] != model:
            continue

        if group is not None and config["group"] != group:
            continue

        filtered[name] = config

    return filtered
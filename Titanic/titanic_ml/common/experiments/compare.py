import pandas as pd
from titanic_ml.common.models.registry import MODEL_REGISTRY

DEFAULT_COMPARE_METRICS = [
    "test_accuracy_mean",
    "test_precision_mean",
    "test_recall_mean",
    "test_f1_mean",
]


def validate_metrics(results_df, metrics):
    missing = [metric for metric in metrics if metric not in results_df.columns]

    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    

def leaderboard(results_df, metric="test_accuracy_mean", top_n=10, only_success=True):
    df = results_df.copy()

    if only_success and "status" in df.columns:
        df = df[df["status"] == "success"]

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results dataframe.")

    columns = [
        "experiment",
        "group",
        "model_name",
        metric,
        "notes",
    ]

    columns = [col for col in columns if col in df.columns]

    return (
        df[columns]
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

def titanic_notes_leaderboard(results_df, top_n=10, only_success=True):
    df = results_df.copy()
    metric = "test_accuracy_mean"

    if only_success and "status" in df.columns:
        df = df[df["status"] == "success"]

    columns = [
        "experiment",
        "model_name",
        metric,
        "test_f1_mean",
    ]

    columns = [col for col in columns if col in df.columns]

    return (
        df[columns]
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    ).to_markdown(index=False)

def get_config_group(experiment_configs):
    if not experiment_configs:
        raise ValueError("No experiment configs provided.")

    groups = {exp.get("group") for exp in experiment_configs}

    if len(groups) != 1:
        raise ValueError(f"Expected one group, found: {groups}")

    return groups.pop()

def compare_experiment_groups(
    results_df,
    reference_group,
    compare_groups=None,
    metrics=None,
    only_success=True,
):
    if metrics is None:
        metrics = DEFAULT_COMPARE_METRICS

    validate_metrics(results_df, metrics)

    df = results_df.copy()

    if only_success and "status" in df.columns:
        df = df[df["status"] == "success"]

    if compare_groups is None:
        compare_groups = sorted(
            group for group in df["group"].unique()
            if group != reference_group
        )

    if isinstance(compare_groups, str):
        compare_groups = [compare_groups]

    reference_df = df[df["group"] == reference_group]

    if reference_df.empty:
        raise ValueError(f"Reference group '{reference_group}' not found.")

    comparison_rows = []

    for compare_group in compare_groups:
        if not isinstance(compare_group, str):
            compare_group = get_config_group(compare_group)
        compare_df = df[df["group"] == compare_group]

        if compare_df.empty:
            raise ValueError(f"Comparison group '{compare_group}' not found.")

        merged = reference_df.merge(
            compare_df,
            on="model_name",
            suffixes=("_reference", "_compare"),
        )

        for _, row in merged.iterrows():
            result = {
                "reference_group": reference_group,
                "compare_group": compare_group,
                "model_name": row["model_name"],
            }

            for metric in metrics:
                reference_value = row[f"{metric}_reference"]
                compare_value = row[f"{metric}_compare"]
                delta = compare_value - reference_value

                result[f"{metric}_reference"] = reference_value
                result[f"{metric}_compare"] = compare_value
                result[f"{metric}_delta"] = delta

            comparison_rows.append(result)

    return pd.DataFrame(comparison_rows)

def summarize_group_comparison(comparison_df, metrics=None):
    if metrics is None:
        metrics = DEFAULT_COMPARE_METRICS

    delta_columns = [f"{metric}_delta" for metric in metrics]

    summary = (
        comparison_df
        .groupby("compare_group")[delta_columns]
        .agg(["mean", "min", "max"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    return summary.reset_index()


def model_progression(results_df, model_name, metric="test_accuracy_mean"):
    df = results_df.copy()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in model registry.")

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results dataframe.")

    model_df = df[df["model_name"] == model_name]

    if model_df.empty:
        raise ValueError(f"Model '{model_name}' not found in results dataframe.")

    columns = ["stage", "feature_group", metric]
    sort_cols = ["stage", "feature_group"]
    return (
        model_df[columns]
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

def analyze_feature_effect(
    comparison_df,
    metric="test_accuracy_mean",
    secondary_metrics=None,
    secondary_positive_threshold=0.01,
    secondary_negative_threshold=-0.01,
    positive_threshold=0.003,
    strong_threshold=0.01,
    negative_threshold=-0.003,
):
    if secondary_metrics is None:
        secondary_metrics = ["test_f1_mean"]

    primary_delta_col = f"{metric}_delta"

    if primary_delta_col not in comparison_df.columns:
        raise ValueError(f"Delta column '{primary_delta_col}' not found.")

    for secondary_metric in secondary_metrics:
        secondary_delta_col = f"{secondary_metric}_delta"

        if secondary_delta_col not in comparison_df.columns:
            raise ValueError(f"Delta column '{secondary_delta_col}' not found.")

    rows = []

    for compare_group, group_df in comparison_df.groupby("compare_group"):
        positive_models = []
        negative_models = []
        neutral_models = []

        recommended_model_deltas = {}
        recommended_model_tradeoffs = {}
        notable_secondary_improvements = {}

        for _, row in group_df.iterrows():
            model_name = row["model_name"]
            primary_delta = row[primary_delta_col]

            secondary_deltas = {
                secondary_metric: row[f"{secondary_metric}_delta"]
                for secondary_metric in secondary_metrics
            }

            if primary_delta >= positive_threshold:
                positive_models.append(model_name)
                recommended_model_deltas[model_name] = round(primary_delta, 3)

                pros = {}
                cons = {}

                for secondary_metric, delta in secondary_deltas.items():
                    rounded_delta = round(delta, 3)

                    if delta >= secondary_positive_threshold:
                        pros[secondary_metric] = rounded_delta
                    elif delta <= secondary_negative_threshold:
                        cons[secondary_metric] = rounded_delta

                recommended_model_tradeoffs[model_name] = {
                    "primary_metric": metric,
                    "primary_delta": round(primary_delta, 3),
                    "secondary_pros": pros,
                    "secondary_cons": cons,
                }

            elif primary_delta <= negative_threshold:
                negative_models.append(model_name)

            else:
                neutral_models.append(model_name)

                notable = {
                    secondary_metric: round(delta, 3)
                    for secondary_metric, delta in secondary_deltas.items()
                    if delta >= secondary_positive_threshold
                }

                if notable:
                    notable_secondary_improvements[model_name] = notable

        mean_delta = round(group_df[primary_delta_col].mean(), 3)
        max_delta = round(group_df[primary_delta_col].max(), 3)
        min_delta = round(group_df[primary_delta_col].min(), 3)

        n_models = len(group_df)
        n_positive = len(positive_models)
        n_negative = len(negative_models)
        n_neutral = len(neutral_models)

        all_positive = n_positive == n_models
        all_negative = n_negative == n_models
        has_positive = n_positive > 0
        has_negative = n_negative > 0
        has_neutral = n_neutral > 0

        if all_positive and mean_delta >= strong_threshold:
            verdict = "strong_general"
        elif all_positive:
            verdict = "general_positive"
        elif all_negative:
            verdict = "general_negative"
        elif has_positive and has_negative:
            verdict = "mixed"
        elif has_positive and has_neutral:
            verdict = "model_specific_positive"
        elif has_negative and has_neutral:
            verdict = "model_specific_negative"
        else:
            verdict = "neutral"

        rows.append({
            "compare_group": compare_group,
            "metric": metric,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "min_delta": min_delta,
            "positive_models": positive_models,
            "neutral_models": neutral_models,
            "negative_models": negative_models,
            "verdict": verdict,
            "recommended_for_all": verdict in ["strong_general", "general_positive"],
            "recommended_model_deltas": recommended_model_deltas,
            "recommended_model_tradeoffs": recommended_model_tradeoffs,
            "notable_secondary_improvements": notable_secondary_improvements,
            "discard": verdict == "general_negative",
        })

    return rows

def domain_best_by_model(
    results_df,
    domain,
    metric="test_accuracy_mean",
    include_baseline=True,
    only_success=True,
):
    df = results_df.copy()

    if only_success and "status" in df.columns:
        df = df[df["status"] == "success"]

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results dataframe.")

    domain_df = df[df["domain"] == domain]

    if include_baseline:
        baseline_df = df[df["group"] == "baseline__raw"]
        domain_df = pd.concat([baseline_df, domain_df], ignore_index=True)

    if domain_df.empty:
        raise ValueError(f"No results found for domain: {domain}")

    best_rows = (
        domain_df
        .sort_values(metric, ascending=False)
        .groupby("model_name", as_index=False)
        .first()
    )

    columns = [
        "model_name",
        "experiment",
        "group",
        "domain",
        metric,
        "test_f1_mean",
    ]

    columns = [col for col in columns if col in best_rows.columns]

    return (
        best_rows[columns]
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )

def domain_best_by_model_with_baseline_delta(
    results_df,
    domain,
    metric="test_accuracy_mean",
):
    best_df = domain_best_by_model(
        results_df=results_df,
        domain=domain,
        metric=metric,
        include_baseline=True,
    )

    baseline = (
        results_df[results_df["group"] == "baseline__raw"]
        [["model_name", metric]]
        .rename(columns={metric: f"{metric}_baseline"})
    )

    best_df = best_df.merge(baseline, on="model_name", how="left")
    best_df[f"{metric}_delta_vs_baseline"] = (
        best_df[metric] - best_df[f"{metric}_baseline"]
    )

    return best_df


METRIC_COLUMNS = {
    "accuracy": "test_accuracy_mean",
    "precision": "test_precision_mean",
    "recall": "test_recall_mean",
    "f1": "test_f1_mean",
}


def recommended_by_domain_for_model(
    results_df,
    model_name,
    thresholds,
    baseline_group="baseline__raw",
    only_success=True,
):
    """
    Selects at most one recommended experiment per domain for a model.

    Experiments must satisfy every metric threshold relative to the model's
    baseline result. Among valid experiments in each domain, the best one is
    selected according to the priority order of the metrics in `thresholds`.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing experiment results.

    model_name : str
        Model to evaluate.

    thresholds : dict[str, float]
        Minimum acceptable delta for each metric.

        Supported keys:
            - "accuracy"
            - "precision"
            - "recall"
            - "f1"

        Dictionary insertion order determines ranking priority.

        Example:
            {
                "accuracy": 0.003,
                "f1": -0.01,
            }

        means:
            1. accuracy delta must be >= 0.003
            2. f1 delta must be >= -0.01
            3. accuracy is the primary ranking metric
            4. f1 is the secondary ranking metric

    baseline_group : str, default="baseline__raw"
        Experiment group used as baseline.

    only_success : bool, default=True
        If True, ignores failed experiments.

    Returns
    -------
    dict
        {
            "df": all model experiments with calculated deltas,
            "recommended_df": best valid experiment per domain,
            "recommended": compact list of recommendations,
        }
    """

    if not thresholds:
        raise ValueError("At least one metric threshold must be provided.")

    invalid_metrics = [
        metric
        for metric in thresholds
        if metric not in METRIC_COLUMNS
    ]

    if invalid_metrics:
        raise ValueError(
            f"Unsupported metrics: {invalid_metrics}. "
            f"Supported metrics are: {list(METRIC_COLUMNS)}"
        )

    df = results_df.copy()

    if only_success and "status" in df.columns:
        df = df[df["status"] == "success"]

    # ---------------------------------------------------------
    # 1. Select the requested model
    # ---------------------------------------------------------

    model_df = df[df["model_name"] == model_name].copy()

    if model_df.empty:
        raise ValueError(
            f"No results found for model '{model_name}'."
        )

    # ---------------------------------------------------------
    # 2. Retrieve exactly one baseline row
    # ---------------------------------------------------------

    baseline_df = model_df[
        model_df["group"] == baseline_group
    ]

    if len(baseline_df) != 1:
        raise ValueError(
            f"Expected exactly one baseline result for model "
            f"'{model_name}' in group '{baseline_group}', "
            f"found {len(baseline_df)}."
        )

    baseline_row = baseline_df.iloc[0]

    # ---------------------------------------------------------
    # 3. Remove baseline from candidate experiments
    # ---------------------------------------------------------

    experiment_df = model_df[
        model_df["group"] != baseline_group
    ].copy()

    if experiment_df.empty:
        raise ValueError(
            f"No non-baseline experiments found for model "
            f"'{model_name}'."
        )

    # Domain is now required metadata for experiment selection.
    missing_domain = experiment_df["domain"].isna()

    if missing_domain.any():
        missing_groups = (
            experiment_df.loc[missing_domain, "group"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Some non-baseline experiments have no domain: "
            f"{missing_groups}"
        )

    # ---------------------------------------------------------
    # 4. Calculate deltas for requested metrics
    # ---------------------------------------------------------

    delta_columns = []

    for metric in thresholds:
        metric_col = METRIC_COLUMNS[metric]

        if metric_col not in experiment_df.columns:
            raise ValueError(
                f"Metric column '{metric_col}' not found "
                "in results dataframe."
            )

        baseline_value = baseline_row[metric_col]

        delta_col = f"{metric}_delta_vs_baseline"

        experiment_df[delta_col] = (
            experiment_df[metric_col] - baseline_value
        )

        delta_columns.append(delta_col)

    # ---------------------------------------------------------
    # 5. Apply every threshold
    # ---------------------------------------------------------

    candidate_mask = pd.Series(
        True,
        index=experiment_df.index,
    )

    for metric, threshold in thresholds.items():
        delta_col = f"{metric}_delta_vs_baseline"

        candidate_mask &= (
            experiment_df[delta_col] >= threshold
        )

    candidate_df = experiment_df[candidate_mask].copy()

    # ---------------------------------------------------------
    # 6. Rank candidates according to metric priority
    # ---------------------------------------------------------

    if candidate_df.empty:
        recommended_df = candidate_df.copy()

    else:
        recommended_df = (
            candidate_df
            .sort_values(
                by=delta_columns,
                ascending=[False] * len(delta_columns),
                kind="stable",
            )
            .drop_duplicates(
                subset="domain",
                keep="first",
            )
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # 7. Build compact recommendation output
    # ---------------------------------------------------------

    recommended = []

    for _, row in recommended_df.iterrows():

        deltas = {
            metric: round(
                row[f"{metric}_delta_vs_baseline"],
                3,
            )
            for metric in thresholds
        }

        recommended.append({
            "domain": row["domain"],
            "group": row["group"],
            "experiment": row["experiment"],
            "deltas": deltas,
        })

    # ---------------------------------------------------------
    # 8. Useful display dataframe
    # ---------------------------------------------------------

    display_columns = [
        "domain",
        "group",
        "experiment",
        "model_name",
        *[
            METRIC_COLUMNS[metric]
            for metric in thresholds
        ],
        *delta_columns,
    ]

    display_columns = [
        col
        for col in display_columns
        if col in experiment_df.columns
    ]

    experiment_df = (
        experiment_df[display_columns]
        .sort_values(
            ["domain", *delta_columns],
            ascending=[True] + [False] * len(delta_columns),
        )
        .reset_index(drop=True)
    )

    recommended_df = recommended_df[
        [
            col
            for col in display_columns
            if col in recommended_df.columns
        ]
    ]

    return {
        "df": experiment_df,
        "recommended_df": recommended_df,
        "recommended": recommended,
    }
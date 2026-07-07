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
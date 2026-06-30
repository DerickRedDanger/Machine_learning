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
    positive_threshold=0.003,
    strong_threshold=0.01,
    negative_threshold=-0.003,
):
    delta_col = f"{metric}_delta"

    if delta_col not in comparison_df.columns:
        raise ValueError(f"Delta column '{delta_col}' not found.")

    rows = []

    for compare_group, group_df in comparison_df.groupby("compare_group"):
        positive_models = []
        negative_models = []
        neutral_models = []
        positive_model_deltas = {}


        

        for _, row in group_df.iterrows():
            model_name = row["model_name"]
            delta = row[delta_col]

            if delta >= positive_threshold:
                positive_models.append(model_name)
                positive_model_deltas[model_name] = round(delta,3)

            elif delta <= negative_threshold:
                negative_models.append(model_name)

            else:
                neutral_models.append(model_name)

        mean_delta = round(group_df[delta_col].mean(),3)
        max_delta = round(group_df[delta_col].max(),3)
        min_delta = round(group_df[delta_col].min(),3)

        n_models = len(group_df)
        n_positive = len(positive_models)
        n_negative = len(negative_models)

        if mean_delta >= strong_threshold and n_positive >= max(1, int(0.7 * n_models)):
            verdict = "strong_general"
        elif n_positive >= max(1, int(0.7 * n_models)):
            verdict = "general_positive"
        elif n_positive > 0 and n_negative > 0:
            verdict = "model_specific_mixed"
        elif n_positive > 0:
            verdict = "model_specific_positive"
        elif n_negative >= max(1, int(0.7 * n_models)):
            verdict = "general_negative"
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
            "recommended_models": positive_model_deltas,
            "discard": verdict in ["general_negative", "neutral"] and max_delta < positive_threshold,
        })

    return rows
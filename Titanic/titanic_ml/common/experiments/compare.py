import pandas as pd

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
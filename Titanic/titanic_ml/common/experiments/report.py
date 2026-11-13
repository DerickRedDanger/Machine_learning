import pandas as pd
import pprint
from pathlib import Path
from titanic_ml.common.experiments.save_load import make_json_safe
from titanic_ml.paths import EXPERIMENT_RESULTS_FILE, EXPERIMENT_CONFIGS_FILE
from titanic_ml.common.experiments.compare import compare_experiment_groups, summarize_group_comparison, leaderboard, titanic_notes_leaderboard
def format_metric(result_row, metric_name):
    mean_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"

    mean = result_row.get(mean_key)
    std = result_row.get(std_key)

    if mean is None or std is None:
        return "N/A"

    return f"{mean} ± {std}"

def experiment_result_to_markdown(result_row):
    if hasattr(result_row, "to_dict"):
        result_row = result_row.to_dict()

    lines = [
        f"Experiment - {result_row.get('experiment', 'N/A')}:",
        "| Field | Value |",
        "|---|---|",
        f"|Train accuracy| {format_metric(result_row, 'train_accuracy')} |",
        f"|Train precision| {format_metric(result_row, 'train_precision')} |",
        f"|Train recall| {format_metric(result_row, 'train_recall')} |",
        f"|Train f1| {format_metric(result_row, 'train_f1')} |",
        f"|Test accuracy| {format_metric(result_row, 'test_accuracy')} |",
        f"|Test precision| {format_metric(result_row, 'test_precision')} |",
        f"|Test recall| {format_metric(result_row, 'test_recall')} |",
        f"|Test f1| {format_metric(result_row, 'test_f1')} |",
    ]
    return "\n".join(lines)

def experiments_summary_to_markdown(results_df):
    summary_rows = []

    for _, row in results_df.iterrows():
        summary_rows.append({
            "experiment": row.get("experiment", "N/A"),
            "model_name": row.get("model_name", "N/A"),
            "status": row.get("status", "N/A"),
            "test_accuracy": format_metric(row, "test_accuracy"),
            "test_precision": format_metric(row, "test_precision"),
            "test_recall": format_metric(row, "test_recall"),
            "test_f1": format_metric(row, "test_f1"),
            "fit_time": row.get("fit_time_mean", "N/A"),
            "notes": row.get("notes", ""),

        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df.to_markdown(index=False)

def baseline_summary_to_markdown(results_df):
    summary_rows = []

    for _, row in results_df.iterrows():
        summary_rows.append({
            "model_name": row.get("model_name", "N/A"),
            "accuracy": format_metric(row, "test_accuracy"),
            "f1": format_metric(row, "test_f1"),
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df.to_markdown(index=False)


def experiment_config_to_markdown(config):
    safe_config = make_json_safe(
        config
    )

    return pprint.pformat(
        safe_config,
        sort_dicts=False,
    )

def experiment_report(results_df, experiment_configs, print_report=False):
    individual_reports = []

    configs_by_name = {
        config["name"]: config
        for config in experiment_configs.values()
    }

    for result_row in results_df.to_dict(orient="records"):
        experiment_name = result_row["experiment"]

        if experiment_name not in configs_by_name:
            raise ValueError(
                f"No configuration found for experiment "
                f"'{experiment_name}'."
            )

        config = configs_by_name[experiment_name]
        experiment_name = result_row.get("experiment", "N/A")

        individual_reports.append({
            "experiment": experiment_name,
            "result_markdown": experiment_result_to_markdown(result_row),
            "config_markdown": experiment_config_to_markdown(config),
        })

    full_report = experiments_summary_to_markdown(results_df)

    if print_report:
        print("Individual Experiment Reports:")

        for report in individual_reports:
            print()
            print(report["result_markdown"])

            print("\nFull configuration:")
            print("```python")
            print(report["config_markdown"])
            print("```")

            print("-" * 40)

        print("\nFull Summary Report:\n")
        print(full_report)

    return individual_reports, full_report


def experiment_group_report_to_markdown(
    results_df,
    experiment_configs,
    experiment_name,
    description="",
    conclusion="",
):
    lines = []

    lines.append(f"### {experiment_name}")
    lines.append("")
    lines.append(description if description else "_Description pending._")
    lines.append("")

    lines.append("#### Result")
    lines.append("")
    lines.append(experiments_summary_to_markdown(results_df))
    lines.append("")

    lines.append("<details>")
    lines.append("<summary>Model details</summary>")
    lines.append("")

    configs_by_name = {
        config["name"]: config
        for config in experiment_configs.values()
    }

    for result_row in results_df.to_dict(orient="records"):
        experiment_name = result_row["experiment"]

        if experiment_name not in configs_by_name:
            raise ValueError(
                f"No configuration found for experiment "
                f"'{experiment_name}'."
            )

        config = configs_by_name[experiment_name]
        ...
        model_name = result_row.get("model_name", "N/A")
        exp_name = result_row.get("experiment", "N/A")

        lines.append(f"##### {model_name}")
        lines.append("")
        lines.append(experiment_result_to_markdown(result_row))
        lines.append("")

        lines.append("<details>")
        lines.append("<summary>Full configuration</summary>")
        lines.append("")
        lines.append("```python")
        lines.append(experiment_config_to_markdown(config))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("#### Conclusion")
    lines.append("")
    lines.append(conclusion if conclusion else "_Conclusion pending._")
    lines.append("")

    lines.append("</details>")

    return "\n".join(lines)

def experiment_group_summary_report(
    results_df,
    reference_group,
    compare_group,
    description="",
    conclusion="",
    metrics=None,
):
    if metrics is None:
        metrics = ["test_accuracy_mean", "test_f1_mean"]

    comparison = compare_experiment_groups(
        results_df=results_df,
        reference_group=reference_group,
        compare_groups=[compare_group],
        metrics=metrics,
    )

    summary = summarize_group_comparison(
        comparison_df=comparison,
        metrics=metrics,
    )

    lines = [
        f"#### {compare_group}",
        "",
        description or "_Description pending._",
        "",
        "<details>",
        "<summary>Comparison details</summary>",
        "",
        "##### Comparison vs baseline",
        "",
        comparison.to_markdown(index=False),
        "",
        "##### Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "</details>",
        "",
        "##### Conclusion",
        "",
        conclusion or "_Conclusion pending._",
    ]

    return "\n".join(lines)

def workflow_report(workflow, conclusion="", description="", top_n=10):
    results_df = workflow["all_results"]
    reference_group = workflow["reference_group"]
    compare_group = workflow["group"]
    comparison = workflow["comparison"]
    summary = workflow["summary"]

    # For when refenrence and compare group are the same, so no comparison or summary is generated
    if comparison is None or summary is None:
        report = [
            f"#### {compare_group}",
            "",
            description or "_Description pending._",
            "",
            "##### Conclusion",
            "",
            conclusion or "_Conclusion pending._",
            "",
            "<details>",
            "<summary>Details</summary>",
            "",
            "##### Result",
            "",
            baseline_summary_to_markdown(workflow["group_results"]),
            "",
            "</details>",

        ]
    else:
        report = [
            f"#### {compare_group}",
            "",
            description or "_Description pending._",
            "",
            "<details>",
            "<summary>Conclusion</summary>",
            "",
            *feature_effect_interpretation(workflow['feature_effect']),
            "",     
            "##### Conclusion",
            "",
            conclusion or "_Conclusion pending._",
            "",
            "</details>",
            "",
            "<details>",
            "<summary>Experiment details</summary>",
            "",
            f"##### Comparison vs {reference_group}",
            "",
            comparison.to_markdown(index=False),
            "",
            "##### Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "</details>",
        ]

    return {
        "report": "\n".join(report),
        "leaderboard": titanic_notes_leaderboard(results_df=results_df, top_n=top_n),
        "feature_effect": workflow['feature_effect']
    }

# Create a report class to centralise the reporting logic and make it easier to extend in the future
# class ExperimentReport:
#     def __init__(self, experiment_result, experiment_config):
#         self.experiment_result = experiment_result
#         self.experiment_config = experiment_config

#     def generate_report(self):
#         individual_report = []
#         for experiment,config in zip(self.experiment_result, self.experiment_config):
#             individual_report.append((experiment.get("experiment", "N/A"), experiment_result_to_markdown(experiment), experiment_config_to_markdown(config)))

#         full_report = experiments_summary_to_markdown(self.experiment_result)
#         return individual_report, full_report


def feature_effect_interpretation(feature_effect):
    if not feature_effect:
        return []

    effect = feature_effect[0]

    section = [
        "",
        "##### Interpretation",
        "",
        f"- Verdict: {effect['verdict']}",
    ]

    if effect["recommended_for_all"]:
        section.extend([
            "- Recommended for all models",
            f"- Mean delta: {effect['mean_delta']}",
        ])

    elif effect["discard"]:
        section.extend([
            "- Discard",
            f"- Max delta: {effect['max_delta']}",
        ])

    else:
        section.append("- Recommended for specific models:")

        for model_name, tradeoff in effect["recommended_model_tradeoffs"].items():
            section.append(f"  - {model_name}: {tradeoff['primary_metric']}: {tradeoff['primary_delta']}")

            if tradeoff["secondary_pros"]:
                section.append("    - Secondary gains:")
                for metric, delta in tradeoff["secondary_pros"].items():
                    section.append(f"      - {metric}: {delta}")

            if tradeoff["secondary_cons"]:
                section.append("    - Secondary losses:")
                for metric, delta in tradeoff["secondary_cons"].items():
                    section.append(f"      - {metric}: {delta}")

        if effect["notable_secondary_improvements"]:
            section.append("- Notable secondary improvements in non-recommended models:")

            for model_name, metrics in effect["notable_secondary_improvements"].items():
                metric_text = ", ".join(
                    f"{metric}: {delta}"
                    for metric, delta in metrics.items()
                )
                section.append(f"  - {model_name}: {metric_text}")

    section.append("")
    return section

# -------------- Hiperparameter Optimization (HPO) Reporting Functions --------------

def summarize_hpo_results(results, decimals=5):
    parameter_columns = [
        column
        for column in results.columns
        if column.startswith("param_")
    ]

    metric_columns = [
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_f1",
        "std_test_f1",
        "mean_train_accuracy",
        "rank_test_accuracy",
    ]

    selected_columns = [
        "experiment",
        *parameter_columns,
        *[
            column
            for column in metric_columns
            if column in results.columns
        ],
    ]

    summary = results[selected_columns].copy()

    summary = summary.rename(
        columns={
            "experiment": "Experiment",
            **{
                column: column.removeprefix("param_model__")
                for column in parameter_columns
            },
            "mean_test_accuracy": "CV Accuracy",
            "std_test_accuracy": "Accuracy Std",
            "mean_test_f1": "CV F1",
            "std_test_f1": "F1 Std",
            "mean_train_accuracy": "Train Accuracy",
            "rank_test_accuracy": "Rank",
        }
    )

    return (
        summary
        .sort_values(
            ["CV Accuracy", "CV F1"],
            ascending=False,
        )
        .round(decimals)
        .reset_index(drop=True)
    )

def has_varying_values(series):
    """
    Return True if a Series contains different values.

    Supports scalar hyperparameters and dictionary values.
    Treats missing scalar values as equivalent.
    """
    if series.empty:
        return False

    first_value = series.iloc[0]

    for value in series.iloc[1:]:

        # Handle missing scalar values.
        first_missing = (
            first_value is None
            or first_value is pd.NA
            or (
                isinstance(first_value, float)
                and pd.isna(first_value)
            )
        )

        value_missing = (
            value is None
            or value is pd.NA
            or (
                isinstance(value, float)
                and pd.isna(value)
            )
        )

        if first_missing and value_missing:
            continue

        if first_missing != value_missing:
            return True

        if value != first_value:
            return True

    return False

def hpo_leaderboard(results, top_n=5, decimals=5):
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # Detect searched hyperparameters.
    param_columns = [
        col for col in results.columns
        if col.startswith("param_")
    ]

    # Keep only parameters that vary across candidates.
    varying_params = [
        col for col in param_columns
        if has_varying_values(results[col])
    ]

    columns = [
        "experiment",
        *varying_params,
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_f1",
        "std_test_f1",
        "rank_test_accuracy",
    ]

    leaderboard = results[columns].copy()

    leaderboard = leaderboard.sort_values(
        [
            "mean_test_accuracy",
            "mean_test_f1",
            "std_test_accuracy",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).head(top_n)

    leaderboard = leaderboard.rename(
        columns={
            "experiment": "Experiment",
            **{
                col: col.removeprefix("param_model__")
                for col in varying_params
            },
            "mean_test_accuracy": "CV Accuracy",
            "std_test_accuracy": "Accuracy Std",
            "mean_test_f1": "CV F1",
            "std_test_f1": "F1 Std",
            "rank_test_accuracy": "Rank",
        }
    )

    return leaderboard.round(decimals).reset_index(drop=True)

# ------------- Threshold Optimization Reporting Functions --------------

def summarize_threshold_candidates(candidates, decimals=5):
    summary = (
        candidates
        .groupby("threshold")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            selected_count=("selected", "sum"),
        )
        .reset_index()
    )

    return summary.round(decimals)

def summarize_threshold_performance(results, decimals=5):
    metrics = ["accuracy", "precision", "recall", "f1"]

    summary = (
        results
        .groupby("method")[metrics]
        .agg(["mean", "std"])
    )

    return summary.round(decimals)
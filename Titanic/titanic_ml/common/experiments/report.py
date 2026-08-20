import pandas as pd
import pprint
from pathlib import Path
from titanic_ml.common.experiments.v0_revised_save_load import make_config_json_safe
from titanic_ml.paths import EXPERIMENT_RESULTS_FILE, EXPERIMENT_CONFIGS_FILE
from titanic_ml.common.experiments.compare import compare_experiment_groups, summarize_group_comparison, leaderboard, titanic_notes_leaderboard
from titanic_ml.common.experiments.presentation import feature_effect_interpretation
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
    safe_config = make_config_json_safe(
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

        lines.append(f"#### {model_name}")
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
        f"### {compare_group}",
        "",
        description or "_Description pending._",
        "",
        "<details>",
        "<summary>Comparison details</summary>",
        "",
        "#### Comparison vs baseline",
        "",
        comparison.to_markdown(index=False),
        "",
        "#### Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "</details>",
        "",
        "#### Conclusion",
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
            f"### {compare_group}",
            "",
            description or "_Description pending._",
            "",
            "#### Conclusion",
            "",
            conclusion or "_Conclusion pending._",
            "",
            "<details>",
            "<summary>Details</summary>",
            "",
            "#### Result",
            "",
            baseline_summary_to_markdown(workflow["group_results"]),
            "",
            "</details>",

        ]
    else:
        report = [
            f"### {compare_group}",
            "",
            description or "_Description pending._",
            "",
            "<details>",
            "<summary>Conclusion</summary>",
            "",
            *feature_effect_interpretation(workflow['feature_effect']),
            "",     
            "#### Conclusion",
            "",
            conclusion or "_Conclusion pending._",
            "",
            "</details>",
            "",
            "<details>",
            "<summary>Experiment details</summary>",
            "",
            f"#### Comparison vs {reference_group}",
            "",
            comparison.to_markdown(index=False),
            "",
            "#### Summary",
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
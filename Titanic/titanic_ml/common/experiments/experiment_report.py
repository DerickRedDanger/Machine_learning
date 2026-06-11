import pandas as pd
import pprint
import pandas as pd
from pathlib import Path
from titanic_ml.paths import EXPERIMENT_RESULTS_FILE, EXPERIMENT_CONFIGS_FILE

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


def experiment_config_to_markdown(config):
    config = config.copy()
    config['feature_engineering'] =[x.__name__ for x in config['feature_engineering']] 
    return pprint.pformat(config, sort_dicts=False)

def experiment_report(results_df, experiment_configs, print_report=False):
    individual_reports = []

    for result_row, config in zip(
        results_df.to_dict(orient="records"),
        experiment_configs
    ):
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

    for result_row, config in zip(
        results_df.to_dict(orient="records"),
        experiment_configs
    ):
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

def save_results(results_df, path=EXPERIMENT_RESULTS_FILE, append=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        old_results = pd.read_csv(path)
        results_df = pd.concat([old_results, results_df], ignore_index=True)

        # Avoid duplicate experiment rows if you rerun the same experiment
        results_df = results_df.drop_duplicates(
            subset=["experiment"],
            keep="last",
        )

    results_df.to_csv(path, index=False)

    return results_df


def load_results(path=EXPERIMENT_RESULTS_FILE):
    return pd.read_csv(path)

import json
from pathlib import Path


def make_config_json_safe(config):
    safe_config = {}

    for key, value in config.items():
        if key == "feature_engineering" and isinstance(value, list):
            safe_config[key] = [
                fn.__name__ if callable(fn) else str(fn)
                for fn in value
            ]
        else:
            safe_config[key] = value

    return safe_config


def save_configs(configs, path=EXPERIMENT_CONFIGS_FILE, append=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_configs = {
        config["name"]: make_config_json_safe(config)
        for config in configs
    }

    if append and path.exists():
        with open(path, "r", encoding="utf-8") as file:
            old_configs = json.load(file)
    else:
        old_configs = {}

    old_configs.update(new_configs)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(old_configs, file, indent=2)

    return old_configs


def load_configs(path=EXPERIMENT_CONFIGS_FILE):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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
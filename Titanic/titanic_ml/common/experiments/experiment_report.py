import pandas as pd
import pprint

def experiment_result_to_markdown(result_row):
    if hasattr(result_row, "to_dict"):
        result_row = result_row.to_dict()

    lines = [
        f"Experiment {result_row.get('experiment', 'N/A')}:",
        "| Field | Value |",
        "|---|---|",
        f"|Train accuracy| {result_row.get('train_accuracy_mean', 'N/A')} ± {result_row.get('train_accuracy_std', 'N/A')} |",
        f"|Train precision| {result_row.get('train_precision_mean', 'N/A')} ± {result_row.get('train_precision_std', 'N/A')} |",
        f"|Train recall| {result_row.get('train_recall_mean', 'N/A')} ± {result_row.get('train_recall_std', 'N/A')} |",
        f"|Train f1| {result_row.get('train_f1_mean', 'N/A')} ± {result_row.get('train_f1_std', 'N/A')} |",
        f"|Test accuracy| {result_row.get('test_accuracy_mean', 'N/A')} ± {result_row.get('test_accuracy_std', 'N/A')} |",
        f"|Test precision| {result_row.get('test_precision_mean', 'N/A')} ± {result_row.get('test_precision_std', 'N/A')} |",
        f"|Test recall| {result_row.get('test_recall_mean', 'N/A')} ± {result_row.get('test_recall_std', 'N/A')} |",
        f"|Test f1| {result_row.get('test_f1_mean', 'N/A')} ± {result_row.get('test_f1_std', 'N/A')} |",
    ]
    return "\n".join(lines)

def format_metric(result_row, metric_name):
    mean_key = f"test_{metric_name}_mean"
    std_key = f"test_{metric_name}_std"

    mean = result_row.get(mean_key)
    std = result_row.get(std_key)

    if mean is None or std is None:
        return "N/A"

    return f"{mean} ± {std}"

def experiments_summary_to_markdown(results_df):
    summary_rows = []

    for _, row in results_df.iterrows():
        summary_rows.append({
            "experiment": row.get("experiment", "N/A"),
            "model_name": row.get("model_name", "N/A"),
            "status": row.get("status", "N/A"),
            "test_accuracy": format_metric(row, "accuracy"),
            "test_precision": format_metric(row, "precision"),
            "test_recall": format_metric(row, "recall"),
            "test_f1": format_metric(row, "f1"),
            "notes": row.get("notes", ""),
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df.to_markdown(index=False)


def experiment_config_to_markdown(config):
    config_text = pprint.pformat(config, sort_dicts=False)
    return config_text
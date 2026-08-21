import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from titanic_ml.common.models.registry import MODEL_REGISTRY
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from titanic_ml.common.experiments.save_load import save_results, load_results, save_configs, load_configs, save_feature_effects, load_feature_effects
from titanic_ml.common.experiments.v0_revised_save_load import save_results, load_results, save_configs, load_configs, save_feature_effects, load_feature_effects
from titanic_ml.common.experiments.compare import compare_experiment_groups, summarize_group_comparison, leaderboard, analyze_feature_effect
from titanic_ml.common.experiments.runner import build_preprocessor, evaluate_model
from titanic_ml.common.experiments.utils import get_experiment_configs, get_config_group

def run_experiments(
    df,
    experiments,
    target,
    verbose=False,
    round_decimals=3,
    debug=False,
):
    results = []

    # New experiment groups are dictionaries.
    # Temporary compatibility while older configs still exist.
    experiment_iter = get_experiment_configs(experiments)
    for exp in experiment_iter:

        if verbose:
            print(f"Running experiment: {exp['name']}")

        if debug:
            print(f"Running config: {exp}")

        try:
            # -------------------------------------------------
            # 1. Validate target
            # -------------------------------------------------

            if target not in df.columns:
                raise ValueError(
                    f"Target column '{target}' "
                    "not found in dataframe."
                )

            # -------------------------------------------------
            # 2. Reject legacy feature engineering
            # -------------------------------------------------

            legacy_fe = exp.get(
                "feature_engineering",
                []
            )

            if legacy_fe:
                raise ValueError(
                    "Legacy 'feature_engineering' "
                    "functions are not supported by "
                    "the CV-safe runner. Convert this "
                    "experiment to 'feature_pipeline'."
                )

            # -------------------------------------------------
            # 3. Split raw predictors / target
            # -------------------------------------------------

            X = df.drop(columns=[target])
            y = df[target]

            # -------------------------------------------------
            # 4. Build preprocessor
            # -------------------------------------------------

            preprocessor = build_preprocessor(
                exp["preprocessing"]
            )

            # -------------------------------------------------
            # 5. Build model
            # -------------------------------------------------

            model_class = MODEL_REGISTRY.get(
                exp["model_name"]
            )

            if model_class is None:
                raise ValueError(
                    f"Model '{exp['model_name']}' "
                    "not found in registry."
                )

            model = model_class(
                **exp.get("model_params", {})
            )

            # -------------------------------------------------
            # 6. Build complete CV-safe pipeline
            # -------------------------------------------------

            feature_pipeline = exp.get(
                "feature_pipeline",
                []
            )

            pipeline_steps = [
                *feature_pipeline,
                ("preprocessor", preprocessor),
                ("model", model),
            ]

            model_pipeline = Pipeline(
                pipeline_steps
            )

            if debug:
                print("Pipeline steps:")

                for step_name, step in pipeline_steps:
                    print(
                        f"  {step_name}: "
                        f"{type(step).__name__}"
                    )

            # -------------------------------------------------
            # 7. Evaluate
            # -------------------------------------------------

            model_result = evaluate_model(
                model_pipeline=model_pipeline,
                X=X,
                y=y,
                evaluation_config=exp["evaluation"],
                round_decimals=round_decimals,
            )

            result = {
                "experiment": exp["name"],
                "stage": exp.get("stage", ""),
                "feature_group": exp.get(
                    "feature_group",
                    "",
                ),
                "model_name": exp["model_name"],
                "group": exp.get("group", ""),
                "status": "success",
                "error_type": "",
                "error_message": "",
                **model_result,
                "domain": exp.get("domain", ""),
                "notes": exp.get("notes", ""),
            }

        except Exception as e:

            result = {
                "experiment": exp.get(
                    "name",
                    "N/A",
                ),
                "stage": exp.get(
                    "stage",
                    "",
                ),
                "feature_group": exp.get(
                    "feature_group",
                    "",
                ),
                "model_name": exp.get(
                    "model_name",
                    "N/A",
                ),
                "group": exp.get(
                    "group",
                    "",
                ),
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "domain": exp.get(
                    "domain",
                    "",
                ),
                "notes": exp.get(
                    "notes",
                    "",
                ),
            }

        if verbose:
            print(
                f"\nExperiment "
                f"'{exp['name']}' results:"
            )

            for metric, value in result.items():
                if metric != "experiment":
                    print(
                        f"  {metric}: {value}"
                    )

            print(
                f"\nExperiment "
                f"'{exp['name']}' completed."
            )
            print("-" * 40)

        results.append(result)

    return pd.DataFrame(results)

def run_experiment_group_workflow(
    df,
    experiment_configs,
    target,
    reference_group="baseline__raw",
    metrics=None,
    save=True,
    verbose=False,
    debug=False,
):
    if metrics is None:
        metrics = [
            "test_accuracy_mean",
            "test_f1_mean",
        ]

    compare_group = get_config_group(
        experiment_configs
    )

    # -------------------------------------------------
    # 1. Run current experiment group
    # -------------------------------------------------

    group_results = run_experiments(
        df=df,
        experiments=experiment_configs,
        target=target,
        verbose=verbose,
        debug=debug,
    )

    # -------------------------------------------------
    # 2. Save/load combined results
    # -------------------------------------------------

    if save:
        all_results = save_results(
            group_results
        )

        save_configs(
            experiment_configs
        )

    else:
        all_results = group_results

    # -------------------------------------------------
    # 3. Compare with reference
    # -------------------------------------------------

    comparison = None
    summary = None

    if compare_group != reference_group:
        comparison = (
            compare_experiment_groups(
                results_df=all_results,
                reference_group=reference_group,
                compare_groups=[
                    compare_group
                ],
                metrics=metrics,
            )
        )

        summary = (
            summarize_group_comparison(
                comparison_df=comparison,
                metrics=metrics,
            )
        )

    # -------------------------------------------------
    # 4. Leaderboard
    # -------------------------------------------------

    top_models = leaderboard(
        results_df=all_results,
        metrics=metrics,
        top_n=30,
    )

    # -------------------------------------------------
    # 5. Feature effect
    # -------------------------------------------------

    feature_effect = None

    if comparison is not None:
        feature_effect = (
            analyze_feature_effect(
                comparison,
                metric=metrics[0],
            )
        )

        if save:
            save_feature_effects(
                feature_effect
            )

    return {
        "group": compare_group,
        "reference_group": reference_group,
        "group_results": group_results,
        "all_results": all_results,
        "comparison": comparison,
        "summary": summary,
        "leaderboard": top_models,
        "feature_effect": feature_effect,
    }
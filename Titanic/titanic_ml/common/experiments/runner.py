import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from titanic_ml.common.models.registry import MODEL_REGISTRY
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from titanic_ml.common.experiments.save_load import save_results, load_results, save_configs, load_configs, save_feature_effects, load_feature_effects
from titanic_ml.common.experiments.save_load import save_results, load_results, save_configs, load_configs, save_feature_effects, load_feature_effects
from titanic_ml.common.experiments.compare import compare_experiment_groups, summarize_group_comparison, leaderboard, analyze_feature_effect
# from titanic_ml.common.experiments.runner import build_preprocessor, evaluate_model
from titanic_ml.common.experiments.utils import get_experiment_configs, get_config_group

def build_preprocessor(preprocessing_config):
    numeric_features = preprocessing_config.get("numeric_features", [])
    onehot_features = preprocessing_config.get("onehot_features", [])
    ordinal_features = preprocessing_config.get("ordinal_features", [])

    numeric_imputer = preprocessing_config.get("numeric_imputer", "median")
    categorical_imputer = preprocessing_config.get("categorical_imputer", "most_frequent")
    scaler = preprocessing_config.get("scaler", None)

    transformers = []

    if numeric_features:
        numeric_steps = []

        if numeric_imputer:
            numeric_steps.append(
                ("imputer", SimpleImputer(strategy=numeric_imputer))
            )

        if scaler == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            numeric_steps.append(("scaler", RobustScaler()))
        elif scaler is not None:
            raise ValueError(f"Unsupported scaler: {scaler}")

        numeric_pipeline = Pipeline(numeric_steps)

        transformers.append(
            ("num", numeric_pipeline, numeric_features)
        )

    if onehot_features:
        onehot_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=categorical_imputer)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers.append(
            ("onehot", onehot_pipeline, onehot_features)
        )

    if ordinal_features:
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=categorical_imputer)),
            ("encoder", OrdinalEncoder()),
        ])

        transformers.append(
            ("ordinal", ordinal_pipeline, ordinal_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

def evaluate_model(model_pipeline, X, y, evaluation_config, round_decimals=None):
    method = evaluation_config.get("method", "cross_validation")

    if method != "cross_validation":
        raise ValueError(f"Unsupported evaluation method: {method}")

    cv = evaluation_config.get("cv", 5)
    scoring = evaluation_config.get("scoring", ["accuracy"])

    if isinstance(scoring, str):
        scoring = [scoring]

    scores = cross_validate(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=evaluation_config.get("return_train_score", False),
        n_jobs=evaluation_config.get("n_jobs", -1),
    )

    result = {}

    for metric in scoring:
        test_scores = scores[f"test_{metric}"]

        result[f"test_{metric}_mean"] = float(np.mean(test_scores))
        result[f"test_{metric}_std"] = float(np.std(test_scores))

        if evaluation_config.get("return_train_score", False):
            train_scores = scores[f"train_{metric}"]
            result[f"train_{metric}_mean"] = float(np.mean(train_scores))
            result[f"train_{metric}_std"] = float(np.std(train_scores))

    result["fit_time_mean"] = float(np.mean(scores["fit_time"]))
    result["score_time_mean"] = float(np.mean(scores["score_time"]))

    # Round results if round_decimals is specified
    if round_decimals is not None:
        result = {
            key: round(value, round_decimals) if isinstance(value, (int, float)) else value
            for key, value in result.items()
        }

    return result

def apply_pre_cv_feature_pipeline(
    X,
    exp,
    context_df=None,
):
    pipeline = exp.get(
        "pre_cv_feature_pipeline",
        [],
    )

    if not pipeline:
        return X

    scope = exp["pre_cv_scope"]

    if scope == "full_prediction_context":
        if context_df is None:
            raise ValueError(
                "Experiment uses pre-CV scope "
                "'full_prediction_context', but no "
                "context_df was provided."
            )

        combined = pd.concat(
            [X, context_df],
            axis=0,
        )

        pre_cv_pipeline = Pipeline(
            pipeline
        )

        transformed = (
            pre_cv_pipeline.fit_transform(
                combined
            )
        )

        return transformed.iloc[:len(X)]

    raise ValueError(
        f"Unsupported pre-CV scope: '{scope}'."
    )

def run_experiments(
    df,
    experiments,
    target,
    verbose=False,
    round_decimals=3,
    debug=False,
    context_df=None,
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
            # 3.1 Apply pre-CV feature pipeline if specified
            # -------------------------------------------------

            pre_cv_feature_pipeline = exp.get(
                "pre_cv_feature_pipeline",
                [],
            )

            if pre_cv_feature_pipeline:
                X = apply_pre_cv_feature_pipeline(
                    X=X,
                    exp=exp,
                    context_df=context_df,
                )
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
                "uses_pre_cv_fe":bool(
                                exp.get(
                                    "pre_cv_feature_pipeline",
                                    [],
                                )
                            ),
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
                "pre_cv_scope": exp.get("pre_cv_scope", None),
                "domain": exp.get("domain", ""),
                "notes": exp.get("notes", ""),
            }

        except Exception as e:

            result = {
                "experiment": exp.get(
                    "name",
                    "N/A",
                ),
                "uses_pre_cv_fe":bool(
                    exp.get(
                        "pre_cv_feature_pipeline",
                        [],
                    )
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
                "pre_cv_scope": exp.get("pre_cv_scope", None),
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
    context_df=None,
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
        context_df=context_df,
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
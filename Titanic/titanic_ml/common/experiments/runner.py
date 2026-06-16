import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from titanic_ml.common.models.registry import MODEL_REGISTRY
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from titanic_ml.common.experiments.save_load import save_results, load_results, save_configs, load_configs
from titanic_ml.common.experiments.compare import compare_experiment_groups, summarize_group_comparison, leaderboard

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

def run_experiments(df, experiments, target, verbose=False, round_decimals=3, debug=False):
    results = []

    for exp in experiments:
        if verbose:
            print(f"Running experiment: {exp['name']}")

        try:
            working_df = df.copy()
            for fn in exp.get("feature_engineering",[]):
                if debug:
                    print(f'Applying feature: {fn.__name__}')
                working_df = fn(working_df)
                if debug:
                    print(f'Features after {fn.__name__}: {working_df.columns.tolist()}')

            features = exp["features"]
            X = working_df[features]
            y = working_df[target]

            preprocessor = build_preprocessor(exp["preprocessing"])

            model_class = MODEL_REGISTRY.get(exp["model_name"])
            if model_class is None:
                raise ValueError(f"Model '{exp['model_name']}' not found in registry.")

            model = model_class(**exp.get("model_params", {}))

            model_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

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
                "feature_group": exp.get("feature_group", ""),
                "model_name": exp["model_name"],
                "group": exp.get("group", ""),
                "status": "success",
                "error_type": "",
                "error_message": "",
                **model_result,
                "notes": exp.get("notes", ""),
            }

        except Exception as e:
            result={
                "experiment": exp["name"],
                "model_name": exp.get("model_name", "N/A"),
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "notes": exp.get("notes", ""),
            }

        if verbose:
            print(f"\nExperiment '{exp['name']}' results:")
            for metric, value in result.items():
                if metric != "experiment":
                    print(f"  {metric}: {value}")

            print(f"\nExperiment '{exp['name']}' completed.")
            print("-" * 40)

        results.append(result)

    return pd.DataFrame(results)

def get_config_group(experiment_configs):
    if not experiment_configs:
        raise ValueError("No experiment configs provided.")

    groups = {exp.get("group") for exp in experiment_configs}

    if len(groups) != 1:
        raise ValueError(f"Expected one group, found: {groups}")

    return groups.pop()

def run_experiment_group_workflow(
    df,
    experiment_configs,
    target,
    reference_group="baseline__raw",
    metrics=None,
    save=True,
    verbose=False,
):
    if metrics is None:
        metrics = ["test_accuracy_mean", "test_f1_mean"]

    compare_group = get_config_group(experiment_configs)

    # 1. Run current experiment group
    group_results = run_experiments(
        df=df,
        experiments=experiment_configs,
        target=target,
        verbose=verbose,
    )

    # 2. Save/load combined results
    if save:
        all_results = save_results(group_results)
        save_configs(experiment_configs)
    else:
        all_results = group_results

    # 3. Compare with reference group
    comparison = None
    summary = None

    if compare_group != reference_group:
        comparison = compare_experiment_groups(
            results_df=all_results,
            reference_group=reference_group,
            compare_groups=[compare_group],
            metrics=metrics,
        )

        summary = summarize_group_comparison(
            comparison_df=comparison,
            metrics=metrics,
        )

    # 4. Leaderboard
    top_models = leaderboard(
        results_df=all_results,
        metric=metrics[0],
        top_n=10,
    )

    return {
        "group": compare_group,
        "reference_group": reference_group,
        "group_results": group_results,
        "all_results": all_results,
        "comparison": comparison,
        "summary": summary,
        "leaderboard": top_models,
    }

# def run_experiment(df, target_col, experiment_config, model_registry, logger=None):
#     """
#     Run one ML experiment and return a structured result dict.
#     """
    ...

    '''
    Example of result dict:
    
    {
    "experiment_name": "rf_title_family",
    "model_name": "rf",
    "model_params": {
        "n_estimators": 200,
        "max_depth": 5,
        "random_state": 42
    },
    "features": [
        "Pclass", "Sex", "Age", "Fare", "Embarked",
        "Title", "Family_size", "Alone"
    ],
    "n_features": 8,
    "cv_accuracy_mean": 0.831,
    "cv_accuracy_std": 0.018,
    "cv_precision_mean": 0.812,
    "cv_recall_mean": 0.770,
    "cv_f1_mean": 0.789,
    "status": "success",
    "error_message": None,
    "notes": "Random forest with selected engineered features"
}

Example of failed result dict:
{
    "experiment_name": "svc_bad_test",
    "model_name": "svc",
    "model_params": {...},
    "features": [...],
    "n_features": 7,
    "cv_accuracy_mean": None,
    "cv_accuracy_std": None,
    "cv_precision_mean": None,
    "cv_recall_mean": None,
    "cv_f1_mean": None,
    "status": "failed",
    "error_message": "ValueError: ...",
    "notes": "Testing unstable configuration"
}'''
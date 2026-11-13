from titanic_ml.common.experiments.utils import get_experiment_configs
from titanic_ml.common.experiments.runner import apply_pre_cv_feature_pipeline, build_model_pipeline

def run_hyperparameter_search(
    df,
    experiments,
    target,
    context_df=None,
):
    """
    Run hyperparameter searches for a group of experiments.

    Returns
    -------
    searches : dict
        Fitted search objects keyed by experiment name.

    results : pd.DataFrame
        Combined candidate-level CV results.
    """

    if not isinstance(experiments, dict):
        raise TypeError(
            "experiments must be a dictionary of experiment configurations."
        )

    if "model_name" in experiments:
        raise ValueError(
            "Received a single experiment configuration. "
            "Wrap it with config_to_group() before running HPO."
        )

    experiment_iter = get_experiment_configs(experiments)

    searches = {}
    result_frames = []

    for exp in experiment_iter:

        experiment_name = exp["name"]

        if experiment_name in searches:
            raise ValueError(
                f"Duplicate experiment name: '{experiment_name}'."
            )
    
        search, results = _run_single_hpo(
            df=df,
            exp=exp,
            target=target,
            context_df=context_df,
        )

        searches[experiment_name] = search
        result_frames.append(results)

    combined_results = pd.concat(
        result_frames,
        ignore_index=True,
    ) if result_frames else pd.DataFrame()

    return searches, combined_results

from sklearn.model_selection import GridSearchCV
import pandas as pd


def _run_single_hpo(
    df,
    exp,
    target,
    context_df=None,
):
    """
    Run a grid search using an existing experiment configuration.

    Returns
    -------
    search : GridSearchCV
        Fitted search object containing the best estimator,
        best parameters, and complete CV results.

    results : pd.DataFrame
        One row per evaluated hyperparameter combination.
    """

    # -------------------------------------------------
    # 1. Validate configuration
    # -------------------------------------------------

    search_config = exp.get("hyperparameter_search")

    if not isinstance(search_config, dict):
        raise ValueError(
            "Experiment must contain a "
            "'hyperparameter_search' dictionary."
        )

    if search_config.get("method") != "grid":
        raise ValueError(
            "Only 'grid' hyperparameter search "
            "is currently supported."
        )

    param_grid = search_config.get("param_grid")

    if not isinstance(param_grid, (dict, list)) or not param_grid:
        raise ValueError(
            "A non-empty 'param_grid' is required."
        )

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    # -------------------------------------------------
    # 2. Prepare predictors and target
    # -------------------------------------------------

    X = df.drop(columns=[target])
    y = df[target]

    if exp.get("pre_cv_feature_pipeline"):
        X = apply_pre_cv_feature_pipeline(
            X=X,
            exp=exp,
            context_df=context_df,
        )

    # -------------------------------------------------
    # 3. Build model pipeline
    # -------------------------------------------------

    model_pipeline = build_model_pipeline(exp)

    # Validate searched parameter names
    valid_params = model_pipeline.get_params(deep=True)

    grids = param_grid if isinstance(param_grid, list) else [param_grid]

    for grid in grids:
        if not isinstance(grid, dict):
            raise TypeError(
                "Each parameter grid must be a dictionary."
            )

        for parameter in grid:
            if parameter not in valid_params:
                raise ValueError(
                    f"Unknown pipeline parameter: '{parameter}'."
                )

    # -------------------------------------------------
    # 4. Read evaluation configuration
    # -------------------------------------------------

    evaluation = exp["evaluation"]

    if evaluation.get("method", "cross_validation") != "cross_validation":
        raise ValueError(
            "Hyperparameter search requires cross-validation."
        )

    scoring = evaluation.get("scoring", ["accuracy"])

    if isinstance(scoring, str):
        scoring = [scoring]

    if not scoring:
        raise ValueError("At least one scoring metric is required.")

    refit = search_config.get("refit", "accuracy")

    if refit not in scoring:
        raise ValueError(
            f"Refit metric '{refit}' is not included in scoring."
        )

    # -------------------------------------------------
    # 5. Execute search
    # -------------------------------------------------

    search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=evaluation.get("cv", 5),
        n_jobs=evaluation.get("n_jobs", -1),
        return_train_score=evaluation.get(
            "return_train_score", True
        ),
        error_score="raise",
    )

    search.fit(X, y)

    
    # -------------------------------------------------
    # 6. Collect results and candidate metadata
    # -------------------------------------------------

    results = pd.DataFrame(search.cv_results_)

    stage = exp["stage"]
    feature_group = exp["feature_group"]
    model_name = exp["model_name"]

    # Keep the original search identity.
    source_search = exp["name"]

    # Generate candidate IDs in search order, not rank order.
    candidate_ids = range(1, len(results) + 1)

    # Generate a unique experiment name for each candidate.
    candidate_names = [
        f"{stage}__{feature_group}{candidate_id:02d}__{model_name}"
        for candidate_id in candidate_ids
    ]

    # Preserve experiment metadata alongside each candidate.
    metadata = {
        "experiment": candidate_names,
        "source_search": source_search,
        "candidate_id": list(candidate_ids),
        "stage": stage,
        "feature_group": feature_group,
        "group": exp["group"],
        "model_name": model_name,
        "domain": exp.get("domain", ""),
    }

    results = pd.concat(
        [
            pd.DataFrame(metadata, index=results.index),
            results,
        ],
        axis=1,
    )

    return search, results

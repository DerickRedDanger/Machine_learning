from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import check_cv

from titanic_ml.common.experiments.runner import apply_pre_cv_feature_pipeline, build_model_pipeline
from titanic_ml.common.experiments.utils import get_experiment_configs

def get_cv_fold(X, y, cv=5, fold=0):
    cv_splitter = check_cv(
        cv=cv,
        y=y,
        classifier=True,
    )

    for fold_idx, (train_idx, validation_idx) in enumerate(
        cv_splitter.split(X, y)
    ):
        if fold_idx == fold:
            return (
                X.iloc[train_idx].copy(),
                X.iloc[validation_idx].copy(),
                y.iloc[train_idx].copy(),
                y.iloc[validation_idx].copy(),
            )

    raise ValueError(
        f"Fold {fold} does not exist for cv={cv}."
    )

def get_feature_pipeline(model_pipeline):
    feature_steps = []

    for step_name, transformer in model_pipeline.steps:
        if step_name == "preprocessor":
            break

        feature_steps.append(
            (step_name, clone(transformer))
        )

    if not feature_steps:
        return None

    return Pipeline(feature_steps)

def inspect_experiment_fold(
    df,
    experiments,
    target,
    context_df=None,
    cv=5,
    fold=0,
):
    # -------------------------------------------------
    # 1. Validate target
    # -------------------------------------------------

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' "
            "not found in dataframe."
        )

    # -------------------------------------------------
    # 2. Split predictors / target
    # -------------------------------------------------

    X = df.drop(columns=[target])
    y = df[target]

    # -------------------------------------------------
    # 3. Apply pre-CV feature engineering
    # -------------------------------------------------
    experiment_iter = get_experiment_configs(experiments)
    exp=None
    for experiment in experiment_iter:
        exp = experiment
        break
        
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
    # 4. Get requested CV fold
    # -------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = get_cv_fold(
        X=X,
        y=y,
        cv=cv,
        fold=fold,
    )

    # -------------------------------------------------
    # 5. Build canonical pipeline
    # -------------------------------------------------

    model_pipeline = build_model_pipeline(exp)

    feature_pipeline = get_feature_pipeline(
        model_pipeline
    )

    # -------------------------------------------------
    # 6. Apply CV feature engineering
    # -------------------------------------------------

    if feature_pipeline is None:
        X_train_engineered = X_train.copy()
        X_validation_engineered = (
            X_validation.copy()
        )

    else:
        X_train_engineered = (
            feature_pipeline.fit_transform(
                X_train,
                y_train,
            )
        )

        X_validation_engineered = (
            feature_pipeline.transform(
                X_validation
            )
        )

    # -------------------------------------------------
    # 7. Return inspection data
    # -------------------------------------------------

    return {
        "fold": fold,
        "X_train_cv_input":X_train,
        "X_validation_cv_input":X_validation,
        "X_train_engineered": (
            X_train_engineered
        ),
        "X_validation_engineered": (
            X_validation_engineered
        ),
        "y_train": y_train,
        "y_validation": y_validation,
    }
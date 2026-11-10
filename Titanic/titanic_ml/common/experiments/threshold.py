import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from titanic_ml.common.experiments.runner import apply_pre_cv_feature_pipeline, build_model_pipeline

def select_best_threshold(
    y_true,
    scores,
    thresholds,
    default_threshold=0.5,
):
    results = []

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
        })

    results = pd.DataFrame(results)

    results["distance_from_default"] = abs(
        results["threshold"] - default_threshold
    )

    ranked = results.sort_values(
        ["accuracy", "f1", "distance_from_default"],
        ascending=[False, False, True],
        kind="stable",
    )

    return float(ranked.iloc[0]["threshold"]), results


def evaluate_threshold_cv(
    X,
    y,
    thresholds,
    pipeline=None,
    exp=None,
    context_df=None,
    outer_cv=None,
    inner_cv=None,
    score_method="predict_proba",
    default_threshold=None,
):
    if score_method not in ("predict_proba", "decision_function"):
        raise ValueError(
            "score_method must be 'predict_proba' or 'decision_function'."
        )

    # -------------------------------------------------
    # 1. Resolve model pipeline and feature preparation
    # -------------------------------------------------

    # Checks both cases where both or none are provided
    if (exp is None) == (pipeline is None):
        raise ValueError(
            "Provide exactly one of 'exp' or 'pipeline'."
        )

    if exp is not None:
        if exp.get("pre_cv_feature_pipeline"):
            X = apply_pre_cv_feature_pipeline(
                X=X,
                exp=exp,
                context_df=context_df,
            )

        pipeline = build_model_pipeline(exp)

    elif context_df is not None:
        raise ValueError(
            "context_df requires an experiment configuration."
        )

    if outer_cv is None:
        outer_cv = StratifiedKFold(n_splits=5)

    if inner_cv is None:
        inner_cv = StratifiedKFold(n_splits=5)

    if default_threshold is None:
        default_threshold = (
            0.5 if score_method == "predict_proba" else 0.0
        )

    rows = []
    candidate_results = []


    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Inner CV: generate out-of-fold scores for threshold selection.
        inner_scores = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=inner_cv,
            method=score_method,
            n_jobs=-1,
        )

        if score_method == "predict_proba":
            inner_scores = inner_scores[:, 1]

        best_threshold, fold_candidates = select_best_threshold(
            y_train,
            inner_scores,
            thresholds,
            default_threshold=default_threshold,
        )

        fold_candidates.insert(0, "outer_fold", fold)
        fold_candidates["selected"] = (
            fold_candidates["threshold"] == best_threshold
        )

        candidate_results.append(fold_candidates)

        # Fit using all outer training observations.
        model = clone(pipeline)
        model.fit(X_train, y_train)

        # Generate scores for the untouched outer validation fold.
        if score_method == "predict_proba":
            outer_scores = model.predict_proba(X_test)[:, 1]
        else:
            outer_scores = model.decision_function(X_test)

        for name, threshold in [
            ("default", default_threshold),
            ("tuned", best_threshold),
        ]:
            predictions = (outer_scores >= threshold).astype(int)

            rows.append({
                "fold": fold,
                "method": name,
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
            })

    return (
        pd.DataFrame(rows),
        pd.concat(candidate_results, ignore_index=True),
    )

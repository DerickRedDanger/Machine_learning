def run_experiment(df, target_col, experiment_config, model_registry, logger=None):
    """
    Run one ML experiment and return a structured result dict.
    """
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
# This file defines the configuration for machine learning experiments on the Titanic dataset.
# Each experiment is represented as a dictionary with specific fields that describe the model, features, preprocessing steps, evaluation method, and other relevant information.

# Current experiment model - WIP

import copy
baseline_config_model = {
        "name": "",
        "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],

        "feature_engineering": None,

        "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
            "onehot_features": ["Sex", "Embarked"],
            "ordinal_features": ["Pclass"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },

        "model_name": "",
        "model_params": {
        },

        "evaluation": {
            "method": "cross_validation",
            "cv": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"],
            "return_train_score": True,
            "n_jobs": -1,
        },
        "notes": "Base logistic regression. Baseline for comparison.",
    }

baseline_models =[
    {   
    "name": "baseline_logreg",
    "model_name": "logreg",
    "model_params": {
        "max_iter": 1000,
        "random_state": 42,},
    },

    {  
    "name": "baseline_knn",
    "model_name":'knn',
    "model_params":{
        "n_neighbors": 5,
        "random_state": 42,
    },
    },

    {   
    "name": "baseline_svc",
    "model_name":'svc',
    "model_params":{
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": False,
        "random_state": 42,
    },
    },

    {   
    "name": "baseline_decision_tree",
    "model_name":'DecisionTreeClassifier',
    "model_params":{
        "max_depth": 4,
        "min_samples_leaf": 5,
        "random_state": 42,
    },
    },

    {
    "name": "baseline_random_forest",
    "model_name":'RandomForestClassifier',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
    },
    },

    {
    "name": "baseline_extra_trees",
    "model_name":'ExtraTreesClassifier',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
    },
    },

    {   
    "name": "baseline_xgb",
    "model_name":'XGBClassifier',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    },
    },
]

baseline_config = []

for model in baseline_models:
    config = copy.deepcopy(baseline_config_model)
    config.update(model)
    baseline_config.append(config)

baseline = [
    {
        "name": "baseline_logreg",
        "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],

        "feature_engineering": None,

        "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
            "onehot_features": ["Sex", "Embarked"],
            "ordinal_features": ["Pclass"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },

        "model_name": "logreg",
        "model_params": {
            "max_iter": 1000,
            "random_state": 42,
        },

        "evaluation": {
            "method": "cross_validation",
            "cv": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"],
            "return_train_score": True,
            "n_jobs": -1,
        },
        "notes": "Base logistic regression. Baseline for comparison.",
    },
]

dummy = [
    {
        "name": "baseline_logreg",
        "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],

        "feature_engineering": None,

        "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
            "onehot_features": ["Sex", "Embarked"],
            "ordinal_features": ["Pclass"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },

        "model_name": "logreg",
        "model_params": {
            "max_iter": 1000,
            "random_state": 42,
        },

        "evaluation": {
            "method": "cross_validation",
            "cv": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"],
            "return_train_score": True,
            "n_jobs": -1,
        },
        "notes": "Base logistic regression. Baseline for comparison.",
    },

    {
        "name": "baseline_logreg",
        "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],

        "feature_engineering": None,

        "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
            "onehot_features": ["Sex", "Embarked"],
            "ordinal_features": ["Pclass"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },

        "model_name": "logreg",
        "model_params": {
            "max_iter": 1000,
            "random_state": 42,
        },

        "evaluation": {
            "method": "cross_validation",
            "cv": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"],
            "return_train_score": True,
            "n_jobs": -1,
        },
        "notes": "Base logistic regression. Baseline for comparison.",
    },
]

EXPERIMENTS = [
    {
        "name": "baseline_logreg",
        "model_name": "logreg",
        "model_params": {
            "max_iter": 1000
        },
        "features": [
            "Pclass", "Sex", "Age", "Fare", "Embarked"
        ],
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
            "encoder": "onehot",
            "handle_outliers": False
        },
        "evaluation": {
            "method": "cross_validation",
            "cv": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"]
        },
        "use_tuning": False,
        "param_grid": None,
        "notes": "Base logistic regression. Baseline for comparission"
    },
]

"""
Field by field meaning:

- name: A unique identifier for the experiment.
Readable, short, and descriptive.
example: "Baseline Logistic Regression", "Random Forest with Tuning", "XGBoost with Feature Engineering"

- model_name: The key corresponding to the model in the MODEL_REGISTRY.
Must match one of the keys defined in the MODEL_REGISTRY dictionary.
example: "logreg", "rf", "xgb"

- model_params: A dictionary of fixed parameters to initialize the model with.
example: {"n_estimators": 200, "max_depth": 5}

- features: A list of feature names to be used in the experiment.
example: ["Pclass", "Sex", "Age", "Fare", "Embarked"],

- preprocessing: A dictionary defining the preprocessing steps to be applied to the experiment before training the model.
example: {"numeric_imputer": "median", "categorical_imputer": "most_frequent", "scaler": "standard", "encoder": "onehot", "handle_outliers": False}

- evaluation: A dictionary defining the evaluation method and metrics to be used for assessing the model's performance.
example: {"method": "cross_validation", "cv": 5, "scoring": ["accuracy", "precision", "recall", "f1"]}

- use_tuning: A boolean indicating whether to perform hyperparameter tuning for this experiment.
example: True, False

- param_grid: A dictionary defining the hyperparameters and their respective values to be used for tuning.
Only relevant if use_tuning is True.
example: {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]}

- notes: A brief description of the experiment and the reasoning behind it, or any additional information about the experiment.
example: "This experiment tests the performance of a Random Forest model with default parameters. Meant to be used as baseline for comparison with tuned models."
"""
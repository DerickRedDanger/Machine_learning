# This file defines the configuration for machine learning experiments on the Titanic dataset.
# Each experiment is represented as a dictionary with specific fields that describe the model, features, preprocessing steps, evaluation method, and other relevant information.

from titanic_ml.feature_engineering import add_family_features, add_has_cabin, add_full_deck, add_title, add_full_title_feature
import copy

# Baseline configuration for experiment config
baseline__config_model = {
        "name": "",
        "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],

        "feature_engineering": [],

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
        "notes": "",
    }

# Baseline configuration for models
baseline__models =[
    {   
    "name": "baseline__raw__logreg",
    "model_name": "logreg",
    "model_params": {
        "max_iter": 1000,
        "random_state": 42,
        },
        "notes": "Base logistic regression, using raw configuration. Baseline for comparison."
        
    },

    {  
    "name": "baseline__raw__knn",
    "model_name":'knn',
    "model_params":{
        "n_neighbors": 5,
        },
        "notes": "Base kneighbors classifier, using raw configuration. Baseline for comparison."
        
    },

    {   
    "name": "baseline__raw__svc",
    "model_name":'svc',
    "model_params":{
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": False,
        "random_state": 42,
        },
        "notes": "Base SVC, using raw configuration. Baseline for comparison.",
    },

    {   
    "name": "baseline__raw__decision_tree",
    "model_name":'decision_tree',
    "model_params":{
        "max_depth": 4,
        "min_samples_leaf": 5,
        "random_state": 42,
        },
        "notes": "Base decision tree, using raw configuration. Baseline for comparison."
    },

    {
    "name": "baseline__raw__random_forest",
    "model_name":'random_forest',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
        },
        "notes": "Base random forest, using raw configuration. Baseline for comparison."
    },

    {
    "name": "baseline__raw__extra_trees",
    "model_name":'extra_trees',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
        },
        "notes": "Base extra trees, using raw configuration. Baseline for comparison."
    },

    {   
    "name": "baseline__raw__xgb",
    "model_name":'xgb',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        },
        "notes": "Base xgb, using raw configuration. Baseline for comparison."
    },
]

# Baseline experiment configuration
baseline__raw = []

for model in baseline__models:
    config = copy.deepcopy(baseline__config_model)
    config.update(model)
    baseline__raw.append(config)

# New Naming convention:
# <stages>__<feature_group>__<model>
import copy

VALID_EXPERIMENT_KEYS = {
    "name",
    "features",
    "feature_engineering",
    "preprocessing",
    "model_name",
    "model_params",
    "evaluation",
    "notes",
}


def validate_experiment_override(override):
    unknown_keys = set(override) - VALID_EXPERIMENT_KEYS

    if unknown_keys:
        raise ValueError(f"Unknown override keys: {unknown_keys}")

    if "features" in override and not isinstance(override["features"], list):
        raise TypeError("'features' must be a list.")

    if "feature_engineering" in override:
        feature_engineering = override["feature_engineering"]

        if not isinstance(feature_engineering, list):
            raise TypeError("'feature_engineering' must be a list of functions.")

        for fn in feature_engineering:
            if not callable(fn):
                raise TypeError(f"Feature engineering item is not callable: {fn}")

    if "preprocessing" in override and not isinstance(override["preprocessing"], dict):
        raise TypeError("'preprocessing' must be a dictionary.")

    if "model_params" in override and not isinstance(override["model_params"], dict):
        raise TypeError("'model_params' must be a dictionary.")

    if "evaluation" in override and not isinstance(override["evaluation"], dict):
        raise TypeError("'evaluation' must be a dictionary.")


def create_experiment_group(stage, feature_group, base_configs, group_override=None):
    if not stage:
        raise ValueError("Experiment group needs a stage, like 'fe01'.")

    if not feature_group:
        raise ValueError("Experiment group needs a feature group, like 'family'.")

    if not base_configs:
        raise ValueError("No base configs provided.")

    group_override = group_override or {}

    validate_experiment_override(group_override)

    configs = copy.deepcopy(base_configs)

    for exp in configs:
        model_name = exp["model_name"]

        exp.update(group_override)
        exp["name"] = f"{stage}__{feature_group}__{model_name}"

    return configs

fe01__family_override = {
    "feature_engineering": [add_family_features],

    "features": [
        "Pclass", "Sex", "Age", "Fare", "Embarked",
        "FamilySize", "IsAlone",
    ],

    "preprocessing": {
        "numeric_features": ["Age", "Fare", "FamilySize", "IsAlone"],
        "onehot_features": ["Sex", "Embarked"],
        "ordinal_features": ["Pclass"],
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scaler": "standard",
    },

    "notes": "Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone.",
}

fe01__family = create_experiment_group(
    stage="fe01",
    feature_group="family",
    base_configs=baseline__raw,
    group_override=fe01__family_override,
)


EXPERIMENTS = [
    {
        "name": "baseline_raw_logreg",
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
        "notes": "Base logistic regression, using raw configuration. Baseline for comparission"
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
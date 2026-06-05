# This file defines the configuration for machine learning experiments on the Titanic dataset.
# Each experiment is represented as a dictionary with specific fields that describe the model, features, preprocessing steps, evaluation method, and other relevant information.

from titanic_ml.feature_engineering import add_family_features, add_has_cabin, add_full_deck, add_title, add_full_title_feature
import copy

# Baseline configuration for experiment config
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
        "notes": "",
    }

# Baseline configuration for models
baseline__models =[
    {   
    "name": "baseline__logreg",
    "model_name": "logreg",
    "model_params": {
        "max_iter": 1000,
        "random_state": 42,
        },
        "notes": "Base logistic regression. Baseline for comparison."
        
    },

    {  
    "name": "baseline__knn",
    "model_name":'knn',
    "model_params":{
        "n_neighbors": 5,
        },
        "notes": "Base kneighbors classifier. Baseline for comparison."
        
    },

    {   
    "name": "baseline__svc",
    "model_name":'svc',
    "model_params":{
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": False,
        "random_state": 42,
        },
        "notes": "Base SVC. Baseline for comparison.",
    },

    {   
    "name": "baseline__decision_tree",
    "model_name":'decision_tree',
    "model_params":{
        "max_depth": 4,
        "min_samples_leaf": 5,
        "random_state": 42,
        },
        "notes": "Base decision tree. Baseline for comparison."
    },

    {
    "name": "baseline__random_forest",
    "model_name":'random_forest',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
        },
        "notes": "Base random forest. Baseline for comparison."
    },

    {
    "name": "baseline__extra_trees",
    "model_name":'extra_trees',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
        },
        "notes": "Base extra trees. Baseline for comparison."
    },

    {   
    "name": "baseline__xgb",
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
        "notes": "Base xgb. Baseline for comparison."
    },
]

# Baseline experiment configuration
baseline__config = []

for model in baseline__models:
    config = copy.deepcopy(baseline_config_model)
    config.update(model)
    baseline__config.append(config)

# New Naming convention:
# <stages>__<feature_group>__<model>

def create_experiment(experiment_name,feature_engineering_name, base_config, model_config_list=None, feature_engineering=None, notes=''):
    if not experiment_name:
        raise ValueError('Experiment needs a name')
    
    if not feature_engineering_name:
        raise ValueError('models needs a feature_engineering name')
    
    if not base_config:
        raise ValueError('No base model added')
    
    base_config = copy.deepcopy(base_config)

    if not model_config_list:
        model_config_list = [[] for x in base_config]

    # print(model_config_list)

    for exp, model in zip(base_config,model_config_list):
        # print(zip(base_config,model_config_list))
        # print(exp)
        if feature_engineering:
            exp.update(feature_engineering)
        if model:
            exp.update(model)

        exp_name = exp['name']
        exp_name = exp_name.split('__')[1]
        exp_name = feature_engineering_name + '__' + exp_name
        exp['name'] = exp_name
        exp['notes'] = f'{experiment_name}__{exp_name}. {notes}'

    return base_config


# Feature engineering 1 - family
fe01 ='fe01'
fe01_fe_name = 'family'
fe01_fe = {"feature_engineering":[add_family_features]}
fe01_note = 'fe01 - Feature engineering 01 experiment - family/alone feature'

fe01__family__config = create_experiment(fe01, fe01_fe_name, baseline__config,feature_engineering=fe01_fe, notes=fe01_note)



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
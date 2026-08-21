from titanic_ml.feature_engineering.sklearn_compatible import AGE_IMPUTER_TITLE, AGE_IMPUTER_TITLE_PCLASS, age_imputer_title, age_imputer_title_pclass, AGE_BIN, age_bin_transformer, TITLE_TRANSFORM, TitleTransformer
from titanic_ml.common.experiments.config_creation import create_config, create_config_group, validate_config_group
import copy

RAW_FEATURES = {
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}

### Old baseline configuration to help with trasition


# List to hold all experiment configurations
ALL_EXPERIMENTS = {}

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
        "stage": "baseline",
        "feature_group": "raw",
        "group": "baseline__raw",
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
    "model_name": 'knn',
    "model_params":{
        "n_neighbors": 5,
        },
        "notes": "Base kneighbors classifier, using raw configuration. Baseline for comparison."
        
    },

    {   
    "name": "baseline__raw__svc",
    "model_name": 'svc',
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
    "model_name": 'decision_tree',
    "model_params":{
        "max_depth": 4,
        "min_samples_leaf": 5,
        "random_state": 42,
        },
        "notes": "Base decision tree, using raw configuration. Baseline for comparison."
    },

    {
    "name": "baseline__raw__random_forest",
    "model_name": 'random_forest',
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
    "model_name": 'extra_trees',
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
    "model_name": 'xgb',
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

# ALL_EXPERIMENTS['baseline__raw'] = baseline__raw


## Old version

# TITLE_FEATURE = {
#     "transformations": [
#         TITLE_TRANSFORM,
#     ],

#     "add": {
#         "preprocessing": {
#             "onehot_features": ["Title"],
#         },
#     },
# }

# TITLE_ONLY = {
#     "transformations": [
#         TITLE_TRANSFORM,
#     ],
# }

# AGE_IMPUTATION_TITLE = {
#     "transformations": [
#         AGE_IMPUTER_TITLE,
#     ],
    
# }

# AGE_IMPUTATION_TITLE_PCLASS = {
#     "transformations": [
#         AGE_IMPUTER_TITLE_PCLASS,
#     ],
# }

# AGE_BIN_FEATURE = {
#     "transformations": [
#         AGE_BIN,
#     ],

#     "add": {
#         "preprocessing": {
#             "ordinal_features": ["Age_bin"],
#         },
#     },
# }



AGE_BIN_TRANSFORM = {
    "id": "age_bin",

    "transformer": age_bin_transformer,

    "requires": [
        "Age",
    ],

    "produces": [
        "Age_bin",
    ],

    "owns": [
        "Age_bin",
    ],
}

## Old Version

# FE06 =[
#     TITLE_ONLY,
#     AGE_IMPUTATION_TITLE,
# ]

# FE11=[
#     AGE_BIN_FEATURE,
# ]

# CB03 = [
#     TITLE_ONLY,
#     AGE_IMPUTATION_TITLE_PCLASS,
#     AGE_BIN_FEATURE,
# ]

FE11 = {
    "transformations": [
        AGE_BIN_TRANSFORM,
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Age_bin",
            ],
        },
    },

    "remove": {
        "preprocessing": {
            "numeric_features": [
                "Age",
            ],
        },
    },
}

FE11_LEGACY_TEST = {
    "transformations": [
        AGE_BIN_TRANSFORM,
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Age_bin",
            ],
        },
    },

    "remove": {
        "preprocessing": {
            "numeric_features": [
                "Age",
                "Fare",
            ],
        },
    },
}

# Converting baseline_raw list into baseline_config dictionary

baseline_config = {
    f"{exp['model_name']}__baseline":
        copy.deepcopy(exp)

    for exp in baseline__raw
}

validate_config_group(
    baseline_config
)

ALL_EXPERIMENTS['baseline__raw'] = baseline_config

# Example config:

fe11_config = create_config_group(
    base_configs=baseline_config,
    patches=[FE11_LEGACY_TEST],
    raw_features=RAW_FEATURES,
    stage="fe11",
    feature_group="age_bin",
    domain="age",
    notes=(
        "Feature engineering 11: "
        "replaces Age with Age_bin."
    ),
)

ALL_EXPERIMENTS['fe11__age_bin'] = fe11_config

FE11_LEGACY_TEST = {
    "transformations": [
        AGE_BIN_TRANSFORM,
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Age_bin",
            ],
        },
    },

    "remove": {
        "preprocessing": {
            "numeric_features": [
                "Age",
                "Fare",
            ],
        },
    },
}

# cb03_config = create_config(
#     base_config=baseline__raw,
#     patches=[
#         TITLE_ONLY,
#         AGE_IMPUTATION_TITLE_PCLASS,
#         AGE_BIN_FEATURE,
#     ],
#     raw_features=RAW_FEATURES,
# )

# config_meant_to_fail = create_config(
#     base_config=baseline__raw,
#     patches=[
#         TITLE_ONLY,
#         AGE_IMPUTATION_TITLE,
#         AGE_IMPUTATION_TITLE_PCLASS,
#     ],
#     raw_features=RAW_FEATURES,
# )
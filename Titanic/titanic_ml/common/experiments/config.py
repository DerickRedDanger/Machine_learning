from titanic_ml.common.experiments.config_creation import create_config, create_config_group, features_from_preprocessing, validate_config_group
import copy
from titanic_ml.feature_engineering.updated import FE

# Dictionary to hold all experiment configurations
ALL_EXPERIMENTS = {}

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

# ============================================================#
# BASELINE CONFIGURATION
# ============================================================#
BASELINE_CONFIG_MODEL = {
    "preprocessing": {
        "numeric_features": [
            "Age",
            "SibSp",
            "Parch",
            "Fare",
        ],
        "onehot_features": [
            "Sex",
            "Embarked",
        ],
        "ordinal_features": [
            "Pclass",
        ],
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scaler": "standard",
    },

    "feature_pipeline": [],

    "evaluation": {
        "method": "cross_validation",
        "cv": 5,
        "scoring": [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ],
        "return_train_score": True,
        "n_jobs": -1,
    },

    "stage": "baseline",
    "feature_group": "raw",
    "group": "baseline__raw",
    "domain": None,
}

# ============================================================#
# BASELINE MODELS
# ============================================================#
BASELINE_MODELS = {
    "logreg": {
        "model_params": {
            "max_iter": 1000,
            "random_state": 42,
        },
        "notes": (
            "Base logistic regression using the raw "
            "feature configuration. Baseline for comparison."
        ),
    },

    "knn": {
        "model_params": {
            "n_neighbors": 5,
        },
        "notes": (
            "Base K-neighbors classifier using the raw "
            "feature configuration. Baseline for comparison."
        ),
    },

    "svc": {
        "model_params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": False,
            "random_state": 42,
        },
        "notes": (
            "Base SVC using the raw feature configuration. "
            "Baseline for comparison."
        ),
    },

    "decision_tree": {
        "model_params": {
            "max_depth": 4,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "notes": (
            "Base decision tree, using raw configuration. "
            "Baseline for comparison."
        ),
    },

    "random_forest": {
        "model_params": {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_leaf": 3,
            "random_state": 42,
            "n_jobs": -1,
        },
        "notes": (
            "Base random forest, using raw configuration."
            "Baseline for comparison."
        ),
    },

    "extra_trees": {
        "model_params": {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_leaf": 3,
            "random_state": 42,
            "n_jobs": -1,
        },
        "notes": (
            "Base extra trees, using raw configuration."
            " Baseline for comparison."
        ),
    },

    "xgb": {
        "model_params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
        },
        "notes": (
            "Base xgb, using raw configuration. "
            "Baseline for comparison."
        ),
    },

}

# ============================================================#
# BASELINE CONFIGURATION GROUP
# ============================================================#

baseline_config = {}

for model_name, model_config in BASELINE_MODELS.items():
    config = copy.deepcopy(
        BASELINE_CONFIG_MODEL
    )

    config["model_name"] = model_name

    config["model_params"] = copy.deepcopy(
        model_config["model_params"]
    )

    config["notes"] = model_config.get(
        "notes",
        "",
    )

    config["name"] = (
        f"baseline__raw__{model_name}"
    )

    config["features"] = (
        features_from_preprocessing(
            config["preprocessing"]
        )
    )

    key = f"{model_name}__raw"

    baseline_config[key] = config

validate_config_group(
    baseline_config
)

ALL_EXPERIMENTS[
    "baseline__raw"
] = baseline_config

# ============================================================
# CONFIGURATION EXAMPLE
# ============================================================
#
# create_config()
#     Builds ONE model configuration from:
#         base config + ordered patches
#
# create_config_group()
#     Applies the same patches to MANY base model configs.
#
# ------------------------------------------------------------
# Example patch
# ------------------------------------------------------------

"""
EXAMPLE_TRANSFORM = {
    "id": "example_transform",

    # Factory/class returning a fresh sklearn transformer.
    "transformer": example_transformer,

    # Columns that must exist before this transformer runs.
    "requires": ["RawFeature"],

    # Columns guaranteed to exist after it runs.
    "produces": ["EngineeredFeature"],

    # Columns this transform exclusively controls/modifies.
    "owns": ["EngineeredFeature"],
}


EXAMPLE_PATCH = {
    "transformations": [
        EXAMPLE_TRANSFORM,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "EngineeredFeature",
            ],
        },
    },

    "remove": {
        "preprocessing": {
            "numeric_features": [
                "RawFeature",
            ],
        },
    },

    "update": {
        # Scalar/config replacement if needed.
    },
}

example_single_config = create_config(
    base_config=baseline_config[
        "logreg__raw"
    ],
    patches=[
        EXAMPLE_PATCH,
    ],
    raw_features=RAW_FEATURES,
    stage="feXX",
    feature_group="example",
    domain="example_domain",
    notes=("Example experiment."
            "And it's notes"),
)

example_group_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        EXAMPLE_PATCH,
    ],
    raw_features=RAW_FEATURES,
    stage="feXX",
    feature_group="example",
    domain="example_domain",
    notes=("Example experiment group."
            "And it's notes."),
)
"""

"""
Use create_config_group when the same experiment definition
is applied to multiple model configurations.

Use create_config when constructing one model-specific
configuration, especially final experiments or heterogeneous
experiment groups.
"""

# Updated Configurations

# FE 01 - Family
fe01__family_patch = {
    "transformations":[
        FE.family
    ],

    "add":{
        "preprocessing":{
            "numerical_features":[
                "Family", "IsAlone"
            ]
        }
    },

    "remove":{
        "numerical_features":[
            "SibSp", "Parch"
        ]
    },

}

fe01_config = create_config_group(
    base_configs=baseline_config,
    patches=[fe01__family_patch],
    raw_features=RAW_FEATURES,
    stage="fe01",
    feature_group="family",
    domain="family",
    notes=(
        "Feature engineering 01: "
        "Replacing SibSp and Parch with Family and IsAlone."
    ),
)

ALL_EXPERIMENTS['fe01__family'] = fe01_config

#Fe 02 - Has Cabin

fe02__has_cabin_patch = {
    "transformations": [
        FE.has_cabin,
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Has_Cabin",
            ],
        },
    },
}

fe02__has_cabin_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe02__has_cabin_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe02",
    feature_group="has_cabin",
    domain="cabin",
    notes=("Feature engineering 02."
            "Adding Has_Cabin"),
)

ALL_EXPERIMENTS["fe02__has_cabin"] = fe02__has_cabin_config

# Fe 03 - Deck

fe03__Deck_patch = {
    "transformations": [
        FE.deck,
    ],

    "add": {
        "preprocessing": {
            "onehot_features": [
                "Deck",
            ],
        },
    },
}

fe03__Deck_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe03__Deck_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe03",
    feature_group="Deck",
    domain="cabin",
    notes=("Feature engineering 03."
            "Adding Deck."),
)

ALL_EXPERIMENTS['Fe03__deck'] = fe03__Deck_config

#Fe 04

fe04__cabin_features_patch = {
    "transformations": [
        FE.has_cabin, FE.deck
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Has_cabin",
            ],
            "onehot_feature":[
                "Deck",
            ]
        },
    },
}

fe04__cabin_features_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe04__cabin_features_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe04",
    feature_group="cabin_features",
    domain="cabin",
    notes=("Feature engineering 04."
            "Adding both Deck and Has_cabin."),
)

ALL_EXPERIMENTS['fe04__cabin_features'] = fe04__cabin_features_config

#Fe 05

fe05__title_patch = {
    "transformations": [
        FE.title,
    ],

    "add": {
        "preprocessing": {
            "onehot_features": [
                "Title",
            ],
        },
    },

}

fe05__title_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe05__title_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe05",
    feature_group="title",
    domain="title",
    notes=("Feature engineering 05."
            "Adding Title."),
)

ALL_EXPERIMENTS['fe05__title'] = fe05__title_config

# Fe 06 - Age imputation by title

fe06_age_imputation_title_patch = {
    "transformations": [
        FE.age_imputer_title,
    ],
}

fe06_age_imputation_title_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe06_age_imputation_title_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe06",
    feature_group="age_imputation_title",
    domain="age",
    notes=("Feature engineering 06."
            "Imputting Age by Title."),
)

ALL_EXPERIMENTS['fe06_age_imputation_title'] = fe06_age_imputation_title_config

# Fe 07 - Age imputation by Title and Pclass

fe07__age_imputation_title_pclass_patch = {
    "transformations": [
        FE.age_imputer_title_pclass,
    ],
}

fe07__age_imputation_title_pclass_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe07__age_imputation_title_pclass_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe04",
    feature_group="age_imputation_title_pclass",
    domain="age",
    notes=("feature engineering 07."
            "Imputting Age by both Title and Pclass."),
)

ALL_EXPERIMENTS['fe07__age_imputation_title_pclass'] = fe07__age_imputation_title_pclass_config

#Fe 08 - Fare per family member

fe08_fare_per_family_member_patch = {
    "transformations": [
        FE.fare_family,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize",
            ],
        },
    },

    "remove": {
        "preprocessing": {
            "numeric_features": [
                "Fare",
            ],
        },
    },

}

fe08_fare_per_family_member_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe08_fare_per_family_member_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe08",
    feature_group="fare_per_family_member",
    domain="fare",
    notes=("Feature engineering 08."
            "Replacing Fare with Fare/FamilySize."),
)

ALL_EXPERIMENTS['fe08_fare_per_family_member'] = fe08_fare_per_family_member_config

# Fe 09 - Ticket group size

fe09__ticket_group_size_patch = {
    "transformations": [
        FE.ticket,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "TicketGroupSize",
            ],
        },
    },
}

fe09__ticket_group_size_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe09__ticket_group_size_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe09",
    feature_group="ticket_group_size",
    domain="ticket",
    notes=("Feature engineering 09."
            "Adding Ticket group size."),
)

ALL_EXPERIMENTS['fe09__ticket_group_size'] = fe09__ticket_group_size_patch_config




FE11 = {
    "transformations": [
        FE.age_bin,
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
        FE.age_bin,
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
        FE.age_bin,
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
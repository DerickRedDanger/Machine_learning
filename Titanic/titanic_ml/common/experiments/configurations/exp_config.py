from titanic_ml.common.experiments.config_creation import create_config, create_config_group, features_from_preprocessing, validate_config_group
import copy
from titanic_ml.feature_engineering.updated import fe as FE
from titanic_ml.pre_cv_feature_engineering import pre_cv_fe as PRE_CV_FE
# Dictionary to hold all experiment configurations
ALL_EXPERIMENTS = {}
ALL_PATCHES={}

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
    
    "pre_cv_feature_pipeline": [],
    "pre_cv_scope": None,

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
    "tags":None,
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
        FE.family,
        FE.is_alone,
    ],

    "add":{
        "preprocessing":{
            "numeric_features":[
                "FamilySize",
                "IsAlone",
            ],
        },
    },

    "remove":{
        "preprocessing":{
            "numeric_features":[
                "SibSp",
                "Parch",
            ],
        },
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
ALL_PATCHES['fe01__family_patch'] = fe01__family_patch

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
ALL_PATCHES["fe02__has_cabin_patch"] = fe02__has_cabin_patch

# Fe 03 - Deck

fe03__deck_patch = {
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

fe03__deck_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe03__deck_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe03",
    feature_group="deck",
    domain="cabin",
    notes=("Feature engineering 03."
            "Adding Deck."),
)

ALL_EXPERIMENTS['fe03__deck'] = fe03__deck_config
ALL_PATCHES['fe03__deck_patch'] = fe03__deck_patch

#Fe 04

fe04__cabin_features_patch = {
    "transformations": [
        FE.has_cabin, FE.deck
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Has_Cabin",
            ],
            "onehot_features":[
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
ALL_PATCHES['fe04__cabin_features_patch'] = fe04__cabin_features_patch

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
ALL_PATCHES['fe05__title_patch'] = fe05__title_patch

# Fe 06 - Age imputation by title

fe06__age_imputation_title_patch = {
    "transformations": [
        FE.title,
        FE.age_imputer_title,
    ],
}

fe06__age_imputation_title_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe06__age_imputation_title_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe06",
    feature_group="age_imputation_title",
    domain="age",
    notes=("Feature engineering 06."
            "Imputting Age by Title."),
)

ALL_EXPERIMENTS['fe06__age_imputation_title'] = fe06__age_imputation_title_config
ALL_PATCHES['fe06__age_imputation_title_patch'] = fe06__age_imputation_title_patch

# Fe 07 - Age imputation by Title and Pclass

fe07__age_imputation_title_pclass_patch = {
    "transformations": [
        FE.title,
        FE.age_imputer_title_pclass,
    ],
}

fe07__age_imputation_title_pclass_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe07__age_imputation_title_pclass_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe07",
    feature_group="age_imputation_title_pclass",
    domain="age",
    notes=("feature engineering 07."
            "Imputting Age by both Title and Pclass."),
)

ALL_EXPERIMENTS['fe07__age_imputation_title_pclass'] = fe07__age_imputation_title_pclass_config
ALL_PATCHES['fe07__age_imputation_title_pclass_patch'] = fe07__age_imputation_title_pclass_patch
#Fe 08 - Fare per family member

fe08__fare_per_family_member_patch = {
    "transformations": [
        FE.family,
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

fe08__fare_per_family_member_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe08__fare_per_family_member_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe08",
    feature_group="fare_per_family_member",
    domain="fare",
    notes=("Feature engineering 08."
            "Replacing Fare with Fare/FamilySize."),
)

ALL_EXPERIMENTS['fe08__fare_per_family_member'] = fe08__fare_per_family_member_config

ALL_PATCHES['fe08__fare_per_family_member_patch'] = fe08__fare_per_family_member_patch

# Fe 09 - Ticket group size

# Fe 10 - Fare per Ticket Member

# Moved to the bottom of the file to avoid circular dependency issues with the new ticket group size experiments.


# Fe 11 - Age Bin

fe11__age_bin_patch = {
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

fe11__age_bin_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe11__age_bin_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe11",
    feature_group="age_bin",
    domain="age",
    notes=("Feature Engineering 11."
            "Replacing Age with Age bin."),
)

ALL_EXPERIMENTS['fe11__age_bin'] = fe11__age_bin_config

ALL_PATCHES['fe11__age_bin_patch'] = fe11__age_bin_patch

# FE 12 - Sex Pclass

fe12__sex_pclass_patch = {
    "transformations": [
        FE.sex_pclass,
    ],

    "add": {
        "preprocessing": {
            "onehot_features": [
                "Sex_Pclass",
            ],
        },
    },

    "remove": {
        "preprocessing": {
            "onehot_features": [
                "Sex",
            ],
            "ordinal_features":[
                "Pclass",
            ],
        },
    },
}

fe12__sex_pclass_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe12__sex_pclass_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe12",
    feature_group="sex_pclass",
    domain="sex_pclass",
    notes=("Feature Engineering 12."
            "Replacing Sex and Pclass with Sex_Pclass."),
)

ALL_EXPERIMENTS['fe12__sex_pclass'] = fe12__sex_pclass_config

ALL_PATCHES['fe12__sex_pclass_patch'] = fe12__sex_pclass_patch

# Combo experiments

# Cb 01 - Age and bins

cb01__age_and_bins_patch = {
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
}

cb01__age_and_bins_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb01__age_and_bins_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb01",
    feature_group="age_and_bins",
    domain="age",
    notes=("Combo Experiment 01."
            "Adding Age bins."),
)

ALL_EXPERIMENTS['cb01__age_and_bins'] = cb01__age_and_bins_config

ALL_PATCHES['cb01__age_and_bins_patch'] = cb01__age_and_bins_patch

# Cb 02 - Age imputed by Title and Bins

cb02__age_imputed_title_and_bins_patch = {
    "transformations": [
        FE.title,
        FE.age_imputer_title,
        FE.age_bin
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Age_bin",
            ],
        },
    },
}

cb02__age_imputed_title_and_bins_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb02__age_imputed_title_and_bins_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb02",
    feature_group="age_imputed_title_and_bins",
    domain="age",
    notes=("Combo Experiment 02."
            "Imputing Age by Title, then adding Age bins."),
)

ALL_EXPERIMENTS['cb02__age_imputed_title_and_bins'] = cb02__age_imputed_title_and_bins_config

ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'] = cb02__age_imputed_title_and_bins_patch

# Cb 03 - Age imputed by Title and Pclass and adding Bins

cb03__age_imputed_title_pclass_and_bins_patch = {
    "transformations": [
        FE.title,
        FE.age_imputer_title_pclass,
        FE.age_bin
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": [
                "Age_bin",
            ],
        },
    },
}

cb03__age_imputed_title_pclass_and_bins_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb03__age_imputed_title_pclass_and_bins_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb03",
    feature_group="age_imputed_title_pclass_and_bins",
    domain="age",
    notes=("Combo Experiment 03."
            "Imputing Age by Title and Pclass then adding Age bin."),
)

ALL_EXPERIMENTS['cb03__age_imputed_title_pclass_and_bins'] = cb03__age_imputed_title_pclass_and_bins_config

ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'] = cb03__age_imputed_title_pclass_and_bins_patch

# Cb 04 - Fare and Fare per Family Member

cb04__fare_and_fare_per_family_patch = {
    "transformations": [
        FE.family,
        FE.fare_family,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize",
            ],
        },
    },
}

cb04__fare_and_fare_per_family_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb04__fare_and_fare_per_family_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb04",
    feature_group="fare_and_fare_per_family",
    domain="fare",
    notes=("Combo Experiment 04."
            "Adding Fare per Family member."),
)

ALL_EXPERIMENTS['cb04__fare_and_fare_per_family'] = cb04__fare_and_fare_per_family_config

ALL_PATCHES['cb04__fare_and_fare_per_family_patch'] = cb04__fare_and_fare_per_family_patch

# Cb 05 - Fare and Fare per Ticket member

# Cb 06 - All Fare features

# Moved to the bottom of the file to avoid circular dependency issues with the new ticket group size experiments.

# Cb 07 - Family features

cb07__family_features_patch = {
    "transformations": [
        FE.family,
        FE.is_alone
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "FamilySize",
                "IsAlone",
            ],
        },
    },
}

cb07__family_features_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb07__family_features_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb07",
    feature_group="family_features",
    domain="family",
    notes=("Combo Experiment 07."
            "Adding Family Size."),
)

ALL_EXPERIMENTS['cb07__family_features'] = cb07__family_features_config

ALL_PATCHES['cb07__family_features_patch'] = cb07__family_features_patch

# Cb 08 - Sex Pclass features

cb08__sex_pclass_features_patch = {
    "transformations": [
        FE.sex_pclass,
    ],

    "add": {
        "preprocessing": {
            "onehot_features": [
                "Sex_Pclass",
            ],
        },
    },
}

cb08__sex_pclass_features_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb08__sex_pclass_features_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb08",
    feature_group="sex_pclass_features",
    domain="sex_pclass",
    notes=("Combo Experiment 08."
            "Adding Sex_Pclass."),
)

ALL_EXPERIMENTS['cb08__sex_pclass_features'] = cb08__sex_pclass_features_config

ALL_PATCHES['cb08__sex_pclass_features_patch'] = cb08__sex_pclass_features_patch

# Ablations

# Ab 01 - Age and bins without fare

ab01__age_and_bins_without_fare_patch= {
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
                "Fare",
            ],
        },
    },
}

ab01__age_and_bins_without_fare_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        ab01__age_and_bins_without_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="ab01",
    feature_group="age_and_bins_without_fare",
    domain="ablation",
    notes=("Ablation Experiment 01."
            "Adding Age bins and removing Fare."),
    tags=set(["ablation"])
)

ALL_EXPERIMENTS['ab01__age_and_bins_without_fare'] = ab01__age_and_bins_without_fare_config

# Ab 02 - Age imputed by title, then Age bin without Fare

ab02__age_imputed_title_and_bins_without_fare_patch = {
    "transformations": [
        FE.title,
        FE.age_imputer_title, 
        FE.age_bin
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
                "Fare",
            ],
        },
    },
}

ab02__age_imputed_title_and_bins_without_fare_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        ab02__age_imputed_title_and_bins_without_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="ab02",
    feature_group="age_imputed_title_and_bins_without_fare",
    domain="ablation",
    notes=("Ablation Experiment."
            "Age imputed by Title then adding Age bin while removing Fare."),
    tags=set(["ablation"])
)

ALL_EXPERIMENTS['ab02__age_imputed_title_and_bins_without_fare'] = ab02__age_imputed_title_and_bins_without_fare_config

# Ab 03 - Age imputed by both Title and Pclass, then age bin, without Fare

ab03__age_imputed_title_pclass_and_bins_without_fare_patch = {
    "transformations": [
        FE.title,
        FE.age_imputer_title_pclass,
        FE.age_bin
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
                "Fare",
            ],
        },
    },
}

ab03__age_imputed_title_pclass_and_bins_without_fare_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        ab03__age_imputed_title_pclass_and_bins_without_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="ab03",
    feature_group="age_imputed_title_pclass_and_bins_without_fare",
    domain="ablation",
    notes=("Ablation Experiment 03."
            "Age imputed by Title and Pclass, then binned, without Fare."),
    tags={"ablation"}
)

ALL_EXPERIMENTS['ab03__age_imputed_title_pclass_and_bins_without_fare'] = ab03__age_imputed_title_pclass_and_bins_without_fare_config

# ablation
# Ab 04 - Age bin without Fare
ab04__age_bin_without_fare_patch = {
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


ab04__age_bin_without_fare_config = create_config_group(
    base_configs=baseline_config,
    patches=[ab04__age_bin_without_fare_patch],
    raw_features=RAW_FEATURES,
    stage="ab04",
    feature_group="age_bin_without_fare",
    domain="ablation",
    notes=(
        "Feature engineering 11: "
        "replaces Age with Age_bin."
    ),
    tags=set(["ablation"]),
)

ALL_EXPERIMENTS['ab04__age_bin_without_fare'] = ab04__age_bin_without_fare_config


# Ab 05 - Sex _ Pclass without SibSp and Parch

ab05__sex_pclass_without_sibsp_parch_patch = {
    "transformations": [
        FE.sex_pclass,
    ],

    "add": {
        "preprocessing": {
            "onehot_features": [
                "Sex_Pclass",
            ],
        },
    },

    "remove": {
        "preprocessing": {
            "onehot_features": [
                "Sex",
            ],
            "ordinal_features":[
                "Pclass",
            ],
            "numeric_features":[
                "SibSp",
                "Parch",
            ]
        },
    },
}

ab05__sex_pclass_without_sibsp_parch_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        ab05__sex_pclass_without_sibsp_parch_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="ab05",
    feature_group="sex_pclass_without_sibsp_parch",
    domain="ablation",
    notes=("Ablation experiment 05."
            "Replacing Sex and Pclass with Sex_Pclass while removing SibSp and Parch."),
    tags=set(["ablation"]),
)

ALL_EXPERIMENTS['ab05__sex_pclass_without_sibsp_parch'] = ab05__sex_pclass_without_sibsp_parch_config


# debugging

# ============================================================
# CONTROL TEST - CB01 with legacy preprocessing column order
#
# Purpose:
# Test whether the RF / ExtraTrees / XGB result differences
# are caused only by preprocessing feature order.
# ============================================================

cb01__legacy_order_control_config = copy.deepcopy(
    cb01__age_and_bins_config
)

for config in cb01__legacy_order_control_config.values():

    # Change ONLY the feature order to match legacy CB01.
    config["preprocessing"]["numeric_features"] = [
        "Fare",
        "SibSp",
        "Parch",
        "Age",
    ]

    # Give the control experiment its own identity.
    model_name = config["model_name"]

    config["stage"] = "control"
    config["feature_group"] = "cb01_legacy_order"
    config["group"] = "control__cb01_legacy_order"
    config["name"] = (
        f"control__cb01_legacy_order__{model_name}"
    )

    # Keep the derived human-readable feature list synchronized.
    config["features"] = features_from_preprocessing(
        config["preprocessing"]
    )


validate_config_group(
    cb01__legacy_order_control_config
)

ALL_EXPERIMENTS[
    "control__cb01_legacy_order"
] = cb01__legacy_order_control_config


# new Ticket groupsize experiment

fe09__ticket_group_size_batch_patch = {
    "transformations": [
        FE.ticket_group_size_batch,
    ],
    "add": {
        "preprocessing": {
            "numeric_features": [
                "TicketGroupSize",
            ],
        },
    },
}


fe09__ticket_group_size_fitted_patch = {
    "transformations": [
        FE.ticket_group_size_fitted,
    ],
    "add": {
        "preprocessing": {
            "numeric_features": [
                "TicketGroupSize",
            ],
        },
    },
}

fe09__ticket_group_size_full_context_patch = {
    "pre_cv_transformations": [
        PRE_CV_FE.ticket_group_size_full_context,
    ],
    "add": {
        "preprocessing": {
            "numeric_features": [
                "TicketGroupSize",
            ],
        },
    },
}

fe09__ticket_group_size_batch_configs = create_config_group(
    base_configs=baseline_config,
    patches=[fe09__ticket_group_size_batch_patch],
    raw_features=RAW_FEATURES,
    stage="fe09",
    feature_group="ticket_group_size_batch",
    domain="ticket",
)

fe09__ticket_group_size_fitted_configs = create_config_group(
    base_configs=baseline_config,
    patches=[fe09__ticket_group_size_fitted_patch],
    raw_features=RAW_FEATURES,
    stage="fe09",
    feature_group="ticket_group_size_fitted",
    domain="ticket",
)

fe09__ticket_group_size_full_context_configs = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe09__ticket_group_size_full_context_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe09",
    feature_group="ticket_group_size_full_context",
    domain="ticket",
    pre_cv_scope="full_prediction_context",
)

ALL_EXPERIMENTS["fe09__ticket_group_size_batch"] = fe09__ticket_group_size_batch_configs

ALL_EXPERIMENTS["fe09__ticket_group_size_fitted"] = fe09__ticket_group_size_fitted_configs

ALL_EXPERIMENTS["fe09__ticket_group_size_full_context"] = fe09__ticket_group_size_full_context_configs

ALL_PATCHES["fe09__ticket_group_size_batch_patch"] = fe09__ticket_group_size_batch_patch

ALL_PATCHES["fe09__ticket_group_size_fitted_patch"] = fe09__ticket_group_size_fitted_patch

ALL_PATCHES["fe09__ticket_group_size_full_context_patch"] = fe09__ticket_group_size_full_context_patch
# fe10

fe10__fare_per_ticket_member_batch_patch = {
    "transformations": [
        FE.ticket_group_size_batch, 
        FE.fare_ticket,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/TicketMember",
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


fe10__fare_per_ticket_member_fitted_patch = {
    "transformations": [
        FE.ticket_group_size_fitted, 
        FE.fare_ticket,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/TicketMember",
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

fe10__fare_per_ticket_member_full_context_patch = {
    "pre_cv_transformations": [
            PRE_CV_FE.ticket_group_size_full_context,
        ],

    "transformations": [
        FE.fare_ticket,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/TicketMember",
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

fe10__fare_per_ticket_member_batch_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe10__fare_per_ticket_member_batch_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe10",
    feature_group="fare_per_ticket_member_batch",
    domain="fare",
    notes=("Features Engineering 10."
            "Replacing Fare with Fare/TicketMember, batch version."),
)

fe10__fare_per_ticket_member_fitted_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe10__fare_per_ticket_member_fitted_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe10",
    feature_group="fare_per_ticket_member_fitted",
    domain="fare",
    notes=("Features Engineering 10."
            "Replacing Fare with Fare/TicketMember, fitted version."),
)

fe10__fare_per_ticket_member_full_context_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe10__fare_per_ticket_member_full_context_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="fe10",
    feature_group="fare_per_ticket_member_full_context",
    domain="fare",
    notes=("Features Engineering 10."
            "Replacing Fare with Fare/TicketMember, full context version."),
    pre_cv_scope="full_prediction_context",
)

ALL_EXPERIMENTS['fe10__fare_per_ticket_member_batch'] = fe10__fare_per_ticket_member_batch_config

ALL_EXPERIMENTS['fe10__fare_per_ticket_member_fitted'] = fe10__fare_per_ticket_member_fitted_config

ALL_EXPERIMENTS['fe10__fare_per_ticket_member_full_context'] = fe10__fare_per_ticket_member_full_context_config

ALL_PATCHES['fe10__fare_per_ticket_member_batch_patch'] = fe10__fare_per_ticket_member_batch_patch

ALL_PATCHES['fe10__fare_per_ticket_member_fitted_patch'] = fe10__fare_per_ticket_member_fitted_patch

ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'] = fe10__fare_per_ticket_member_full_context_patch

# CB05

cb05__fare_and_fare_per_ticket_batch_patch = {
    "transformations": [
        FE.ticket_group_size_batch,
        FE.fare_ticket,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/TicketMember",
            ],
        },
    },
}

cb05__fare_and_fare_per_ticket_fitted_patch = {
    "transformations": [
        FE.ticket_group_size_fitted,
        FE.fare_ticket,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/TicketMember",
            ],
        },
    },
}

cb05__fare_and_fare_per_ticket_full_context_patch = {
    "pre_cv_transformations": [
            PRE_CV_FE.ticket_group_size_full_context,
        ],

    "transformations": [
        FE.fare_ticket,
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/TicketMember",
            ],
        },
    },
}

cb05__fare_and_fare_per_ticket_batch_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb05__fare_and_fare_per_ticket_batch_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb05",
    feature_group="fare_and_fare_per_ticket_batch",
    domain="fare",
    notes=("Combo Experiment 05."
            "Adding Fare per Ticket member, batch version."),
)


cb05__fare_and_fare_per_ticket_fitted_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb05__fare_and_fare_per_ticket_fitted_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb05",
    feature_group="fare_and_fare_per_ticket_fitted",
    domain="fare",
    notes=("Combo Experiment 05."
            "Adding Fare per Ticket member, fitted version."),
)


cb05__fare_and_fare_per_ticket_full_context_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb05__fare_and_fare_per_ticket_full_context_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb05",
    feature_group="fare_and_fare_per_ticket_full_context",
    domain="fare",
    notes=("Combo Experiment 05."
            "Adding Fare per Ticket member, full context version."),
    pre_cv_scope="full_prediction_context",
)

ALL_EXPERIMENTS['cb05__fare_and_fare_per_ticket_batch'] = cb05__fare_and_fare_per_ticket_batch_config

ALL_EXPERIMENTS['cb05__fare_and_fare_per_ticket_fitted'] = cb05__fare_and_fare_per_ticket_fitted_config

ALL_EXPERIMENTS['cb05__fare_and_fare_per_ticket_full_context'] = cb05__fare_and_fare_per_ticket_full_context_config

ALL_PATCHES['cb05__fare_and_fare_per_ticket_batch_patch'] = cb05__fare_and_fare_per_ticket_batch_patch

ALL_PATCHES['cb05__fare_and_fare_per_ticket_fitted_patch'] = cb05__fare_and_fare_per_ticket_fitted_patch

ALL_PATCHES['cb05__fare_and_fare_per_ticket_full_context_patch'] = cb05__fare_and_fare_per_ticket_full_context_patch

# Cb06

cb06__all_fare_features_batch_patch = {
    "transformations": [
        FE.family,
        FE.ticket_group_size_batch,
        FE.fare_family,
        FE.fare_ticket
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize", "Fare/TicketMember",
            ],
        },
    },
}

cb06__all_fare_features_fitted_patch = {
    "transformations": [
        FE.family,
        FE.ticket_group_size_fitted,
        FE.fare_family,
        FE.fare_ticket
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize", "Fare/TicketMember",
            ],
        },
    },
}

cb06__all_fare_features_full_context_patch = {
    "pre_cv_transformations": [
        PRE_CV_FE.ticket_group_size_full_context,
    ],

    "transformations": [
        FE.family,
        FE.fare_family,
        FE.fare_ticket
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize", "Fare/TicketMember",
            ],
        },
    },
}

cb06__all_fare_features_batch_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb06__all_fare_features_batch_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb06",
    feature_group="all_fare_features_batch",
    domain="fare",
    pre_cv_scope= None,
    notes=("Combo Experiment 06."
            "Adding Fare/FamilySize and Fare/TicketMember."),
)


cb06__all_fare_features_fitted_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb06__all_fare_features_fitted_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb06",
    feature_group="all_fare_features_fitted",
    domain="fare",
    pre_cv_scope= None,
    notes=("Combo Experiment 06."
            "Adding Fare/FamilySize and Fare/TicketMember."),
)


cb06__all_fare_features_full_context_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb06__all_fare_features_full_context_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb06",
    feature_group="all_fare_features_full_context",
    domain="fare",
    pre_cv_scope="full_prediction_context",
    notes=("Combo Experiment 06."
            "Adding Fare/FamilySize and Fare/TicketMember."),
    
)

ALL_EXPERIMENTS['cb06__all_fare_features_batch'] = cb06__all_fare_features_batch_config

ALL_EXPERIMENTS['cb06__all_fare_features_fitted'] = cb06__all_fare_features_fitted_config

ALL_EXPERIMENTS['cb06__all_fare_features_full_context'] = cb06__all_fare_features_full_context_config

ALL_PATCHES['cb06__all_fare_features_batch_patch'] = cb06__all_fare_features_batch_patch

ALL_PATCHES['cb06__all_fare_features_fitted_patch'] = cb06__all_fare_features_fitted_patch

ALL_PATCHES['cb06__all_fare_features_full_context_patch'] = cb06__all_fare_features_full_context_patch
# Cb09

cb09__fare_per_family_and_ticket_batch_patch = {
    "transformations": [
        FE.family,
        FE.ticket_group_size_batch,
        FE.fare_family,
        FE.fare_ticket
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize", "Fare/TicketMember",
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

cb09__fare_per_family_and_ticket_fitted_patch = {
    "transformations": [
        FE.family,
        FE.ticket_group_size_fitted,
        FE.fare_family,
        FE.fare_ticket
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize", "Fare/TicketMember",
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

cb09__fare_per_family_and_ticket_full_context_patch = {
    "pre_cv_transformations": [
        PRE_CV_FE.ticket_group_size_full_context,
    ],

    "transformations": [
        FE.family,
        FE.fare_family,
        FE.fare_ticket
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize", "Fare/TicketMember",
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

cb09__fare_per_family_and_ticket_batch_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb09__fare_per_family_and_ticket_batch_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb09",
    feature_group="fare_per_family_and_ticket_batch",
    domain="fare",
    pre_cv_scope= None,
    notes=("Combo Experiment 09."
            "Adding Fare/FamilySize and Fare/TicketMember, without raw fare."),
)


cb09__fare_per_family_and_ticket_fitted_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb09__fare_per_family_and_ticket_fitted_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb09",
    feature_group="fare_per_family_and_ticket_fitted",
    domain="fare",
    pre_cv_scope= None,
    notes=("Combo Experiment 09."
            "Adding Fare/FamilySize and Fare/TicketMember, without raw fare."),
)


cb09__fare_per_family_and_ticket_full_context_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        cb09__fare_per_family_and_ticket_full_context_patch,
    ],
    raw_features=RAW_FEATURES,
    stage="cb09",
    feature_group="fare_per_family_and_ticket_full_context",
    domain="fare",
    pre_cv_scope="full_prediction_context",
    notes=("Combo Experiment 09."
            "Adding Fare/FamilySize and Fare/TicketMember, without raw fare."),
    
)

ALL_EXPERIMENTS['cb09__fare_per_family_and_ticket_batch'] = cb09__fare_per_family_and_ticket_batch_config

ALL_EXPERIMENTS['cb09__fare_per_family_and_ticket_fitted'] = cb09__fare_per_family_and_ticket_fitted_config

ALL_EXPERIMENTS['cb09__fare_per_family_and_ticket_full_context'] = cb09__fare_per_family_and_ticket_full_context_config

ALL_PATCHES['cb09__fare_per_family_and_ticket_batch_patch'] = cb09__fare_per_family_and_ticket_batch_patch

ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch'] = cb09__fare_per_family_and_ticket_fitted_patch

ALL_PATCHES['cb09__fare_per_family_and_ticket_full_context_patch'] = cb09__fare_per_family_and_ticket_full_context_patch

# test configuration

test_family_ticket_batch_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe01__family_patch, fe09__ticket_group_size_batch_patch, cb09__fare_per_family_and_ticket_batch_patch
    ],
    raw_features=RAW_FEATURES,
    stage="test",
    feature_group="family_ticket_batch",
    domain="test",
    pre_cv_scope= None,
    notes=("test to compare values between family and ticket batch."
            ),
)


test_family_ticket_fitted_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        fe01__family_patch, fe09__ticket_group_size_fitted_patch, cb09__fare_per_family_and_ticket_fitted_patch
    ],
    raw_features=RAW_FEATURES,
    stage="test",
    feature_group="family_ticket_fitted",
    domain="test",
    pre_cv_scope= None,
    notes=("test to compare values between family and ticket fitted."),
)


test_family_ticket_full_context_patch = {
    "pre_cv_transformations": [
        PRE_CV_FE.ticket_group_size_full_context,
    ],

    "transformations": [
        FE.family,
        FE.fare_family,
        FE.fare_ticket
    ],

    "add": {
        "preprocessing": {
            "numeric_features": [
                "Fare/FamilySize", "Fare/TicketMember", "FamilySize", "TicketGroupSize"
            ],
        },
    },
}



test_family_ticket_full_context_config = create_config_group(
    base_configs=baseline_config,
    patches=[
        test_family_ticket_full_context_patch
    ],
    raw_features=RAW_FEATURES,
    stage="test",
    feature_group="family_ticket_full_context",
    domain="test",
    pre_cv_scope="full_prediction_context",
    notes=("test to compare values between family and ticket full context."),
    
)

ALL_EXPERIMENTS['test__family_ticket_batch'] = test_family_ticket_batch_config

ALL_EXPERIMENTS['test__family_ticket_fitted'] = test_family_ticket_fitted_config

ALL_EXPERIMENTS['test__family_ticket_full_context'] = test_family_ticket_full_context_config


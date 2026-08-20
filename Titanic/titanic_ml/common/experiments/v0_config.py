from titanic_ml.feature_engineering.sklearn_compatible import AGE_IMPUTER_TITLE, AGE_IMPUTER_TITLE_PCLASS, age_imputer_title, age_imputer_title_pclass, AGE_BIN, age_bin_transformer, TITLE_TRANSFORM, TitleTransformer
from titanic_ml.common.experiments.v0_config_compose import create_config, create_config_group, validate_config_group
from titanic_ml.common.experiments.config import baseline__raw
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
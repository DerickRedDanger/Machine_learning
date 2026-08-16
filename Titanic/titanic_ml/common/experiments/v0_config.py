

from titanic_ml.feature_engineering.sklearn_compatible import AGE_IMPUTER_TITLE, AGE_IMPUTER_TITLE_PCLASS, age_imputer_title, age_imputer_title_pclass, AGE_BIN, age_bin_transformer, TITLE_TRANSFORM, TitleTransformer


TITLE_FEATURE = {
    "transformations": [
        TITLE_TRANSFORM,
    ],

    "add": {
        "preprocessing": {
            "onehot_features": ["Title"],
        },
    },
}

TITLE_ONLY = {
    "transformations": [
        TITLE_TRANSFORM,
    ],
}

AGE_IMPUTATION_TITLE = {
    "transformations": [
        AGE_IMPUTER_TITLE,
    ],
    
}

AGE_IMPUTATION_TITLE_PCLASS = {
    "transformations": [
        AGE_IMPUTER_TITLE_PCLASS,
    ],
}

AGE_BIN_FEATURE = {
    "transformations": [
        AGE_BIN,
    ],

    "add": {
        "preprocessing": {
            "ordinal_features": ["Age_bin"],
        },
    },
}

FE06 =[
    TITLE_ONLY,
    AGE_IMPUTATION_TITLE,
]

CB03 = [
    TITLE_ONLY,
    AGE_IMPUTATION_TITLE_PCLASS,
    AGE_BIN_FEATURE,
]

from titanic_ml.common.experiments.v0_config import create_config, baseline_config, RAW_FEATURES

# Example config:

cb03_config = create_config(
    base_config=baseline_config,
    patches=[
        TITLE_ONLY,
        AGE_IMPUTATION_TITLE_PCLASS,
        AGE_BIN_FEATURE,
    ],
    raw_features=RAW_FEATURES,
)

config_meant_to_fail = create_config(
    base_config=baseline_config,
    [
        TITLE_ONLY,
        AGE_IMPUTATION_TITLE,
        AGE_IMPUTATION_TITLE_PCLASS,
    ],
    RAW_FEATURES,
)
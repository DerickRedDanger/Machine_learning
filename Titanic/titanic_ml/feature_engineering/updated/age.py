from titanic_ml.common.utils.sklearn_compatible.GroupedImputer import GroupedImputer
from titanic_ml.common.utils.sklearn_compatible.BinTransformer import BinTransformer


def age_imputer_title_transform():
    return GroupedImputer(
        target_col="Age",
        group_cols=["Title"],
        agg_func="median",
        fallback_to_global=True,
    )

ADD_AGE_IMPUTED_TITLE = {
    "id": "age_imputer_title",
    "transformer": age_imputer_title_transform,
    "requires": ["Age", "Title"],
    "produces": ["Age"],
    "owns": ["Age"],
}

def age_imputer_title_pclass_transform():
    return GroupedImputer(
        target_col="Age",
        group_cols=["Title", "Pclass"],
        agg_func="median",
        fallback_to_global=True,
    )

ADD_AGE_IMPUTED_TITLE_PCLASS = {
    "id": "age_imputer_title_pclass",
    "transformer": age_imputer_title_pclass_transform,
    "requires": ["Age", "Title", "Pclass"],
    "produces": ["Age"],
    "owns": ["Age"],
}

def age_bin_transform():
    return BinTransformer(
        source_col="Age",
        output_col="Age_bin",
        bins=[0, 14, 35, 60, 100],
        labels=["0", "2", "3", "1"],
        right=False,
    )

ADD_AGE_BIN = {
    "id": "age_bin",
    "transformer": age_bin_transform,
    "requires": ["Age"],
    "produces": ["Age_bin"],
}
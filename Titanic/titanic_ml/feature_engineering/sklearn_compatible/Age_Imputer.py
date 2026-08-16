from titanic_ml.sklearn_compatible.GroupedImputer import GroupedImputer

def age_imputer_title():
    return GroupedImputer(
        target_col="Age",
        group_cols=["Title"],
        agg_func="median",
        fallback_to_global=True,
    )

AGE_IMPUTER_TITLE = {
    "id": "age_imputer_title",
    "transformer": age_imputer_title,
    "requires": ["Age", "Title"],
    "produces": ["Age"],
    "owns": ["Age"],
}

def age_imputer_title_pclass():
    return GroupedImputer(
        target_col="Age",
        group_cols=["Title", "Pclass"],
        agg_func="median",
        fallback_to_global=True,
    )

AGE_IMPUTER_TITLE_PCLASS = {
    "id": "age_imputer_title_pclass",
    "transformer": age_imputer_title_pclass,
    "requires": ["Age", "Title", "Pclass"],
    "produces": ["Age"],
    "owns": ["Age"],
}
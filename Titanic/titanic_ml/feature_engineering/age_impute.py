from titanic_ml.common.utils.reusable_functions import grouped_impute
from titanic_ml.feature_engineering import add_full_title_feature

def age_imputed_by_title(df):
    df = add_full_title_feature(df)

    df = grouped_impute(
        df=df,
        target_cols=["Age"],
        group_cols=["Title"],
        agg_func="median",
        fallback_to_global=True,
    )

    return df

def age_imputed_by_title_pclass(df):
    df = add_full_title_feature(df)

    df = grouped_impute(
        df=df,
        target_cols=["Age"],
        group_cols=["Title", 'Pclass'],
        agg_func="median",
        fallback_to_global=True,
    )

    return df
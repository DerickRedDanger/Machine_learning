def grouped_impute(
    df,
    target_cols,
    group_cols,
    agg_func="median",
    inplace=False,
    fallback_to_global=True,
    default_fill_value=None,
):
    if not inplace:
        df = df.copy()

    if isinstance(target_cols, str):
        target_cols = [target_cols]

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    for col in target_cols:
        if agg_func == "mode":
            def fill_group(x):
                mode = x.mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else None
                return x.fillna(fill_value)

            df[col] = df.groupby(group_cols)[col].transform(fill_group)

        elif agg_func == "median":
            df[col] = df[col].fillna(
                df.groupby(group_cols)[col].transform("median")
            )

        elif agg_func == "mean":
            df[col] = df[col].fillna(
                df.groupby(group_cols)[col].transform("mean")
            )

        else:
            raise ValueError(f"Unsupported agg_func: {agg_func}")

        if fallback_to_global and df[col].isna().sum() > 0:
            if agg_func == "mode":
                global_mode = df[col].mode(dropna=True)
                global_value = global_mode.iloc[0] if not global_mode.empty else None
            elif agg_func == "median":
                global_value = df[col].median()
            elif agg_func == "mean":
                global_value = df[col].mean()

            if global_value is not None:
                df[col] = df[col].fillna(global_value)

        if default_fill_value is not None and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(default_fill_value)

    return df
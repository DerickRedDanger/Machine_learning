import pandas as pd

def quick_description(df):
    """
    Provides a quick overview of the DataFrame, including:
    - DataFrame info
    - Shape, columns, and data types
    - Descriptive statistics for numeric and categorical features
    - Missing values count

    Parameters:
    df (pd.DataFrame): The DataFrame to be described.

    Returns:
    None: Prints the description to the console.
    """
    print('DataFrame info:')
    df.info()
    print('------------------------')
    print('DataFrame shape:', df.shape)
    print('------------------------')
    print('DataFrame columns:', df.columns.tolist())
    print('------------------------')
    print('DataFrame dtypes:')
    print(df.dtypes)
    print('------------------------')

    print('Describe numeric features:')
    print(df.describe(include=['number']))
    print('------------------------')

    print('Describe categorical features:')
    print(df.describe(include=['object', 'category']))
    print('------------------------')

    print('Missing values:')
    print(df.isnull().sum())
    print('------------------------')

    print('DataFrame head:')
    print(df.head())
    print('------------------------')


def format_missing_index(index):
    return ["MISSING" if pd.isna(idx) else idx for idx in index]


def safe_top_value(series):
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return None
    value = counts.index[0]
    return "MISSING" if pd.isna(value) else value


def safe_bottom_value(series):
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return None
    value = counts.index[-1]
    return "MISSING" if pd.isna(value) else value

def classify_skew(skew):

    if pd.isna(skew):
        return None

    skew = round(abs(skew), 3)

    if skew < 0.5:
        return "low skew"

    if skew < 1:
        return "moderate skew"

    return "high skew"

def classify_dominance(dominance):
    # dominance = dominance.round(3)

    if dominance == 100:
        return "constant"

    if dominance >= 95:
        return "almost_constant"

    if dominance >= 30:
        return "some_dominance"

    return "balanced"

def classify_kurtosis(kurtosis):
    
    if pd.isna(kurtosis):
        return None
    
    kurtosis = round(abs(kurtosis), 3)

    if kurtosis < 0:
        return "light_tails"

    if kurtosis < 3:
        return "normal_tails"

    return "heavy_tails"

def classify_cardinality(cardinality_pct, unique):
    if unique == 0:
        return "empty"

    if unique == 1:
        return "constant"

    if cardinality_pct >= 95:
        return "potential_id"

    if unique <= 10:
        return "low_cardinality"

    if unique <= 50:
        return "moderate_cardinality"

    return "high_cardinality"


def summarize_dataframe(df):

    cardinality = (df.nunique(dropna=False) / len(df) * 100).round(2)
    unique = df.nunique(dropna=False)
    cardinality_labels = pd.Series({
        col: classify_cardinality(cardinality[col], unique[col])
        for col in df.columns
    })
    dominance = df.apply(lambda s: s.value_counts(dropna=False, normalize=True).iloc[0] * 100 if not s.empty else None)
    dominance_labels = dominance.apply(classify_dominance)
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null_count": df.notna().sum(),
        "missing_count": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": unique,
        "cardinality_%": cardinality,
        "cardinality_label": cardinality_labels,
        "top_value": df.apply(safe_top_value),
        "dominance_%": dominance,
        "dominance_label": dominance_labels,
        "bottom_value": df.apply(safe_bottom_value),
    })

    return summary
# .sort_values(by="missing_%", ascending=False)

def summarize_dataframe_health(df):
    rows_with_missing = df.isna().any(axis=1).sum()
    duplicate_rows = df.duplicated().sum()

    return pd.Series({
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": duplicate_rows,
        "duplicate_%": round(duplicate_rows / len(df) * 100, 2),
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_%": round(rows_with_missing / len(df) * 100, 2),
        "total_missing_values": df.isna().sum().sum(),
        "dataFrame_columns": df.columns.tolist(),
    })

def summarize_categorical_column(df, feature, target=None, max_display=20):
    feature_series = df[feature].fillna("MISSING")

    counts = feature_series.value_counts(dropna=False)
    percents = (feature_series.value_counts(dropna=False, normalize=True) * 100).round(2)

    summary = pd.DataFrame({
        "count": counts,
        "percent": percents,
    })

    if target is not None:
        target_summary = pd.crosstab(
            feature_series,
            df[target],
            normalize="index"
        ).round(4) * 100

        target_summary.columns = [
            f"{target}_{col}_%" for col in target_summary.columns
        ]

        summary = summary.join(target_summary)

    if len(summary) <= max_display:
        return summary

    top_n = max_display // 2

    separator = pd.DataFrame(
        {col: ["..."] for col in summary.columns},
        index=["..."]
    )

    return pd.concat([
        summary.head(top_n),
        separator,
        summary.tail(top_n),
    ])

def summarize_numerical_column(series):
    summary = series.describe()

    clean = series.dropna()

    if clean.empty:
        extra = pd.Series({
            "missing_count": series.isna().sum(),
            "missing_%": round(series.isna().mean() * 100, 2),
            "outlier_count": 0,
            "outlier_%": 0,
            "skew": None,
            "skew_classification": None,
            "kurtosis": None,
            "kurtosis_classification": None,
        })
        return pd.concat([summary, extra])

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (clean < lower) | (clean > upper)

    skew = clean.skew()
    kurtosis = clean.kurtosis()
    skew_classification = classify_skew(skew)
    kurtosis_classification = classify_kurtosis(kurtosis)

    extra = pd.Series({
        "missing_count": series.isna().sum(),
        "missing_%": round(series.isna().mean() * 100, 2),
        "outlier_count": mask.sum(),
        "outlier_%": round(mask.mean() * 100, 2),
        "skew": skew,
        "skew_classification": skew_classification,
        "kurtosis": kurtosis,
        "kurtosis_classification": kurtosis_classification,
    })

    return pd.concat([summary, extra])

def run_eda(df, target=None, display=True, max_display=20, method="pearson"):
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns
    corr_df = df[numerical_cols].corr(method=method).round(3)
    target_corr_df = None
    
    if target is not None and target in corr_df.columns:
        target_corr_df = corr_df[target].drop(target).round(3).sort_values(key=lambda s: s.abs(), ascending=False).to_frame()

        # Remove target from categorical and numerical columns if present
        categorical_cols = [
            col for col in categorical_cols
            if col != target
        ]
        numerical_cols = [
            col for col in numerical_cols
            if col != target
        ]

    eda_results = {
        "dataFrame_summary": summarize_dataframe(df),
        "dataFrame_health": summarize_dataframe_health(df),
        "categorical_summary": {
            col: summarize_categorical_column(df, feature=col, max_display=max_display, target=target)
            for col in categorical_cols
        },
        "numerical_summary": {
            col: summarize_numerical_column(df[col])
            for col in numerical_cols
        },
        "correlation_matrix": corr_df,
    }

    if target_corr_df is not None:
        eda_results["target_correlation"] = target_corr_df

    if display:
        print('Dataframe Health')
        print(eda_results["dataFrame_health"])
        print("="*60)
        print("DataFrame Summary:")
        print(eda_results["dataFrame_summary"])
        print("="*60)
        print("Categorical columns summary:")
        for col in categorical_cols:
            print(f"\n=== {col} ===")
            print(eda_results["categorical_summary"][col])
        print("="*60)
        print("\nNumerical columns summary:")
        for col in numerical_cols:
            print(f"\n=== {col} ===")
            print(eda_results["numerical_summary"][col])
        print("="*60)
        print(f"\nCorrelation matrix ({method}):")
        print(eda_results["correlation_matrix"])
        print("="*60)
        if target_corr_df is not None:
            print(f"\n=== {target} ===")
            print(f"\nCorrelation of features with target '{target}':")
            print(eda_results["target_correlation"])
            
    return eda_results
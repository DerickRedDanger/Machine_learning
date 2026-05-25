import pandas as pd

def sample_dataframe(
    df,
    head=5,
    random=10,
    tail=5,
    random_state=42
):
    top = df.head(head)
    bottom = df.tail(tail)

    middle = df.iloc[head:max(head, len(df) - tail)]

    if middle.empty:
        rand = middle
    else:
        rand = middle.sample(
            n=min(random, len(middle)),
            random_state=random_state
        ).sort_index()

    return {
        "head": top,
        "random": rand,
        "tail": bottom,
    }

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

def display_memory_usage(df):
    df.memory_usage(deep=True)
    return(f"Total memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


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
        "memory_usage": display_memory_usage(df),
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

def get_rare_categories(
    series,
    threshold=0.01,
    max_unique=50,
    max_cardinality_pct=30,
):
    unique = series.nunique(dropna=False)
    cardinality_pct = unique / len(series) * 100

    if (
        unique > max_unique
        or cardinality_pct > max_cardinality_pct
    ):
        return "No rare categories - too many unique values"

    freq = (
        series.fillna("MISSING")
        .value_counts(normalize=True)
    )

    rare = freq[freq < threshold]

    if rare.empty:
        return "No rare categories"

    return pd.DataFrame({
        "count": (
            series.fillna("MISSING")
            .value_counts()
            .loc[rare.index]
        ),
        "percent": (rare * 100).round(2),
    })

def run_eda(df,
            target=None,
            display=True,
            max_display=20,
            method="pearson",
            head=5,
            random=10,
            tail=5,
            random_state=42,
            display_advanced=False,
):
    sample = sample_dataframe(df, head=head, random=random, tail=tail, random_state=random_state)
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
        "sample": sample,
        "categorical_rare": {
            col: get_rare_categories(df[col])
            for col in categorical_cols
        },
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

        for sample_type, sample_df in sample.items():
            print(f"\n=== Sample: {sample_type} ===")
            print(sample_df)

        if display_advanced:
            print("\n=== Advanced Summary ===")
            print('='*60)
            print("\nRare categories in categorical columns:")
            for col in categorical_cols:
                if col in eda_results["categorical_rare"] and eda_results["categorical_rare"][col] is not None:
                    print(f"\n--- {col} ---")
                    print(eda_results["categorical_rare"][col])
    return eda_results


# ================================== WIP Functions ==================================
# WIP: Add function to show examples of rows containing specific values in a column
def show_examples(df, column, values=None, n=5, contains=True, random_state=42):
    series = df[column]

    if values is None:
        return df.sample(n=min(n, len(df)), random_state=random_state)

    if not isinstance(values, list):
        values = [values]

    if contains and series.dtype == "object":
        pattern = "|".join(map(str, values))
        mask = series.fillna("").astype(str).str.contains(pattern, case=False, regex=True)
    else:
        mask = series.isin(values)

    return df[mask].head(n)

# WIP: Add function to summarize text length in a column
def summarize_text_length(series):
    text = series.dropna().astype(str)

    return pd.Series({
        "min_len": text.str.len().min(),
        "mean_len": round(text.str.len().mean(), 2),
        "median_len": text.str.len().median(),
        "max_len": text.str.len().max(),
    })
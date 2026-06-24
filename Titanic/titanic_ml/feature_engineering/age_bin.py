import pandas as pd
def add_age_bin(df):
    df = df.copy()
    bins = [0, 14, 35, 60, 100]
    labels = ['0', '2', '3', '1']
    df['Age_bin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df
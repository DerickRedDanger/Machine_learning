def add_sex_pclass(df):
    df = df.copy()

    df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
    return df
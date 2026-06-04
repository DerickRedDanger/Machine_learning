def add_family_features(df):
    df = df.copy()

    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Alone'] = (df['Family_size'] == 1).astype(int)
    
    return df
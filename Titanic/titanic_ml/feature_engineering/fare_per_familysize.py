from titanic_ml.feature_engineering import add_family_features
def add_fare_per_familysize(df):
    df = df.copy()

    if "FamilySize" not in df.columns:
        df = add_family_features(df)

    df['Fare/FamilySize'] = df['Fare']/df['FamilySize']
    return df
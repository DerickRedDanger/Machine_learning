def add_fare_per_familysize(df):
    df = df.copy()
    df['Fare/FamilySize'] = df['Fare']/df['FamilySize']
    return df
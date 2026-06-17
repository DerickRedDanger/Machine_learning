def add_has_cabin(df):
    df = df.copy()

    df['Has_Cabin'] = df['Deck'].notnull().astype(int)
    # Filling Deck's Nan with 'None'


    return df

def add_deck(df):
    df = df.copy()
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('None')
    return df

def add_full_deck(df):
    df = add_deck(df)
    df = add_has_cabin(df)

    return df
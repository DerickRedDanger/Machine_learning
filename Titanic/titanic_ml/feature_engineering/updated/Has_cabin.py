
from titanic_ml.common.utils.sklearn_compatible.HasFeature import HasFeatureTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def has_cabin_transformer():
    return HasFeatureTransformer(
        source_col='Cabin',
        output_col='Has_Cabin'
    )

HAS_CABIN = {
    "id": "has_cabin",
    "transformer": has_cabin_transformer,
    "requires": ["Cabin"],
    "produces": ["Has_Cabin"],
}


class HasCabinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Has_Cabin'] = X['Cabin'].notnull().astype(int)
        return X


def add_has_cabin(df):
    df = df.copy()

    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    # Filling Deck's Nan with 'None'


    return df

def add_deck(df):
    df = df.copy()
    df['Deck'] = df['Cabin'].str[0]

# df['Deck'].value_counts(dropna=False):
#     Deck
# NaN    687
# C       59
# B       47
# D       33
# E       32
# A       15
# F       13
# G        4
# T        1

    df['Deck'] = df['Deck'].fillna('None')

    # Fusing G and T into 'Rare' due to low counts
    rare_decks = [
    'G', 'T'
    ] 
    df['Deck'] = df['Deck'].replace(rare_decks, 'Rare')
    return df

def add_full_cabin_features(df):
    df = add_deck(df)
    df = add_has_cabin(df)

    return df
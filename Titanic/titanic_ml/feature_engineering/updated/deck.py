from sklearn.base import BaseEstimator, TransformerMixin

class AddDeckTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if 'Cabin' not in X.columns:
            raise ValueError("Source column 'Cabin' not found.")
        return self

    def transform(self, X):
        X = X.copy()
        X['Deck'] = X['Cabin'].str[0]

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

        X['Deck'] = X['Deck'].fillna('None')

        # Fusing G and T into 'Rare' due to low counts
        rare_decks = ['G', 'T']

        X['Deck'] = X['Deck'].replace(rare_decks, 'Rare')
        return X

ADD_DECK = {
    "id": "add_deck",
    "stage": "cv",
    "transformer": AddDeckTransformer,
    "requires": ["Cabin"],
    "produces": ["Deck"],
}

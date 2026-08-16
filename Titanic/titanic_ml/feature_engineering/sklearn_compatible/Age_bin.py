from titanic_ml.sklearn_compatible.BinTransformer import BinTransformer

def age_bin_transformer():
    return BinTransformer(
        source_col="Age",
        output_col="Age_bin",
        bins=[0, 14, 35, 60, 100],
        labels=["0", "2", "3", "1"],
        right=False,
    )

AGE_BIN = {
    "id": "age_bin",
    "transformer": age_bin_transformer,
    "requires": ["Age"],
    "produces": ["Age_bin"],
    "owns": ["Age_bin"],
}
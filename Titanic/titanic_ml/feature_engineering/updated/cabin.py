
from titanic_ml.common.utils.sklearn_compatible.HasFeature import HasFeatureTransformer

def has_cabin_transform():
    return HasFeatureTransformer(
        source_col='Cabin',
        output_col='Has_Cabin'
    )

ADD_HAS_CABIN = {
    "id": "has_cabin",
    "transformer": has_cabin_transform,
    "requires": ["Cabin"],
    "produces": ["Has_Cabin"],
}

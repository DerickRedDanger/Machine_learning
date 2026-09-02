from titanic_ml.common.utils.sklearn_compatible.SumColumns import SumColumnsTransformer
from titanic_ml.common.utils.sklearn_compatible.IsOne import IsOneTransformer


def family_feature_transform():
    return SumColumnsTransformer(
        source_cols=["SibSp", "Parch"],
        output_col="FamilySize",
        constant=1,
    )


ADD_FAMILY_SIZE = {
    "id": "family_size",
    "stage": "cv",
    "transformer":family_feature_transform,
    "requires": ["SibSp", "Parch"],
    "produces": ["FamilySize"],
}

def is_alone_transform():
    return IsOneTransformer(
        source_col='FamilySize',
        output_col='IsAlone'
    )

ADD_IS_ALONE = {
    "id": "is_alone",
    "stage": "cv",
    "transformer": is_alone_transform,
    "requires": ["FamilySize"],
    "produces": ["IsAlone"],
}
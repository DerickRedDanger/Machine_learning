from titanic_ml.common.utils.sklearn_compatible.FeatureDividedByFeature import FeatureDividedByFeatureTransformer

def fare_per_familysize_transform():
    return FeatureDividedByFeatureTransformer(
        numerator_col='Fare',
        denominator_col='FamilySize',
        output_col='Fare/FamilySize'
    )

ADD_FARE_PER_FAMILYSIZE = {
    "id": "fare_per_familysize",
    "stage": "cv",
    "transformer": fare_per_familysize_transform,
    "requires": ["Fare", "FamilySize"],
    "produces": ["Fare/FamilySize"],
}

def fare_per_ticket_transform():
    return FeatureDividedByFeatureTransformer(
        numerator_col='Fare',
        denominator_col='TicketGroupSize',
        output_col='Fare/TicketMember'
    )

ADD_FARE_PER_TICKET = {
    "id": "fare_per_ticket",
    "stage": "cv",
    "transformer": fare_per_ticket_transform,
    "requires": ["Fare", "TicketGroupSize"],
    "produces": ["Fare/TicketMember"],
}
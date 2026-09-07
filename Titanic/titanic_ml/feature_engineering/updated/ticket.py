from titanic_ml.common.utils.sklearn_compatible.BatchValueCountTransformer import (
    BatchValueCountTransformer,
)


def ticket_group_size_batch_transform():
    return BatchValueCountTransformer(
        source_col="Ticket",
        output_col="TicketGroupSize",
    )


ADD_TICKET_GROUP_SIZE_BATCH = {
    "id": "ticket_group_size_batch",
    "stage": "cv",
    "transformer": ticket_group_size_batch_transform,
    "requires": ["Ticket"],
    "produces": ["TicketGroupSize"],
    "owns": ["TicketGroupSize"],
}

from titanic_ml.common.utils.sklearn_compatible.FittedValueCountTransformer import FittedValueCountTransformer


def ticket_group_size_fitted_transform():
    return FittedValueCountTransformer(
        source_col="Ticket",
        output_col="TicketGroupSize",
    )


ADD_TICKET_GROUP_SIZE_FITTED = {
    "id": "ticket_group_size_fitted",
    "stage": "cv",
    "transformer": ticket_group_size_fitted_transform,
    "requires": ["Ticket"],
    "produces": ["TicketGroupSize"],
    "owns": ["TicketGroupSize"],
}
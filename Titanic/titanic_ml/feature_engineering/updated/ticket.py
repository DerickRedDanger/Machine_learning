from titanic_ml.common.utils.sklearn_compatible.ValueCountTransformer import (
    ValueCountTransformer,
)


def ticket_group_size_transform():
    return ValueCountTransformer(
        source_col="Ticket",
        output_col="TicketGroupSize",
    )


ADD_TICKET_GROUP_SIZE = {
    "id": "ticket_group_size",
    "stage": "cv",
    "transformer": ticket_group_size_transform,
    "requires": ["Ticket"],
    "produces": ["TicketGroupSize"],
    "owns": ["TicketGroupSize"],
}
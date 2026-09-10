from titanic_ml.common.utils.sklearn_compatible.FittedValueCountTransformer import FittedValueCountTransformer


def ticket_group_size_full_context_transform():
    return FittedValueCountTransformer(
        source_col="Ticket",
        output_col="TicketGroupSize",
    )


ADD_TICKET_GROUP_SIZE_FULL_CONTEXT = {
    "id": "ticket_group_size_full_context",
    "stage": "pre_cv",
    "transformer": ticket_group_size_full_context_transform,
    "requires": ["Ticket"],
    "produces": ["TicketGroupSize"],
    "owns": ["TicketGroupSize"],
    "tags":{"full_context"}
}
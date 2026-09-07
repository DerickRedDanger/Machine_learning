from .ticket import ADD_TICKET_GROUP_SIZE_FULL_CONTEXT

ticket_group_size_full_context = ADD_TICKET_GROUP_SIZE_FULL_CONTEXT

ALL_FEATURES = [
    ticket_group_size_full_context
]

for feature in ALL_FEATURES:
    if feature["stage"] != "pre_cv":
        raise ValueError(f"Pre CV Feature Engineering {feature['id']} has stage {feature['stage']}, expected 'pre_cv'")
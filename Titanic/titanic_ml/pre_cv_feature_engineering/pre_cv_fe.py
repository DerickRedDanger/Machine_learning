

ALL_FEATURES = [
 
]

for feature in ALL_FEATURES:
    if feature["stage"] != "pre_cv":
        raise ValueError(f"Pre CV Feature Engineering {feature['id']} has stage {feature['stage']}, expected 'pre_cv'")
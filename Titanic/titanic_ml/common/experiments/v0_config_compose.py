import copy


def create_config(
    base_config,
    patches,
    raw_features,
):
    config = copy.deepcopy(base_config)

    # Final sklearn-compatible FE pipeline
    feature_pipeline = []

    # Validation state
    available_features = set(raw_features)
    used_transform_ids = set()
    owned_features = {}

    # -----------------------------------------------------
    # 1. Process patches in execution order
    # -----------------------------------------------------

    for patch in patches:

        # ---------------------------------------------
        # Transformations
        # ---------------------------------------------

        for transformation in patch.get(
            "transformations",
            [],
        ):
            transform_id = transformation["id"]

            # Same exact transformation requested twice:
            # safely reuse the first occurrence.
            if transform_id in used_transform_ids:
                continue

            requires = set(
                transformation.get("requires", [])
            )

            missing_requirements = (
                requires - available_features
            )

            if missing_requirements:
                raise ValueError(
                    f"Transformation '{transform_id}' "
                    "requires unavailable features: "
                    f"{sorted(missing_requirements)}"
                )

            # Check competing ownership
            for feature in transformation.get(
                "owns",
                [],
            ):
                if feature in owned_features:
                    previous_owner = (
                        owned_features[feature]
                    )

                    raise ValueError(
                        f"Transformation "
                        f"'{transform_id}' attempts to "
                        f"modify '{feature}', but it is "
                        "already owned by "
                        f"'{previous_owner}'."
                    )

            # Fresh sklearn transformer
            transformer_factory = (
                transformation["transformer"]
            )

            transformer = transformer_factory()

            feature_pipeline.append(
                (transform_id, transformer)
            )

            used_transform_ids.add(
                transform_id
            )

            for feature in transformation.get(
                "owns",
                [],
            ):
                owned_features[feature] = (
                    transform_id
                )

            available_features.update(
                transformation.get(
                    "produces",
                    [],
                )
            )

        # ---------------------------------------------
        # Config modifications
        # ---------------------------------------------

        _apply_add(
            config,
            patch.get("add", {}),
        )

        _apply_remove(
            config,
            patch.get("remove", {}),
        )

        _apply_update(
            config,
            patch.get("update", {}),
        )

    # -----------------------------------------------------
    # 2. Attach resolved FE pipeline
    # -----------------------------------------------------

    config["feature_pipeline"] = (
        feature_pipeline
    )

    # -----------------------------------------------------
    # 3. Validate features actually exposed to model
    # -----------------------------------------------------

    model_features = set(
        features_from_preprocessing(
            config["preprocessing"]
        )
    )

    unavailable_model_features = (
        model_features - available_features
    )

    if unavailable_model_features:
        raise ValueError(
            "Preprocessing requests features that "
            "are not available after feature "
            "engineering: "
            f"{sorted(unavailable_model_features)}"
        )

    # Human-readable derived metadata
    config["features"] = (
        features_from_preprocessing(
            config["preprocessing"]
        )
    )

    return config
# Append Helper
def _append_unique(target_list, values):
    for value in values:
        if value not in target_list:
            target_list.append(value)

# Add Helper
def _apply_add(config, additions):
    for section, changes in additions.items():

        if section not in config:
            config[section] = {}

        for key, value in changes.items():

            if isinstance(value, list):
                current = config[section].setdefault(
                    key,
                    []
                )

                _append_unique(
                    current,
                    value,
                )

            else:
                if key not in config[section]:
                    config[section][key] = value
# Remove Helper
def _apply_remove(config, removals):
    for section, changes in removals.items():

        if section not in config:
            continue

        for key, values in changes.items():

            if key not in config[section]:
                continue

            if not isinstance(
                config[section][key],
                list,
            ):
                raise ValueError(
                    f"Cannot remove list values "
                    f"from non-list config field "
                    f"'{section}.{key}'."
                )

            config[section][key] = [
                item
                for item
                in config[section][key]
                if item not in values
            ]

# Update Helper
def _apply_update(config, updates):
    for section, changes in updates.items():

        if section not in config:
            config[section] = {}

        for key, value in changes.items():
            config[section][key] = value

def features_from_preprocessing(preprocessing):
    feature_groups = [
        "numeric_features",
        "onehot_features",
        "ordinal_features",
    ]

    features = []
    seen = set()

    for group in feature_groups:
        for feature in preprocessing.get(group, []):
            if feature in seen:
                raise ValueError(
                    f"Feature '{feature}' appears in more than one "
                    "preprocessing group."
                )

            seen.add(feature)
            features.append(feature)

    return features